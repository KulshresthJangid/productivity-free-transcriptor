"""Long-audio pipeline: decode -> split on silence -> transcribe -> stream.

Everything lives in a per-request temp dir that is deleted when the request
ends. Nothing is kept except the transcript text sent back to the browser.
"""
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import wave
from pathlib import Path

import numpy as np

import cleanup
from hinglish import to_hinglish

MODEL_SIZE = os.getenv("WHISPER_MODEL", "large-v3")
DEVICE = os.getenv("WHISPER_DEVICE", "auto")            # auto | cpu | cuda
COMPUTE = os.getenv("WHISPER_COMPUTE", "auto")          # auto: float16 on GPU, int8 on CPU
# Models are saved here once and reused on every start (mount it as a volume in Docker).
MODEL_DIR = os.getenv("MODEL_DIR", str(Path(__file__).parent / "models"))
CHUNK_SECONDS = int(os.getenv("CHUNK_SECONDS", "600"))  # target chunk length
TMP_PREFIX = "hinglish-tx-"
SR = 16000

# Roman-script Hinglish seed: nudges Whisper to write Hinglish, not Devanagari.
SEED_PROMPT = (
    "Haan toh aaj hum baat karenge ki kaise ye kaam hota hai. "
    "Dekho, main bol raha hoon ki ye bilkul theek hai, aur English words "
    "jaise meeting, project, update bhi aise hi likhe jaate hain."
)

MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", str(min(8, os.cpu_count() or 2))))
_models = {}
_resolved = None  # (device, compute) once the auto-detect warm-up has run
_model_lock = threading.Lock()
_job_lock = threading.Lock()  # one transcription at a time keeps RAM predictable


def get_model(workers=1):
    """workers > 1 builds a model that can run that many chunks at once (weights are shared)."""
    global _resolved
    with _model_lock:
        if workers in _models:
            return _models[workers]
        from faster_whisper import WhisperModel
        threads = max(1, int(os.getenv("CPU_THREADS", os.cpu_count() or 2)) // workers)

        def build(dev, comp):
            kw = dict(device=dev, compute_type=comp, num_workers=workers,
                      cpu_threads=threads, download_root=MODEL_DIR)
            try:  # already saved? load straight from disk, no network check
                return WhisperModel(MODEL_SIZE, local_files_only=True, **kw)
            except Exception:
                print(f"Downloading model '{MODEL_SIZE}' to {MODEL_DIR} (one time)...")
                return WhisperModel(MODEL_SIZE, **kw)

        if _resolved is None and DEVICE == "auto":
            # "auto" can pick a GPU whose CUDA libraries are missing; the failure only
            # shows up on the first encode, so warm up once and fall back to CPU.
            m = build("auto", COMPUTE)
            try:
                list(m.transcribe(np.zeros(SR, dtype=np.float32))[0])
                _resolved = ("auto", COMPUTE)
            except RuntimeError:
                print("GPU not usable, falling back to CPU (int8).")
                _resolved = ("cpu", "int8")
                m = build(*_resolved)
        else:
            _resolved = _resolved or (DEVICE, COMPUTE)
            m = build(*_resolved)
        _models[workers] = m
        return m


def sweep_stale(max_age=3600):
    """Remove temp dirs left behind by a crash or killed request."""
    root = Path(tempfile.gettempdir())
    for p in root.glob(TMP_PREFIX + "*"):
        try:
            if time.time() - p.stat().st_mtime > max_age:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def new_workdir() -> Path:
    return Path(tempfile.mkdtemp(prefix=TMP_PREFIX))


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, errors="replace")


def probe_duration(src: Path):
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
              "-of", "default=nw=1:nk=1", str(src)])
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def decode_to_wav(src: Path, dst: Path, denoise=True):
    """Any audio/video container -> 16 kHz mono PCM wav (optionally noise-reduced).
    Generator: yields percent done (0-100) while ffmpeg works."""
    total = probe_duration(src)
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-vn", "-map", "0:a:0?", "-ac", "1"]
    if denoise:
        # cut rumble and hiss outside the speech band, then spectral noise reduction
        cmd += ["-af", "highpass=f=80,lowpass=f=7500,afftdn=nf=-25"]
    cmd += ["-ar", str(SR), "-c:a", "pcm_s16le", "-progress", "pipe:1", "-nostats", str(dst)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    try:
        for line in proc.stdout:
            if total and line.startswith("out_time_us="):
                try:
                    yield min(99.0, int(line.split("=")[1]) / 1e6 / total * 100)
                except ValueError:
                    pass
        proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
    if proc.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
        raise ValueError("Could not read audio from this file (no audio track or unsupported format).")


def find_silences(wav: Path):
    r = _run(["ffmpeg", "-nostdin", "-i", str(wav), "-af",
              "silencedetect=noise=-35dB:d=0.4", "-f", "null", "-"])
    starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r.stderr)]
    return list(zip(starts, ends))


def plan_chunks(duration, silences):
    """Cut near CHUNK_SECONDS, but in the middle of a pause so no word is split."""
    chunks, start = [], 0.0
    mids = [(s + e) / 2 for s, e in silences]
    while duration - start > CHUNK_SECONDS * 1.2:
        lo, hi = start + CHUNK_SECONDS * 0.7, start + CHUNK_SECONDS * 1.2
        cands = [m for m in mids if lo <= m <= hi]
        target = start + CHUNK_SECONDS
        cut = min(cands, key=lambda m: abs(m - target)) if cands else target
        chunks.append((start, cut))
        start = cut
    chunks.append((start, duration))
    return chunks


def read_slice(wav_path: Path, t0, t1):
    with wave.open(str(wav_path), "rb") as w:
        w.setpos(int(t0 * SR))
        raw = w.readframes(int((t1 - t0) * SR))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _transcribe_chunk(model, wav, t0, t1, lang, hinglish, clean, context, repeats):
    """Yield (segment_dict, text) for one chunk."""
    audio = read_slice(wav, t0, t1)
    if hinglish:
        prompt = (SEED_PROMPT + " " + context).strip()
    else:
        prompt = context or None
    segs, _ = model.transcribe(
        audio, language=lang, task="transcribe", beam_size=5,
        initial_prompt=prompt, vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=False,
        temperature=[0.0, 0.2, 0.4], compression_ratio_threshold=2.2,
        no_speech_threshold=0.6,
    )
    for s in segs:
        text = s.text.strip()
        if hinglish:
            text = to_hinglish(text)
        if clean:
            text = cleanup.clean_text(text)
            if cleanup.is_junk(s, text) or not repeats.keep(text):
                continue
        if not text:
            continue
        yield {"type": "segment", "start": t0 + s.start, "end": t0 + s.end, "text": text}


def transcribe_stream(src: Path, workdir: Path, language="hi", hinglish=True,
                      denoise=True, clean=True, parallel=False):
    """Yield event dicts: status, info, segment, progress, done."""
    with _job_lock:
        wav = workdir / "audio.wav"
        yield {"type": "status", "message": "Reading audio..."}
        for pct in decode_to_wav(src, wav, denoise):
            yield {"type": "progress", "pct": pct}
        src.unlink(missing_ok=True)  # original upload no longer needed

        with wave.open(str(wav), "rb") as w:
            duration = w.getnframes() / SR
        if duration < 0.1:
            raise ValueError("The file contains no audio.")
        yield {"type": "status", "message": "Finding pauses to split on..."}
        chunks = plan_chunks(duration, find_silences(wav))
        yield {"type": "info", "duration": duration, "chunks": len(chunks)}

        workers = min(len(chunks), MAX_PARALLEL) if parallel else 1
        yield {"type": "status", "message": f"Loading model ({MODEL_SIZE}); first run downloads it..."}
        model = get_model(workers)
        lang = None if language == "auto" else language

        if workers <= 1:
            context, repeats = "", cleanup.RepeatFilter()
            for idx, (t0, t1) in enumerate(chunks):
                yield {"type": "status", "message": f"Transcribing part {idx + 1} of {len(chunks)}..."}
                texts = []
                for ev in _transcribe_chunk(model, wav, t0, t1, lang, hinglish, clean, context, repeats):
                    texts.append(ev["text"])
                    yield ev
                    yield {"type": "progress", "pct": min(99.0, ev["end"] / duration * 100)}
                context = " ".join(texts)[-200:]
        else:
            yield {"type": "status", "message": f"TURBO: {workers} parts at once..."}
            import queue
            from concurrent.futures import ThreadPoolExecutor
            n = len(chunks)
            q = queue.Queue()
            bufs, done, pcts = [[] for _ in chunks], [False] * n, [0.0] * n
            lens = [t1 - t0 for t0, t1 in chunks]

            def work(i, t0, t1):
                try:
                    # Parts are independent here (no text carried between them).
                    for ev in _transcribe_chunk(model, wav, t0, t1, lang, hinglish, clean, "",
                                                cleanup.RepeatFilter()):
                        q.put(("seg", i, ev))
                    q.put(("done", i, None))
                except Exception as e:  # surface in the main loop
                    q.put(("err", i, e))

            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                for i, (t0, t1) in enumerate(chunks):
                    pool.submit(work, i, t0, t1)
                head = 0
                while head < n:
                    try:
                        kind, i, x = q.get(timeout=1)
                    except queue.Empty:
                        yield {"type": "ping"}   # also lets us notice a closed connection
                        continue
                    if kind == "err":
                        raise x
                    if kind == "seg":
                        bufs[i].append(x)
                        pcts[i] = min(1.0, (x["end"] - chunks[i][0]) / lens[i])
                    else:
                        done[i], pcts[i] = True, 1.0
                    # text is shown in order: stream the earliest unfinished part live
                    while head < n:
                        for ev in bufs[head]:
                            yield ev
                        bufs[head].clear()
                        if done[head]:
                            head += 1
                        else:
                            break
                    yield {"type": "progress", "pct": min(99.0, sum(p * l for p, l in zip(pcts, lens)) / duration * 100)}
                    yield {"type": "parts", "pcts": [round(p * 100) for p in pcts]}
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        yield {"type": "done"}

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
        threads = max(1, (os.cpu_count() or 2) // workers)

        def build(dev, comp):
            return WhisperModel(MODEL_SIZE, device=dev, compute_type=comp,
                                num_workers=workers, cpu_threads=threads)

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


def decode_to_wav(src: Path, dst: Path, denoise=True):
    """Any audio/video container -> 16 kHz mono PCM wav (optionally noise-reduced)."""
    cmd = ["ffmpeg", "-nostdin", "-y", "-i", str(src), "-vn", "-map", "0:a:0?", "-ac", "1"]
    if denoise:
        # cut rumble and hiss outside the speech band, then spectral noise reduction
        cmd += ["-af", "highpass=f=80,lowpass=f=7500,afftdn=nf=-25"]
    r = _run(cmd + ["-ar", str(SR), "-c:a", "pcm_s16le", str(dst)])
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 100:
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
        decode_to_wav(src, wav, denoise)
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
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                # Parts are independent here (no text carried between them), each with its own repeat filter.
                futures = [pool.submit(lambda t0=t0, t1=t1: list(_transcribe_chunk(
                    model, wav, t0, t1, lang, hinglish, clean, "", cleanup.RepeatFilter())))
                    for t0, t1 in chunks]
                for idx, fut in enumerate(futures):   # emit in order so the text reads correctly
                    for ev in fut.result():
                        yield ev
                    yield {"type": "progress", "pct": min(99.0, (idx + 1) / len(chunks) * 100)}
                    yield {"type": "status", "message": f"TURBO: {idx + 1} of {len(chunks)} parts done..."}
            finally:
                pool.shutdown(wait=False, cancel_futures=True)
        yield {"type": "done"}

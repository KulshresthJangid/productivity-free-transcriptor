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

_model = None
_model_lock = threading.Lock()
_job_lock = threading.Lock()  # one transcription at a time keeps RAM predictable


def get_model():
    global _model
    with _model_lock:
        if _model is None:
            from faster_whisper import WhisperModel
            m = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE)
            if DEVICE == "auto":
                # "auto" can pick a GPU whose CUDA libraries are missing; the failure only
                # shows up on the first encode, so warm up once and fall back to CPU.
                try:
                    list(m.transcribe(np.zeros(SR, dtype=np.float32))[0])
                except RuntimeError:
                    print("GPU not usable, falling back to CPU (int8).")
                    m = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
            _model = m
        return _model


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


def decode_to_wav(src: Path, dst: Path):
    """Any audio/video container -> 16 kHz mono PCM wav."""
    r = _run(["ffmpeg", "-nostdin", "-y", "-i", str(src), "-vn", "-map", "0:a:0?",
              "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)])
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


def transcribe_stream(src: Path, workdir: Path, language="hi", hinglish=True):
    """Yield event dicts: status, info, segment, progress, done."""
    with _job_lock:
        wav = workdir / "audio.wav"
        yield {"type": "status", "message": "Reading audio..."}
        decode_to_wav(src, wav)
        src.unlink(missing_ok=True)  # original upload no longer needed

        with wave.open(str(wav), "rb") as w:
            duration = w.getnframes() / SR
        if duration < 0.1:
            raise ValueError("The file contains no audio.")
        yield {"type": "status", "message": "Finding pauses to split on..."}
        chunks = plan_chunks(duration, find_silences(wav))
        yield {"type": "info", "duration": duration, "chunks": len(chunks)}

        yield {"type": "status", "message": f"Loading model ({MODEL_SIZE}); first run downloads it..."}
        model = get_model()
        lang = None if language == "auto" else language
        context = ""
        for idx, (t0, t1) in enumerate(chunks):
            yield {"type": "status", "message": f"Transcribing part {idx + 1} of {len(chunks)}..."}
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
            texts = []
            for s in segs:
                text = s.text.strip()
                if hinglish:
                    text = to_hinglish(text)
                if not text:
                    continue
                texts.append(text)
                yield {"type": "segment", "start": t0 + s.start, "end": t0 + s.end, "text": text}
                yield {"type": "progress", "pct": min(99.0, (t0 + s.end) / duration * 100)}
            context = " ".join(texts)[-200:]
            del audio
        yield {"type": "done"}

# Hinglish Transcriptor

Free, self-hosted transcription of any audio/video file into **Hinglish** (Hindi written in Roman letters, English words kept as English).

- **Any format**: anything ffmpeg can read (mp3, wav, m4a, ogg, opus, flac, mp4, mkv, mov, webm, amr, ...).
- **Any length**: the file is decoded, split at natural pauses into ~10 minute parts, and transcribed one part at a time, so a multi-hour recording works in roughly constant memory. Text streams to the page as it is produced.
- **Nothing stored**: no database. The upload and every intermediate file live in a temp folder that is deleted when the request ends (and swept on startup if the server crashed). The transcript exists only in your browser until you download it.
- **Free and local**: runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper) on your machine; no API keys.

## Run

1. Install [Python 3.10+](https://www.python.org/downloads/) and [ffmpeg](https://ffmpeg.org/download.html) (`winget install Gyan.FFmpeg`).
2. Then:

   ```
   python -m venv .venv
   .venv\Scripts\activate        # Linux/Mac: source .venv/bin/activate
   pip install -r requirements.txt
   uvicorn app:app --port 8000
   ```
3. Open http://localhost:8000 and drop a file. The first run downloads the model (about 3 GB for `large-v3`).

## Settings (environment variables)

| Variable | Default | Notes |
|---|---|---|
| `WHISPER_MODEL` | `large-v3` | Most accurate. On CPU-only machines use `medium` or `small` for speed. |
| `WHISPER_DEVICE` | `auto` | `cpu` or `cuda` |
| `WHISPER_COMPUTE` | `auto` | e.g. `float16` (GPU), `int8` (CPU) |
| `CHUNK_SECONDS` | `600` | Target length of each part |

## How Hinglish works

Whisper has no "Hinglish" mode. We transcribe as Hindi with a Roman-script Hinglish prompt (which nudges it to write Latin letters and keep English words), and any Devanagari that still slips through is converted to Roman by `hinglish.py`.

## Honest limits

No speech-to-text is 100% error-free. Accuracy drops with noise, overlapping speakers and heavy accents, and Hinglish spelling is inherently non-standard (`nahi` / `nahin`). `large-v3` on a GPU is the most accurate setup; `large-v3` on CPU is slower than real time.

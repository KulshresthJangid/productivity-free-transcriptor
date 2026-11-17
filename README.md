# Hinglish Transcriptor

Free, self-hosted transcription of any audio/video file into **Hinglish** (Hindi written in Roman letters, English words kept as English), with user accounts.

- **Any format**: anything ffmpeg can read (mp3, wav, m4a, ogg, opus, flac, mp4, mkv, mov, webm, amr, ...).
- **Any length**: the file is decoded, split at natural pauses into ~10 minute parts, and transcribed one part at a time, so a multi-hour recording works in roughly constant memory. Text streams to the page live.
- **Audio is never stored**: uploads and intermediate files live in a per-request temp folder that is deleted when the request ends (and swept on startup after a crash). MongoDB holds only user accounts (bcrypt-hashed passwords) and each user's transcript *text*.
- **Free and local**: runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper); no API keys.

## Stack

FastAPI backend, MongoDB, plain HTML/JS front end, login via an httpOnly JWT cookie.

## Run with Docker (recommended)

1. `copy .env.example .env`, then edit `.env`: set `MONGO_PASSWORD`, and `JWT_SECRET` (32+ random characters; generate with `openssl rand -hex 32`, or in PowerShell: `-join ((48..57)+(97..102) | Get-Random -Count 64 | % {[char]$_})`).
2. `docker compose up -d --build`
3. Open http://localhost:8000, sign up, drop a file.

The model (about 3 GB for `large-v3`) is downloaded once, when the app first starts, into the `models/` folder next to the code, and reused on every later start, rebuild or `docker compose down -v`. It then also works offline. Set `MODEL_DIR` to store it elsewhere, or `PRELOAD_MODEL=false` to download on first upload instead. On a CPU-only machine set `WHISPER_MODEL=small` or `medium` in `.env`.

Useful commands: `docker compose logs -f app`, `docker compose down` (keeps data), `docker compose down -v` (also wipes users and transcripts).

## Run without Docker

Needs Python 3.10+, ffmpeg and a MongoDB. Start only the database with `docker compose up -d mongo` (add a `ports: ["27017:27017"]` entry to the mongo service first), then:

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set JWT_SECRET=<32+ random chars>
set MONGO_URI=mongodb://<user>:<password>@localhost:27017/?authSource=admin
uvicorn app:app --port 8000
```

## API

| Method | Path | Notes |
|---|---|---|
| POST | `/api/register`, `/api/login` | JSON `{email, password}`, sets session cookie |
| POST | `/api/logout`, GET `/api/me` | |
| POST | `/transcribe` | multipart `file`, `language`, `hinglish`; streams NDJSON events |
| GET | `/api/transcripts`, `/api/transcripts/{id}` | your history only |
| DELETE | `/api/transcripts/{id}` | |

## Deploying publicly

Put it behind HTTPS (Caddy/nginx) and set `COOKIE_SECURE=true` on the app. Anyone who can register can use your CPU/GPU, so add invite-only signup or rate limits before exposing it.

## Honest limits

No speech-to-text is 100% error-free. Accuracy drops with noise, overlapping speakers and heavy accents, and Hinglish spelling is non-standard (`nahi` / `nahin`). `large-v3` on a GPU is most accurate; on CPU it is slower than real time.

# Hinglish Transcriptor

Free, self-hosted transcription of any audio/video file into **Hinglish** (Hindi written in Roman letters, English words kept as English), with user accounts.

- **Any format**: anything ffmpeg can read (mp3, wav, m4a, ogg, opus, flac, mp4, mkv, mov, webm, amr, ...).
- **Any length**: the file is decoded, split at natural pauses into ~10 minute parts, and transcribed one part at a time, so a multi-hour recording works in roughly constant memory. Text streams to the page live.
- **Audio is never stored**: uploads and intermediate files live in a per-request temp folder that is deleted when the request ends (and swept on startup after a crash). MongoDB holds only user accounts (bcrypt-hashed passwords) and each user's transcript *text*.
- **Free and local**: runs [faster-whisper](https://github.com/SYSTRAN/faster-whisper); no API keys.

## Stack

FastAPI backend, MongoDB, plain HTML/JS front end, login via an httpOnly JWT cookie.

## Run with Docker (recommended)

1. `cp .env.example .env`, then edit `.env`: set `MONGO_PASSWORD`, `JWT_SECRET` (32+ random characters, e.g. `openssl rand -hex 32`) and `DATA_DIR` (the mount point of your big disk, e.g. `/mnt/data1tb`).
2. `./deploy.sh`
3. Open http://localhost:8000, sign up, drop a file.

`deploy.sh` refuses to run if `DATA_DIR` is on the same disk as `/`, so nothing large lands on the system SSD. Under `DATA_DIR` it creates:

| Folder | Holds |
|---|---|
| `models/` | the Whisper model (about 3 GB for `large-v3`), downloaded once and reused on every restart, rebuild or `down -v`; works offline afterwards |
| `mongo/` | the database (users, transcript text) |
| `tmp/` | uploads and temporary audio while a file is processed (deleted right after) |

Docker's own image layers (about 1-2 GB) stay in `/var/lib/docker`. To move those too, set `"data-root"` in `/etc/docker/daemon.json` to a folder on the big disk and restart Docker.

On a CPU-only machine set `WHISPER_MODEL=small` or `medium` in `.env`. Set `PRELOAD_MODEL=false` to download on the first upload instead of at startup.

Useful commands: `docker compose logs -f app`, `docker compose down` (keeps data), `docker compose down` then delete `$DATA_DIR/mongo` to wipe users and transcripts.

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
| POST | `/upload/init` | JSON `{filename, size}`; returns `{id, chunk_size, chunks}` (chunked upload, used by the page) |
| PUT | `/upload/{id}/{n}` | raw bytes of piece `n` (20 MB); retry-safe |
| POST | `/transcribe/{id}` | form `language`, `hinglish`, ...; reassembled file, streams NDJSON events |
| DELETE | `/upload/{id}` | cancel and delete an upload |
| GET | `/api/transcripts`, `/api/transcripts/{id}` | your history only |
| DELETE | `/api/transcripts/{id}` | |

## Large files

Cloudflare's free plan rejects request bodies over 100 MB, so the page uploads in 20 MB pieces and the server reassembles them (up to `MAX_UPLOAD_BYTES`, default 4 GB). Abandoned uploads are deleted after 6 hours.

## Deploying publicly

Put it behind HTTPS (Caddy/nginx) and set `COOKIE_SECURE=true` on the app. Anyone who can register can use your CPU/GPU, so add invite-only signup or rate limits before exposing it.

## Honest limits

No speech-to-text is 100% error-free. Accuracy drops with noise, overlapping speakers and heavy accents, and Hinglish spelling is non-standard (`nahi` / `nahin`). `large-v3` on a GPU is most accurate; on CPU it is slower than real time.

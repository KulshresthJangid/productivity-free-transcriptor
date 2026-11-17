import json
import os
import shutil
import tempfile
import time
import uuid
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from pymongo import AsyncMongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import iterate_in_threadpool

import auth
import transcriber

STATIC = Path(__file__).parent / "static"
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "transcriptor")


@asynccontextmanager
async def lifespan(app):
    auth.check_secret()
    transcriber.sweep_stale()
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found on PATH. Install it; nothing works without it.")
    if os.getenv("PRELOAD_MODEL", "true").lower() == "true":
        # download (first time only) and load the model now, so the first upload doesn't wait
        threading.Thread(target=transcriber.get_model, daemon=True).start()
    client = AsyncMongoClient(MONGO_URI)
    app.state.db = client[MONGO_DB]
    await app.state.db.users.create_index("email", unique=True)
    await app.state.db.transcripts.create_index([("user_id", 1), ("created_at", -1)])
    yield
    await client.close()


app = FastAPI(title="Hinglish Transcriptor", lifespan=lifespan)


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


def public(user):
    return {"id": str(user["_id"]), "email": user["email"]}


# ---------- auth ----------
@app.post("/api/register")
async def register(body: Credentials, response: Response):
    db = app.state.db
    doc = {"email": body.email.lower(), "password": auth.hash_password(body.password),
           "created_at": datetime.now(timezone.utc)}
    try:
        res = await db.users.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "An account with this email already exists")
    auth.set_session(response, str(res.inserted_id))
    return {"id": str(res.inserted_id), "email": doc["email"]}


@app.post("/api/login")
async def login(body: Credentials, response: Response):
    user = await app.state.db.users.find_one({"email": body.email.lower()})
    if not user or not auth.verify_password(body.password, user["password"]):
        raise HTTPException(401, "Wrong email or password")
    auth.set_session(response, str(user["_id"]))
    return public(user)


@app.post("/api/logout")
async def logout(response: Response):
    response.delete_cookie(auth.COOKIE)
    return {"ok": True}


@app.get("/api/me")
async def me(user=Depends(auth.current_user)):
    return public(user)


# ---------- saved transcripts (text only; audio is never stored) ----------
@app.get("/api/transcripts")
async def list_transcripts(user=Depends(auth.current_user)):
    cur = app.state.db.transcripts.find({"user_id": user["_id"]}, {"segments": 0}).sort(
        "created_at", DESCENDING).limit(200)
    return [{"id": str(t["_id"]), "title": t["title"], "duration": t.get("duration"),
             "created_at": t["created_at"].isoformat()} async for t in cur]


def _oid(s):
    try:
        return ObjectId(s)
    except Exception:
        raise HTTPException(404, "Not found")


@app.get("/api/transcripts/{tid}")
async def get_transcript(tid: str, user=Depends(auth.current_user)):
    t = await app.state.db.transcripts.find_one({"_id": _oid(tid), "user_id": user["_id"]})
    if not t:
        raise HTTPException(404, "Not found")
    return {"id": str(t["_id"]), "title": t["title"], "segments": t["segments"]}


@app.delete("/api/transcripts/{tid}")
async def delete_transcript(tid: str, user=Depends(auth.current_user)):
    res = await app.state.db.transcripts.delete_one({"_id": _oid(tid), "user_id": user["_id"]})
    if not res.deleted_count:
        raise HTTPException(404, "Not found")
    return {"ok": True}


# ---------- transcription ----------
def stream_transcription(user, title, src, workdir, language, hinglish, denoise, clean, parallel):
    """Runs the pipeline on src and streams NDJSON events; always deletes workdir at the end."""
    async def events():
        segments, duration = [], None
        try:
            async for ev in iterate_in_threadpool(
                    transcriber.transcribe_stream(src, workdir, language, hinglish, denoise, clean, parallel)):
                if ev["type"] == "segment":
                    segments.append({"start": ev["start"], "end": ev["end"], "text": ev["text"]})
                elif ev["type"] == "info":
                    duration = ev["duration"]
                elif ev["type"] == "done" and segments:
                    res = await app.state.db.transcripts.insert_one({
                        "user_id": user["_id"], "title": title, "duration": duration,
                        "segments": segments, "created_at": datetime.now(timezone.utc)})
                    ev = {"type": "done", "id": str(res.inserted_id)}
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # report to the UI instead of a dead stream
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(events(), media_type="application/x-ndjson")


# Small files: one request. (Cloudflare rejects request bodies over 100 MB, so the page
# uses the chunked upload below for anything big.)
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("hi"),
                     hinglish: bool = Form(True), denoise: bool = Form(True),
                     clean: bool = Form(True), parallel: bool = Form(False),
                     user=Depends(auth.current_user)):
    workdir = transcriber.new_workdir()
    src = workdir / "upload.bin"
    try:
        with open(src, "wb") as out:  # stream to disk in 1 MB pieces, never the whole file in RAM
            while block := await file.read(1024 * 1024):
                out.write(block)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    return stream_transcription(user, file.filename or "Untitled", src, workdir,
                                language, hinglish, denoise, clean, parallel)


# ---------- chunked upload (each piece stays far below Cloudflare's 100 MB request cap) ----------
CHUNK_SIZE = 20 * 1024 * 1024
MAX_UPLOAD = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 ** 3)))   # 4 GB
UPLOAD_TTL = 6 * 3600          # abandoned uploads are deleted after this
UPLOADS = {}                   # upload id -> {user, dir, path, size, name, chunks, received, created}


def _drop_upload(uid):
    up = UPLOADS.pop(uid, None)
    if up:
        shutil.rmtree(up["dir"], ignore_errors=True)


def _get_upload(uid, user):
    up = UPLOADS.get(uid)
    if not up or up["user"] != str(user["_id"]):
        raise HTTPException(404, "Upload not found (it may have expired). Start again.")
    return up


class UploadInit(BaseModel):
    filename: str = Field(max_length=300)
    size: int = Field(gt=0)


@app.post("/upload/init")
async def upload_init(body: UploadInit, user=Depends(auth.current_user)):
    if body.size > MAX_UPLOAD:
        raise HTTPException(413, f"File too large (limit {MAX_UPLOAD // 1024 ** 3} GB)")
    now = time.time()
    for uid in [u for u, v in UPLOADS.items() if now - v["created"] > UPLOAD_TTL]:
        _drop_upload(uid)
    uid = uuid.uuid4().hex
    d = Path(tempfile.mkdtemp(prefix=transcriber.TMP_PREFIX + "up-"))
    path = d / "upload.bin"
    with open(path, "wb") as f:
        f.truncate(body.size)       # sparse file; pieces are written at their own offsets
    UPLOADS[uid] = {"user": str(user["_id"]), "dir": d, "path": path, "size": body.size,
                    "name": body.filename, "chunks": -(-body.size // CHUNK_SIZE),
                    "received": set(), "created": now}
    return {"id": uid, "chunk_size": CHUNK_SIZE, "chunks": UPLOADS[uid]["chunks"]}


@app.put("/upload/{uid}/{index}")
async def upload_chunk(uid: str, index: int, request: Request, user=Depends(auth.current_user)):
    up = _get_upload(uid, user)
    if not 0 <= index < up["chunks"]:
        raise HTTPException(400, "Bad chunk number")
    offset = index * CHUNK_SIZE
    expected = min(CHUNK_SIZE, up["size"] - offset)
    written = 0
    with open(up["path"], "r+b") as f:
        f.seek(offset)
        async for block in request.stream():
            written += len(block)
            if written > expected:
                raise HTTPException(400, "Chunk larger than expected")
            f.write(block)
    if written != expected:
        raise HTTPException(400, f"Chunk incomplete ({written} of {expected} bytes)")
    up["received"].add(index)
    return {"received": len(up["received"]), "chunks": up["chunks"]}


@app.delete("/upload/{uid}")
async def upload_cancel(uid: str, user=Depends(auth.current_user)):
    _get_upload(uid, user)
    _drop_upload(uid)
    return {"ok": True}


@app.post("/transcribe/{uid}")
async def transcribe_uploaded(uid: str, language: str = Form("hi"),
                              hinglish: bool = Form(True), denoise: bool = Form(True),
                              clean: bool = Form(True), parallel: bool = Form(False),
                              user=Depends(auth.current_user)):
    up = _get_upload(uid, user)
    if len(up["received"]) != up["chunks"]:
        raise HTTPException(400, f"Upload incomplete ({len(up['received'])} of {up['chunks']} pieces)")
    workdir = transcriber.new_workdir()
    src = workdir / "upload.bin"
    try:
        shutil.move(str(up["path"]), src)   # same disk: an instant rename, no copy
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    name = up["name"]
    _drop_upload(uid)
    return stream_transcription(user, name or "Untitled", src, workdir,
                                language, hinglish, denoise, clean, parallel)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

import asyncio
import os
import shutil
import tempfile
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from bson import ObjectId
from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field
from pymongo import AsyncMongoClient, DESCENDING
from pymongo.errors import DuplicateKeyError
from starlette.concurrency import run_in_threadpool

import auth
import jobs
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
    client = AsyncMongoClient(MONGO_URI)
    app.state.db = client[MONGO_DB]
    await app.state.db.users.create_index("email", unique=True)
    await app.state.db.transcripts.create_index([("user_id", 1), ("created_at", -1)])
    await app.state.db.jobs.create_index([("user_id", 1), ("status", 1)])
    await app.state.db.jobs.create_index([("status", 1), ("created_at", 1)])
    await app.state.db.job_events.create_index([("job_id", 1), ("seq", 1)], unique=True)
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


# ---------- jobs: the web app only enqueues; the worker container does the heavy work ----------
MAX_ACTIVE_PER_USER = int(os.getenv("MAX_ACTIVE_PER_USER", "2"))
MAX_QUEUE = int(os.getenv("MAX_QUEUE", "20"))


class JobOptions(BaseModel):
    language: str = Field("hi", pattern=r"^(auto|[a-z]{2,3})$")
    hinglish: bool = True
    denoise: bool = True
    clean: bool = True
    parallel: bool = False


@app.post("/transcribe/{uid}")
async def start_job(uid: str, opts: JobOptions, user=Depends(auth.current_user)):
    up = _get_upload(uid, user)
    if len(up["received"]) != up["chunks"]:
        raise HTTPException(400, f"Upload incomplete ({len(up['received'])} of {up['chunks']} pieces)")
    db = app.state.db
    if await db.jobs.count_documents(
            {"user_id": user["_id"], "status": {"$in": ["queued", "running"]}}) >= MAX_ACTIVE_PER_USER:
        raise HTTPException(429, f"You already have {MAX_ACTIVE_PER_USER} files waiting or running. Wait for one to finish.")
    if await db.jobs.count_documents({"status": "queued"}) >= MAX_QUEUE:
        raise HTTPException(503, "The queue is full right now. Please try again in a while.")

    oid = ObjectId()
    jobdir = jobs.JOBS_ROOT / str(oid)
    jobdir.mkdir(parents=True)
    try:
        shutil.move(str(up["path"]), jobdir / "upload.bin")   # same disk: an instant rename
    except Exception:
        shutil.rmtree(jobdir, ignore_errors=True)
        raise
    name = up["name"] or "Untitled"
    _drop_upload(uid)
    await db.jobs.insert_one({"_id": oid, "user_id": user["_id"], "title": name, "status": "queued",
                              "options": opts.model_dump(), "attempts": 0,
                              "created_at": datetime.now(timezone.utc)})
    try:
        await run_in_threadpool(jobs.publish, str(oid))
    except Exception as e:
        print("queue publish failed:", repr(e), flush=True)
        await db.jobs.update_one({"_id": oid}, {"$set": {"status": "error"}})
        shutil.rmtree(jobdir, ignore_errors=True)
        raise HTTPException(503, "The job queue is unavailable. Please try again shortly.")
    return {"job_id": str(oid)}


@app.get("/api/jobs/active")
async def active_job(user=Depends(auth.current_user)):
    j = await app.state.db.jobs.find_one(
        {"user_id": user["_id"], "status": {"$in": ["queued", "running"]}}, sort=[("created_at", -1)])
    return {"job": {"id": str(j["_id"]), "title": j["title"]} if j else None}


@app.delete("/api/jobs/{jid}")
async def cancel_job(jid: str, user=Depends(auth.current_user)):
    res = await app.state.db.jobs.find_one_and_update(
        {"_id": _oid(jid), "user_id": user["_id"], "status": {"$in": ["queued", "running"]}},
        {"$set": {"status": "cancelled", "finished_at": datetime.now(timezone.utc)}})
    if res and res["status"] == "queued":   # a running job is stopped by the worker itself
        shutil.rmtree(jobs.JOBS_ROOT / jid, ignore_errors=True)
    return {"ok": True}


@app.websocket("/ws/jobs/{jid}")
async def job_socket(ws: WebSocket, jid: str, after: int = 0):
    """Streams a job's events (replaying from `after`), then live. Reconnect-safe."""
    await ws.accept()
    db = app.state.db
    origin, host = ws.headers.get("origin"), ws.headers.get("host")
    if origin and host and urlparse(origin).netloc != host:
        return await ws.close(4403)
    user = await auth.user_from_token(ws.cookies.get(auth.COOKIE), db)
    if not user:
        return await ws.close(4401)
    try:
        oid = ObjectId(jid)
    except Exception:
        return await ws.close(4404)
    job = await db.jobs.find_one({"_id": oid, "user_id": user["_id"]}, {"status": 1, "created_at": 1})
    if not job:
        return await ws.close(4404)

    last, last_ahead, last_sent = after, None, time.time()
    try:
        while True:
            batch = [d async for d in
                     db.job_events.find({"job_id": oid, "seq": {"$gt": last}}).sort("seq", 1).limit(500)]
            for d in batch:
                await ws.send_json({"seq": d["seq"], **d["ev"]})
            if batch:
                last, last_sent = batch[-1]["seq"], time.time()
            job = await db.jobs.find_one({"_id": oid}, {"status": 1, "created_at": 1})
            if job["status"] == "queued":
                ahead = await db.jobs.count_documents(
                    {"status": "queued", "created_at": {"$lt": job["created_at"]}}) + \
                    await db.jobs.count_documents({"status": "running"})
                if ahead != last_ahead:
                    await ws.send_json({"type": "queue", "ahead": ahead})
                    last_ahead, last_sent = ahead, time.time()
            elif job["status"] in jobs.TERMINAL and not batch:
                await ws.send_json({"type": "end", "status": job["status"]})
                return await ws.close()
            if time.time() - last_sent > 20:   # keep proxies from closing an idle socket
                await ws.send_json({"type": "ping"})
                last_sent = time.time()
            try:   # doubles as a 1 s sleep that notices the browser leaving immediately
                msg = await asyncio.wait_for(ws.receive(), 1.0)
                if msg["type"] == "websocket.disconnect":
                    return
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        return


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

import json
import os
import shutil
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from bson import ObjectId
from fastapi import Depends, FastAPI, File, Form, HTTPException, Response, UploadFile
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
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("hi"),
                     hinglish: bool = Form(True), denoise: bool = Form(True),
                     clean: bool = Form(True), user=Depends(auth.current_user)):
    workdir = transcriber.new_workdir()
    src = workdir / "upload.bin"
    try:
        with open(src, "wb") as out:  # stream to disk in 1 MB pieces, never the whole file in RAM
            while block := await file.read(1024 * 1024):
                out.write(block)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    title = file.filename or "Untitled"

    async def events():
        segments, duration = [], None
        try:
            async for ev in iterate_in_threadpool(
                    transcriber.transcribe_stream(src, workdir, language, hinglish, denoise, clean)):
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


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

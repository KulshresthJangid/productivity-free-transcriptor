import json
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import transcriber

STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_app):
    transcriber.sweep_stale()
    if shutil.which("ffmpeg") is None:
        print("WARNING: ffmpeg not found on PATH. Install it; nothing works without it.")
    yield


app = FastAPI(title="Hinglish Transcriptor", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("hi"),
                     hinglish: bool = Form(True)):
    workdir = transcriber.new_workdir()
    src = workdir / "upload.bin"
    try:
        with open(src, "wb") as out:  # stream to disk in 1 MB pieces, never the whole file in RAM
            while block := await file.read(1024 * 1024):
                out.write(block)
    except Exception:
        shutil.rmtree(workdir, ignore_errors=True)
        raise

    def events():
        try:
            for ev in transcriber.transcribe_stream(src, workdir, language, hinglish):
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # report to the UI instead of a dead stream
            yield json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False) + "\n"
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return StreamingResponse(events(), media_type="application/x-ndjson")


app.mount("/static", StaticFiles(directory=STATIC), name="static")

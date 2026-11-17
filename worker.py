"""Transcription worker: takes ONE job at a time from RabbitMQ and runs it.

The web app only accepts uploads and enqueues; all the heavy CPU work happens here, in a
container with a CPU cap, so a long file can never starve the rest of the server. Progress
and text are written to MongoDB (job_events) and the web app relays them to browsers over a
WebSocket, so a closed tab or dropped connection never loses a job.
"""
import json
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pika
from bson import ObjectId
from pymongo import MongoClient

import jobs
import transcriber

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "transcriptor")
MAX_ATTEMPTS = 2   # a job interrupted by a restart is retried once, then failed (no crash loops)


def now():
    return datetime.now(timezone.utc)


class Cancelled(Exception):
    pass


class Emitter:
    """Buffers events and writes them to Mongo about once a second (progress is coalesced)."""

    def __init__(self, db, oid):
        self.db, self.oid = db, oid
        last = db.job_events.find_one({"job_id": oid}, sort=[("seq", -1)])
        self.seq = last["seq"] if last else 0
        self.pending, self.last_flush = [], time.time()

    def add(self, ev):
        if ev["type"] in ("progress", "parts"):   # only the latest value matters
            self.pending = [p for p in self.pending if p["type"] != ev["type"]]
        self.pending.append(ev)
        if ev["type"] in ("done", "error"):
            self.flush(check_cancel=False)
        else:
            self.tick()

    def tick(self):
        if time.time() - self.last_flush >= 1.0:
            self.flush()

    def flush(self, check_cancel=True):
        if self.pending:
            docs = []
            for ev in self.pending:
                self.seq += 1
                docs.append({"job_id": self.oid, "seq": self.seq, "ev": ev})
            self.db.job_events.insert_many(docs)
            self.pending = []
        self.last_flush = time.time()
        if check_cancel:
            j = self.db.jobs.find_one({"_id": self.oid}, {"status": 1})
            if not j or j["status"] == "cancelled":
                raise Cancelled()


def run_job(db, job_id):
    oid = ObjectId(job_id)
    job = db.jobs.find_one_and_update(
        {"_id": oid, "status": "queued"},
        {"$set": {"status": "running", "started_at": now()}, "$inc": {"attempts": 1}},
        return_document=True)
    if not job:   # cancelled while waiting, or already handled
        shutil.rmtree(jobs.JOBS_ROOT / job_id, ignore_errors=True)
        return
    jobdir = jobs.JOBS_ROOT / job_id
    src, o = jobdir / "upload.bin", job["options"]
    em, final = Emitter(db, oid), "done"
    if job["attempts"] > 1:
        em.add({"type": "reset"})     # browsers clear what they showed and replay from the start
    segments, duration = [], None
    try:
        for ev in transcriber.transcribe_stream(src, jobdir, o["language"], o["hinglish"],
                                                o["denoise"], o["clean"], o["parallel"]):
            if ev["type"] == "ping":
                em.tick()
                continue
            if ev["type"] == "segment":
                segments.append({"start": ev["start"], "end": ev["end"], "text": ev["text"]})
            elif ev["type"] == "info":
                duration = ev["duration"]
            elif ev["type"] == "done" and segments:
                res = db.transcripts.insert_one({
                    "user_id": job["user_id"], "title": job["title"], "duration": duration,
                    "segments": segments, "created_at": now()})
                ev = {"type": "done", "id": str(res.inserted_id)}
            em.add(ev)
    except Cancelled:
        final = "cancelled"
    except Exception as e:
        traceback.print_exc()
        final = "error"
        em.add({"type": "error", "message": str(e) or e.__class__.__name__})
    finally:
        try:
            em.flush(check_cancel=False)
        except Exception:
            traceback.print_exc()
        db.jobs.update_one({"_id": oid}, {"$set": {"status": final, "finished_at": now()}})
        shutil.rmtree(jobdir, ignore_errors=True)
    print(f"job {job_id} -> {final}", flush=True)


def recover(db, ch):
    """After a restart: requeue jobs that were running when we died (once), drop orphans."""
    for j in db.jobs.find({"status": "running"}):
        jid = str(j["_id"])
        if j.get("attempts", 1) < MAX_ATTEMPTS and (jobs.JOBS_ROOT / jid / "upload.bin").exists():
            db.jobs.update_one({"_id": j["_id"]}, {"$set": {"status": "queued"}})
            ch.basic_publish("", jobs.QUEUE, json.dumps({"job_id": jid}),
                             pika.BasicProperties(delivery_mode=2))
            print(f"requeued interrupted job {jid}", flush=True)
        else:
            db.job_events.insert_one({"job_id": j["_id"], "seq": 1 + (db.job_events.count_documents({"job_id": j["_id"]})),
                                      "ev": {"type": "error", "message": "The server restarted while processing. Please upload again."}})
            db.jobs.update_one({"_id": j["_id"]}, {"$set": {"status": "error", "finished_at": now()}})
            shutil.rmtree(jobs.JOBS_ROOT / jid, ignore_errors=True)
    if jobs.JOBS_ROOT.exists():   # leftover folders of finished jobs
        for d in jobs.JOBS_ROOT.iterdir():
            try:
                st = db.jobs.find_one({"_id": ObjectId(d.name)}, {"status": 1})
            except Exception:
                st = None
            if not st or st["status"] in jobs.TERMINAL:
                shutil.rmtree(d, ignore_errors=True)


def main():
    db = MongoClient(MONGO_URI)[MONGO_DB]
    jobs.JOBS_ROOT.mkdir(parents=True, exist_ok=True)
    transcriber.sweep_stale()
    if os.getenv("PRELOAD_MODEL", "true").lower() == "true":
        transcriber.get_model()   # download (first time only) and load before the first job
    print("worker ready", flush=True)
    t = None
    while True:
        try:
            conn = jobs.connect()
            ch = conn.channel()
            jobs.declare(ch)
            recover(db, ch)
            while True:
                # basic_get (not a long-lived consumer): nothing is prefetched, so a second job
                # is never held while one is running. The message is acked right away; MongoDB is
                # the source of truth for job state, and recover() handles a crash mid-job.
                # (Acking early also avoids RabbitMQ's 30-minute consumer_timeout on long files.)
                method, _, body = ch.basic_get(jobs.QUEUE, auto_ack=False)
                if method is None:
                    conn.sleep(2)
                    continue
                ch.basic_ack(method.delivery_tag)
                job_id = json.loads(body)["job_id"]
                t = threading.Thread(target=run_job, args=(db, job_id), daemon=True)
                t.start()
                while t.is_alive():
                    conn.process_data_events(time_limit=1)   # keeps AMQP heartbeats flowing
        except (pika.exceptions.AMQPError, OSError) as e:
            print(f"RabbitMQ connection problem: {e!r}; retrying in 5s", flush=True)
            if t is not None and t.is_alive():
                t.join()          # never start a second job while one is still running
            time.sleep(5)


if __name__ == "__main__":
    main()

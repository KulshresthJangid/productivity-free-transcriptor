"""Shared by the web app and the worker: where job files live and how jobs are queued (RabbitMQ)."""
import json
import os
import tempfile
from pathlib import Path

import pika

RABBIT_URL = os.getenv("RABBIT_URL", "")
QUEUE = "transcribe"
# Same folder in both containers (bind-mounted TMPDIR): the web app moves the finished upload
# into jobs/<id>/upload.bin and the worker picks it up from there.
JOBS_ROOT = Path(tempfile.gettempdir()) / "jobs"
TERMINAL = ("done", "error", "cancelled")


def connect():
    p = pika.URLParameters(RABBIT_URL)
    p.connection_attempts, p.retry_delay, p.socket_timeout = 3, 2, 10
    return pika.BlockingConnection(p)


def declare(channel):
    channel.queue_declare(QUEUE, durable=True)   # survives a RabbitMQ restart


def publish(job_id: str):
    """Persistent, confirmed publish. Raises if the broker did not accept the message."""
    conn = connect()
    try:
        ch = conn.channel()
        declare(ch)
        ch.confirm_delivery()
        ch.basic_publish("", QUEUE, json.dumps({"job_id": job_id}),
                         pika.BasicProperties(delivery_mode=2, content_type="application/json"),
                         mandatory=True)
    finally:
        conn.close()

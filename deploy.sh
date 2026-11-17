#!/usr/bin/env bash
# Deploys the transcriptor so that everything big lives on the large data disk,
# never on the system SSD:
#   $DATA_DIR/models  - Whisper model (downloaded once)
#   $DATA_DIR/tmp     - uploads and temporary audio while a file is being processed
#
# Usage:   DATA_DIR=/mnt/data1tb ./deploy.sh
#   or set DATA_DIR in .env.   Set ALLOW_SYSTEM_DISK=1 to skip the safety check (not recommended).
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] || { echo "ERROR: .env not found. Run: cp .env.example .env  and fill it in."; exit 1; }

# DATA_DIR: environment wins, then .env
if [ -z "${DATA_DIR:-}" ]; then
  DATA_DIR="$(grep -E '^DATA_DIR=' .env | tail -n1 | cut -d= -f2- | tr -d '\r"' || true)"
fi
[ -n "${DATA_DIR:-}" ] || { echo "ERROR: DATA_DIR is not set. Example: DATA_DIR=/mnt/data1tb ./deploy.sh"; exit 1; }
DATA_DIR="${DATA_DIR%/}"

# --- make sure this really is the big disk, not the SSD ---
[ -d "$DATA_DIR" ] || { echo "ERROR: $DATA_DIR does not exist. Is the 1 TB disk mounted?"; exit 1; }
if [ "${ALLOW_SYSTEM_DISK:-0}" != "1" ]; then
  if [ "$(stat -c %d "$DATA_DIR")" = "$(stat -c %d /)" ]; then
    echo "ERROR: $DATA_DIR is on the same disk as / (the system SSD)."
    echo "       Mount the 1 TB disk and point DATA_DIR at it (check with: lsblk -f ; df -h)."
    exit 1
  fi
fi
AVAIL_GB=$(df -BG --output=avail "$DATA_DIR" | tail -n1 | tr -dc '0-9')
[ "$AVAIL_GB" -ge 20 ] || { echo "ERROR: only ${AVAIL_GB} GB free on $DATA_DIR (need at least 20)."; exit 1; }
echo "Using $DATA_DIR (${AVAIL_GB} GB free) for models and temp files."

# --- folders ---
mkdir -p "$DATA_DIR/models" "$DATA_DIR/tmp"
chmod 1777 "$DATA_DIR/tmp"

# remember the choice for later `docker compose` commands
if grep -qE '^DATA_DIR=' .env; then
  sed -i "s#^DATA_DIR=.*#DATA_DIR=$DATA_DIR#" .env
else
  printf '\nDATA_DIR=%s\n' "$DATA_DIR" >> .env
fi

# --- start ---
docker compose up -d --build

echo "Waiting for the app..."
for _ in $(seq 1 60); do
  if curl -fs -o /dev/null http://localhost:8000/; then
    echo "Up: http://localhost:8000"
    echo "The first start downloads the model into $DATA_DIR/models (follow: docker compose logs -f app)."
    exit 0
  fi
  sleep 2
done
echo "App did not answer in 2 minutes. Check: docker compose logs app"
exit 1

#!/usr/bin/env bash
# One-time: gives the transcriptor its own RabbitMQ vhost + user on the EXISTING `rabbitmq`
# container (no root credentials needed: rabbitmqctl authenticates inside the container),
# then writes RABBIT_URL into .env. Safe to re-run (it resets the user's password).
# Run on the server, from the repo folder:   bash scripts/setup-rabbitmq.sh
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] || cp .env.example .env
CONTAINER="${RABBIT_CONTAINER:-rabbitmq}"
VHOST=transcriptor
USER_NAME=transcriptor
PW=$(openssl rand -hex 24)   # hex only, so it is URL-safe

docker exec "$CONTAINER" rabbitmqctl list_vhosts -q | grep -qx "$VHOST" \
  || docker exec "$CONTAINER" rabbitmqctl add_vhost "$VHOST"
if docker exec "$CONTAINER" rabbitmqctl list_users -q | cut -f1 | grep -qx "$USER_NAME"; then
  docker exec "$CONTAINER" rabbitmqctl change_password "$USER_NAME" "$PW"
else
  docker exec "$CONTAINER" rabbitmqctl add_user "$USER_NAME" "$PW"
fi
docker exec "$CONTAINER" rabbitmqctl set_permissions -p "$VHOST" "$USER_NAME" ".*" ".*" ".*"

sed -i '/^RABBIT_URL=/d' .env
echo "RABBIT_URL=amqp://${USER_NAME}:${PW}@host.docker.internal:5672/${VHOST}" >> .env
chmod 600 .env
echo "RABBIT_URL written to .env (vhost '$VHOST', user '$USER_NAME'). Password not shown."

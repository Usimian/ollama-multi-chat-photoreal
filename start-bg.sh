#!/usr/bin/env bash
# Start the chat server + Caddy TLS proxy in the background.
# PIDs in /tmp/chatapp.pid and /tmp/chatapp-caddy.pid, logs in /tmp/chatapp.log.
cd "$(dirname "$0")"

if [ -f /tmp/chatapp.pid ] && kill -0 "$(cat /tmp/chatapp.pid)" 2>/dev/null; then
  echo "already running (pid $(cat /tmp/chatapp.pid))"
  exit 1
fi

export TLS_HOST="$(hostname).local"

nohup .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 \
  > /tmp/chatapp.log 2>&1 &
echo $! > /tmp/chatapp.pid

nohup caddy run --config ./Caddyfile --adapter caddyfile \
  >> /tmp/chatapp.log 2>&1 &
echo $! > /tmp/chatapp-caddy.pid

sleep 1
echo "started — https://${TLS_HOST}:8443 — logs: /tmp/chatapp.log"

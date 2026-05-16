#!/usr/bin/env bash
# Start the chat server in the background. PID is written to /tmp/chatapp.pid,
# logs to /tmp/chatapp.log.
cd "$(dirname "$0")"

if [ -f /tmp/chatapp.pid ] && kill -0 "$(cat /tmp/chatapp.pid)" 2>/dev/null; then
  echo "already running (pid $(cat /tmp/chatapp.pid))"
  exit 1
fi

nohup .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 \
  > /tmp/chatapp.log 2>&1 &
echo $! > /tmp/chatapp.pid
sleep 1
echo "started pid $(cat /tmp/chatapp.pid) — http://127.0.0.1:8765 — logs: /tmp/chatapp.log"

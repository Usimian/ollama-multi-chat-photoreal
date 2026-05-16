#!/usr/bin/env bash
# Stop the background chat server.
if [ ! -f /tmp/chatapp.pid ]; then
  echo "no pid file — not running?"
  exit 0
fi

pid=$(cat /tmp/chatapp.pid)
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  echo "stopped pid $pid"
else
  echo "pid $pid not running"
fi
rm -f /tmp/chatapp.pid

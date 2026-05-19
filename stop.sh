#!/usr/bin/env bash
# Stop the chat server and Caddy proxy (started by either start.sh or start-bg.sh).
stopped=0
for f in /tmp/chatapp.pid /tmp/chatapp-caddy.pid; do
  [ -f "$f" ] || continue
  pid=$(cat "$f")
  if kill -0 "$pid" 2>/dev/null; then
    kill "$pid" && echo "stopped pid $pid ($f)"
    stopped=1
  fi
  rm -f "$f"
done
[ $stopped -eq 0 ] && echo "nothing to stop"
exit 0

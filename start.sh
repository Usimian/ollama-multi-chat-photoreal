#!/usr/bin/env bash
# Start the chat server (127.0.0.1:8765) + Caddy TLS proxy (:8443) in the
# foreground. Ctrl+C or ./stop.sh stops both.
cd "$(dirname "$0")"

export TLS_HOST="$(hostname).local"
echo "LAN URL:  https://${TLS_HOST}:8443  (accept the self-signed cert on first visit)"
echo "Local:    http://127.0.0.1:8765"

.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 &
server_pid=$!
echo $server_pid > /tmp/chatapp.pid

caddy run --config ./Caddyfile --adapter caddyfile &
caddy_pid=$!
echo $caddy_pid > /tmp/chatapp-caddy.pid

trap 'kill $server_pid $caddy_pid 2>/dev/null; wait 2>/dev/null; rm -f /tmp/chatapp.pid /tmp/chatapp-caddy.pid; exit 0' INT TERM

wait

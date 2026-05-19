#!/usr/bin/env bash
# Start the chat server (127.0.0.1:8765) and a TLS reverse proxy (:8443).
# Ctrl+C stops both. iPhone/LAN access: https://<hostname>.local:8443
cd "$(dirname "$0")"

export TLS_HOST="$(hostname).local"
echo "LAN URL:  https://${TLS_HOST}:8443  (accept the self-signed cert on first visit)"
echo "Local:    http://127.0.0.1:8765"

.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 &
server_pid=$!

trap 'kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null; exit 0' INT TERM

caddy run --config ./Caddyfile --adapter caddyfile

#!/usr/bin/env bash
# Start the chat server (127.0.0.1:8765) + Caddy TLS proxy (:8443) in the
# foreground. Ctrl+C or ./stop.sh stops both.
# Pass --verbose (-v) to stream the Ditto worker's detailed output and torch
# warnings to the terminal; default is a sparse subsystem checklist only.
cd "$(dirname "$0")"

for arg in "$@"; do
  case "$arg" in
    -v|--verbose) export AVATAR_VERBOSE=1 ;;
  esac
done

# Host/IP for the access banner, which the server prints at the end of its
# warmup (so it lands below the startup logs and stays visible).
export TLS_HOST="$(hostname).local"
export LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')"

.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 &
server_pid=$!
echo $server_pid > /tmp/chatapp.pid

caddy run --config ./Caddyfile --adapter caddyfile &
caddy_pid=$!
echo $caddy_pid > /tmp/chatapp-caddy.pid

trap 'kill $server_pid $caddy_pid 2>/dev/null; wait 2>/dev/null; rm -f /tmp/chatapp.pid /tmp/chatapp-caddy.pid; exit 0' INT TERM

wait

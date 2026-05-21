#!/usr/bin/env bash
# Start the chat server (127.0.0.1:8765) + Caddy TLS proxy (:8443) in the
# foreground. Ctrl+C or ./stop.sh stops both.
cd "$(dirname "$0")"

export TLS_HOST="$(hostname).local"
LAN_IP="$(ip route get 1.1.1.1 2>/dev/null | awk '{print $7; exit}')"

print_urls() {
  echo
  echo "================================================================"
  echo "  Open the app at:"
  echo "    This machine:    http://127.0.0.1:8765"
  echo "    Other devices:   https://${TLS_HOST}:8443   (accept the cert on first visit)"
  [ -n "$LAN_IP" ] && \
  echo "                     https://${LAN_IP}:8443   (if .local doesn't resolve)"
  echo "================================================================"
  echo
}

print_urls

.venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765 &
server_pid=$!
echo $server_pid > /tmp/chatapp.pid

caddy run --config ./Caddyfile --adapter caddyfile &
caddy_pid=$!
echo $caddy_pid > /tmp/chatapp-caddy.pid

trap 'kill $server_pid $caddy_pid 2>/dev/null; wait 2>/dev/null; rm -f /tmp/chatapp.pid /tmp/chatapp-caddy.pid; exit 0' INT TERM

# Re-print after the servers dump their startup logs so the URLs stay visible.
( sleep 3; print_urls ) &

wait

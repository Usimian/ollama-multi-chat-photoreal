#!/usr/bin/env bash
# Expose the local chat server to the internet via ngrok, password-gated by the
# basic-auth traffic policy in ngrok-policy.yml. Ctrl+C to stop the tunnel.
#
# One-time setup:
#   1. Create a free ngrok account at https://dashboard.ngrok.com
#   2. ngrok config add-authtoken <YOUR_TOKEN>
#   3. (optional) edit ngrok-policy.yml to change the username/password
#
# Run the chat server (./start.sh) first, then this in another terminal.
# ngrok terminates TLS, so the public URL is real HTTPS — the iPhone mic works
# from anywhere. The free tier gives a new random URL each run.
cd "$(dirname "$0")"

if ! command -v ngrok >/dev/null 2>&1; then
  echo "ngrok not installed. Install with: sudo snap install ngrok"
  exit 1
fi
if [ ! -f ngrok-policy.yml ]; then
  echo "ngrok-policy.yml missing. Copy the template and set credentials:"
  echo "  cp ngrok-policy.yml.example ngrok-policy.yml"
  exit 1
fi

exec ngrok http 8765 --traffic-policy-file ngrok-policy.yml

#!/usr/bin/env bash
# Start the chat server in the foreground. Ctrl+C to stop.
cd "$(dirname "$0")"
exec .venv/bin/uvicorn server:app --host 127.0.0.1 --port 8765

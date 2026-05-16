"""End-to-end smoke (no browser): create a conv with avatar_mode, connect ws,
send a short user_message, count binary frames + tts_chunks, save first frame.

Run AFTER `./start.sh` is up on port 8765:
    .venv/bin/python smoke_e2e.py
"""
import asyncio
import json
import os
import sys
import time

import httpx
import websockets

BASE = "http://127.0.0.1:8765"
WS = "ws://127.0.0.1:8765"
OUT = os.path.join(os.path.dirname(__file__), "tmp")
MODEL = "qwen2.5:7b"  # small, fast, in your installed list


async def main() -> int:
    os.makedirs(OUT, exist_ok=True)

    async with httpx.AsyncClient(timeout=10) as http:
        spec = {
            "title": "e2e-smoke",
            "mode": "user_llm",
            "participants": [
                {"name": "me", "kind": "user"},
                {"name": "bot", "kind": "llm", "model": MODEL, "system": "Reply with at least three full sentences."},
            ],
            "avatar_mode": True,
        }
        r = await http.post(f"{BASE}/api/conversations", json=spec)
        r.raise_for_status()
        conv = r.json()
        cid = conv["id"]
        print(f"[smoke] conv created: {cid}")

    try:
        async with websockets.connect(f"{WS}/ws/{cid}", max_size=None) as ws:
            print("[smoke] ws connected, waiting for avatar_ready…")
            t_open = time.perf_counter()
            ready_ms = None
            n_frames = 0
            n_tts = 0
            first_frame_path = None
            sent = False

            while True:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=2.0)
                except asyncio.TimeoutError:
                    elapsed = time.perf_counter() - t_open
                    print(f"  [t+{elapsed:.1f}s] silence; frames={n_frames} tts={n_tts}")
                    if elapsed > 60:
                        print("[smoke] giving up after 60s")
                        break
                    continue

                if isinstance(msg, bytes):
                    if msg[:1] == b"\x01":
                        n_frames += 1
                        if first_frame_path is None:
                            first_frame_path = os.path.join(OUT, "e2e_first.jpg")
                            with open(first_frame_path, "wb") as f:
                                f.write(msg[1:])
                            print(f"  [t+{time.perf_counter()-t_open:.1f}s] FIRST FRAME → {first_frame_path} ({len(msg)-1} bytes)")
                        if n_frames % 25 == 0:
                            print(f"  [t+{time.perf_counter()-t_open:.1f}s] frames={n_frames} tts={n_tts}")
                    continue

                ev = json.loads(msg)
                t = ev.get("type")
                if t == "avatar_ready":
                    ready_ms = (time.perf_counter() - t_open) * 1000
                    print(f"  [t+{ready_ms/1000:.1f}s] avatar_ready")
                    if not sent:
                        await ws.send(json.dumps({"type": "user_message", "content": "Tell me three interesting facts about octopuses, briefly."}))
                        sent = True
                        print("  [smoke] sent user_message")
                elif t == "tts_chunk":
                    n_tts += 1
                    print(f"  [t+{time.perf_counter()-t_open:.1f}s] tts_chunk #{ev.get('seq')}: {ev.get('text','')[:60]!r}")
                elif t == "state" and not ev.get("running", True):
                    # LLM reply ended; give the avatar pipeline time to drain.
                    print(f"  [t+{time.perf_counter()-t_open:.1f}s] LLM done; draining 8s…")
                    drain_end = time.perf_counter() + 20
                    while time.perf_counter() < drain_end:
                        try:
                            m = await asyncio.wait_for(ws.recv(), timeout=1.0)
                            if isinstance(m, bytes) and m[:1] == b"\x01":
                                n_frames += 1
                                if first_frame_path is None:
                                    first_frame_path = os.path.join(OUT, "e2e_first.jpg")
                                    with open(first_frame_path, "wb") as f:
                                        f.write(m[1:])
                                    print(f"  [t+{time.perf_counter()-t_open:.1f}s] FIRST FRAME → {first_frame_path} ({len(m)-1} bytes)")
                        except asyncio.TimeoutError:
                            pass
                    print(f"[smoke] FINAL frames={n_frames} tts={n_tts}")
                    break

            success = n_frames > 0 and n_tts > 0
            print(f"[smoke] {'PASS' if success else 'FAIL'}")
            return 0 if success else 1
    finally:
        async with httpx.AsyncClient(timeout=10) as http:
            await http.delete(f"{BASE}/api/conversations/{cid}")
        print(f"[smoke] deleted conv {cid}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

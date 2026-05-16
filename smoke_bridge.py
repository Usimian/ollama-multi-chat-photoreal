"""Exercise DittoBridge end-to-end: synthesize one sentence with Kokoro at 24 kHz,
feed it into the bridge, count frames, save first/last JPEG.

Run from the photoreal venv:
    .venv/bin/python smoke_bridge.py
"""
import asyncio
import io
import os
import time

import numpy as np
import soundfile as sf

from photoreal import DittoBridge

HERE = os.path.dirname(os.path.abspath(__file__))
REF_IMG = os.path.expanduser("~/ditto-talkinghead/example/image.png")
OUT_DIR = os.path.join(HERE, "tmp")
TEXT = "Hello there. This is a quick test of the photoreal talking head pipeline."


def synth_kokoro(text: str, voice: str = "af_heart") -> np.ndarray:
    """Return float32 mono 24 kHz."""
    from kokoro import KPipeline
    pipe = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    parts = []
    for _g, _p, audio in pipe(text, voice=voice):
        a = audio if isinstance(audio, np.ndarray) else np.array(audio)
        parts.append(a)
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts).astype(np.float32)


async def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[smoke] synthesizing with kokoro…")
    t0 = time.perf_counter()
    pcm_24k = await asyncio.to_thread(synth_kokoro, TEXT)
    print(f"[smoke] kokoro: {len(pcm_24k)} samples ({len(pcm_24k)/24000:.2f}s) in {(time.perf_counter()-t0)*1000:.0f} ms")

    bridge = DittoBridge()
    print("[smoke] starting bridge (launches worker)…")
    t0 = time.perf_counter()
    await bridge.start()
    print(f"[smoke] worker up in {(time.perf_counter()-t0)*1000:.0f} ms")

    try:
        conv_id = "smoke"
        t_open = time.perf_counter()
        await bridge.open_session(conv_id, ref_image_path=REF_IMG, max_size=512)
        print(f"[smoke] session ready in {(time.perf_counter()-t_open)*1000:.0f} ms")

        # collect frames concurrently
        frames: list[dict] = []
        ttff_holder: dict = {}
        t_audio_start = time.perf_counter()

        async def collect():
            async for fr in bridge.frames(conv_id):
                if not frames:
                    ttff_holder["ttff_ms"] = (time.perf_counter() - t_audio_start) * 1000
                frames.append(fr)

        collector = asyncio.create_task(collect())

        # feed in real-time-ish: 200 ms of 24k audio = 4800 samples
        FEED_CHUNK = 4800
        for i in range(0, len(pcm_24k), FEED_CHUNK):
            await bridge.feed_audio_24k(conv_id, pcm_24k[i:i + FEED_CHUNK])
        await bridge.end(conv_id)
        print(f"[smoke] all audio fed in {(time.perf_counter()-t_audio_start)*1000:.0f} ms")

        # give the worker time to drain
        try:
            await asyncio.wait_for(asyncio.sleep(10), timeout=10)
        except asyncio.TimeoutError:
            pass

        await bridge.close_session(conv_id)
        await asyncio.wait_for(collector, timeout=5)

        print(f"[smoke] got {len(frames)} frames")
        if frames:
            print(f"[smoke] TTFF = {ttff_holder.get('ttff_ms', float('nan')):.0f} ms (audio→first frame)")
            # save first + last
            for tag, fr in (("bridge_first", frames[0]), ("bridge_last", frames[-1])):
                p = os.path.join(OUT_DIR, f"{tag}.jpg")
                with open(p, "wb") as f:
                    f.write(fr["jpeg"])
                print(f"[smoke] wrote {p}")
            return 0
        else:
            print("[smoke] NO FRAMES — failure")
            return 1
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

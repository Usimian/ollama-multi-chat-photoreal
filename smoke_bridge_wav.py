"""Bridge transport smoke — use ditto's example audio.wav (not kokoro), so any
failure isolates the bridge/UDS plumbing."""
import asyncio
import os
import time

import numpy as np
import soundfile as sf

from photoreal import DittoBridge

REF_IMG = os.path.expanduser("~/ditto-talkinghead/example/image.png")
WAV = os.path.expanduser("~/ditto-talkinghead/example/audio.wav")
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")


def load_24k(path: str) -> np.ndarray:
    pcm, sr = sf.read(path, dtype="float32", always_2d=False)
    if pcm.ndim == 2:
        pcm = pcm.mean(axis=1)
    # pretend it's 24k input (bridge resamples 24→16); but real file is 16k.
    # The bridge's job is to resample; simulate by treating it as 24k won't be
    # exact, but the goal here is "does any audio op trigger frames?"
    if sr != 24000:
        n = int(round(len(pcm) * 24000 / sr))
        xp = np.linspace(0, 1, len(pcm), endpoint=False)
        x = np.linspace(0, 1, n, endpoint=False)
        pcm = np.interp(x, xp, pcm).astype(np.float32)
    return pcm


async def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    pcm_24k = load_24k(WAV)
    print(f"[wav] loaded {len(pcm_24k)} samples @ 24k ({len(pcm_24k)/24000:.2f}s)")

    bridge = DittoBridge()
    await bridge.start()
    try:
        cid = "wavsmoke"
        await bridge.open_session(cid, ref_image_path=REF_IMG, max_size=512)
        print("[wav] session ready")

        frames: list[dict] = []
        async def collect():
            async for fr in bridge.frames(cid):
                frames.append(fr)
        col = asyncio.create_task(collect())

        t0 = time.perf_counter()
        FEED = 4800
        for i in range(0, len(pcm_24k), FEED):
            await bridge.feed_audio_24k(cid, pcm_24k[i:i+FEED])
        await bridge.end(cid)
        print(f"[wav] fed in {(time.perf_counter()-t0)*1000:.0f} ms; sleeping 12s for drain")
        await asyncio.sleep(12)
        await bridge.close_session(cid)
        await asyncio.wait_for(col, timeout=5)
        print(f"[wav] got {len(frames)} frames")
        if frames:
            with open(os.path.join(OUT_DIR, "wav_first.jpg"), "wb") as f:
                f.write(frames[0]["jpeg"])
            with open(os.path.join(OUT_DIR, "wav_last.jpg"), "wb") as f:
                f.write(frames[-1]["jpeg"])
        return 0 if frames else 1
    finally:
        await bridge.shutdown()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

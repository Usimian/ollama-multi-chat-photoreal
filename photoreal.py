"""DittoBridge — async client to ditto_worker.py over Unix domain socket.

Owns the worker subprocess and a single UDS connection. Per-conversation
sessions are multiplexed by conv_id. Audio is buffered, resampled 24k→16k
and split into 200 ms chunks of 3200 float32 samples before sending.

Typical use:

    bridge = DittoBridge()
    await bridge.start()
    await bridge.open_session("conv-1", ref_image_path="/path/to/face.png")
    async for frame in bridge.frames("conv-1"):
        ...                                   # frame = {"seq", "ts_ms", "jpeg"}
    # elsewhere:
    await bridge.feed_audio_24k("conv-1", pcm_f32_24k)
    await bridge.end("conv-1")
    await bridge.close_session("conv-1")
"""
from __future__ import annotations

import asyncio
import os
import struct
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import msgpack
import numpy as np

# Where the ditto-talkinghead checkout lives — used for the venv that has
# torch+onnxruntime+the SDK, and as the import path for the SDK modules.
# Override with $DITTO_DIR if you have it installed elsewhere.
DITTO_DIR = os.path.abspath(os.path.expanduser(os.environ.get("DITTO_DIR", "~/ditto-talkinghead")))
DITTO_PY = os.path.join(DITTO_DIR, ".venv/bin/python")
# The worker itself is vendored in this repo so deployment doesn't depend on
# files inside the upstream Ditto checkout.
DITTO_WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "worker", "ditto_worker.py")
DEFAULT_SOCK = "/tmp/ditto.sock"
WORKER_LOG = "/tmp/ditto-worker.log"
# When false, the worker's chatty output is redirected to WORKER_LOG so the
# server terminal stays readable. Set AVATAR_VERBOSE=1 (start.sh --verbose) to
# let it stream to the terminal instead.
VERBOSE = os.environ.get("AVATAR_VERBOSE", "") not in ("", "0", "false", "False")

CHUNK_16K = 3200            # 200 ms @ 16 kHz mono
SENTINEL_END = object()


def _resample_linear(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Cheap linear resample. Adequate for speech; replace with polyphase if quality matters."""
    if src_rate == dst_rate:
        return pcm.astype(np.float32, copy=False)
    n_out = int(round(len(pcm) * dst_rate / src_rate))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=len(pcm), endpoint=False)
    x = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x, xp, pcm).astype(np.float32)


@dataclass
class _Session:
    conv_id: str
    frame_q: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=256))
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    error: Optional[str] = None
    buf_16k: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))


class DittoBridge:
    def __init__(self, sock_path: str = DEFAULT_SOCK, worker_python: str = DITTO_PY, worker_script: str = DITTO_WORKER):
        self.sock_path = sock_path
        self.worker_python = worker_python
        self.worker_script = worker_script
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._send_lock = asyncio.Lock()
        self._reader_task: Optional[asyncio.Task] = None
        self._sessions: dict[str, _Session] = {}
        self._started = asyncio.Event()
        self._log = None

    # ---------- lifecycle ----------

    async def start(self, startup_timeout: float = 60.0) -> None:
        if self._proc is not None:
            return
        if os.path.exists(self.sock_path):
            os.unlink(self.sock_path)
        # Unless verbose, redirect the worker's very chatty output (mediapipe/EGL
        # banners, the SDK's setup-kwargs dump, onnxruntime shape warnings, tqdm
        # progress) to a logfile so the server terminal stays readable.
        self._log = None if VERBOSE else open(WORKER_LOG, "w")
        self._proc = await asyncio.create_subprocess_exec(
            self.worker_python, self.worker_script,
            "--sock", self.sock_path,
            "--ditto-dir", DITTO_DIR,
            cwd=DITTO_DIR,
            stdout=self._log,
            stderr=(asyncio.subprocess.STDOUT if self._log else None),
        )
        # wait for socket
        deadline = asyncio.get_event_loop().time() + startup_timeout
        while not os.path.exists(self.sock_path):
            if self._proc.returncode is not None:
                raise RuntimeError(f"ditto_worker exited early rc={self._proc.returncode}")
            if asyncio.get_event_loop().time() > deadline:
                raise TimeoutError("ditto_worker never created socket")
            await asyncio.sleep(0.1)
        self._reader, self._writer = await asyncio.open_unix_connection(self.sock_path)
        self._reader_task = asyncio.create_task(self._read_loop(), name="ditto-bridge-reader")
        self._started.set()

    async def shutdown(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception:
                pass
        if self._reader_task is not None:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._proc is not None and self._proc.returncode is None:
            self._proc.terminate()
            try:
                await asyncio.wait_for(self._proc.wait(), timeout=5)
            except asyncio.TimeoutError:
                self._proc.kill()
                await self._proc.wait()
        for sess in self._sessions.values():
            await sess.frame_q.put(SENTINEL_END)
        self._sessions.clear()
        if self._log is not None:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None
        self._proc = None
        self._reader = None
        self._writer = None

    # ---------- session ops ----------

    async def open_session(self, conv_id: str, ref_image_path: str, max_size: int = 512, ready_timeout: float = 300.0) -> None:
        if conv_id in self._sessions:
            await self.close_session(conv_id)
        sess = _Session(conv_id=conv_id)
        self._sessions[conv_id] = sess
        await self._send({"op": "open", "conv_id": conv_id, "ref_image_path": ref_image_path, "max_size": int(max_size)})
        try:
            await asyncio.wait_for(sess.ready.wait(), timeout=ready_timeout)
        except asyncio.TimeoutError:
            self._sessions.pop(conv_id, None)
            raise
        if sess.error:
            self._sessions.pop(conv_id, None)
            raise RuntimeError(f"open failed: {sess.error}")

    async def feed_audio_24k(self, conv_id: str, pcm_f32_24k: np.ndarray) -> None:
        sess = self._sessions.get(conv_id)
        if sess is None:
            raise KeyError(conv_id)
        pcm_16k = _resample_linear(np.asarray(pcm_f32_24k, dtype=np.float32), 24000, 16000)
        sess.buf_16k = np.concatenate([sess.buf_16k, pcm_16k])
        while len(sess.buf_16k) >= CHUNK_16K:
            window = sess.buf_16k[:CHUNK_16K]
            sess.buf_16k = sess.buf_16k[CHUNK_16K:]
            await self._send({"op": "audio", "conv_id": conv_id, "pcm_f32_16k": window.tobytes()})

    async def end(self, conv_id: str) -> None:
        sess = self._sessions.get(conv_id)
        if sess is None:
            return
        if len(sess.buf_16k) > 0:
            tail = np.concatenate([sess.buf_16k, np.zeros(CHUNK_16K - len(sess.buf_16k), dtype=np.float32)])
            sess.buf_16k = np.zeros(0, dtype=np.float32)
            await self._send({"op": "audio", "conv_id": conv_id, "pcm_f32_16k": tail.tobytes()})
        await self._send({"op": "end", "conv_id": conv_id})

    async def close_session(self, conv_id: str) -> None:
        if conv_id not in self._sessions:
            return
        await self._send({"op": "close", "conv_id": conv_id})
        sess = self._sessions.pop(conv_id)
        await sess.frame_q.put(SENTINEL_END)

    async def frames(self, conv_id: str) -> AsyncIterator[dict]:
        sess = self._sessions.get(conv_id)
        if sess is None:
            raise KeyError(conv_id)
        while True:
            item = await sess.frame_q.get()
            if item is SENTINEL_END:
                return
            yield item

    # ---------- wire ----------

    async def _send(self, msg: dict) -> None:
        assert self._writer is not None, "bridge not started"
        payload = msgpack.packb(msg, use_bin_type=True)
        async with self._send_lock:
            self._writer.write(struct.pack(">I", len(payload)) + payload)
            await self._writer.drain()

    async def _read_loop(self) -> None:
        assert self._reader is not None
        try:
            while True:
                hdr = await self._reader.readexactly(4)
                (n,) = struct.unpack(">I", hdr)
                body = await self._reader.readexactly(n)
                msg = msgpack.unpackb(body, raw=False)
                conv_id = msg.get("conv_id", "")
                sess = self._sessions.get(conv_id)
                if sess is None:
                    continue
                t = msg.get("type")
                if t == "frame":
                    try:
                        sess.frame_q.put_nowait(msg)
                    except asyncio.QueueFull:
                        # drop oldest to keep up under backpressure
                        try:
                            sess.frame_q.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                        sess.frame_q.put_nowait(msg)
                elif t == "ready":
                    sess.ready.set()
                elif t == "error":
                    sess.error = msg.get("message", "unknown error")
                    sess.ready.set()
                    await sess.frame_q.put(SENTINEL_END)
        except asyncio.IncompleteReadError:
            pass
        except asyncio.CancelledError:
            raise
        finally:
            for sess in self._sessions.values():
                await sess.frame_q.put(SENTINEL_END)

"""UDS worker that wraps Ditto's StreamSDK for the photoreal chatbot.

This file is vendored INTO the photoreal repo but the SDK + model weights it
imports live in a separate Ditto install (``ditto-talkinghead``). The install
path is configurable so this file can be moved without breaking imports:
  1) ``--ditto-dir`` CLI arg
  2) ``$DITTO_DIR`` env var
  3) Fallback: ``~/ditto-talkinghead``

Protocol (length-prefixed msgpack, uint32 big-endian length):

  client → worker
    {"op": "open",  "conv_id": str, "ref_image_path": str, "max_size": int=512}
    {"op": "audio", "conv_id": str, "pcm_f32_16k": bytes}   # any length, mono float32
    {"op": "end",   "conv_id": str}                          # flush trailing chunk
    {"op": "close", "conv_id": str}

  worker → client
    {"type": "ready", "conv_id": str}
    {"type": "frame", "conv_id": str, "seq": int, "ts_ms": int, "jpeg": bytes}
    {"type": "error", "conv_id": str, "message": str}

Single-client UDS server. One Session per conv_id; sessions hold a StreamSDK in
online mode. The SDK's mp4 writer is intercepted so frames become JPEG messages
instead of file writes.
"""

import argparse
import os
import socket
import struct
import sys
import tempfile
import threading
import time
import traceback


def _resolve_ditto_dir(cli_dir: str | None) -> str:
    """--ditto-dir > $DITTO_DIR > ~/ditto-talkinghead."""
    if cli_dir:
        return os.path.abspath(os.path.expanduser(cli_dir))
    env = os.environ.get("DITTO_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.expanduser("~/ditto-talkinghead")


# Bootstrap the import path before pulling Ditto modules. We parse argv twice
# (once here, once below) so this still works whether or not the user passes
# the flag.
_cli_dir = None
for i, a in enumerate(sys.argv):
    if a == "--ditto-dir" and i + 1 < len(sys.argv):
        _cli_dir = sys.argv[i + 1]
        break
DITTO_DIR = _resolve_ditto_dir(_cli_dir)
sys.path.insert(0, DITTO_DIR)

import cv2          # noqa: E402 — must follow sys.path edit
import msgpack      # noqa: E402
import numpy as np  # noqa: E402
from stream_pipeline_online import StreamSDK  # noqa: E402

CFG_PKL = os.path.join(DITTO_DIR, "checkpoints/ditto_cfg/v0.4_hubert_cfg_pytorch.pkl")
DATA_ROOT = os.path.join(DITTO_DIR, "checkpoints/ditto_pytorch")

CHUNKSIZE = (3, 5, 2)
SAMPLES_PER_STRIDE = CHUNKSIZE[1] * 640          # 3200 samples = 200 ms @ 16 kHz
SPLIT_LEN = int(sum(CHUNKSIZE) * 0.04 * 16000) + 80  # 6480
# Two-part pre-pad before any real audio:
#   * CHUNKSIZE[0] * 640 = 1920 samples (120 ms) — wav2feat's "past" context.
#   * WARMUP_SKIP_FRAMES * 640 — covers the audio2motion model's "warmup
#     skip" in online mode. stream_pipeline_online.py sets
#     res_kp_seq_valid_start = seq_frames - fuse_length = 80-10 = 70 on the
#     first chunk, so its first emitted frame corresponds to audio_feat[70].
#     BUT the pipeline also pre-seeds audio_feat with overlap_v2 (=10) silence
#     features at init, so real audio actually starts at audio_feat[80]. To
#     make the first emitted frame (audio_feat[70]) line up with real audio
#     time 0, we pad 70 - 10 = 60 frames of silence. Without this pad, lipsync
#     is offset by ~2.8 s; with the wrong count, by a fraction of a second.
WARMUP_SKIP_FRAMES = 60
LEADING_PAD = (CHUNKSIZE[0] + WARMUP_SKIP_FRAMES) * 640


def _read_msg(sock: socket.socket):
    hdr = b""
    while len(hdr) < 4:
        chunk = sock.recv(4 - len(hdr))
        if not chunk:
            return None
        hdr += chunk
    (n,) = struct.unpack(">I", hdr)
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return msgpack.unpackb(bytes(buf), raw=False)


def _send_msg(sock: socket.socket, lock: threading.Lock, msg: dict) -> None:
    payload = msgpack.packb(msg, use_bin_type=True)
    with lock:
        sock.sendall(struct.pack(">I", len(payload)) + payload)


class Session:
    def __init__(self, conv_id: str, ref_image_path: str, sock, send_lock, max_size: int = 512):
        self.conv_id = conv_id
        self.sock = sock
        self.send_lock = send_lock
        self.seq = 0
        self.t_open = time.perf_counter()

        tmp_dir = tempfile.mkdtemp(prefix=f"ditto_{conv_id}_")
        discard_mp4 = os.path.join(tmp_dir, "discard.mp4")

        self.sdk = StreamSDK(CFG_PKL, DATA_ROOT)
        self.sdk.setup(ref_image_path, discard_mp4, online_mode=True, max_size=max_size)
        self.sdk.setup_Nd(N_d=-1, fade_in=-1, fade_out=-1, ctrl_info={})

        self._patch_writer()
        self._audio_buf = np.zeros((LEADING_PAD,), dtype=np.float32)
        self._got_audio = False

    def _patch_writer(self) -> None:
        writer = self.sdk.writer
        if not hasattr(writer, "writer") or not hasattr(writer.writer, "append_data"):
            raise RuntimeError("writer surface changed; can't intercept frames")

        def on_frame(img):
            # imageio writers expect RGB uint8 HxWx3; cv2.imencode wants BGR.
            bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                return
            ts_ms = int((time.perf_counter() - self.t_open) * 1000)
            _send_msg(self.sock, self.send_lock, {
                "type": "frame",
                "conv_id": self.conv_id,
                "seq": self.seq,
                "ts_ms": ts_ms,
                "jpeg": buf.tobytes(),
            })
            self.seq += 1

        writer.writer.append_data = on_frame

    def feed(self, pcm_f32_16k: np.ndarray) -> None:
        if pcm_f32_16k.size:
            self._got_audio = True
        self._audio_buf = np.concatenate([self._audio_buf, pcm_f32_16k])
        while len(self._audio_buf) >= SPLIT_LEN:
            window = self._audio_buf[:SPLIT_LEN].copy()
            self.sdk.run_chunk(window, chunksize=CHUNKSIZE)
            self._audio_buf = self._audio_buf[SAMPLES_PER_STRIDE:]

    def end(self) -> None:
        if self._got_audio and len(self._audio_buf) > 0:
            pad_len = SPLIT_LEN - len(self._audio_buf)
            window = np.concatenate([self._audio_buf, np.zeros((max(pad_len, 0),), dtype=np.float32)])[:SPLIT_LEN]
            self.sdk.run_chunk(window, chunksize=CHUNKSIZE)
        self._audio_buf = np.zeros((LEADING_PAD,), dtype=np.float32)
        self._got_audio = False

    def close(self) -> None:
        try:
            self.sdk.close()
        except Exception:
            traceback.print_exc()
        # Free the SDK's GPU memory. StreamSDK reserves several GB of CUDA
        # memory per instance; without dropping the reference and emptying
        # torch's cache, opening a fresh session each turn accumulates until
        # the GPU is exhausted and frame production silently stops.
        self.sdk = None
        try:
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass


def handle_client(sock: socket.socket) -> None:
    sessions: dict[str, Session] = {}
    send_lock = threading.Lock()
    try:
        while True:
            msg = _read_msg(sock)
            if msg is None:
                break
            op = msg.get("op")
            conv_id = msg.get("conv_id", "")
            try:
                if op == "open":
                    if conv_id in sessions:
                        sessions[conv_id].close()
                    sessions[conv_id] = Session(
                        conv_id, msg["ref_image_path"], sock, send_lock,
                        max_size=int(msg.get("max_size", 512)),
                    )
                    _send_msg(sock, send_lock, {"type": "ready", "conv_id": conv_id})
                elif op == "audio":
                    pcm = np.frombuffer(msg["pcm_f32_16k"], dtype=np.float32)
                    sessions[conv_id].feed(pcm)
                elif op == "end":
                    sessions[conv_id].end()
                elif op == "close":
                    s = sessions.pop(conv_id, None)
                    if s:
                        s.close()
                else:
                    _send_msg(sock, send_lock, {"type": "error", "conv_id": conv_id, "message": f"unknown op {op!r}"})
            except KeyError:
                _send_msg(sock, send_lock, {"type": "error", "conv_id": conv_id, "message": "no such session"})
            except Exception as e:
                traceback.print_exc()
                _send_msg(sock, send_lock, {"type": "error", "conv_id": conv_id, "message": str(e)})
    finally:
        for s in sessions.values():
            s.close()


def serve(sock_path: str) -> None:
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    os.chmod(sock_path, 0o600)
    srv.listen(1)
    print(f"[ditto_worker] listening on {sock_path} (ditto_dir={DITTO_DIR})", flush=True)
    try:
        while True:
            try:
                conn, _ = srv.accept()
            except KeyboardInterrupt:
                break
            print("[ditto_worker] client connected", flush=True)
            try:
                handle_client(conn)
            except KeyboardInterrupt:
                break
            finally:
                conn.close()
                print("[ditto_worker] client disconnected", flush=True)
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sock", default="/tmp/ditto.sock")
    ap.add_argument("--ditto-dir", default=None,
                    help="Path to ditto-talkinghead checkout (overrides $DITTO_DIR; default ~/ditto-talkinghead)")
    args = ap.parse_args()
    serve(args.sock)

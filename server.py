"""Multi-conversation Ollama playground.

Run: uv run --with-requirements requirements.txt uvicorn server:app --reload
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import re
import uuid
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Sparse terminal by default: hide third-party deprecation/user warnings (torch,
# via Kokoro) so the warmup checklist stays readable. The subsystem checklist
# still prints. Run with --verbose (AVATAR_VERBOSE=1) to see everything.
VERBOSE = os.environ.get("AVATAR_VERBOSE", "") not in ("", "0", "false", "False")
if not VERBOSE:
    warnings.filterwarnings("ignore", category=UserWarning, module=r"torch\..*")
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\..*")

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from photoreal import DittoBridge
import tools as web_tools
import memory as avatar_memory

OLLAMA_URL = "http://localhost:11434"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent / "data"
ASSET_DIR = Path(__file__).parent / "assets"
AVATAR_REF = ASSET_DIR / "avatar_ref.jpg"
DATA_DIR.mkdir(exist_ok=True)


# ---------- Data model ----------

@dataclass
class Participant:
    name: str
    kind: str                       # "user" | "llm"
    model: Optional[str] = None
    system: Optional[str] = None

@dataclass
class Message:
    id: str
    sender: int                     # participant index; -2 = system note
    content: str
    thinking: str = ""

@dataclass
class Conversation:
    id: str
    title: str
    mode: str                       # "user_llm" | "llm_llm"
    participants: list[Participant]
    messages: list[Message] = field(default_factory=list)
    max_turns: int = 20
    turns_taken: int = 0            # counts LLM turns only
    avatar_mode: bool = False       # enable TTS + viseme broadcasting
    tts_voice: str = "af_heart"     # Kokoro voice id
    thinking_mode: bool = False     # send think:True to Ollama (slower but more rigorous)
    running: bool = False
    _task: Optional[asyncio.Task] = None
    _stop: asyncio.Event = field(default_factory=asyncio.Event)
    _abort: asyncio.Event = field(default_factory=asyncio.Event)

    def to_public(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "mode": self.mode,
            "participants": [asdict(p) for p in self.participants],
            "messages": [asdict(m) for m in self.messages],
            "max_turns": self.max_turns,
            "turns_taken": self.turns_taken,
            "avatar_mode": self.avatar_mode,
            "tts_voice": self.tts_voice,
            "thinking_mode": self.thinking_mode,
            "running": self.running,
        }


# ---------- In-memory store ----------

conversations: dict[str, Conversation] = {}
sockets: dict[str, set[WebSocket]] = {}


def persist(conv: Conversation) -> None:
    path = DATA_DIR / f"{conv.id}.json"
    tmp = path.with_suffix(".tmp")
    data = conv.to_public()
    data["running"] = False  # never persist live state
    tmp.write_text(json.dumps(data, indent=2))
    tmp.replace(path)


def delete_persisted(cid: str) -> None:
    path = DATA_DIR / f"{cid}.json"
    if path.exists():
        path.unlink()


def load_all() -> None:
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            data = json.loads(path.read_text())
            participants = [Participant(**p) for p in data["participants"]]
            messages = [Message(**m) for m in data.get("messages", [])]
            conv = Conversation(
                id=data["id"],
                title=data["title"],
                mode=data["mode"],
                participants=participants,
                messages=messages,
                max_turns=int(data.get("max_turns", 20)),
                turns_taken=int(data.get("turns_taken", 0)),
                avatar_mode=bool(data.get("avatar_mode", False)),
                tts_voice=str(data.get("tts_voice", "af_heart")),
                thinking_mode=bool(data.get("thinking_mode", False)),
            )
            conversations[conv.id] = conv
        except Exception as e:
            print(f"failed to load {path}: {e}")


async def broadcast(conv_id: str, event: dict) -> None:
    dead = []
    for ws in sockets.get(conv_id, set()):
        try:
            await ws.send_json(event)
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets[conv_id].discard(ws)


async def broadcast_bytes(conv_id: str, payload: bytes) -> None:
    dead = []
    for ws in sockets.get(conv_id, set()):
        try:
            await ws.send_bytes(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        sockets[conv_id].discard(ws)


# ---------- Ditto avatar bridge ----------

_bridge: Optional[DittoBridge] = None
_bridge_lock = asyncio.Lock()
_avatar_sessions: dict[str, asyncio.Task] = {}    # cid -> frame fanout task
_avatar_starting: dict[str, asyncio.Lock] = {}    # cid -> lock guarding open


async def _get_bridge() -> DittoBridge:
    """Lazy-start the singleton DittoBridge + worker subprocess."""
    global _bridge
    async with _bridge_lock:
        if _bridge is None:
            br = DittoBridge()
            await br.start()
            _bridge = br
        return _bridge


async def _frame_fanout(cid: str) -> None:
    """Pull frames from the bridge and broadcast as binary ws messages.
    Binary payload format: 1-byte type tag (0x01 = jpeg frame) + JPEG bytes."""
    bridge = await _get_bridge()
    try:
        async for fr in bridge.frames(cid):
            await broadcast_bytes(cid, b"\x01" + fr["jpeg"])
    except Exception as e:
        print(f"[avatar {cid}] fanout ended: {e!r}")


async def ensure_avatar_session(cid: str) -> None:
    """Open a Ditto session for this conv if not already running."""
    lock = _avatar_starting.setdefault(cid, asyncio.Lock())
    async with lock:
        if cid in _avatar_sessions and not _avatar_sessions[cid].done():
            return
        if not AVATAR_REF.exists():
            print(f"[avatar {cid}] ref image missing at {AVATAR_REF}; avatar disabled")
            return
        bridge = await _get_bridge()
        await bridge.open_session(cid, ref_image_path=str(AVATAR_REF), max_size=512)
        _avatar_sessions[cid] = asyncio.create_task(_frame_fanout(cid), name=f"avatar-{cid}")
        await broadcast(cid, {"type": "avatar_ready"})


async def close_avatar_session(cid: str) -> None:
    # Acquire the SAME per-cid lock ensure_avatar_session uses, so a close can
    # never interleave with an in-flight open. Without this, a close that lands
    # while open_session is still awaiting the worker leaves _avatar_sessions
    # pointing at a fanout whose bridge session was already torn down — the
    # session looks open but produces no frames (the intermittent "no lips" bug).
    lock = _avatar_starting.setdefault(cid, asyncio.Lock())
    async with lock:
        task = _avatar_sessions.pop(cid, None)
        if task and not task.done():
            task.cancel()
        if _bridge is not None:
            try:
                await _bridge.close_session(cid)
            except Exception:
                pass


# ---------- Ollama ----------

async def ollama_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))


async def ollama_stream(model: str, messages: list[dict], think: bool = True, tools: Optional[list] = None):
    """Yield (content_delta, thinking_delta, tool_calls) from Ollama /api/chat.
    `tool_calls` is None except on the chunk where the model requests tools.
    `think` matches conv.thinking_mode."""
    payload = {"model": model, "messages": messages, "stream": True, "think": bool(think)}
    if tools:
        payload["tools"] = tools
    async with httpx.AsyncClient(timeout=None) as client:
        async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                m = obj.get("message", {})
                yield m.get("content", ""), m.get("thinking", ""), m.get("tool_calls") or None
                if obj.get("done"):
                    return


# ---------- Speech (STT + TTS) ----------

_whisper = None
_kokoro_pipelines: dict[str, object] = {}


def _get_whisper():
    global _whisper
    if _whisper is None:
        from faster_whisper import WhisperModel
        _whisper = WhisperModel("small.en", device="cuda", compute_type="float16")
    return _whisper


def _get_kokoro(lang_code: str = "a"):
    if lang_code not in _kokoro_pipelines:
        from kokoro import KPipeline
        _kokoro_pipelines[lang_code] = KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")
    return _kokoro_pipelines[lang_code]


def _transcribe(audio_bytes: bytes) -> str:
    """Run faster-whisper on raw audio bytes (any format ffmpeg/libsndfile reads)."""
    model = _get_whisper()
    bio = io.BytesIO(audio_bytes)
    segments, _info = model.transcribe(bio, beam_size=1, vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def _synthesize(text: str, voice: str = "af_heart") -> bytes:
    """Return WAV bytes for the given text."""
    pipe = _get_kokoro("a")
    audio_parts = []
    for _g, _p, audio in pipe(text, voice=voice):
        audio_parts.append(audio if isinstance(audio, np.ndarray) else np.array(audio))
    if not audio_parts:
        return b""
    audio = np.concatenate(audio_parts).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# Sentence-boundary regex: matches up to (and including) sentence-ending punctuation
# followed by whitespace. Minimum length guard avoids splitting on abbreviations.
SENTENCE_END_RE = re.compile(r"[^.!?\n]*?[.!?\n](?=\s|$)", re.DOTALL)


def _pop_sentences(buffer: str, min_len: int = 8) -> tuple[list[str], str]:
    """Pull complete sentences off the front of `buffer`. Return (sentences, remainder)."""
    sentences: list[str] = []
    pos = 0
    for m in SENTENCE_END_RE.finditer(buffer):
        chunk = buffer[pos:m.end()].strip()
        if len(chunk) >= min_len:
            sentences.append(chunk)
            pos = m.end()
    return sentences, buffer[pos:]


async def _tts_worker(conv: Conversation, msg_id: str, queue: asyncio.Queue) -> None:
    """Pop sentences, synthesize, broadcast tts_chunk events. If a Ditto avatar
    session is open for this conv, also feed the PCM to the bridge so the
    avatar lip-syncs in step with the audio the browser plays.

    Respects conv._abort: once set, pending sentences are drained without
    synthesis so the worker stops both audio and bridge frames promptly."""
    seq = 0
    while True:
        sentence = await queue.get()
        if sentence is None:
            return
        if conv._abort.is_set():
            queue.task_done()
            continue
        clean = _scrub_for_tts(sentence)
        if not clean:
            # Sentence was emojis-only — nothing to speak.
            queue.task_done()
            continue
        try:
            wav = await asyncio.to_thread(_synthesize, clean, conv.tts_voice)
        except Exception as e:
            print(f"[tts {conv.id}] synthesize error: {e!r}")
            queue.task_done()
            continue
        if conv._abort.is_set():
            queue.task_done()
            continue
        if wav:
            await broadcast(conv.id, {
                "type": "tts_chunk",
                "id": msg_id,
                "seq": seq,
                "text": sentence,
                "audio_b64": base64.b64encode(wav).decode("ascii"),
            })
            if conv.id in _avatar_sessions:
                try:
                    pcm, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=False)
                    if pcm.ndim == 2:
                        pcm = pcm.mean(axis=1)
                    if sr != 24000:
                        n = int(round(len(pcm) * 24000 / sr))
                        x_old = np.linspace(0, 1, len(pcm), endpoint=False)
                        x_new = np.linspace(0, 1, n, endpoint=False)
                        pcm = np.interp(x_new, x_old, pcm).astype(np.float32)
                    bridge = await _get_bridge()
                    await bridge.feed_audio_24k(conv.id, pcm)
                except Exception as e:
                    print(f"[avatar {conv.id}] feed failed: {e!r}")
            seq += 1
        queue.task_done()


# Strip emojis and most pictographic symbols before TTS — kokoro tries to
# pronounce them ("rolling on the floor laughing face") which is jarring.
# Covers the major emoji blocks; rare unmapped codepoints just pass through.
EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"   # emoticons
    "\U0001F300-\U0001F5FF"   # symbols & pictographs
    "\U0001F680-\U0001F6FF"   # transport
    "\U0001F700-\U0001F77F"   # alchemical
    "\U0001F780-\U0001F7FF"   # geometric ext
    "\U0001F800-\U0001F8FF"   # arrows ext
    "\U0001F900-\U0001F9FF"   # supplemental symbols & pictographs
    "\U0001FA00-\U0001FA6F"   # chess, symbols & pictographs ext-a
    "\U0001FA70-\U0001FAFF"   # symbols & pictographs ext-b
    "\U00002600-\U000026FF"   # misc symbols
    "\U00002700-\U000027BF"   # dingbats
    "\U0001F1E0-\U0001F1FF"   # flags
    "‍"                    # ZWJ
    "️"                    # variation selector-16
    "]+",
    flags=re.UNICODE,
)


MD_MARKUP_RE = re.compile(r"[*_`#~]+")


def _scrub_for_tts(text: str) -> str:
    text = EMOJI_RE.sub(" ", text)
    text = MD_MARKUP_RE.sub("", text)
    return text.strip()


def build_messages_for(conv: Conversation, speaker_idx: int) -> list[dict]:
    """Transcript from one participant's POV.

    Its own prior messages are 'assistant', everyone else's are 'user'.
    Thinking is controlled solely via Ollama's `think: true/false` API
    parameter — no model-side directives, no system-prompt nudging.
    """
    speaker = conv.participants[speaker_idx]
    out: list[dict] = []
    system_parts = []
    if (speaker.system or "").strip():
        system_parts.append(speaker.system.strip())
    mem = avatar_memory.load_memory(speaker.name)
    if mem:
        system_parts.append(
            "Things you remember about the user from past conversations:\n" + mem +
            "\n\nWhen you learn a new durable fact worth keeping, save it with the remember tool."
        )
    if system_parts:
        out.append({"role": "system", "content": "\n\n".join(system_parts)})
    for m in conv.messages:
        if m.sender < 0:
            continue  # internal note (was -1 referee or -2 system)
        if not m.content.strip():
            # Skip empty turns. Sending an empty assistant message confuses
            # chat models (they try to "complete" it, dumping reasoning into
            # the wrong field). Skipping empty user messages just keeps the
            # transcript clean.
            continue
        if m.sender == speaker_idx:
            out.append({"role": "assistant", "content": m.content})
        else:
            other = conv.participants[m.sender]
            prefix = f"[{other.name}]: " if conv.mode == "llm_llm" else ""
            out.append({"role": "user", "content": prefix + m.content})
    return out


# ---------- Turn execution ----------

async def run_llm_turn(conv: Conversation, speaker_idx: int) -> str:
    """Stream one LLM turn from Ollama. qwen3.6 routes reasoning into the
    `thinking` field (visible in the UI under "show thinking") and the actual
    reply into `content` (streamed as text + TTS'd as sentences complete)."""
    speaker = conv.participants[speaker_idx]
    assert speaker.kind == "llm" and speaker.model

    msg = Message(id=str(uuid.uuid4()), sender=speaker_idx, content="")
    # Build the request BEFORE appending the new (empty) placeholder. If we
    # appended first, Ollama would receive a trailing empty assistant message,
    # which qwen3 interprets as "complete this empty turn" and dumps its
    # reasoning into content. Verified empirically with qwen3.6:35b.
    request_messages = build_messages_for(conv, speaker_idx)
    conv.messages.append(msg)
    await broadcast(conv.id, {"type": "message_start", "message": asdict(msg)})

    tts_queue: Optional[asyncio.Queue] = asyncio.Queue() if conv.avatar_mode else None
    tts_task: Optional[asyncio.Task] = (
        asyncio.create_task(_tts_worker(conv, msg.id, tts_queue)) if tts_queue else None
    )
    sentence_buffer = ""
    # Working message list for the tool loop. The model may ask to call tools
    # (web_search / fetch_url); we run them, append the results, and re-query
    # until it produces a normal answer. Capped so a tool-happy model can't loop
    # forever. Casual messages never trigger a call — the model only uses tools
    # when it needs info past its knowledge cutoff.
    convo = list(request_messages)
    MAX_TOOL_ROUNDS = 5
    try:
        for _round in range(MAX_TOOL_ROUNDS):
            round_content = ""
            tool_calls = None
            async for c_delta, t_delta, tcs in ollama_stream(
                speaker.model, convo, think=conv.thinking_mode,
                tools=web_tools.TOOL_SCHEMAS + [avatar_memory.REMEMBER_TOOL_SCHEMA],
            ):
                if conv._abort.is_set():
                    break
                if tcs:
                    tool_calls = tcs
                # Ollama splits the stream: c_delta is answer text, t_delta is
                # reasoning. Tool-call rounds carry no content, so streaming it
                # straight to TTS is safe.
                if c_delta:
                    round_content += c_delta
                    msg.content += c_delta
                    await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": c_delta})
                    if tts_queue is not None:
                        sentence_buffer += c_delta
                        sentences, sentence_buffer = _pop_sentences(sentence_buffer)
                        for s in sentences:
                            await tts_queue.put(s)
                if t_delta:
                    msg.thinking += t_delta
                    await broadcast(conv.id, {"type": "thinking", "id": msg.id, "delta": t_delta})

            if conv._abort.is_set() or not tool_calls:
                break

            # Model requested tools. Record its tool-call turn, run each tool,
            # feed the results back, and loop for another round.
            convo.append({"role": "assistant", "content": round_content, "tool_calls": tool_calls})
            for tc in tool_calls:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args or "{}")
                    except json.JSONDecodeError:
                        args = {}
                note = f"\n🔎 {name}({', '.join(f'{k}={v}' for k, v in args.items())})\n"
                msg.thinking += note
                await broadcast(conv.id, {"type": "thinking", "id": msg.id, "delta": note})
                if name == "remember":
                    result = avatar_memory.append_memory(speaker.name, args.get("fact", ""))
                else:
                    result = await web_tools.execute_tool(name, args)
                convo.append({"role": "tool", "content": result, "tool_name": name})
    except Exception as e:
        msg.content += f"\n\n[error: {e}]"
        await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": f"\n\n[error: {e}]"})

    # Tail flush: any remaining partial sentence in the clean buffer.
    if tts_queue is not None:
        if sentence_buffer.strip():
            await tts_queue.put(sentence_buffer.strip())
        await tts_queue.put(None)
        if tts_task is not None:
            try:
                await asyncio.wait_for(tts_task, timeout=60)
            except asyncio.TimeoutError:
                tts_task.cancel()
        # Push 3 seconds of silence into Ditto so its LMDM lookahead buffer
        # (~80 frames) gets flushed — otherwise the last ~3s of real audio
        # never produces frames, and the avatar stops moving before the
        # audio finishes.
        if not conv._abort.is_set() and conv.id in _avatar_sessions:
            try:
                silence = np.zeros(24000 * 3, dtype=np.float32)
                bridge = await _get_bridge()
                await bridge.feed_audio_24k(conv.id, silence)
            except Exception as e:
                print(f"[avatar {conv.id}] tail-silence failed: {e!r}")

    await broadcast(conv.id, {"type": "message_end", "id": msg.id})
    persist(conv)
    print(f"[{conv.title}] {speaker.name} ◀ {msg.content.strip()}", flush=True)
    return msg.content


async def llm_llm_loop(conv: Conversation) -> None:
    """Ping-pong between the two LLM participants until stopped or max_turns."""
    try:
        # Find which LLM should go next: whoever isn't the last speaker, or idx 0 if empty.
        idx = 0
        if conv.messages:
            last = next((m for m in reversed(conv.messages) if m.sender >= 0), None)
            if last is not None:
                idx = 1 - last.sender  # assumes 2 LLMs

        while (
            not conv._stop.is_set()
            and conv.turns_taken < conv.max_turns
        ):
            await run_llm_turn(conv, idx)
            conv.turns_taken += 1
            await broadcast(conv.id, {"type": "state", "turns_taken": conv.turns_taken})

            if conv._stop.is_set():
                break

            idx = 1 - idx
            await asyncio.sleep(0)  # yield
    finally:
        conv.running = False
        conv._stop.clear()
        conv._abort.clear()
        await broadcast(conv.id, {"type": "state", "running": False, "turns_taken": conv.turns_taken})


# ---------- FastAPI app ----------

app = FastAPI()


def _print_urls() -> None:
    """Print the access banner last (after the warmup checklist) so it stays
    visible. Host/IP come from start.sh via env; skip if launched directly."""
    tls = os.environ.get("TLS_HOST")
    if not tls:
        return
    lan = os.environ.get("LAN_IP", "")
    lines = [
        "",
        "================================================================",
        "  Open the app at:",
        "    This machine:    http://127.0.0.1:8765",
        f"    Other devices:   https://{tls}:8443   (accept the cert on first visit)",
    ]
    if lan:
        lines.append(f"                     https://{lan}:8443   (if .local doesn't resolve)")
    lines += ["================================================================", ""]
    print("\n".join(lines), flush=True)


async def _warm_step(label: str, coro) -> None:
    """Run one warmup step and print an aligned status line as it completes."""
    import time
    t0 = time.perf_counter()
    try:
        await coro
        print(f"  {label:.<26} ✓  ({time.perf_counter() - t0:.1f}s)", flush=True)
    except Exception as e:
        print(f"  {label:.<26} ✗  {e!r}", flush=True)


async def _ollama_load(model: str) -> None:
    async with httpx.AsyncClient(timeout=120) as client:
        await client.post(f"{OLLAMA_URL}/api/chat",
                          json={"model": model, "messages": [], "stream": False})


async def _ditto_warm() -> None:
    bridge = await _get_bridge()
    await bridge.open_session("__warmup__", ref_image_path=str(AVATAR_REF), max_size=512)
    await bridge.close_session("__warmup__")


async def _warmup() -> None:
    """Preload the heavy subsystems in the background so the first real query
    doesn't race cold-start. The server is already serving by the time this
    runs. Each step is independent; a failure is shown and the rest continue."""
    # Let uvicorn finish printing its own startup lines first, so the checklist
    # below prints as one contiguous block instead of interleaving with them.
    await asyncio.sleep(0.8)
    print("\n──────────── warming up subsystems ────────────", flush=True)
    await _warm_step("Whisper (STT)", asyncio.to_thread(_get_whisper))
    await _warm_step("Kokoro (TTS)", asyncio.to_thread(_synthesize, "Ready.", "af_heart"))
    if AVATAR_REF.exists():
        await _warm_step("Ditto avatar", _ditto_warm())
    models = {p.model for c in conversations.values()
              for p in c.participants if p.kind == "llm" and p.model}
    for m in sorted(models):
        await _warm_step(f"Ollama: {m}", _ollama_load(m))
    print("──────────── ready for first query ────────────", flush=True)
    if not VERBOSE:
        print("  (--verbose for detail · worker log: /tmp/ditto-worker.log)", flush=True)
    _print_urls()


@app.on_event("startup")
async def on_startup() -> None:
    load_all()
    print(f"loaded {len(conversations)} conversation(s) from {DATA_DIR}")
    asyncio.create_task(_warmup())


@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Tear down the Ditto bridge so the worker subprocess doesn't leak.
    Without this, every uvicorn restart orphans a worker holding many GB
    of GPU memory until manually killed."""
    global _bridge
    for conv in conversations.values():
        if conv._task and not conv._task.done():
            conv._abort.set()
            conv._task.cancel()
    for cid in list(_avatar_sessions.keys()):
        try:
            await close_avatar_session(cid)
        except Exception:
            pass
    if _bridge is not None:
        try:
            await _bridge.shutdown()
        except Exception as e:
            print(f"bridge shutdown error: {e!r}")
        _bridge = None
        print("ditto bridge shut down")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.middleware("http")
async def no_cache_static(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith(("/static/", "/assets/")):
        response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/api/models")
async def api_models():
    try:
        return {"models": await ollama_models()}
    except Exception as e:
        return {"models": [], "error": str(e)}


@app.post("/api/stt")
async def api_stt(file: UploadFile):
    audio = await file.read()
    try:
        text = await asyncio.to_thread(_transcribe, audio)
    except Exception as e:
        return {"text": "", "error": str(e)}
    return {"text": text}


@app.get("/api/conversations")
async def api_list():
    return [{"id": c.id, "title": c.title, "mode": c.mode, "running": c.running} for c in conversations.values()]


@app.post("/api/conversations")
async def api_create(spec: dict):
    cid = str(uuid.uuid4())[:8]
    participants = [Participant(**p) for p in spec["participants"]]
    conv = Conversation(
        id=cid,
        title=spec.get("title") or f"conv-{cid}",
        mode=spec["mode"],
        participants=participants,
        max_turns=int(spec.get("max_turns", 20)),
        avatar_mode=bool(spec.get("avatar_mode", False)),
        tts_voice=str(spec.get("tts_voice", "af_heart")),
        thinking_mode=bool(spec.get("thinking_mode", False)),
    )
    conversations[cid] = conv
    persist(conv)
    return conv.to_public()


@app.get("/api/conversations/{cid}")
async def api_get(cid: str):
    conv = conversations.get(cid)
    if not conv:
        return {"error": "not found"}
    return conv.to_public()


@app.patch("/api/conversations/{cid}")
async def api_patch(cid: str, patch: dict):
    conv = conversations.get(cid)
    if not conv:
        return {"error": "not found"}
    if "title" in patch:
        conv.title = patch["title"] or conv.title
    if "max_turns" in patch:
        conv.max_turns = int(patch["max_turns"])
    if "avatar_mode" in patch:
        conv.avatar_mode = bool(patch["avatar_mode"])
    if "tts_voice" in patch:
        conv.tts_voice = str(patch["tts_voice"])
    if "thinking_mode" in patch:
        conv.thinking_mode = bool(patch["thinking_mode"])
    for pu in patch.get("participants", []):
        i = int(pu["index"])
        if 0 <= i < len(conv.participants):
            p = conv.participants[i]
            if "name" in pu: p.name = pu["name"]
            if "model" in pu: p.model = pu["model"]
            if "system" in pu: p.system = pu["system"]
    persist(conv)
    await broadcast(cid, {"type": "snapshot", "conversation": conv.to_public()})
    return conv.to_public()


@app.delete("/api/conversations/{cid}")
async def api_delete(cid: str):
    conv = conversations.pop(cid, None)
    if conv:
        conv._stop.set()
        if conv._task:
            conv._task.cancel()
    await close_avatar_session(cid)
    delete_persisted(cid)
    return {"ok": True}


@app.websocket("/ws/{cid}")
async def ws_endpoint(ws: WebSocket, cid: str):
    await ws.accept()
    conv = conversations.get(cid)
    if not conv:
        await ws.send_json({"type": "error", "msg": "no such conversation"})
        await ws.close()
        return

    sockets.setdefault(cid, set()).add(ws)
    await ws.send_json({"type": "snapshot", "conversation": conv.to_public()})

    if conv.avatar_mode:
        asyncio.create_task(ensure_avatar_session(cid))

    try:
        while True:
            data = await ws.receive_json()
            t = data.get("type")

            if t == "user_message":
                idx = next((i for i, p in enumerate(conv.participants) if p.kind == "user"), None)
                if idx is None:
                    await ws.send_json({"type": "error", "msg": "no user participant"})
                    continue
                msg = Message(id=str(uuid.uuid4()), sender=idx, content=data["content"])
                conv.messages.append(msg)
                print(f"[{conv.title}] {conv.participants[idx].name} ▶ {msg.content}", flush=True)
                await broadcast(cid, {"type": "message_start", "message": asdict(msg)})
                await broadcast(cid, {"type": "message_end", "id": msg.id})
                persist(conv)

                # In user↔LLM mode, auto-reply with the LLM participant.
                if conv.mode == "user_llm" and not conv.running:
                    llm_idx = next((i for i, p in enumerate(conv.participants) if p.kind == "llm"), None)
                    if llm_idx is not None:
                        # A user message starts a fresh turn — clear any leftover
                        # stop/abort from a previous reply.
                        conv._stop.clear()
                        conv._abort.clear()
                        if conv.avatar_mode:
                            if conv.turns_taken > 0:
                                # Reset Ditto each turn with a fresh session so the
                                # previous turn's lookahead can't bleed in (clean
                                # lip-sync). The worker frees GPU memory on close, so
                                # this no longer accumulates; the shared open/close
                                # lock keeps it race-free. Await the reopen so audio
                                # isn't fed before the session is ready; cap the wait
                                # so a slow reopen can't hang the reply.
                                await close_avatar_session(cid)
                                try:
                                    await asyncio.wait_for(ensure_avatar_session(cid), timeout=8.0)
                                except asyncio.TimeoutError:
                                    asyncio.create_task(ensure_avatar_session(cid))
                                except Exception as e:
                                    print(f"[avatar {cid}] reopen failed: {e!r}")
                            else:
                                asyncio.create_task(ensure_avatar_session(cid))
                        conv.running = True
                        await broadcast(cid, {"type": "state", "running": True})

                        async def reply():
                            try:
                                await run_llm_turn(conv, llm_idx)
                                conv.turns_taken += 1
                            finally:
                                conv.running = False
                                await broadcast(cid, {"type": "state", "running": False, "turns_taken": conv.turns_taken})

                        conv._task = asyncio.create_task(reply())

            elif t == "start":
                if conv.mode == "llm_llm" and not conv.running:
                    if "max_turns" in data:
                        conv.max_turns = int(data["max_turns"])
                    conv._stop.clear()
                    conv._abort.clear()
                    conv.running = True
                    await broadcast(cid, {"type": "state", "running": True})
                    conv._task = asyncio.create_task(llm_llm_loop(conv))

            elif t == "stop":
                conv._stop.set()
                await broadcast(cid, {"type": "state", "running": False})

            elif t == "abort":
                conv._stop.set()
                conv._abort.set()
                # Close the Ditto session entirely. The SDK has no mid-stream
                # reset — close() joins its worker threads, the only clean
                # way to flush its pipeline state. Next user_message reopens
                # in the background (~5s warm, ~30s cold under VRAM pressure).
                await close_avatar_session(cid)
                await broadcast(cid, {"type": "state", "running": False})

            elif t == "reset_turns":
                conv.turns_taken = 0
                await broadcast(cid, {"type": "state", "turns_taken": 0})
                persist(conv)

            elif t == "clear_messages":
                conv._stop.set()
                conv._abort.set()
                conv.messages = []
                conv.turns_taken = 0
                conv.running = False
                persist(conv)
                await broadcast(cid, {"type": "snapshot", "conversation": conv.to_public()})

            elif t == "seed":
                # Insert a seed message from a given participant without triggering a reply.
                idx = int(data["sender"])
                msg = Message(id=str(uuid.uuid4()), sender=idx, content=data["content"])
                conv.messages.append(msg)
                await broadcast(cid, {"type": "message_start", "message": asdict(msg)})
                await broadcast(cid, {"type": "message_end", "id": msg.id})
                persist(conv)

    except WebSocketDisconnect:
        pass
    finally:
        sockets.get(cid, set()).discard(ws)
        # When the last client for this conversation leaves (e.g. you switched
        # to another chat), close its Ditto session. Otherwise sessions leak —
        # each chat you visit leaves one running on the worker, piling up GPU
        # memory until the worker degrades and animation stops everywhere.
        if not sockets.get(cid):
            try:
                await close_avatar_session(cid)
            except Exception:
                pass


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/assets", StaticFiles(directory=ASSET_DIR), name="assets")

"""Multi-conversation Ollama playground.

Run: uv run --with-requirements requirements.txt uvicorn server:app --reload
Then open http://localhost:8000
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
import soundfile as sf
from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

OLLAMA_URL = "http://localhost:11434"
STATIC_DIR = Path(__file__).parent / "static"
DATA_DIR = Path(__file__).parent / "data"
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
    sender: int                     # participant index; -1 = referee; -2 = system note
    content: str
    thinking: str = ""

@dataclass
class Referee:
    model: str
    system: str
    cadence: int = 2                # run every N turns
    intervene: bool = False         # if True, inject into participants' context

@dataclass
class Conversation:
    id: str
    title: str
    mode: str                       # "user_llm" | "llm_llm"
    participants: list[Participant]
    messages: list[Message] = field(default_factory=list)
    referee: Optional[Referee] = None
    max_turns: int = 20
    turns_taken: int = 0            # counts LLM turns only
    avatar_mode: bool = False       # enable TTS + viseme broadcasting
    tts_voice: str = "af_heart"     # Kokoro voice id
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
            "referee": asdict(self.referee) if self.referee else None,
            "max_turns": self.max_turns,
            "turns_taken": self.turns_taken,
            "avatar_mode": self.avatar_mode,
            "tts_voice": self.tts_voice,
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
            referee = Referee(**data["referee"]) if data.get("referee") else None
            conv = Conversation(
                id=data["id"],
                title=data["title"],
                mode=data["mode"],
                participants=participants,
                messages=messages,
                referee=referee,
                max_turns=int(data.get("max_turns", 20)),
                turns_taken=int(data.get("turns_taken", 0)),
                avatar_mode=bool(data.get("avatar_mode", False)),
                tts_voice=str(data.get("tts_voice", "af_heart")),
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


# ---------- Ollama ----------

async def ollama_models() -> list[str]:
    async with httpx.AsyncClient(timeout=10) as client:
        r = await client.get(f"{OLLAMA_URL}/api/tags")
        r.raise_for_status()
        return sorted(m["name"] for m in r.json().get("models", []))


async def ollama_stream(model: str, messages: list[dict]):
    """Yield (content_delta, thinking_delta) pairs from Ollama /api/chat."""
    payload = {"model": model, "messages": messages, "stream": True}
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
                yield m.get("content", ""), m.get("thinking", "")
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
    """Pop sentences, synthesize, broadcast tts_chunk events."""
    seq = 0
    while True:
        sentence = await queue.get()
        if sentence is None:
            return
        try:
            wav = await asyncio.to_thread(_synthesize, sentence, conv.tts_voice)
        except Exception as e:
            print(f"TTS error: {e}")
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
            seq += 1
        queue.task_done()


THINK_TAG_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


async def strip_think_tags(conv: Conversation, msg: Message) -> None:
    cleaned = THINK_TAG_RE.sub("", msg.content).lstrip()
    if cleaned != msg.content:
        msg.content = cleaned
        await broadcast(conv.id, {"type": "replace_content", "id": msg.id, "content": cleaned})


REPLY_NUDGE = (
    "If you reason inside <think>...</think> tags, you MUST write your actual "
    "reply AFTER the closing </think> tag. Keep reasoning concise. The reply "
    "outside the think tags is what the other participant will see — never "
    "leave it empty."
)


def build_messages_for(conv: Conversation, speaker_idx: int) -> list[dict]:
    """Transcript from one participant's POV.

    Its own prior messages are 'assistant', everyone else's (including referee
    interventions) are 'user'. Referee messages are only included as visible
    context if referee.intervene is True.
    """
    speaker = conv.participants[speaker_idx]
    out: list[dict] = []
    system_text = (speaker.system or "").strip()
    system_text = f"{system_text}\n\n{REPLY_NUDGE}" if system_text else REPLY_NUDGE
    out.append({"role": "system", "content": system_text})
    for m in conv.messages:
        if m.sender == -2:
            continue  # internal note
        if m.sender == -1:
            if conv.referee and conv.referee.intervene:
                out.append({"role": "user", "content": f"[REFEREE]: {m.content}"})
            continue
        if m.sender == speaker_idx:
            out.append({"role": "assistant", "content": m.content})
        else:
            other = conv.participants[m.sender]
            prefix = f"[{other.name}]: " if conv.mode == "llm_llm" else ""
            out.append({"role": "user", "content": prefix + m.content})
    return out


def build_messages_for_referee(conv: Conversation) -> list[dict]:
    assert conv.referee
    transcript_lines = []
    for m in conv.messages:
        if m.sender == -2:
            continue
        if m.sender == -1:
            transcript_lines.append(f"REFEREE (you, earlier): {m.content}")
        else:
            name = conv.participants[m.sender].name
            transcript_lines.append(f"{name}: {m.content}")
    transcript = "\n".join(transcript_lines) or "(no messages yet)"
    return [
        {"role": "system", "content": f"{conv.referee.system}\n\n{REPLY_NUDGE}"},
        {"role": "user", "content": f"Transcript so far:\n\n{transcript}\n\nProvide your commentary."},
    ]


# ---------- Turn execution ----------

async def run_llm_turn(conv: Conversation, speaker_idx: int) -> str:
    speaker = conv.participants[speaker_idx]
    assert speaker.kind == "llm" and speaker.model

    msg = Message(id=str(uuid.uuid4()), sender=speaker_idx, content="")
    conv.messages.append(msg)
    await broadcast(conv.id, {"type": "message_start", "message": asdict(msg)})

    tts_enabled = conv.avatar_mode
    tts_queue: Optional[asyncio.Queue] = asyncio.Queue() if tts_enabled else None
    tts_task: Optional[asyncio.Task] = (
        asyncio.create_task(_tts_worker(conv, msg.id, tts_queue)) if tts_queue else None
    )
    sentence_buffer = ""
    last_content_offset = 0  # position in msg.content up to which we've considered sentences

    try:
        async for c_delta, t_delta in ollama_stream(speaker.model, build_messages_for(conv, speaker_idx)):
            if conv._abort.is_set():
                break
            if c_delta:
                msg.content += c_delta
                await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": c_delta})
                if tts_queue is not None:
                    # Operate on the suffix of content that hasn't been split into sentences yet,
                    # but exclude any leading think-block (server strips after stream completes).
                    sentence_buffer += c_delta
                    # Drop everything up to and including a closing </think> if present in the buffer.
                    if "</think>" in sentence_buffer:
                        sentence_buffer = sentence_buffer.split("</think>", 1)[1].lstrip()
                    if "<think>" in sentence_buffer:
                        # we're inside a think block — drop the buffer; will reset on </think>
                        continue
                    sentences, sentence_buffer = _pop_sentences(sentence_buffer)
                    for s in sentences:
                        await tts_queue.put(s)
            if t_delta:
                msg.thinking += t_delta
                await broadcast(conv.id, {"type": "thinking", "id": msg.id, "delta": t_delta})
    except Exception as e:
        msg.content += f"\n\n[error: {e}]"
        await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": f"\n\n[error: {e}]"})

    # Flush remaining buffer as a final sentence (if any).
    if tts_queue is not None:
        if "</think>" in sentence_buffer:
            sentence_buffer = sentence_buffer.split("</think>", 1)[1].lstrip()
        if sentence_buffer.strip() and "<think>" not in sentence_buffer:
            await tts_queue.put(sentence_buffer.strip())
        await tts_queue.put(None)  # signal end
        if tts_task is not None:
            try:
                await asyncio.wait_for(tts_task, timeout=60)
            except asyncio.TimeoutError:
                tts_task.cancel()

    await strip_think_tags(conv, msg)
    if not msg.content.strip() and msg.thinking.strip():
        msg.content = msg.thinking
        msg.thinking = ""
        await broadcast(conv.id, {"type": "replace_content", "id": msg.id, "content": msg.content})
        await broadcast(conv.id, {"type": "promote_thinking", "id": msg.id})

    await broadcast(conv.id, {"type": "message_end", "id": msg.id})
    persist(conv)
    return msg.content


async def run_referee(conv: Conversation) -> None:
    if not conv.referee:
        return
    msg = Message(id=str(uuid.uuid4()), sender=-1, content="")
    conv.messages.append(msg)
    await broadcast(conv.id, {"type": "message_start", "message": asdict(msg)})
    try:
        async for c_delta, t_delta in ollama_stream(conv.referee.model, build_messages_for_referee(conv)):
            if conv._abort.is_set():
                break
            if c_delta:
                msg.content += c_delta
                await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": c_delta})
            if t_delta:
                msg.thinking += t_delta
                await broadcast(conv.id, {"type": "thinking", "id": msg.id, "delta": t_delta})
    except Exception as e:
        msg.content += f"\n\n[referee error: {e}]"
        await broadcast(conv.id, {"type": "token", "id": msg.id, "delta": f"\n\n[referee error: {e}]"})
    await strip_think_tags(conv, msg)
    if not msg.content.strip() and msg.thinking.strip():
        msg.content = msg.thinking
        msg.thinking = ""
        await broadcast(conv.id, {"type": "replace_content", "id": msg.id, "content": msg.content})
        await broadcast(conv.id, {"type": "promote_thinking", "id": msg.id})
    await broadcast(conv.id, {"type": "message_end", "id": msg.id})
    persist(conv)


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

            if conv.referee and conv.turns_taken % conv.referee.cadence == 0:
                await run_referee(conv)

            idx = 1 - idx
            await asyncio.sleep(0)  # yield
    finally:
        conv.running = False
        conv._stop.clear()
        conv._abort.clear()
        await broadcast(conv.id, {"type": "state", "running": False, "turns_taken": conv.turns_taken})


# ---------- FastAPI app ----------

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    load_all()
    print(f"loaded {len(conversations)} conversation(s) from {DATA_DIR}")


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


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
    referee = Referee(**spec["referee"]) if spec.get("referee") else None
    conv = Conversation(
        id=cid,
        title=spec.get("title") or f"conv-{cid}",
        mode=spec["mode"],
        participants=participants,
        referee=referee,
        max_turns=int(spec.get("max_turns", 20)),
        avatar_mode=bool(spec.get("avatar_mode", False)),
        tts_voice=str(spec.get("tts_voice", "af_heart")),
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
    for pu in patch.get("participants", []):
        i = int(pu["index"])
        if 0 <= i < len(conv.participants):
            p = conv.participants[i]
            if "name" in pu: p.name = pu["name"]
            if "model" in pu: p.model = pu["model"]
            if "system" in pu: p.system = pu["system"]
    if "referee" in patch:
        r = patch["referee"]
        if r is None:
            conv.referee = None
        elif conv.referee:
            if "model" in r: conv.referee.model = r["model"]
            if "system" in r: conv.referee.system = r["system"]
            if "cadence" in r: conv.referee.cadence = int(r["cadence"])
            if "intervene" in r: conv.referee.intervene = bool(r["intervene"])
        else:
            conv.referee = Referee(**r)
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
                await broadcast(cid, {"type": "message_start", "message": asdict(msg)})
                await broadcast(cid, {"type": "message_end", "id": msg.id})
                persist(conv)

                # In user↔LLM mode, auto-reply with the LLM participant.
                if conv.mode == "user_llm" and not conv.running:
                    llm_idx = next((i for i, p in enumerate(conv.participants) if p.kind == "llm"), None)
                    if llm_idx is not None:
                        conv.running = True
                        await broadcast(cid, {"type": "state", "running": True})

                        async def reply():
                            try:
                                await run_llm_turn(conv, llm_idx)
                                conv.turns_taken += 1
                                if conv.referee and conv.turns_taken % conv.referee.cadence == 0:
                                    await run_referee(conv)
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


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

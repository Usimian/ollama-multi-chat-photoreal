# Ollama Multi-Chat

A small web app for running multiple simultaneous conversations against a local
[Ollama](https://ollama.com) instance. You can chat with a model yourself,
watch two models argue with each other, and optionally drop a third "referee"
model into the conversation to observe or moderate.

Each conversation can use a different model per participant, and Ollama keeps
the models resident in VRAM so switching between conversations doesn't pay a
reload cost (assuming `OLLAMA_MAX_LOADED_MODELS` is high enough).

## Prerequisites

- Ollama running on `http://localhost:11434` with at least one model pulled.
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv) for installing deps.
- `espeak-ng` system package, required by Kokoro TTS:
  `sudo apt install espeak-ng`.
- Recommended Ollama env (the bigger your VRAM budget, the more headroom):
  - `OLLAMA_MAX_LOADED_MODELS=3` — so referee + two agents can coexist.
  - `OLLAMA_KEEP_ALIVE=15m` — keep models warm between turns.
  - `OLLAMA_NUM_PARALLEL=2` — lowering this trades a bit of throughput for a
    much smaller per-model KV cache. Useful if you run with very long contexts.

## Install

```
cd ollama-multi-chat
uv venv .venv
uv pip install --python .venv -r requirements.txt
```

## Run

```
./start.sh         # foreground; Ctrl+C to stop
./start-bg.sh      # background; logs to /tmp/chatapp.log, pid in /tmp/chatapp.pid
./stop.sh          # stops the background instance
```

Then open <http://127.0.0.1:8765>.

## Modes

**user ↔ llm** — classic chat. Type, get a streamed reply. Enter sends,
Shift+Enter inserts a newline.

**llm ↔ llm** — two LLM participants ping-pong. Set a system prompt for each
to give them personas/goals, choose a `max_turns` limit, hit ▶ to start.
Pause (⏸) lets the current turn finish then stops. **abort** cuts the current
stream immediately. **reset turns** rewinds the counter without clearing
messages. **clear** wipes the transcript but keeps participant config.

## Referee

Optional third LLM that watches the conversation. Triggers every `cadence`
LLM turns.

- **Observe only** (default): commentary shows up in the UI as yellow italic
  bubbles. The agents don't see it.
- **Inject into participants**: the referee's output is added to the agents'
  next-turn context as `[REFEREE]: …`, so they can react to it.

The referee is configured at conversation creation and can be edited later
via the **edit** button.

The **hide referee** button toggles visibility of referee bubbles in the
chat. It does NOT stop the referee from running (so any "inject" effect
still happens) — purely a UI declutter.

## Reasoning ("thinking") models

For models that emit `<think>…</think>` blocks (qwen3.x, deepseek-r1,
gpt-oss):

- Thinking is rendered into a separate gray italic block beneath each
  message, hidden by default.
- Toggle **show thinking** to watch it stream live.
- A system-prompt nudge is automatically appended to every LLM participant
  asking the model to keep its reply outside the think tags. If a model
  still emits only thinking (some qwen personas do this), the server falls
  back to promoting the thinking content into the bubble so the chat doesn't
  stall.
- Leading `<think>…</think>` tags accidentally emitted into `content` are
  stripped after each turn.

## Avatar + voice mode (user ↔ llm)

When you tick **avatar + voice mode** while creating or editing a
`user ↔ llm` conversation, the chat UI flips into a voice-first layout:

- A 3D ReadyPlayerMe avatar is rendered with three.js.
- The mic button (and the **spacebar**) record your voice via the browser's
  MediaRecorder. On release, the clip is posted to `/api/stt`,
  faster-whisper transcribes it, and the transcript is sent as your turn.
- As the LLM streams its reply, sentences are pulled off as soon as they
  complete and synthesized through Kokoro. Audio chunks are streamed back
  over the websocket and played in order with no audible gap.
- The avatar's jaw/mouth blendshapes are driven by the RMS of the currently
  playing audio (simple amplitude lipsync — good enough for v1).

System requirements added by this mode:
- `espeak-ng` installed system-wide (`sudo apt install espeak-ng`).
- Browser model + voice weights downloaded on first use (~470 MB for
  Whisper small.en, ~330 MB for Kokoro-82M, cached in
  `~/.cache/huggingface`).

To use a different avatar, edit `DEFAULT_AVATAR_URL` in `static/avatar.js`
or point it at any ReadyPlayerMe glb with ARKit/Oculus visemes baked in:
`https://models.readyplayer.me/<your-id>.glb?morphTargets=ARKit,Oculus%20Visemes`.

Available voices: `af_heart`, `af_bella`, `am_michael`, `am_adam`,
`bf_emma`, `bm_george` (see Kokoro docs for the full list).

## Persistence

Every change is auto-saved as a JSON file under `./data/<conv-id>.json`.
On server startup all files in that directory are loaded back into memory,
so transcripts, participants, system prompts, and referee config survive
restarts. The files are human-readable; you can edit or delete them by hand.

Live state (whether the loop is running) is **not** persisted — every
conversation boots as idle. You'll click ▶ again to resume an llm↔llm loop.

## File layout

```
server.py           FastAPI + websocket backend
requirements.txt    Python deps
start.sh            Foreground run script
start-bg.sh         Background run script
stop.sh             Stop script for background instance
static/
  index.html        Single-page UI
  app.js            Client logic and websocket handling
  style.css         Dark theme
data/               Auto-created; one JSON file per conversation
```

## REST endpoints

| Method | Path                                  | Purpose                                |
| ------ | ------------------------------------- | -------------------------------------- |
| GET    | `/`                                   | Serves the SPA                         |
| GET    | `/api/models`                         | Lists installed Ollama models          |
| POST   | `/api/stt`                            | Multipart audio → `{text}` (Whisper)   |
| GET    | `/api/conversations`                  | Lists conversations (id, title, mode)  |
| POST   | `/api/conversations`                  | Creates a new conversation             |
| GET    | `/api/conversations/{id}`             | Returns full conversation              |
| PATCH  | `/api/conversations/{id}`             | Updates title / participants / referee |
| DELETE | `/api/conversations/{id}`             | Deletes (in-memory + on-disk)          |
| WS     | `/ws/{id}`                            | Live event stream for a conversation   |

## WebSocket events

**Client → server:** `user_message`, `start`, `stop`, `abort`,
`reset_turns`, `clear_messages`, `seed`.

**Server → client:** `snapshot`, `message_start`, `token`, `thinking`,
`tts_chunk`, `replace_content`, `promote_thinking`, `message_end`, `state`,
`error`.

"""Per-avatar long-term memory, stored as human-readable markdown.

One file per avatar (keyed by participant name) under data/memory/. The file
is loaded into the system prompt each turn so the avatar always knows what it
remembers, and the model appends to it via the `remember` tool when it learns
something worth keeping across conversations.
"""
from __future__ import annotations

import re
from pathlib import Path

MEMORY_DIR = Path(__file__).parent / "data" / "memory"

# Tool the model calls to persist a durable fact. Dispatched in run_llm_turn
# (it needs the speaker's name as the file key, which tool args don't carry).
REMEMBER_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a durable fact about the user or your relationship with them "
            "so you'll know it in future conversations — their name, preferences, "
            "important things they tell you. Use it whenever you learn something "
            "worth keeping long-term. Don't save trivia or one-off details."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "fact": {
                    "type": "string",
                    "description": "The fact to remember, as a short standalone sentence.",
                },
            },
            "required": ["fact"],
        },
    },
}


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-") or "default"


def memory_path(name: str) -> Path:
    return MEMORY_DIR / f"{_slug(name)}.md"


def load_memory(name: str) -> str:
    p = memory_path(name)
    return p.read_text().strip() if p.exists() else ""


def append_memory(name: str, fact: str) -> str:
    """Append one fact as a markdown bullet. Returns a short status for the model."""
    fact = " ".join((fact or "").split())
    if not fact:
        return "Nothing to remember."
    p = memory_path(name)
    existing = p.read_text() if p.exists() else ""
    if fact.lower() in existing.lower():
        return "Already remembered that."
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        if existing and not existing.endswith("\n"):
            f.write("\n")
        f.write(f"- {fact}\n")
    return f"Remembered: {fact}"

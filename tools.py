"""Web tools the LLM can call: search the web and fetch page text.

Exposed to Ollama as function tools. The model decides when to use them, so
ordinary chat never triggers a network call — they only fire when the model
needs information past its knowledge cutoff.
"""
from __future__ import annotations

import asyncio
import json

import httpx

# Tool schemas in Ollama's function-calling format. Sent with each /api/chat
# request; the model emits tool_calls referencing these by name.
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web for current information that may be past your "
                "knowledge cutoff (news, prices, weather, recent events, etc.). "
                "Returns a list of result titles, URLs, and snippets. Follow up "
                "with fetch_url to read a promising result in full."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "max_results": {
                        "type": "integer",
                        "description": "How many results to return (default 5, max 10).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch a web page and return its main readable text. Use after "
                "web_search to read a specific result, or for a URL the user gave you."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The full URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
]


def _web_search(query: str, max_results: int = 5) -> str:
    from ddgs import DDGS

    max_results = max(1, min(int(max_results or 5), 10))
    rows = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            rows.append({
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "snippet": r.get("body", ""),
            })
    if not rows:
        return f"No results for {query!r}."
    return json.dumps(rows, ensure_ascii=False)


def _fetch_url(url: str, max_chars: int = 6000) -> str:
    import trafilatura

    with httpx.Client(timeout=15, follow_redirects=True, headers={"User-Agent": "Mozilla/5.0"}) as client:
        resp = client.get(url)
        resp.raise_for_status()
        html = resp.text
    text = trafilatura.extract(html, include_links=False, include_comments=False) or ""
    text = text.strip()
    if not text:
        return f"Could not extract readable text from {url}."
    if len(text) > max_chars:
        text = text[:max_chars] + "\n…[truncated]"
    return text


async def execute_tool(name: str, args: dict) -> str:
    """Run a tool by name off the event loop. Always returns a string for the
    model; tool errors are returned as text so the model can react, not raised."""
    try:
        if name == "web_search":
            return await asyncio.to_thread(_web_search, args.get("query", ""), args.get("max_results", 5))
        if name == "fetch_url":
            return await asyncio.to_thread(_fetch_url, args.get("url", ""))
        return f"Unknown tool {name!r}."
    except Exception as e:
        return f"Tool {name} failed: {e!r}"

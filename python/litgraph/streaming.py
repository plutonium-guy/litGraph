"""LangChain-shaped event taxonomy on top of native litGraph streams.

`astream_events(stream)` translates litGraph's native event objects
(`{kind, text, …}` duck-typed) into the LangChain `astream_events` v2
shape so code written against LangChain's typed event vocabulary
ports without rewriting the listener.

Native event kinds → LangChain event names:

    text                  → on_chat_model_stream
    tool_call_delta       → on_tool_start / on_tool_stream
    tool_call_complete    → on_tool_end
    thinking              → on_chat_model_stream (kind="thinking")
    usage                 → on_chat_model_end (with usage payload)
    finish                → on_chat_model_end

Emits LangChain v2 envelope shape:

    {
        "event":   str,            # e.g. "on_chat_model_stream"
        "name":    str,            # the source identifier
        "run_id":  str,            # uuid4 per stream
        "data":    {"chunk": ...} or {"output": ...} or {"input": ...},
    }

Designed for porting in the easy direction. Native code keeps using
the native stream; LangChain-port code calls `astream_events`.
"""
from __future__ import annotations

import uuid
from typing import Any, AsyncIterable, Iterable, Iterator


__all__ = ["astream_events", "stream_events"]


_LC_EVENT_BY_KIND = {
    "text": "on_chat_model_stream",
    "thinking": "on_chat_model_stream",
    "tool_call_delta": "on_tool_stream",
    "tool_call_complete": "on_tool_end",
    "usage": "on_chat_model_end",
    "finish": "on_chat_model_end",
}


def _envelope(event: str, name: str, run_id: str, data: dict[str, Any]) -> dict[str, Any]:
    return {"event": event, "name": name, "run_id": run_id, "data": data}


def _native_to_lc(ev: Any, run_id: str, name: str) -> dict[str, Any] | None:
    """Translate one native event to its LangChain envelope. Returns
    None for kinds with no LC equivalent (silently dropped)."""
    kind = getattr(ev, "kind", None)
    if kind is None and isinstance(ev, dict):
        kind = ev.get("kind") or ev.get("type")
    if not isinstance(kind, str):
        return None
    lc = _LC_EVENT_BY_KIND.get(kind)
    if lc is None:
        return None

    text = getattr(ev, "text", None)
    if text is None and isinstance(ev, dict):
        text = ev.get("text") or ev.get("content")

    if kind == "tool_call_complete":
        result = getattr(ev, "result", None)
        if result is None and isinstance(ev, dict):
            result = ev.get("result") or ev.get("output")
        return _envelope(lc, name, run_id, {"output": result})
    if kind == "tool_call_delta":
        return _envelope(lc, name, run_id, {"chunk": text or ""})
    if kind in ("text", "thinking"):
        return _envelope(lc, name, run_id, {"chunk": text or "", "kind": kind})
    if kind in ("finish", "usage"):
        usage = getattr(ev, "usage", None)
        if usage is None and isinstance(ev, dict):
            usage = ev.get("usage")
        finish_reason = getattr(ev, "finish_reason", None)
        if finish_reason is None and isinstance(ev, dict):
            finish_reason = ev.get("finish_reason")
        payload: dict[str, Any] = {}
        if usage is not None:
            payload["usage"] = usage
        if finish_reason is not None:
            payload["finish_reason"] = finish_reason
        return _envelope(lc, name, run_id, {"output": payload})
    return None


def stream_events(
    stream: Iterable[Any], *, name: str = "litgraph", run_id: str | None = None
) -> Iterator[dict[str, Any]]:
    """Synchronous translator. Wraps a sync iterator of native events."""
    rid = run_id or str(uuid.uuid4())
    yield _envelope("on_chat_model_start", name, rid, {"input": None})
    saw_finish = False
    for ev in stream:
        out = _native_to_lc(ev, rid, name)
        if out is None:
            continue
        if out["event"] == "on_chat_model_end":
            saw_finish = True
        yield out
    if not saw_finish:
        # Some streams end without a finish event; emit a synthetic
        # one so listeners that wait for `on_chat_model_end` always
        # see it.
        yield _envelope("on_chat_model_end", name, rid, {"output": {}})


async def astream_events(
    stream: AsyncIterable[Any], *, name: str = "litgraph", run_id: str | None = None
):
    """Async translator. Mirrors LangChain's `astream_events` shape."""
    rid = run_id or str(uuid.uuid4())
    yield _envelope("on_chat_model_start", name, rid, {"input": None})
    saw_finish = False
    async for ev in stream:
        out = _native_to_lc(ev, rid, name)
        if out is None:
            continue
        if out["event"] == "on_chat_model_end":
            saw_finish = True
        yield out
    if not saw_finish:
        yield _envelope("on_chat_model_end", name, rid, {"output": {}})

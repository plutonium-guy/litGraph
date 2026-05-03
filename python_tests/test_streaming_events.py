"""Tests for litgraph.streaming — astream_events / stream_events shim."""
from __future__ import annotations

from types import SimpleNamespace

from litgraph.streaming import stream_events


def _ev(kind: str, **extra) -> SimpleNamespace:
    return SimpleNamespace(kind=kind, **extra)


def test_emits_start_then_end_when_stream_empty():
    out = list(stream_events([]))
    assert [e["event"] for e in out] == ["on_chat_model_start", "on_chat_model_end"]


def test_text_kind_becomes_chat_model_stream():
    out = list(stream_events([_ev("text", text="hello")]))
    chunks = [e for e in out if e["event"] == "on_chat_model_stream"]
    assert len(chunks) == 1
    assert chunks[0]["data"]["chunk"] == "hello"
    assert chunks[0]["data"]["kind"] == "text"


def test_thinking_also_chat_model_stream_with_kind():
    out = list(stream_events([_ev("thinking", text="reasoning")]))
    chunks = [e for e in out if e["event"] == "on_chat_model_stream"]
    assert chunks[0]["data"]["kind"] == "thinking"


def test_tool_call_complete_becomes_on_tool_end():
    out = list(stream_events([
        _ev("tool_call_delta", text='{"a": 1'),
        _ev("tool_call_complete", result={"sum": 3}),
    ]))
    kinds = [e["event"] for e in out]
    assert "on_tool_stream" in kinds
    assert "on_tool_end" in kinds
    end = next(e for e in out if e["event"] == "on_tool_end")
    assert end["data"]["output"] == {"sum": 3}


def test_finish_emits_chat_model_end_once():
    out = list(stream_events([
        _ev("text", text="hi"),
        _ev("finish", finish_reason="stop"),
    ]))
    ends = [e for e in out if e["event"] == "on_chat_model_end"]
    # One from finish, NOT a synthetic one (it saw a real finish).
    assert len(ends) == 1
    assert ends[0]["data"]["output"]["finish_reason"] == "stop"


def test_synthetic_finish_added_when_missing():
    out = list(stream_events([_ev("text", text="hi")]))
    ends = [e for e in out if e["event"] == "on_chat_model_end"]
    assert len(ends) == 1


def test_unknown_kind_dropped_silently():
    out = list(stream_events([_ev("not_a_real_kind", text="x")]))
    # Only start + synthetic end.
    assert [e["event"] for e in out] == ["on_chat_model_start", "on_chat_model_end"]


def test_run_id_consistent_across_envelope():
    out = list(stream_events([_ev("text", text="x")], run_id="fixed-id"))
    rids = {e["run_id"] for e in out}
    assert rids == {"fixed-id"}


def test_name_propagates():
    out = list(stream_events([], name="my-graph"))
    names = {e["name"] for e in out}
    assert names == {"my-graph"}


def test_dict_events_supported():
    """Native events may be dicts (from JSON-shape sources)."""
    out = list(stream_events([
        {"kind": "text", "text": "from-dict"},
        {"kind": "finish", "finish_reason": "stop"},
    ]))
    chunks = [e for e in out if e["event"] == "on_chat_model_stream"]
    assert chunks[0]["data"]["chunk"] == "from-dict"


def test_usage_event_carries_payload():
    out = list(stream_events([_ev("usage", usage={"prompt": 1, "completion": 2})]))
    ends = [e for e in out if e["event"] == "on_chat_model_end"]
    assert ends[0]["data"]["output"]["usage"] == {"prompt": 1, "completion": 2}

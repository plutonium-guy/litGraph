"""Tests for `litgraph.testing` — MockChatModel, MockEmbeddings, MockTool.

These mocks are public API for users writing tests against litGraph
without an LLM credential. The tests here lock the contract.
"""
from __future__ import annotations

import math

from litgraph.testing import MockChatModel, MockEmbeddings, MockTool


# ---- MockChatModel ----

def test_mock_chat_returns_scripted_replies_in_order():
    m = MockChatModel(replies=["alpha", "beta"])
    assert m.invoke([{"role": "user", "content": "x"}])["content"] == "alpha"
    assert m.invoke([{"role": "user", "content": "y"}])["content"] == "beta"


def test_mock_chat_cycles_replies_when_exhausted():
    m = MockChatModel(replies=["a", "b"])
    outs = [m.invoke([{"role": "user", "content": str(i)}])["content"] for i in range(5)]
    assert outs == ["a", "b", "a", "b", "a"]


def test_mock_chat_records_calls_for_assertions():
    m = MockChatModel(replies=["ok"])
    m.invoke([{"role": "user", "content": "ping"}])
    m.invoke([{"role": "user", "content": "pong"}])
    assert len(m.calls) == 2
    assert m.calls[0][0]["content"] == "ping"
    assert m.calls[1][0]["content"] == "pong"


def test_mock_chat_returns_assistant_role_and_usage():
    m = MockChatModel(replies=["ok"], usage={"prompt": 5, "completion": 1, "total": 6})
    out = m.invoke([{"role": "user", "content": "hi"}])
    assert out["role"] == "assistant"
    assert out["tool_calls"] == []
    assert out["usage"] == {"prompt": 5, "completion": 1, "total": 6}


def test_mock_chat_dict_reply_overrides_full_message():
    m = MockChatModel(replies=[{"role": "assistant", "content": "rich", "tool_calls": [{"name": "f"}]}])
    out = m.invoke([{"role": "user", "content": "x"}])
    assert out["content"] == "rich"
    assert out["tool_calls"] == [{"name": "f"}]


def test_mock_chat_stream_yields_word_events_then_finish():
    m = MockChatModel(replies=["one two three"])
    events = list(m.stream([{"role": "user", "content": "x"}]))
    text_events = [e for e in events if e.kind == "text"]
    assert len(text_events) == 3
    assert text_events[0].text.strip() == "one"
    assert events[-1].kind == "finish"
    assert events[-1].finish_reason == "stop"


def test_mock_chat_on_invoke_callback_fires():
    seen: list[list] = []
    m = MockChatModel(replies=["ok"], on_invoke=lambda msgs: seen.append(msgs))
    m.invoke([{"role": "user", "content": "spy"}])
    assert len(seen) == 1
    assert seen[0][0]["content"] == "spy"


def test_mock_chat_with_structured_output_returns_self():
    m = MockChatModel(replies=["x"])
    assert m.with_structured_output(dict) is m


# ---- MockEmbeddings ----

def test_mock_embeddings_deterministic_same_text_same_vec():
    e = MockEmbeddings(dim=8)
    a = e.embed(["hello"])[0]
    b = e.embed(["hello"])[0]
    assert a == b


def test_mock_embeddings_l2_normalised():
    e = MockEmbeddings(dim=16)
    v = e.embed(["some text"])[0]
    norm = math.sqrt(sum(x * x for x in v))
    assert abs(norm - 1.0) < 1e-6


def test_mock_embeddings_different_text_different_vec():
    e = MockEmbeddings(dim=8)
    a, b = e.embed(["one", "two"])
    assert a != b


def test_mock_embeddings_dim_is_respected():
    e = MockEmbeddings(dim=32)
    v = e.embed(["x"])[0]
    assert len(v) == 32


def test_mock_embeddings_records_calls():
    e = MockEmbeddings(dim=4)
    e.embed(["a", "b"])
    e.embed(["c"])
    assert e.calls == [["a", "b"], ["c"]]


def test_mock_embeddings_zero_dim_rejected():
    import pytest
    with pytest.raises(ValueError):
        MockEmbeddings(dim=0)


def test_mock_embeddings_query_alias():
    e = MockEmbeddings(dim=8)
    v_batch = e.embed(["only"])[0]
    v_query = e.embed_query("only")
    assert v_batch == v_query


# ---- MockTool ----

def test_mock_tool_returns_fixed_value():
    t = MockTool("add", returns={"sum": 42})
    assert t.invoke({"a": 1, "b": 2}) == {"sum": 42}


def test_mock_tool_records_invocations():
    t = MockTool("add", returns=None)
    t.invoke({"a": 1})
    t.invoke({"a": 2})
    assert t.calls == [{"a": 1}, {"a": 2}]


def test_mock_tool_side_effect_overrides_returns():
    t = MockTool("compute", returns="ignored", side_effect=lambda args: args["a"] * 2)
    assert t.invoke({"a": 21}) == 42


def test_mock_tool_run_alias_works():
    t = MockTool("x", returns=1)
    assert t.run({}) == 1
    assert t({}) == 1
    assert len(t.calls) == 2


def test_mock_tool_default_schema_accepts_anything():
    t = MockTool("any")
    assert t.schema["type"] == "object"
    assert t.schema["additionalProperties"] is True


def test_mock_tool_custom_schema_preserved():
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    t = MockTool("x", schema=schema)
    assert t.schema == schema

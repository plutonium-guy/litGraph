"""Native deterministic ChatModel for offline agent and harness tests."""
from __future__ import annotations

import pytest

from litgraph import create_agent
from litgraph.agents import ReactAgent
from litgraph.providers import ScriptedChatModel
from litgraph.testing import ScriptedChatModel as NativeModelFromTesting
from litgraph.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def test_direct_invoke_consumes_replies_and_records_calls():
    model = ScriptedChatModel(["one", "two"])
    first = model.invoke([{"role": "user", "content": "a"}])
    second = model.invoke(
        [{"role": "user", "content": "b"}],
        temperature=0.2,
        max_tokens=50,
    )
    assert first["text"] == "one"
    assert second["text"] == "two"
    assert model.call_count == 2
    assert model.remaining == 0
    assert model.calls[1]["messages"][0]["content"] == "b"
    assert model.calls[1]["options"] == {
        "temperature": pytest.approx(0.2),
        "max_tokens": 50,
    }


def test_testing_namespace_exports_native_model():
    assert NativeModelFromTesting is ScriptedChatModel


def test_exhaustion_fails_instead_of_hiding_extra_model_calls():
    model = ScriptedChatModel(["only"])
    model.invoke([])
    with pytest.raises(RuntimeError, match="scripted replies exhausted"):
        model.invoke([])


def test_cycle_and_reset_are_explicit():
    model = ScriptedChatModel(["same"], cycle=True)
    assert [model.invoke([])["text"] for _ in range(3)] == ["same"] * 3
    assert model.remaining is None
    model.reset()
    assert model.call_count == 0


def test_scripted_error_surfaces_as_provider_failure():
    model = ScriptedChatModel([{"error": "planned outage"}])
    with pytest.raises(RuntimeError, match="planned outage"):
        model.invoke([])


def test_stream_chunks_text_and_emits_done():
    model = ScriptedChatModel(["hello"], stream_chunk_size=2)
    events = list(model.stream([{"role": "user", "content": "hi"}]))
    assert [event["text"] for event in events if event["type"] == "delta"] == [
        "he",
        "ll",
        "o",
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["text"] == "hello"


def test_react_agent_runs_tool_loop_without_http_server():
    model = ScriptedChatModel(
        [
            {
                "tool_calls": [
                    {
                        "id": "add-1",
                        "name": "add",
                        "arguments": {"a": 2, "b": 3},
                    }
                ]
            },
            "5",
        ]
    )
    agent = ReactAgent(model, [add])
    state = agent.invoke("What is 2 + 3?")
    assert state["messages"][-1]["content"] == "5"
    assert model.call_count == 2
    second_turn = model.calls[1]["messages"]
    assert any(message.get("tool_call_id") == "add-1" for message in second_turn)


def test_batteries_included_harness_runs_fully_offline():
    harness = create_agent(
        ScriptedChatModel(["offline answer"]),
        instructions="Answer deterministically.",
    )
    result = harness.run("question")
    assert result.success is True
    assert result.output == "offline answer"

"""Batteries-included Python agent harness."""
from __future__ import annotations

import json

import pytest

from litgraph.harness import AgentHarness, create_agent


class _Agent:
    def __init__(self, answers=None):
        self.answers = answers or {}
        self.calls = []

    def invoke(self, user_input):
        self.calls.append(user_input)
        answer = self.answers.get(user_input, f"answer: {user_input}")
        return {"messages": [{"role": "assistant", "content": answer}]}

    def stream_tokens(self, user_input):
        yield {"type": "token_delta", "text": "hello"}
        yield {
            "type": "final",
            "messages": [{"role": "assistant", "content": "hello"}],
        }


def test_run_normalizes_native_agent_state():
    agent = _Agent()
    result = AgentHarness(agent).run("hi")
    assert result.success is True
    assert result.output == "answer: hi"
    assert result.state["messages"][-1]["content"] == "answer: hi"
    assert result.elapsed_ms >= 0
    assert agent.calls == ["hi"]


def test_run_can_capture_errors_without_raising():
    class Broken:
        def invoke(self, _):
            raise RuntimeError("offline")

    result = AgentHarness(Broken()).run("hi", raise_errors=False)
    assert result.success is False
    assert result.output == ""
    assert result.error == "RuntimeError: offline"


def test_run_raises_by_default():
    class Broken:
        def invoke(self, _):
            raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        AgentHarness(Broken()).run("hi")


def test_trace_is_jsonl_and_event_hook_receives_records(tmp_path):
    events = []
    trace = tmp_path / "traces" / "agent.jsonl"
    harness = AgentHarness(_Agent(), trace_path=trace, on_event=events.append)
    harness.run("trace me")
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["type"] for row in rows] == ["run_start", "run_end"]
    assert rows[0]["run_id"] == rows[1]["run_id"]
    assert len(events) == 2


def test_stream_prefers_token_stream_and_traces_events(tmp_path):
    trace = tmp_path / "stream.jsonl"
    events = list(AgentHarness(_Agent(), trace_path=trace).stream("hi"))
    assert [event["type"] for event in events] == ["token_delta", "final"]
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [row["type"] for row in rows] == [
        "run_start",
        "agent_event",
        "agent_event",
        "run_end",
    ]


def test_evaluate_uses_agent_output():
    harness = AgentHarness(_Agent({"France?": "Paris", "Japan?": "Tokyo"}))
    report = harness.evaluate(
        [
            {"input": "France?", "expected": "Paris"},
            {"input": "Japan?", "expected": "Tokyo"},
        ],
        scorers=[{"name": "exact_match"}],
        max_parallel=1,
    )
    assert report["aggregate"]["means"]["exact_match"] == 1.0


def test_build_composes_native_deep_agent(monkeypatch):
    import litgraph.deep_agent as deep_agent

    built = _Agent()
    captured = {}

    def fake_create(model, **kwargs):
        captured["model"] = model
        captured.update(kwargs)
        return built

    monkeypatch.setattr(deep_agent, "create_deep_agent", fake_create)
    harness = create_agent(
        "model",
        tools=["tool"],
        instructions="be precise",
        agents_md_path="AGENTS.md",
        skills_dir="skills",
        max_iterations=7,
    )
    assert harness.agent is built
    assert captured == {
        "model": "model",
        "tools": ["tool"],
        "system_prompt": "be precise",
        "agents_md_path": "AGENTS.md",
        "skills_dir": "skills",
        "max_iterations": 7,
        "with_planning": True,
        "with_vfs": True,
    }


def test_requires_invoke_and_positive_iteration_limit(monkeypatch):
    with pytest.raises(TypeError, match="invoke"):
        AgentHarness(object())
    with pytest.raises(ValueError, match="positive"):
        AgentHarness.build("model", max_iterations=0)

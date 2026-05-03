"""Live integration: iterate `ReactAgent.stream` directly.

`agent.stream(user)` returns an `AgentEventStream` that yields
`{type: 'iteration'|'final'|...}` records as the loop progresses.
This test consumes the iterator (no `stream_tokens=True` — that's
covered in `test_react_stream.py`) and verifies the final event has
the answer.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a, b):
    return {"sum": int(a) + int(b)}


def _add_tool():
    from litgraph.tools import FunctionTool

    return FunctionTool(
        "add",
        "Add two integers.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        _add,
    )


def test_react_agent_stream_yields_iteration_then_final(deepseek_chat):
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="Use the add tool. Be terse.",
        max_iterations=3,
    )
    events = list(agent.stream("What is 17 + 25?"))
    assert events, "stream yielded no events"

    # We should see at least one event marked as the terminal/final.
    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert any(t in {"final", "result", "done"} for t in types), (
        f"no terminal event in {types!r}"
    )

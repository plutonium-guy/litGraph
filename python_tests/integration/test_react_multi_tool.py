"""Live integration: `ReactAgent` chooses between multiple tools.

The agent is given two tools (`add` + `subtract`); we ask it
arithmetic questions that should pick the correct one. Verifies
DeepSeek's tool-call routing is wired through the agent loop.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a, b):
    return {"result": int(a) + int(b)}


def _sub(a, b):
    return {"result": int(a) - int(b)}


def _add_tool():
    from litgraph.tools import FunctionTool

    return FunctionTool(
        "add",
        "Add two integers. Returns {result: int}.",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        _add,
    )


def _sub_tool():
    from litgraph.tools import FunctionTool

    return FunctionTool(
        "subtract",
        "Subtract two integers. Returns {result: int}.",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        _sub,
    )


def test_react_agent_picks_add(deepseek_chat):
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [_add_tool(), _sub_tool()],
        system_prompt="Use the right tool for the operation. Be terse.",
        max_iterations=4,
    )
    state = agent.invoke("What is 12 + 30? Use a tool.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "42" in (text or ""), f"agent answer wrong: {final!r}"


def test_react_agent_picks_subtract(deepseek_chat):
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [_add_tool(), _sub_tool()],
        system_prompt="Use the right tool for the operation. Be terse.",
        max_iterations=4,
    )
    state = agent.invoke("What is 50 - 8? Use a tool.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "42" in (text or ""), f"agent answer wrong: {final!r}"

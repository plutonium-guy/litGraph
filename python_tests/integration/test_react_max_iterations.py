"""Live integration: `ReactAgent` honours `max_iterations`.

Set a low cap and a deliberately tool-heavy prompt; the agent must
terminate (not infinite-loop) and the final state must report the
trace within the cap.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _noop(input):
    return {"echo": str(input)}


def _noop_tool():
    from litgraph.tools import FunctionTool

    return FunctionTool(
        "noop",
        "Echoes the input string.",
        {
            "type": "object",
            "properties": {"input": {"type": "string"}},
            "required": ["input"],
        },
        _noop,
    )


def test_react_agent_terminates_within_max_iterations(deepseek_chat):
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [_noop_tool()],
        system_prompt="Use the noop tool repeatedly. Never stop on your own.",
        max_iterations=2,
    )
    # Even with an instruction to loop forever, the cap must stop us.
    state = agent.invoke("Call noop with 'x' over and over.")
    msgs = state["messages"]
    # Trace contains the user, plus assistant/tool turns up to the cap.
    assert len(msgs) >= 1
    # We didn't hang; that's the test's main contract.
    # Tool turns should be bounded — count `tool` messages.
    tool_msgs = [
        m for m in msgs
        if (m.get("role") if isinstance(m, dict) else None) == "tool"
    ]
    # max_iterations=2 should produce at most ~2 tool turns.
    assert len(tool_msgs) <= 4, f"too many tool turns for cap=2: {len(tool_msgs)}"

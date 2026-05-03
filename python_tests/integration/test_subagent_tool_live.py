"""Live integration: `SubagentTool` lets a parent agent spawn a sub-agent.

The parent has one tool (`research_agent`) which is itself a ReactAgent
configured for math. The parent delegates an arithmetic question and
returns the sub-agent's answer.
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


def test_parent_agent_delegates_to_subagent(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import SubagentTool

    sub = ReactAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="You are a math sub-agent. Use the add tool. Reply with just the number.",
        max_iterations=4,
    )

    sub_tool = SubagentTool(
        "math_subagent",
        "Delegates an arithmetic question to a math-specialist sub-agent.",
        sub,
    )

    parent = ReactAgent(
        deepseek_chat,
        [sub_tool],
        system_prompt=(
            "You delegate math questions to the math_subagent tool. "
            "Pass the user's question through. Be terse."
        ),
        max_iterations=4,
    )

    state = parent.invoke("What is 17 + 25? Use the math_subagent.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "42" in (text or ""), f"parent failed to delegate: {final!r}"

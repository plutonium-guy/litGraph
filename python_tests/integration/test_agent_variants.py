"""Live integration: agent flavours beyond ReactAgent.

Covers TextReActAgent (transcript mode), PlanAndExecuteAgent (plan
+ step executor), and CritiqueReviseAgent (self-improvement loop).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a: int, b: int):
    return {"sum": int(a) + int(b)}


def _add_tool():
    from litgraph.tools import FunctionTool
    return FunctionTool(
        "add", "Add two integers.",
        {"type": "object",
         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
         "required": ["a", "b"]},
        _add,
    )


def test_text_react_agent_completes_arithmetic(deepseek_chat):
    """Transcript-mode ReAct: thought / action / observation / final."""
    from litgraph.agents import TextReActAgent

    agent = TextReActAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="Use the `add` tool for arithmetic.",
        max_iterations=4,
    )
    result = agent.invoke("What is 17 + 25? Use the add tool.")
    # `result` is a TextReactResult — has `final_answer` + `trace`.
    answer = result.final_answer if hasattr(result, "final_answer") else str(result)
    assert "42" in (answer or ""), f"final_answer lacked 42: {answer!r}"


def test_plan_and_execute_agent_finishes(deepseek_chat):
    """Plan-and-execute: planner emits a numbered list of steps;
    executor runs each. We just assert the agent terminates with
    a string-typed answer."""
    from litgraph.agents import PlanAndExecuteAgent

    agent = PlanAndExecuteAgent(
        planner=deepseek_chat,
        executor=deepseek_chat,
        tools=[_add_tool()],
        max_iterations_per_step=3,
    )
    result = agent.invoke("Compute 2+2 then 3+3. Reply with both sums.")
    text = result.final_answer if hasattr(result, "final_answer") else str(result)
    assert isinstance(text, str)
    assert text.strip()

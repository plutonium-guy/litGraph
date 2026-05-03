"""Live integration: built-in `CalculatorTool` via DeepSeek `ReactAgent`.

`CalculatorTool()` ships with the framework — sandboxed math expression
evaluator, no I/O. Verifies the agent picks it up and the answer
flows through.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_react_agent_uses_builtin_calculator(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import CalculatorTool

    agent = ReactAgent(
        deepseek_chat,
        [CalculatorTool()],
        system_prompt="Use the calculator tool for any arithmetic. Be terse.",
        max_iterations=4,
    )
    state = agent.invoke("What is 144 / 12? Use the calculator.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "12" in (text or ""), f"calculator answer wrong: {final!r}"

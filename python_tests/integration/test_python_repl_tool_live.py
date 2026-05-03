"""Live integration: `PythonReplTool` invoked via a DeepSeek `ReactAgent`.

The agent receives a math task, calls `python_repl` with code, and
returns the result. Exercises:
- Sandboxed subprocess execution
- ReactAgent's tool dispatch on a real provider
"""
from __future__ import annotations

import tempfile

import pytest


pytestmark = pytest.mark.integration


def test_react_agent_uses_python_repl_for_math(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import PythonReplTool

    with tempfile.TemporaryDirectory() as workdir:
        repl = PythonReplTool(working_dir=workdir, timeout_s=10)
        agent = ReactAgent(
            deepseek_chat,
            [repl],
            system_prompt=(
                "You have a python_repl tool. For math beyond simple arithmetic, "
                "WRITE PYTHON CODE that prints the answer and call the tool. "
                "Read the tool's stdout for the value, then reply with just the "
                "number."
            ),
            max_iterations=5,
        )
        # Pick a number the model is unlikely to compute correctly without help.
        state = agent.invoke("What is 19 * 23 * 17? Use the python_repl tool.")
        msgs = state["messages"]
        final = msgs[-1]
        text = final.get("content", "") if isinstance(final, dict) else str(final)
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        # 19 * 23 * 17 = 7429
        assert "7429" in (text or ""), f"agent answer wrong: {final!r}"

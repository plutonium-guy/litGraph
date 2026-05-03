"""Live integration: LangChain-compat `AgentExecutor` shim.

Builds a ReactAgent under the hood. Accepts the LangChain constructor
shape (`agent`-with-`.llm`, `tools`, `verbose`, `max_iterations`) and
forwards the loop. Verifies a minimal port-from-LangChain works.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


class _FakeAgent:
    """Stand-in for a LangChain `Agent` object — only `.llm` is read by
    the shim when `tools` is provided."""

    def __init__(self, llm):
        self.llm = llm


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


def test_agent_executor_runs_react_loop(deepseek_chat):
    from litgraph.compat import AgentExecutor

    fake = _FakeAgent(llm=deepseek_chat)
    executor = AgentExecutor(
        agent=fake,
        tools=[_add_tool()],
        verbose=True,  # ignored by shim — should not raise
        max_iterations=4,
    )
    # Most LangChain executors take a `.invoke({"input": "..."})` shape
    # OR direct string. Accept whatever the shim returns and verify the
    # answer threads through.
    out = executor.invoke("What is 7 + 35? Use the add tool.")
    text = (
        out if isinstance(out, str)
        else (out.get("output") or out.get("content") or out.get("text") or str(out))
    )
    assert "42" in str(text), f"executor result lacked answer: {out!r}"

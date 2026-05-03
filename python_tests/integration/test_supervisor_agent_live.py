"""Live integration: `SupervisorAgent` routes to named worker `ReactAgent`s.

The supervisor LLM picks a worker and hands off via the `handoff_to_<worker>`
tool. We register a `math` worker and a `chitchat` worker and verify
the supervisor delegates to the right one.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a, b):
    return {"result": int(a) + int(b)}


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


def test_supervisor_routes_to_math_worker(deepseek_chat):
    from litgraph.agents import ReactAgent, SupervisorAgent

    math_worker = ReactAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="You are the math expert. Use the add tool for arithmetic.",
        max_iterations=4,
    )
    chitchat_worker = ReactAgent(
        deepseek_chat,
        [],
        system_prompt="You are the chitchat agent. Be friendly and brief.",
        max_iterations=2,
    )

    sup = SupervisorAgent(
        deepseek_chat,
        {"math": math_worker, "chitchat": chitchat_worker},
        system_prompt=(
            "You delegate to workers. Route arithmetic to 'math', "
            "everything else to 'chitchat'."
        ),
        max_hops=4,
    )
    names = sup.worker_names()
    assert "math" in names and "chitchat" in names

    out = sup.invoke("What is 17 + 25?")
    # Supervisor returns a string or dict — tolerate both.
    text = out if isinstance(out, str) else (
        out.get("content") or out.get("text") or str(out)
    )
    assert "42" in str(text), f"supervisor failed to delegate: {out!r}"

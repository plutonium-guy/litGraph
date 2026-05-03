"""Live integration: `agents_extras.SwarmAgent` and `BigToolAgent` with DeepSeek.

`SwarmAgent` is the swarm-style multi-agent coordinator (peers, not
hierarchy). `BigToolAgent` retrieves the top-K relevant tools per turn
from a large pool. Both expand on the basic `ReactAgent` shape.
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


def test_swarm_agent_invoke_entry_agent(deepseek_chat):
    """`SwarmAgent` delegates to its `entry` agent. Fixed in iter 372:
    SwarmAgent now extracts the latest user message and passes a
    string when the inner agent is a known native ReactAgent (which
    only accepts `invoke(user: str)`)."""
    from litgraph.agents import ReactAgent
    from litgraph.agents_extras import SwarmAgent

    math_agent = ReactAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="You answer arithmetic. Use the add tool. Be terse.",
        max_iterations=3,
    )
    chitchat = ReactAgent(
        deepseek_chat,
        [],
        system_prompt="You handle non-math. Be terse.",
        max_iterations=2,
    )

    swarm = SwarmAgent(
        agents={"math": math_agent, "chitchat": chitchat},
        entry="math",
    )

    out = swarm.invoke("What is 17 + 25?")
    msgs = out["messages"] if isinstance(out, dict) else []
    last = msgs[-1] if msgs else out
    text = last.get("content", "") if isinstance(last, dict) else str(last)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "42" in (text or ""), f"swarm entry agent failed: {out!r}"


@pytest.mark.skip(reason="BigToolAgent needs an embeddings provider; DeepSeek has none — covered when an embedding key is supplied")
def test_big_tool_agent_constructible(deepseek_chat):  # pragma: no cover
    """`BigToolAgent(agent_factory, tools, embeddings, k=...)` requires
    a real embeddings provider to score the tool catalogue. DeepSeek
    doesn't expose embeddings, so this path is gated on a separate
    provider key (OpenAI/Cohere/Voyage/Jina/FastEmbed). See
    INTEGRATION_TESTS.md → Conditionally testable."""
    pass

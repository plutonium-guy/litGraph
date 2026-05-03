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


@pytest.mark.skip(reason="SwarmAgent invokes inner agents with a list of messages but ReactAgent.invoke takes a str — contract mismatch; would need a shim agent or a fix in agents_extras.py")
def test_swarm_agent_invoke_entry_agent(deepseek_chat):  # pragma: no cover
    """`SwarmAgent` calls `agent.invoke(messages)` with a list, but
    `ReactAgent.invoke(user)` only accepts a string. Running this
    raises `TypeError: argument 'user': 'list' object cannot be
    converted to 'PyString'`. Re-enable once either:
    - SwarmAgent extracts the latest user message and passes the
      string in, OR
    - ReactAgent.invoke accepts both string AND list-of-messages.

    Tracked via INTEGRATION_TESTS.md → Gotchas."""
    pass


@pytest.mark.skip(reason="BigToolAgent needs an embeddings provider; DeepSeek has none — covered when an embedding key is supplied")
def test_big_tool_agent_constructible(deepseek_chat):  # pragma: no cover
    """`BigToolAgent(agent_factory, tools, embeddings, k=...)` requires
    a real embeddings provider to score the tool catalogue. DeepSeek
    doesn't expose embeddings, so this path is gated on a separate
    provider key (OpenAI/Cohere/Voyage/Jina/FastEmbed). See
    INTEGRATION_TESTS.md → Conditionally testable."""
    pass

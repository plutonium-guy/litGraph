"""Live integration: `agents_extras.SwarmAgent` and `BigToolAgent`.

`SwarmAgent` is the swarm-style multi-agent coordinator (peers, not
hierarchy). `BigToolAgent` retrieves the top-K relevant tools per turn
from a large pool. Both expand on the basic `ReactAgent` shape.
"""
from __future__ import annotations

import pytest

from ._capabilities import NO_EMBEDDINGS_REASON, SUPPORTS_EMBEDDINGS


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


@pytest.mark.skipif(not SUPPORTS_EMBEDDINGS, reason=NO_EMBEDDINGS_REASON)
def test_big_tool_agent_selects_relevant_tools(deepseek_chat):
    """`BigToolAgent` embeds every tool's `(name + description)` once and
    hands the inner agent only the top-k for the query.

    Needs a real embeddings provider, so it is gated on
    `LITGRAPH_TEST_EMBED_MODEL` rather than skipped outright."""
    from litgraph.agents import ReactAgent
    from litgraph.agents_extras import BigToolAgent
    from litgraph.embeddings import OpenAIEmbeddings
    from litgraph.tools import FunctionTool

    from ._capabilities import EMBED_MODEL, embed_base_url, embed_dimensions

    embeddings = OpenAIEmbeddings(
        api_key="ollama",
        model=EMBED_MODEL,
        dimensions=embed_dimensions(),
        base_url=embed_base_url(),
    )

    def _tool(name: str, description: str):
        return FunctionTool(
            name,
            description,
            {"type": "object", "properties": {"x": {"type": "string"}}},
            lambda x=None: {"tool": name},
        )

    tools = [
        _tool("add", "Add two integers together."),
        _tool("send_email", "Send an email to a recipient."),
        _tool("resize_image", "Resize an image to given dimensions."),
        _tool("translate_text", "Translate text between languages."),
        _tool("book_flight", "Book a flight between two airports."),
    ]

    agent = BigToolAgent(
        lambda selected: ReactAgent(
            deepseek_chat, selected, system_prompt="Use a tool.", max_iterations=2
        ),
        tools,
        embeddings,
        k=2,
    )

    # Constructing it embeds the catalogue — the bug this guards against
    # was an AttributeError here, because native providers expose
    # `embed_documents`, not `embed`.
    assert len(agent._tool_vecs) == len(tools)

    picked = [t.name for t in agent._select("translate this sentence into French")]
    assert len(picked) == 2
    assert "translate_text" in picked, f"retrieval missed the obvious tool: {picked}"

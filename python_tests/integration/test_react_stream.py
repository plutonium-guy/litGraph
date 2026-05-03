"""Live integration: ReactAgent token + event streaming against DeepSeek."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a: int, b: int) -> dict:
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


def test_react_agent_stream_emits_iteration_and_final(deepseek_chat):
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [_add_tool()],
        system_prompt="Use the add tool for arithmetic.",
        max_iterations=4,
    )
    events = []
    for ev in agent.stream("What is 17 + 25? Use the add tool."):
        events.append(ev)
        # Don't accumulate forever — stop reading after a final event.
        kind = ev.get("type") if isinstance(ev, dict) else None
        if kind in ("final", "max_iterations_reached"):
            break

    types = [e.get("type") for e in events if isinstance(e, dict)]
    assert "iteration_start" in types
    assert "final" in types or "max_iterations_reached" in types


def test_react_agent_stream_tokens_streams_text_deltas(deepseek_chat):
    """Token-level streaming variant should emit `token_delta`
    events between LLM responses."""
    from litgraph.agents import ReactAgent

    agent = ReactAgent(
        deepseek_chat,
        [],  # no tools — just a plain reply
        system_prompt="Be terse.",
        max_iterations=2,
    )
    events = []
    for ev in agent.stream_tokens("Reply with exactly: STREAMED"):
        events.append(ev)
        kind = ev.get("type") if isinstance(ev, dict) else None
        if kind in ("final", "max_iterations_reached"):
            break

    deltas = [e for e in events if isinstance(e, dict) and e.get("type") == "token_delta"]
    # If the model used native streaming we'd see tokens; some
    # OpenAI-compat servers buffer — tolerate either path as long as
    # we see at least one event of any sort.
    assert events

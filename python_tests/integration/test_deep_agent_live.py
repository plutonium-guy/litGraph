"""Live integration: `create_deep_agent` end-to-end against DeepSeek.

`create_deep_agent` wires `PlanningTool` + `VirtualFilesystemTool`
(both auto-injected) onto a `ReactAgent`. Verify the factory's defaults
work and the agent answers a small task.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_create_deep_agent_default_factory(deepseek_chat):
    from litgraph.deep_agent import create_deep_agent

    agent = create_deep_agent(
        deepseek_chat,
        system_prompt=(
            "You are a terse assistant. For trivial questions, answer "
            "directly without invoking the planning tool."
        ),
        max_iterations=4,
    )
    state = agent.invoke("Reply with exactly: pong")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "pong" in (text or "").lower(), f"agent reply unexpected: {final!r}"


def test_create_deep_agent_without_planning_or_vfs(deepseek_chat):
    """Factory accepts `with_planning=False, with_vfs=False`. Agent
    becomes a plain ReactAgent with no auto-injected tools."""
    from litgraph.deep_agent import create_deep_agent

    agent = create_deep_agent(
        deepseek_chat,
        system_prompt="Be terse. Reply with exactly the word: ok",
        with_planning=False,
        with_vfs=False,
        max_iterations=2,
    )
    state = agent.invoke("ping")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert text.strip(), f"agent produced no text: {final!r}"

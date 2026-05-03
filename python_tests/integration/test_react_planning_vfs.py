"""Live integration: `PlanningTool` + `VirtualFilesystemTool` via DeepSeek.

Both tools are part of the deep-agents primitive set. Verify they're
instantiable, accepted by `ReactAgent`, and survive a real round-trip.
The `snapshot()` helpers expose state outside the tool-call protocol —
useful for assertions.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_planning_and_vfs_attach_to_agent(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import PlanningTool, VirtualFilesystemTool

    plan = PlanningTool()
    vfs = VirtualFilesystemTool(max_total_bytes=64_000)

    # Sanity: empty starting state.
    assert plan.snapshot() == []
    assert vfs.snapshot() == {}
    assert vfs.total_bytes() == 0

    agent = ReactAgent(
        deepseek_chat,
        [plan, vfs],
        system_prompt=(
            "You have a planning tool and a virtual filesystem. For trivial "
            "questions answer directly without invoking either tool."
        ),
        max_iterations=3,
    )
    state = agent.invoke("Reply with exactly: ok")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert text.strip(), f"agent produced no text: {final!r}"

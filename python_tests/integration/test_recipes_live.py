"""Live integration: `litgraph.recipes` patterns against DeepSeek."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_recipes_summarize_short_text(deepseek_chat):
    from litgraph.recipes import summarize

    short = "Photosynthesis converts light energy into chemical energy stored in glucose."
    out = summarize(
        short,
        model=deepseek_chat,
        chunk_size=2_000,  # short text → one chunk
    )
    assert isinstance(out["summary"], str)
    assert len(out["summary"]) > 0
    # Either word should appear in a faithful summary.
    lower = out["summary"].lower()
    assert "photosynthesis" in lower or "energy" in lower or "glucose" in lower


def test_recipes_multi_agent_routes(deepseek_chat):
    from litgraph.recipes import multi_agent

    class _Worker:
        def __init__(self, name):
            self.name = name
            self.calls = []
        def invoke(self, q):
            self.calls.append(q)
            return f"{self.name} answers: {q}"

    billing = _Worker("billing")
    tech = _Worker("tech")

    ma = multi_agent(
        {"billing": billing, "tech": tech},
        supervisor_model=deepseek_chat,
    )
    out = ma.invoke("My subscription was double-charged last month.")
    # Supervisor should pick "billing" — but tolerate either; both
    # are real workers and the assertion is just that *some* worker
    # was picked + invoked.
    assert out["chosen_role"] in ("billing", "tech")
    chosen = billing if out["chosen_role"] == "billing" else tech
    assert chosen.calls, f"chosen worker {chosen.name!r} got no calls"

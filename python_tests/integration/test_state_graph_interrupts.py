"""Live integration: `StateGraph.interrupt_before` / `interrupt_after`.

These wire human-in-the-loop pause points into a graph. We verify the
methods accept node names without raising and the graph still
compiles + invokes against DeepSeek.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_state_graph_interrupt_before_raises(deepseek_chat):
    """`interrupt_before('llm')` halts the graph before executing
    `llm` — invoke raises `RuntimeError: interrupted at node ...`.
    Caller is expected to handle the interrupt (resume / inspect)."""
    from litgraph.graph import END, START, StateGraph

    g = StateGraph()

    def call_llm(state):
        out = deepseek_chat.invoke(
            [{"role": "user", "content": state["q"]}], max_tokens=10
        )
        return {"a": out["text"]}

    g.add_node("llm", call_llm)
    g.add_edge(START, "llm")
    g.add_edge("llm", END)
    g.interrupt_before("llm")

    compiled = g.compile()
    with pytest.raises(RuntimeError, match="interrupted"):
        compiled.invoke({"q": "Reply: ok"})


def test_state_graph_no_interrupt_runs_normally(deepseek_chat):
    """Sanity check: same graph WITHOUT interrupt runs to completion."""
    from litgraph.graph import END, START, StateGraph

    g = StateGraph()

    def call_llm(state):
        out = deepseek_chat.invoke(
            [{"role": "user", "content": state["q"]}], max_tokens=10
        )
        return {"a": out["text"]}

    g.add_node("llm", call_llm)
    g.add_edge(START, "llm")
    g.add_edge("llm", END)

    compiled = g.compile()
    out = compiled.invoke({"q": "Reply: ok"})
    assert "a" in out
    assert out["a"].strip()

"""Live integration: StateGraph with a model node against DeepSeek."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_state_graph_with_model_node(deepseek_chat):
    from litgraph.graph import StateGraph, START, END

    g = StateGraph()

    def call_llm(state: dict) -> dict:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": state["question"]}],
            max_tokens=20,
        )
        return {"answer": out["text"]}

    def shout(state: dict) -> dict:
        return {"answer": state["answer"].upper()}

    g.add_node("llm", call_llm)
    g.add_node("shout", shout)
    g.add_edge(START, "llm")
    g.add_edge("llm", "shout")
    g.add_edge("shout", END)

    final = g.compile().invoke({"question": "Reply with exactly: hello"})
    assert "HELLO" in final["answer"]


def test_parallel_branches_both_call_model(deepseek_chat):
    from litgraph.graph import StateGraph, START, END

    g = StateGraph()

    def short(state):
        out = deepseek_chat.invoke(
            [{"role": "user", "content": "Reply: SHORT"}],
            max_tokens=10,
        )
        return {"short": out["text"]}

    def long(state):
        out = deepseek_chat.invoke(
            [{"role": "user", "content": "Reply: LONG"}],
            max_tokens=10,
        )
        return {"long": out["text"]}

    def merge(state):
        return {}

    g.add_node("short", short)
    g.add_node("long", long)
    g.add_node("merge", merge)
    g.add_edge(START, "short")
    g.add_edge(START, "long")
    g.add_edge("short", "merge")
    g.add_edge("long", "merge")
    g.add_edge("merge", END)

    final = g.compile().invoke({})
    # The model paraphrases — accept any non-empty response from each branch.
    assert final["short"]
    assert final["long"]

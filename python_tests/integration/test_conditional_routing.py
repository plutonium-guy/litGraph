"""Live integration: `StateGraph.add_conditional_edges` driven by model output.

The router callable inspects state and returns the next node name.
DeepSeek classifies the input as `math` or `text` and the graph
routes accordingly.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _build_router_graph(deepseek_chat):
    from litgraph.graph import END, START, StateGraph

    g = StateGraph()

    def classify(state: dict) -> dict:
        out = deepseek_chat.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Classify the user message as exactly one word: "
                        "either 'math' if it's an arithmetic question, "
                        "or 'text' otherwise. Reply with just the word."
                    ),
                },
                {"role": "user", "content": state["question"]},
            ],
            max_tokens=5,
        )
        label = out["text"].strip().lower().rstrip(".")
        label = label.split()[0] if label else "text"
        return {"label": label}

    def math_branch(_state: dict) -> dict:
        return {"answer": "MATH-PATH"}

    def text_branch(_state: dict) -> dict:
        return {"answer": "TEXT-PATH"}

    def route(state: dict) -> str:
        return "math_branch" if state["label"] == "math" else "text_branch"

    g.add_node("classify", classify)
    g.add_node("math_branch", math_branch)
    g.add_node("text_branch", text_branch)
    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route)
    g.add_edge("math_branch", END)
    g.add_edge("text_branch", END)
    return g.compile()


def test_conditional_edges_routes_to_math_branch(deepseek_chat):
    compiled = _build_router_graph(deepseek_chat)
    final = compiled.invoke({"question": "What is 2 + 2?"})
    assert final["answer"] == "MATH-PATH", f"expected MATH, got {final!r}"


def test_conditional_edges_routes_to_text_branch(deepseek_chat):
    compiled = _build_router_graph(deepseek_chat)
    final = compiled.invoke({"question": "Tell me a joke."})
    assert final["answer"] == "TEXT-PATH", f"expected TEXT, got {final!r}"

"""Pause a graph mid-flight, edit state, and resume.

Demonstrates the "human-in-the-loop" pattern: a graph runs to a
designated interrupt point, the caller inspects + mutates state, then
calls `compiled.resume(...)` to continue. Backed by an in-memory
checkpointer for the example; swap `SqliteSaver("./graph.db")` for
durability across processes.

Run:  python examples/checkpoint_resume.py
"""
from litgraph.graph import StateGraph, START, END


def plan(state: dict) -> dict:
    return {"plan": ["draft", "review"], "step": 0}


def draft(state: dict) -> dict:
    return {"draft": f"draft v{state.get('step', 0)}: hello world"}


def review(state: dict) -> dict:
    return {"reviewed": state["draft"].upper()}


def main() -> None:
    g = StateGraph()
    g.add_node("plan", plan)
    g.add_node("draft", draft)
    g.add_node("review", review)
    g.add_edge(START, "plan")
    g.add_edge("plan", "draft")
    g.add_edge("draft", "review")
    g.add_edge("review", END)

    # Pause before review so the caller can inspect / edit the draft.
    g.interrupt_before("review")
    compiled = g.compile()

    config = {"thread_id": "demo"}
    out = compiled.invoke({}, config=config)
    print("paused at:", out.get("__interrupt__", "no pending interrupt"))
    print("draft so far:", out.get("draft"))

    # Resume with an edited draft (HITL mutation):
    final = compiled.resume({"draft": "hand-edited: HELLO HUMAN"}, config=config)
    print("\nfinal:", final.get("reviewed"))


if __name__ == "__main__":
    main()

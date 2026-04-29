"""Subgraph composition — embed a CompiledGraph as a single node.

Production motivation: hierarchical / multi-agent workflows where each
"team" is its own graph, composed at the top level. LangGraph supports this
via `add_node` taking a CompiledStateGraph; we expose `add_subgraph` for
the same pattern (also clearer at call sites — readers see the composition
intent, not just a generic node)."""
from litgraph.graph import END, START, StateGraph


def test_subgraph_runs_to_completion_then_state_merges_back():
    # Child increments n twice; parent inc by 10 → run child → inc by 100.
    # Final n = 0 + 10 + 2 + 100 = 112.
    child = StateGraph()
    child.add_node("c1", lambda s: {"n": s["n"] + 1})
    child.add_node("c2", lambda s: {"n": s["n"] + 1})
    child.add_edge(START, "c1")
    child.add_edge("c1", "c2")
    child.add_edge("c2", END)
    child_compiled = child.compile()

    parent = StateGraph()
    parent.add_node("pre", lambda s: {"n": s["n"] + 10})
    parent.add_subgraph("team", child_compiled)
    parent.add_node("post", lambda s: {"n": s["n"] + 100})
    parent.add_edge(START, "pre")
    parent.add_edge("pre", "team")
    parent.add_edge("team", "post")
    parent.add_edge("post", END)

    out = parent.compile().invoke({"n": 0})
    assert out["n"] == 112


def test_subgraph_can_be_reused_across_multiple_parents():
    """The compiled subgraph is referenced via Arc internally — using it as
    a node in two different parent graphs works without reconstruction."""
    child = StateGraph()
    child.add_node("inc", lambda s: {"n": s["n"] + 5})
    child.add_edge(START, "inc")
    child.add_edge("inc", END)
    child_compiled = child.compile()

    p1 = StateGraph()
    p1.add_subgraph("worker", child_compiled)
    p1.add_edge(START, "worker")
    p1.add_edge("worker", END)
    assert p1.compile().invoke({"n": 0})["n"] == 5

    p2 = StateGraph()
    p2.add_node("seed", lambda s: {"n": 100})
    p2.add_subgraph("worker", child_compiled)
    p2.add_edge(START, "seed")
    p2.add_edge("seed", "worker")
    p2.add_edge("worker", END)
    assert p2.compile().invoke({})["n"] == 105


def test_subgraph_error_bubbles_up_to_parent():
    """If the subgraph fails, the parent invoke must surface the error
    rather than silently continuing with stale state."""
    def boom(_state):
        raise RuntimeError("child node exploded")

    child = StateGraph()
    child.add_node("boom", boom)
    child.add_edge(START, "boom")
    child.add_edge("boom", END)
    child_compiled = child.compile()

    parent = StateGraph()
    parent.add_subgraph("team", child_compiled)
    parent.add_edge(START, "team")
    parent.add_edge("team", END)

    try:
        parent.compile().invoke({})
    except RuntimeError as e:
        # The error path is parent → subgraph → child node failure.
        assert "exploded" in str(e), f"got: {e}"
    else:
        raise AssertionError("expected RuntimeError from child failure")


if __name__ == "__main__":
    fns = [
        test_subgraph_runs_to_completion_then_state_merges_back,
        test_subgraph_can_be_reused_across_multiple_parents,
        test_subgraph_error_bubbles_up_to_parent,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

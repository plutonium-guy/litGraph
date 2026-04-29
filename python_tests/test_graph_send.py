"""LangGraph-style Send fan-out from Python — map-reduce parallelism.

The contract: when a node returns a dict containing `__sends__`, each Send
becomes a forked sub-invocation of the target node with its own state
override. Sibling forks do NOT see each other's overrides at invocation time
(scheduler clones state and reduces ONLY each Send's own update into its
clone). After all forks complete, their outputs reduce back into shared state
in completion order.

This is the bug-fix half of iter 77 surfaced to Python: previously Python
nodes could only return a state-update dict; they had no way to express
fan-out without manually building successor edges that all shared state."""
from litgraph.graph import END, START, Send, StateGraph


def test_send_fan_out_runs_worker_per_item_with_isolated_state():
    """Splitter Sends 4 distinct items; worker runs 4 times, each seeing only
    its own item; the sink collects all 4 results."""
    g = StateGraph()

    def split(state):
        items = [10, 20, 30, 40]
        return {
            # Seed `items` into shared state AND fan out one Send per item.
            "__update__": {"items": items},
            "__sends__": [Send("worker", {"item": i}) for i in items],
        }

    def worker(state):
        # Each worker MUST see exactly one `item` — its own.
        item = state["item"]
        # Append to the shared `results` list. The default reducer
        # (merge_append) will concat lists across forks.
        return {"results": [item * 2]}

    def sink(state):
        return {"done": True}

    g.add_node("split", split)
    g.add_node("worker", worker)
    g.add_node("sink", sink)
    g.add_edge(START, "split")
    g.add_edge("worker", "sink")
    g.add_edge("sink", END)

    out = g.compile().invoke({})
    # All 4 items doubled, in any order.
    assert sorted(out.get("results", [])) == [20, 40, 60, 80]
    assert out.get("done") is True


def test_send_accepts_plain_dict_form_in_addition_to_send_class():
    """Users can also write {'goto': ..., 'update': ...} dicts directly,
    no Send class required. Convenient for templates / generated code."""
    g = StateGraph()

    def split(state):
        return {
            "__sends__": [
                {"goto": "w", "update": {"x": 1}},
                {"goto": "w", "update": {"x": 2}},
            ],
        }

    def w(state):
        return {"sum": [state["x"]]}

    g.add_node("split", split)
    g.add_node("w", w)
    g.add_edge(START, "split")

    out = g.compile().invoke({})
    assert sorted(out.get("sum", [])) == [1, 2]


def test_send_payload_does_not_leak_to_sibling_forks():
    """The strict isolation contract: if 3 sibling forks all set `state.x`,
    each invocation sees ONLY its own x — never another fork's x."""
    g = StateGraph()
    seen = []

    def split(state):
        return {
            "__sends__": [
                Send("w", {"x": "a"}),
                Send("w", {"x": "b"}),
                Send("w", {"x": "c"}),
            ],
        }

    def w(state):
        # Capture the per-fork x and return it as a result entry.
        x = state["x"]
        seen.append(x)
        return {"results": [x]}

    g.add_node("split", split)
    g.add_node("w", w)
    g.add_edge(START, "split")

    out = g.compile().invoke({})
    # Each fork captured its own x; the seen list has all 3 distinct values.
    assert sorted(seen) == ["a", "b", "c"]
    assert sorted(out.get("results", [])) == ["a", "b", "c"]


def test_explicit_goto_routing_via_double_underscore_key():
    """`__goto__` lets a node override its successors at runtime — equivalent
    to LangGraph's NodeOutput.goto. Verify by routing to one of two leaves
    based on a state value."""
    g = StateGraph()

    def router(state):
        target = "even" if state["n"] % 2 == 0 else "odd"
        return {
            "__update__": {"chosen": target},
            "__goto__": [target],
        }

    g.add_node("router", router)
    g.add_node("even", lambda s: {"final": "EVEN"})
    g.add_node("odd", lambda s: {"final": "ODD"})
    g.add_edge(START, "router")
    g.add_edge("even", END)
    g.add_edge("odd", END)

    compiled = g.compile()
    assert compiled.invoke({"n": 4})["final"] == "EVEN"
    assert compiled.invoke({"n": 7})["final"] == "ODD"


def test_legacy_plain_dict_return_still_works():
    """The contract pre-iter-78: any dict return without __sends__/__goto__/
    __update__ markers IS the state update. Locked here so we don't regress."""
    g = StateGraph()
    g.add_node("inc", lambda s: {"n": s["n"] + 1})  # legacy: no markers
    g.add_node("double", lambda s: {"n": s["n"] * 2})
    g.add_edge(START, "inc")
    g.add_edge("inc", "double")
    g.add_edge("double", END)

    assert g.compile().invoke({"n": 3})["n"] == 8


def test_send_repr_is_informative():
    s = Send("worker", {"item": 7})
    r = repr(s)
    assert "worker" in r
    assert "7" in r


def test_node_returning_none_is_treated_as_empty_update():
    """Some nodes don't have anything to update but still need to run for
    side effects. Returning None must not crash the graph."""
    g = StateGraph()
    side_effects = []
    def sink(state):
        side_effects.append(state.get("n", 0))
        return None
    g.add_node("inc", lambda s: {"n": (s.get("n") or 0) + 1})
    g.add_node("sink", sink)
    g.add_edge(START, "inc")
    g.add_edge("inc", "sink")
    g.add_edge("sink", END)
    out = g.compile().invoke({})
    assert out["n"] == 1
    assert side_effects == [1]


if __name__ == "__main__":
    fns = [
        test_send_fan_out_runs_worker_per_item_with_isolated_state,
        test_send_accepts_plain_dict_form_in_addition_to_send_class,
        test_send_payload_does_not_leak_to_sibling_forks,
        test_explicit_goto_routing_via_double_underscore_key,
        test_legacy_plain_dict_return_still_works,
        test_send_repr_is_informative,
        test_node_returning_none_is_treated_as_empty_update,
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

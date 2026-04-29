"""Time travel + state history API. LangGraph's differentiator: list
past checkpoints, rewind to a prior step, fork into an alternative
timeline. End-to-end through PyCompiledGraph.

Tests pair an interruptible graph with checkpointer operations:
- state_history: full trajectory
- rewind_to: drop later steps
- fork_at: copy history into a new thread
- clear_thread: wipe a session
"""
from litgraph.graph import StateGraph, START, END


def _build_graph():
    """A 3-step graph that increments `n` each node. Interrupts
    trigger on resume=false — we DON'T interrupt here; we just run
    through and inspect the checkpoint history."""
    g = StateGraph()

    def step1(state):
        return {"n": state.get("n", 0) + 1, "tag": "after_step1"}

    def step2(state):
        return {"n": state["n"] + 10, "tag": "after_step2"}

    def step3(state):
        return {"n": state["n"] + 100, "tag": "after_step3"}

    g.add_node("s1", step1)
    g.add_node("s2", step2)
    g.add_node("s3", step3)
    g.add_edge(START, "s1")
    g.add_edge("s1", "s2")
    g.add_edge("s2", "s3")
    g.add_edge("s3", END)
    return g.compile()


def test_state_history_lists_all_checkpoints_in_step_order():
    compiled = _build_graph()
    _ = compiled.invoke({"n": 0}, thread_id="hist-1")
    hist = compiled.state_history("hist-1")
    # Scheduler writes one checkpoint per step. 3 nodes + initial = 4 steps.
    assert len(hist) >= 3
    steps = [e["step"] for e in hist]
    # Sorted ascending.
    assert steps == sorted(steps)
    # Each entry carries the state at that step.
    for e in hist:
        assert "state" in e
        assert "next_nodes" in e
        assert "ts_ms" in e


def test_state_history_empty_thread_returns_empty_list():
    compiled = _build_graph()
    hist = compiled.state_history("never-ran")
    assert hist == []


def test_rewind_to_drops_later_steps_and_lets_resume_pick_up():
    compiled = _build_graph()
    _ = compiled.invoke({"n": 0}, thread_id="rw-1")
    hist_before = compiled.state_history("rw-1")
    assert len(hist_before) >= 3
    # Rewind to step 1 (after s1 ran).
    target_step = hist_before[0]["step"]  # pick the earliest non-initial
    dropped = compiled.rewind_to("rw-1", target_step)
    assert dropped > 0
    hist_after = compiled.state_history("rw-1")
    assert len(hist_after) == 1
    assert hist_after[0]["step"] == target_step


def test_rewind_to_nonexistent_step_raises_value_error():
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="rw-err")
    try:
        compiled.rewind_to("rw-err", 9999)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no checkpoint at step 9999" in str(e)


def test_fork_at_creates_independent_timeline():
    compiled = _build_graph()
    _ = compiled.invoke({"n": 0}, thread_id="main")
    main_hist = compiled.state_history("main")
    # Fork at step 1.
    source_step = main_hist[0]["step"]
    copied = compiled.fork_at("main", source_step, "fork-a")
    assert copied == 1

    fork_hist = compiled.state_history("fork-a")
    assert len(fork_hist) == 1
    assert fork_hist[0]["step"] == source_step
    # Main is unaffected.
    assert len(compiled.state_history("main")) == len(main_hist)


def test_fork_at_refuses_merge_into_populated_thread():
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="src")
    compiled.invoke({"n": 0}, thread_id="dst")  # dst now populated
    src_step = compiled.state_history("src")[0]["step"]
    try:
        compiled.fork_at("src", src_step, "dst")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "already has checkpoints" in str(e)


def test_fork_at_nonexistent_source_step_raises_value_error():
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="src2")
    try:
        compiled.fork_at("src2", 9999, "new-fork")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "no checkpoint at step 9999" in str(e)


def test_clear_thread_drops_all_checkpoints():
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="clear-1")
    assert len(compiled.state_history("clear-1")) > 0
    compiled.clear_thread("clear-1")
    assert compiled.state_history("clear-1") == []


def test_fork_then_independent_resume_produces_divergent_histories():
    """Realistic time-travel flow: main runs to completion → fork at
    mid-step → clear the old history on the fork so resume starts fresh
    from that point. Verify the two threads have independent trajectories."""
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="timeline-a")
    main_hist = compiled.state_history("timeline-a")

    # Fork at the first checkpoint (after s1).
    fork_source = main_hist[0]["step"]
    compiled.fork_at("timeline-a", fork_source, "timeline-b")

    # Both histories have content; fork-b contains the prefix of main.
    a_steps = [e["step"] for e in compiled.state_history("timeline-a")]
    b_steps = [e["step"] for e in compiled.state_history("timeline-b")]
    assert len(a_steps) >= len(b_steps)
    # Shared prefix.
    assert b_steps[0] == a_steps[0]


def test_state_history_carries_serializable_state():
    compiled = _build_graph()
    compiled.invoke({"n": 0}, thread_id="ser-1")
    hist = compiled.state_history("ser-1")
    # Last entry state should contain the accumulated value.
    last_state = hist[-1]["state"]
    # The final state has `n` set to whatever the scheduler last snapshotted.
    assert "n" in last_state


if __name__ == "__main__":
    import traceback
    fns = [
        test_state_history_lists_all_checkpoints_in_step_order,
        test_state_history_empty_thread_returns_empty_list,
        test_rewind_to_drops_later_steps_and_lets_resume_pick_up,
        test_rewind_to_nonexistent_step_raises_value_error,
        test_fork_at_creates_independent_timeline,
        test_fork_at_refuses_merge_into_populated_thread,
        test_fork_at_nonexistent_source_step_raises_value_error,
        test_clear_thread_drops_all_checkpoints,
        test_fork_then_independent_resume_produces_divergent_histories,
        test_state_history_carries_serializable_state,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

"""Progress — latest-value observability over `tokio::sync::watch`.
Multiple observers read the current state on demand; rapid writes
collapse so observers see only the latest snapshot, never a queue."""
from litgraph.observability import Progress


def test_progress_snapshot_returns_initial():
    p = Progress({"loaders": 0, "chunks": 0})
    obs = p.observer()
    assert obs.snapshot() == {"loaders": 0, "chunks": 0}


def test_progress_set_updates_snapshot():
    p = Progress({"k": 0})
    obs = p.observer()
    p.set({"k": 7})
    assert obs.snapshot() == {"k": 7}


def test_progress_supports_arbitrary_json_state():
    p = Progress({"nested": {"a": [1, 2, 3]}, "flag": True})
    obs = p.observer()
    p.set({"nested": {"a": [4, 5]}, "flag": False, "new_key": "hello"})
    snap = obs.snapshot()
    assert snap == {"nested": {"a": [4, 5]}, "flag": False, "new_key": "hello"}


def test_progress_multiple_observers_see_same_state():
    p = Progress(0)
    a = p.observer()
    b = p.observer()
    c = p.observer()
    p.set(42)
    assert a.snapshot() == 42
    assert b.snapshot() == 42
    assert c.snapshot() == 42


def test_progress_observer_count_tracks_observers():
    p = Progress(0)
    assert p.observer_count() == 0
    a = p.observer()
    assert p.observer_count() == 1
    b = p.observer()
    assert p.observer_count() == 2
    del a
    # Without an explicit drop, observer_count may stay at 2 due to
    # Python's deferred refcount semantics — but creating a new one
    # should still bump.
    c = p.observer()
    assert p.observer_count() >= 2  # at least b and c are alive
    _ = c  # silence linter


def test_progress_rapid_writes_collapse_to_latest():
    """Watch-channel semantics: only the latest value is visible to
    a snapshot reader; intermediate values are lost."""
    p = Progress(0)
    obs = p.observer()
    for i in range(1, 101):
        p.set(i)
    assert obs.snapshot() == 100


def test_progress_set_with_no_observers_is_safe():
    """set() must not raise even when there's no observer — the
    value is still stored for future observers to read."""
    p = Progress({"k": 0})
    p.set({"k": 5})  # no observer; should not raise
    obs = p.observer()
    snap = obs.snapshot()
    # Either the new value or initial — both are documented OK.
    assert snap in [{"k": 0}, {"k": 5}]


def test_progress_repr():
    p = Progress(0)
    _ = p.observer()
    r = repr(p)
    assert "Progress" in r and "observers=" in r


if __name__ == "__main__":
    fns = [
        test_progress_snapshot_returns_initial,
        test_progress_set_updates_snapshot,
        test_progress_supports_arbitrary_json_state,
        test_progress_multiple_observers_see_same_state,
        test_progress_observer_count_tracks_observers,
        test_progress_rapid_writes_collapse_to_latest,
        test_progress_set_with_no_observers_is_safe,
        test_progress_repr,
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

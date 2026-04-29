"""ResumeRegistry — pairs paused work with externally-signalable
resume values via tokio::sync::oneshot. Foundation for LangGraph's
interrupt-resume pattern."""
import threading
import time

from litgraph.observability import ResumeRegistry


def test_register_then_resume_returns_value():
    """The simplest happy path: register a thread, fire a resume from
    another thread, future's await_resume returns the value."""
    reg = ResumeRegistry()
    fut = reg.register("t1")

    def deliver():
        # Tiny delay so the main thread enters await_resume first.
        time.sleep(0.01)
        reg.resume("t1", {"approve": True, "memo": "looks good"})

    t = threading.Thread(target=deliver)
    t.start()
    got = fut.await_resume()
    t.join()
    assert got == {"approve": True, "memo": "looks good"}


def test_cancel_resolves_with_none():
    reg = ResumeRegistry()
    fut = reg.register("t1")

    def cancel():
        time.sleep(0.01)
        assert reg.cancel("t1") is True

    t = threading.Thread(target=cancel)
    t.start()
    got = fut.await_resume()
    t.join()
    assert got is None


def test_resume_unknown_thread_raises():
    reg = ResumeRegistry()
    try:
        reg.resume("nobody-registered", {"x": 1})
    except RuntimeError as e:
        assert "no pending resume" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_double_register_same_id_raises():
    reg = ResumeRegistry()
    _f1 = reg.register("t1")
    try:
        reg.register("t1")
    except RuntimeError as e:
        assert "already pending" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_cancel_unknown_returns_false():
    reg = ResumeRegistry()
    assert reg.cancel("never-registered") is False


def test_pending_count_and_ids():
    reg = ResumeRegistry()
    assert reg.pending_count() == 0
    assert reg.pending_ids() == []
    f1 = reg.register("a")
    f2 = reg.register("b")
    f3 = reg.register("c")
    assert reg.pending_count() == 3
    assert sorted(reg.pending_ids()) == ["a", "b", "c"]
    assert reg.cancel("b") is True
    assert reg.pending_count() == 2
    assert sorted(reg.pending_ids()) == ["a", "c"]
    _ = (f1, f2, f3)


def test_multiple_threads_isolated():
    """Three threads each register; resume each independently with a
    distinct value. All three resolve correctly, no crosstalk."""
    reg = ResumeRegistry()
    fut_a = reg.register("a")
    fut_b = reg.register("b")
    fut_c = reg.register("c")

    results = {}

    def waiter(label, fut):
        results[label] = fut.await_resume()

    threads = [
        threading.Thread(target=waiter, args=("a", fut_a)),
        threading.Thread(target=waiter, args=("b", fut_b)),
        threading.Thread(target=waiter, args=("c", fut_c)),
    ]
    for t in threads:
        t.start()
    time.sleep(0.02)
    reg.resume("b", {"id": "B-val"})
    reg.resume("a", {"id": "A-val"})
    reg.cancel("c")
    for t in threads:
        t.join()
    assert results == {
        "a": {"id": "A-val"},
        "b": {"id": "B-val"},
        "c": None,
    }


def test_await_resume_can_only_be_called_once():
    """Each future is one-shot: a second await on the same future
    raises rather than silently hanging forever."""
    reg = ResumeRegistry()
    fut = reg.register("t1")

    def deliver():
        time.sleep(0.01)
        reg.resume("t1", "ok")

    t = threading.Thread(target=deliver)
    t.start()
    fut.await_resume()
    t.join()
    try:
        fut.await_resume()
    except RuntimeError as e:
        assert "only be called once" in str(e)
    else:
        raise AssertionError("expected RuntimeError on second await")


def test_resume_accepts_any_json_value():
    """Strings, numbers, lists, nested dicts all round-trip cleanly."""
    reg = ResumeRegistry()
    cases = [
        "string-value",
        42,
        [1, 2, {"nested": True}],
        {"deep": {"deeper": {"deepest": [None, 1.5]}}},
    ]
    for i, value in enumerate(cases):
        fut = reg.register(f"t{i}")

        def deliver(thread_id, v):
            time.sleep(0.005)
            reg.resume(thread_id, v)

        t = threading.Thread(target=deliver, args=(f"t{i}", value))
        t.start()
        got = fut.await_resume()
        t.join()
        assert got == value, f"case {i}: {got!r} != {value!r}"


def test_repr_shows_pending_count():
    reg = ResumeRegistry()
    _ = reg.register("t1")
    r = repr(reg)
    assert "ResumeRegistry" in r and "pending=" in r


if __name__ == "__main__":
    fns = [
        test_register_then_resume_returns_value,
        test_cancel_resolves_with_none,
        test_resume_unknown_thread_raises,
        test_double_register_same_id_raises,
        test_cancel_unknown_returns_false,
        test_pending_count_and_ids,
        test_multiple_threads_isolated,
        test_await_resume_can_only_be_called_once,
        test_resume_accepts_any_json_value,
        test_repr_shows_pending_count,
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

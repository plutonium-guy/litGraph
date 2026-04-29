"""ShutdownSignal — Notify-backed coordination for graceful shutdown.
N-waiter, single-fire EDGE; late waiters resolve instantly after the
signal has already gone off."""
import threading
import time

from litgraph.observability import ShutdownSignal


def test_unsignaled_state_is_false():
    s = ShutdownSignal()
    assert s.is_signaled() is False


def test_signal_flips_state():
    s = ShutdownSignal()
    s.signal()
    assert s.is_signaled() is True


def test_wait_returns_after_signal():
    """A pending wait in another thread wakes when signal() fires."""
    s = ShutdownSignal()
    completed = threading.Event()

    def waiter():
        s.wait()
        completed.set()

    t = threading.Thread(target=waiter)
    t.start()
    # Tiny pause so the waiter has a chance to park.
    time.sleep(0.02)
    assert not completed.is_set(), "wait() should not resolve before signal"
    s.signal()
    completed.wait(timeout=1.0)
    t.join(timeout=1.0)
    assert completed.is_set(), "wait() never resolved after signal"


def test_late_wait_resolves_immediately():
    """Wait called AFTER signal returns instantly — no Notify
    replay needed because the flag-check fast-path catches it."""
    s = ShutdownSignal()
    s.signal()
    started = time.time()
    s.wait()
    elapsed = time.time() - started
    assert elapsed < 0.1, f"late wait took {elapsed:.3f}s, expected ~0"


def test_many_concurrent_waiters_all_wake():
    s = ShutdownSignal()
    n = 8
    completed = [threading.Event() for _ in range(n)]

    def waiter(i):
        s.wait()
        completed[i].set()

    threads = [threading.Thread(target=waiter, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    time.sleep(0.05)
    s.signal()
    for ev in completed:
        ev.wait(timeout=1.0)
    for t in threads:
        t.join(timeout=1.0)
    assert all(ev.is_set() for ev in completed), "some waiters never woke"


def test_double_signal_idempotent():
    s = ShutdownSignal()
    s.signal()
    s.signal()  # must not raise/deadlock
    assert s.is_signaled()


def test_repr_shows_state():
    s = ShutdownSignal()
    assert "ShutdownSignal(signaled=false)" in repr(s)
    s.signal()
    assert "ShutdownSignal(signaled=true)" in repr(s)


if __name__ == "__main__":
    fns = [
        test_unsignaled_state_is_false,
        test_signal_flips_state,
        test_wait_returns_after_signal,
        test_late_wait_resolves_immediately,
        test_many_concurrent_waiters_all_wake,
        test_double_signal_idempotent,
        test_repr_shows_state,
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

"""multiplex_chat_streams — fan N chat-model token streams into a
single Python iterator with per-event model labels.

Tests use OpenAIChat against unreachable hosts to exercise the
init-failure path; deterministic streaming behaviour is covered by
the Rust test suite (mock streaming models in pure Rust)."""
from litgraph.agents import multiplex_chat_streams
from litgraph.providers import OpenAIChat


def _broken(label: str):
    return OpenAIChat(
        api_key=f"sk-{label}",
        model="gpt-4o-mini",
        base_url="http://127.0.0.1:1",
    )


def test_multiplex_rejects_empty_models():
    try:
        for _ in multiplex_chat_streams([], [{"role": "user", "content": "hi"}]):
            break
    except ValueError as e:
        assert "at least one" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_multiplex_rejects_non_tuple_entries():
    try:
        for _ in multiplex_chat_streams(
            [_broken("a")],  # not a (label, model) tuple
            [{"role": "user", "content": "hi"}],
        ):
            break
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")


def test_multiplex_init_failures_emit_tagged_error_events():
    """Every inner model points at unreachable host — each one's
    init-or-stream failure should arrive as a tagged-error dict on
    the iterator, not raise a Python exception."""
    a = _broken("a")
    b = _broken("b")
    seen_labels = set()
    saw_error_event = False
    for ev in multiplex_chat_streams(
        [("model-a", a), ("model-b", b)],
        [{"role": "user", "content": "hi"}],
    ):
        if ev.get("type") == "error":
            saw_error_event = True
            seen_labels.add(ev["model_label"])
    assert saw_error_event, "expected at least one tagged-error event"
    # Both labels should report failures (against unreachable hosts).
    assert seen_labels == {"model-a", "model-b"}, seen_labels


def test_multiplex_iterator_terminates_after_all_done():
    """Iterator must reach StopIteration cleanly even when every
    inner model errors at init."""
    a = _broken("a")
    events = list(
        multiplex_chat_streams(
            [("a", a)],
            [{"role": "user", "content": "hi"}],
        )
    )
    # Stream ended (StopIteration). At least one event collected
    # (the init-error tagged event for "a").
    assert len(events) >= 1


def test_multiplex_event_carries_model_label_field():
    a = _broken("a")
    for ev in multiplex_chat_streams(
        [("alpha", a)],
        [{"role": "user", "content": "hi"}],
    ):
        assert "model_label" in ev
        assert ev["model_label"] == "alpha"
        break  # one event is enough


if __name__ == "__main__":
    fns = [
        test_multiplex_rejects_empty_models,
        test_multiplex_rejects_non_tuple_entries,
        test_multiplex_init_failures_emit_tagged_error_events,
        test_multiplex_iterator_terminates_after_all_done,
        test_multiplex_event_carries_model_label_field,
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

"""PostgresChatHistory — distributed durable chat. Integration tests
skip without PG_DSN env var; class-API tests run unconditionally."""
import os

from litgraph.memory import PostgresChatHistory


def _dsn():
    return os.environ.get("PG_DSN")


def test_class_exposed_on_litgraph_memory():
    """Sanity: PyPostgresChatHistory is registered + has the expected API."""
    assert hasattr(PostgresChatHistory, "connect")
    assert hasattr(PostgresChatHistory, "session")
    assert hasattr(PostgresChatHistory, "append")
    assert hasattr(PostgresChatHistory, "append_all")
    assert hasattr(PostgresChatHistory, "messages")
    assert hasattr(PostgresChatHistory, "clear")
    assert hasattr(PostgresChatHistory, "delete_session")
    assert hasattr(PostgresChatHistory, "set_system")
    assert hasattr(PostgresChatHistory, "message_count")
    assert hasattr(PostgresChatHistory, "list_sessions")


def test_connect_malformed_dsn_raises_runtime_error():
    """Malformed DSN should fail fast at parse time (no network)."""
    try:
        PostgresChatHistory.connect("not-a-postgres-dsn", "test-session")
        raise AssertionError("expected error on malformed DSN")
    except (RuntimeError, ValueError):
        pass


def test_append_messages_roundtrips():
    """Integration: append then read back. Skips if no PG_DSN."""
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "test-roundtrip")
    h.delete_session()  # start clean
    h.append({"role": "user", "content": "hi"})
    h.append({"role": "assistant", "content": "hello"})
    msgs = h.messages()
    assert len(msgs) == 2
    assert msgs[0]["content"] == "hi"
    assert msgs[1]["content"] == "hello"
    h.delete_session()


def test_system_pin_prepended():
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "test-pin")
    h.delete_session()
    h.set_system({"role": "system", "content": "you are helpful"})
    h.append({"role": "user", "content": "q"})
    msgs = h.messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "you are helpful"
    h.delete_session()


def test_clear_drops_messages_keeps_pin():
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "test-clear")
    h.delete_session()
    h.set_system({"role": "system", "content": "pin"})
    h.append({"role": "user", "content": "x"})
    h.clear()
    msgs = h.messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "pin"
    h.delete_session()


def test_session_clone_addresses_different_session_same_pool():
    dsn = _dsn()
    if not dsn:
        return
    h_a = PostgresChatHistory.connect(dsn, "test-sess-a")
    h_b = h_a.session("test-sess-b")
    h_a.delete_session()
    h_b.delete_session()
    h_a.append({"role": "user", "content": "from-a"})
    h_b.append({"role": "user", "content": "from-b"})
    assert h_a.messages()[0]["content"] == "from-a"
    assert h_b.messages()[0]["content"] == "from-b"
    h_a.delete_session()
    h_b.delete_session()


def test_message_count_zero_for_empty_session():
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "test-empty-count")
    h.delete_session()
    assert h.message_count() == 0


def test_session_id_getter():
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "user-42")
    assert h.session_id == "user-42"


def test_repr_includes_session_id():
    dsn = _dsn()
    if not dsn:
        return
    h = PostgresChatHistory.connect(dsn, "user-99")
    assert "user-99" in repr(h)


if __name__ == "__main__":
    import traceback
    fns = [
        test_class_exposed_on_litgraph_memory,
        test_connect_malformed_dsn_raises_runtime_error,
        test_append_messages_roundtrips,
        test_system_pin_prepended,
        test_clear_drops_messages_keeps_pin,
        test_session_clone_addresses_different_session_same_pool,
        test_message_count_zero_for_empty_session,
        test_session_id_getter,
        test_repr_includes_session_id,
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

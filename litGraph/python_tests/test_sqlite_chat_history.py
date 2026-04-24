"""SqliteChatHistory — durable conversation memory across process restarts.

Direct LangChain `SQLChatMessageHistory` parity. Per-message storage,
multi-session isolation by session_id, system pin separate from history."""
import os
import tempfile

from litgraph.memory import SqliteChatHistory


def test_append_and_messages_round_trip_in_order():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.append({"role": "user", "content": "hello"})
    h.append({"role": "assistant", "content": "hi there"})
    h.append({"role": "user", "content": "how are you"})
    msgs = h.messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "hi there"
    assert msgs[2]["content"] == "how are you"


def test_append_all_bulk_insert_keeps_order():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.append_all([
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
    ])
    msgs = h.messages()
    assert [m["content"] for m in msgs] == ["a", "b", "c"]


def test_system_pin_prepended_to_messages():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.set_system({"role": "system", "content": "You are helpful."})
    h.append({"role": "user", "content": "hi"})
    msgs = h.messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You are helpful."


def test_clear_drops_messages_but_keeps_system_pin():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.set_system({"role": "system", "content": "persona"})
    h.append({"role": "user", "content": "x"})
    h.append({"role": "user", "content": "y"})
    h.clear()
    msgs = h.messages()
    # Pin survives — represents agent persona, not memory.
    assert len(msgs) == 1
    assert msgs[0]["content"] == "persona"


def test_delete_session_drops_everything_including_pin():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.set_system({"role": "system", "content": "persona"})
    h.append({"role": "user", "content": "x"})
    h.delete_session()
    assert h.messages() == []


def test_sessions_isolated_by_session_id():
    h1 = SqliteChatHistory.in_memory(session_id="s1")
    h2 = h1.session("s2")  # share connection, different session
    h1.append({"role": "user", "content": "alpha"})
    h2.append({"role": "user", "content": "beta"})
    h2.append({"role": "user", "content": "gamma"})
    assert h1.message_count() == 1
    assert h2.message_count() == 2
    assert h1.messages()[0]["content"] == "alpha"


def test_durability_across_process_restart():
    """The promise: chat survives a 'process restart'. Real file, write
    messages, drop the handle (simulating exit), reopen the file, observe
    everything still there."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    path = tmp.name
    try:
        h = SqliteChatHistory.open(path, session_id="user-42")
        h.set_system({"role": "system", "content": "be terse"})
        h.append({"role": "user", "content": "first question"})
        h.append({"role": "assistant", "content": "first answer"})
        del h  # simulate process exit
        # Reopen.
        h2 = SqliteChatHistory.open(path, session_id="user-42")
        msgs = h2.messages()
        assert len(msgs) == 3
        assert msgs[0]["content"] == "be terse"
        assert msgs[1]["content"] == "first question"
        assert msgs[2]["content"] == "first answer"
        # Append after restart picks up at next seq — no PRIMARY KEY collision.
        h2.append({"role": "user", "content": "follow-up"})
        assert h2.message_count() == 3  # excludes pin
        msgs = h2.messages()
        assert msgs[-1]["content"] == "follow-up"
    finally:
        os.unlink(path)


def test_list_sessions_enumerates_all_sessions_in_store():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.append({"role": "user", "content": "a"})
    h.session("s2").set_system({"role": "system", "content": "p"})
    h.session("s3").append({"role": "user", "content": "b"})
    sessions = h.list_sessions()
    # UNION + ORDER BY — sorted, dedup'd.
    assert sessions == ["s1", "s2", "s3"]


def test_message_count_excludes_system_pin():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.set_system({"role": "system", "content": "persona"})
    h.append({"role": "user", "content": "a"})
    h.append({"role": "user", "content": "b"})
    assert h.message_count() == 2  # not 3


def test_set_system_none_clears_the_pin():
    h = SqliteChatHistory.in_memory(session_id="s1")
    h.set_system({"role": "system", "content": "first"})
    h.set_system(None)
    h.append({"role": "user", "content": "x"})
    msgs = h.messages()
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"


if __name__ == "__main__":
    fns = [
        test_append_and_messages_round_trip_in_order,
        test_append_all_bulk_insert_keeps_order,
        test_system_pin_prepended_to_messages,
        test_clear_drops_messages_but_keeps_system_pin,
        test_delete_session_drops_everything_including_pin,
        test_sessions_isolated_by_session_id,
        test_durability_across_process_restart,
        test_list_sessions_enumerates_all_sessions_in_store,
        test_message_count_excludes_system_pin,
        test_set_system_none_clears_the_pin,
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

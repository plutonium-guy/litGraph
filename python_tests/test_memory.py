"""ConversationMemory: BufferMemory + TokenBufferMemory with system pinning."""
from litgraph.memory import BufferMemory, TokenBufferMemory


def test_buffer_keeps_last_n_messages():
    m = BufferMemory(max_messages=3)
    for i in range(5):
        m.append({"role": "user", "content": f"u{i}"})
    msgs = m.messages()
    assert len(msgs) == 3
    assert [x["content"] for x in msgs] == ["u2", "u3", "u4"]


def test_buffer_pins_system_message():
    m = BufferMemory(max_messages=2)
    m.append({"role": "system", "content": "be helpful"})
    for i in range(5):
        m.append({"role": "user", "content": f"u{i}"})
    msgs = m.messages()
    assert len(msgs) == 3, msgs
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "be helpful"
    assert [x["content"] for x in msgs[1:]] == ["u3", "u4"]


def test_buffer_clear_keeps_system_pin():
    m = BufferMemory(max_messages=5)
    m.append({"role": "system", "content": "sys"})
    m.append({"role": "user", "content": "hi"})
    m.clear()
    assert m.messages() == [{"role": "system", "content": "sys"}]


def test_buffer_set_system_overrides_and_removes():
    m = BufferMemory(max_messages=5)
    m.append({"role": "system", "content": "v1"})
    m.set_system({"role": "system", "content": "v2"})
    assert m.messages()[0]["content"] == "v2"
    m.set_system(None)
    assert m.messages() == []


def test_token_buffer_evicts_oldest_when_over_budget():
    counter = lambda msg: len(msg["content"])  # 1 token per char
    m = TokenBufferMemory(max_tokens=10, counter=counter)
    m.append({"role": "user", "content": "aaaaa"})  # 5
    m.append({"role": "user", "content": "bbb"})    # 8 total
    m.append({"role": "user", "content": "ccc"})    # 11 → evict aaaaa
    msgs = m.messages()
    assert len(msgs) == 2
    assert [x["content"] for x in msgs] == ["bbb", "ccc"]


def test_token_buffer_pins_system_against_budget():
    counter = lambda msg: len(msg["content"])
    m = TokenBufferMemory(max_tokens=20, counter=counter)
    m.append({"role": "system", "content": "system_prompt_here"})  # 18 tokens
    m.append({"role": "user", "content": "hello world"})           # 11 → 29, must evict
    msgs = m.messages()
    # System pinned, user evicted.
    assert len(msgs) == 1
    assert msgs[0]["role"] == "system"


def test_token_counter_callback_receives_role_and_content():
    seen = []
    def counter(msg):
        seen.append((msg["role"], msg["content"]))
        return len(msg["content"])
    m = TokenBufferMemory(max_tokens=100, counter=counter)
    m.append({"role": "user", "content": "hello"})
    _ = m.messages()
    assert ("user", "hello") in seen


def test_buffer_memory_full_chat_loop_shape():
    """End-to-end: simulate 3-turn conversation, verify the final messages()
    shape is what you'd hand straight to chat.invoke()."""
    m = BufferMemory(max_messages=10)
    m.append({"role": "system", "content": "be terse"})
    m.append({"role": "user", "content": "hi"})
    m.append({"role": "assistant", "content": "hello"})
    m.append({"role": "user", "content": "what is 2+2"})
    m.append({"role": "assistant", "content": "4"})
    msgs = m.messages()
    assert len(msgs) == 5
    assert msgs[0]["role"] == "system"
    assert [x["role"] for x in msgs[1:]] == ["user", "assistant", "user", "assistant"]


def test_buffer_to_bytes_round_trip_via_classmethod():
    """Save → reload via from_bytes classmethod; preserves system pin + history."""
    m = BufferMemory(max_messages=10)
    m.append({"role": "system", "content": "be helpful"})
    m.append({"role": "user", "content": "hi"})
    m.append({"role": "assistant", "content": "hello"})
    blob = m.to_bytes()
    assert isinstance(blob, bytes) and len(blob) > 0
    m2 = BufferMemory.from_bytes(10, blob)
    msgs = m2.messages()
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system"
    assert msgs[2]["content"] == "hello"


def test_buffer_restore_preserves_existing_max_messages():
    """Snapshot with 8 messages restored into 3-cap → drops oldest 5."""
    big = BufferMemory(max_messages=20)
    for i in range(8):
        big.append({"role": "user", "content": f"u{i}"})
    blob = big.to_bytes()
    small = BufferMemory.from_bytes(3, blob)
    msgs = small.messages()
    assert len(msgs) == 3
    assert msgs[0]["content"] == "u5"
    assert msgs[2]["content"] == "u7"


def test_buffer_restore_into_existing_instance():
    """`.restore(bytes)` on an existing instance overwrites state but keeps cap."""
    m = BufferMemory(max_messages=5)
    m.append({"role": "user", "content": "old"})
    src = BufferMemory(max_messages=5)
    src.append({"role": "system", "content": "fresh sys"})
    src.append({"role": "user", "content": "new"})
    m.restore(src.to_bytes())
    msgs = m.messages()
    assert [x["role"] for x in msgs] == ["system", "user"]
    assert msgs[1]["content"] == "new"


def test_token_buffer_to_bytes_round_trip_keeps_counter():
    """TokenBufferMemory persistence — counter callback bound at construction
    is reused after restore (no need to re-pass it on the wire)."""
    counter = lambda msg: len(msg["content"])
    m = TokenBufferMemory(max_tokens=100, counter=counter)
    m.append({"role": "system", "content": "sys"})
    m.append({"role": "user", "content": "hello"})
    blob = m.to_bytes()

    target = TokenBufferMemory(max_tokens=100, counter=counter)
    target.restore(blob)
    msgs = target.messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["content"] == "hello"

    # Counter still works post-restore: appending more triggers eviction.
    target.append({"role": "user", "content": "x" * 200})  # over 100 tokens
    after = target.messages()
    # System pin + the new big message; old "hello" evicted.
    assert any(m["role"] == "system" for m in after)
    assert not any(m["content"] == "hello" for m in after)


def test_to_bytes_blob_is_json_decodable():
    """Sanity: the wire format is JSON, so ops can grep/jq stored sessions."""
    import json
    m = BufferMemory(max_messages=5)
    m.append({"role": "system", "content": "marker-sys"})
    m.append({"role": "user", "content": "marker-user"})
    decoded = json.loads(m.to_bytes())
    assert decoded["version"] == 1
    assert decoded["system"]["role"] == "system"
    assert any("marker-user" in str(blk) for blk in decoded["history"][0]["content"])


def test_from_bytes_rejects_garbage_and_wrong_version():
    try:
        BufferMemory.from_bytes(10, b"not valid json")
    except RuntimeError as e:
        assert "deserialize" in str(e)
    else:
        raise AssertionError("expected RuntimeError")

    import json
    bad_version = json.dumps({"version": 9999, "system": None, "history": []}).encode()
    try:
        BufferMemory.from_bytes(10, bad_version)
    except RuntimeError as e:
        assert "version mismatch" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    fns = [
        test_buffer_keeps_last_n_messages,
        test_buffer_pins_system_message,
        test_buffer_clear_keeps_system_pin,
        test_buffer_set_system_overrides_and_removes,
        test_token_buffer_evicts_oldest_when_over_budget,
        test_token_buffer_pins_system_against_budget,
        test_token_counter_callback_receives_role_and_content,
        test_buffer_memory_full_chat_loop_shape,
        test_buffer_to_bytes_round_trip_via_classmethod,
        test_buffer_restore_preserves_existing_max_messages,
        test_buffer_restore_into_existing_instance,
        test_token_buffer_to_bytes_round_trip_keeps_counter,
        test_to_bytes_blob_is_json_decodable,
        test_from_bytes_rejects_garbage_and_wrong_version,
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

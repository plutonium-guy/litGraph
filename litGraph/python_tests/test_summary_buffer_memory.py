"""SummaryBufferMemory — rolling recent buffer + LLM-distilled running
summary of evicted turns. LangChain parity: ConversationSummaryBufferMemory.
Compaction is decoupled from append(); caller calls .compact(model) when ready."""
import http.server
import json
import threading

from litgraph.memory import SummaryBufferMemory
from litgraph.providers import OpenAIChat


class _Fake(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []
    REPLY = "summary-of-old-turns"

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED.append(json.loads(body))
        payload = {
            "id": "x", "object": "chat.completion", "model": "gpt-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.REPLY},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(reply="summary-of-old-turns"):
    _Fake.CAPTURED = []
    _Fake.REPLY = reply
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _client(port):
    return OpenAIChat(api_key="k", model="gpt-x",
                      base_url=f"http://127.0.0.1:{port}/v1")


def test_no_compact_below_cap():
    srv, port = _spawn()
    try:
        mem = SummaryBufferMemory(max_recent_messages=10, summarize_chunk=2)
        mem.append({"role": "user", "content": "hi"})
        mem.append({"role": "assistant", "content": "hello"})
        assert not mem.needs_compact()
        ran = mem.compact(_client(port))
        assert ran is False
        assert mem.running_summary() is None
    finally:
        srv.shutdown()
    # Server never called.
    assert len(_Fake.CAPTURED) == 0


def test_compact_folds_oldest_into_summary():
    srv, port = _spawn(reply="greetings were exchanged")
    try:
        mem = SummaryBufferMemory(max_recent_messages=3, summarize_chunk=2)
        mem.append({"role": "user", "content": "hi"})
        mem.append({"role": "assistant", "content": "hello"})
        mem.append({"role": "user", "content": "how are you"})
        mem.append({"role": "assistant", "content": "good"})
        assert mem.needs_compact()

        ran = mem.compact(_client(port))
        assert ran is True
        assert mem.running_summary() == "greetings were exchanged"
        assert mem.recent_len() == 2

        msgs = mem.messages()
        # [summary_system, ...recent]
        assert msgs[0]["role"] == "system"
        assert "greetings were exchanged" in msgs[0]["content"]
        assert msgs[1]["content"] == "how are you"
        assert msgs[2]["content"] == "good"
    finally:
        srv.shutdown()


def test_system_pin_survives_compaction():
    srv, port = _spawn()
    try:
        mem = SummaryBufferMemory(max_recent_messages=1, summarize_chunk=2)
        mem.set_system({"role": "system", "content": "you are helpful"})
        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})
        mem.append({"role": "user", "content": "c"})
        mem.compact(_client(port))

        msgs = mem.messages()
        # [system_pin, summary_system, ...recent]
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system"
        assert roles[1] == "system"
        # Caller pin first, summary second.
        assert msgs[0]["content"] == "you are helpful"
        assert "summary" in msgs[1]["content"].lower()
    finally:
        srv.shutdown()


def test_compact_all_empties_buffer():
    srv, port = _spawn(reply="everything summarized")
    try:
        mem = SummaryBufferMemory(max_recent_messages=100, summarize_chunk=10)
        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})
        mem.append({"role": "user", "content": "c"})
        assert not mem.needs_compact()

        ran = mem.compact_all(_client(port))
        assert ran is True
        assert mem.recent_len() == 0
        assert mem.running_summary() == "everything summarized"
    finally:
        srv.shutdown()


def test_clear_wipes_summary_and_buffer():
    srv, port = _spawn()
    try:
        mem = SummaryBufferMemory(max_recent_messages=1, summarize_chunk=2)
        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})
        mem.append({"role": "user", "content": "c"})
        mem.compact(_client(port))
        assert mem.running_summary() is not None

        mem.clear()
        assert mem.recent_len() == 0
        assert mem.running_summary() is None
    finally:
        srv.shutdown()


def test_repr_reflects_state():
    srv, port = _spawn()
    try:
        mem = SummaryBufferMemory(max_recent_messages=10, summarize_chunk=4)
        r1 = repr(mem)
        assert "SummaryBufferMemory" in r1
        assert "summary=no" in r1

        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})
        mem.compact_all(_client(port))
        r2 = repr(mem)
        assert "summary=yes" in r2
    finally:
        srv.shutdown()


def test_snapshot_roundtrip_preserves_summary():
    srv, port = _spawn(reply="preserved-summary")
    try:
        mem = SummaryBufferMemory(max_recent_messages=1, summarize_chunk=2)
        mem.set_system({"role": "system", "content": "sys"})
        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})
        mem.append({"role": "user", "content": "c"})
        mem.compact(_client(port))

        blob = mem.to_bytes()
        # Construct a fresh instance from the blob.
        mem2 = SummaryBufferMemory.from_bytes(1, 2, blob)
        assert mem2.running_summary() == "preserved-summary"
        msgs = mem2.messages()
        # System pin + summary + recent all come back.
        assert any(m["content"] == "sys" for m in msgs)
        assert any("preserved-summary" in m["content"] for m in msgs)
    finally:
        srv.shutdown()


def test_second_compact_extends_prior_summary():
    """Second call to compact() must pass the FIRST summary to the LLM
    so it can extend rather than overwrite."""
    srv, port = _spawn(reply="first")
    try:
        mem = SummaryBufferMemory(max_recent_messages=1, summarize_chunk=1)
        mem.append({"role": "user", "content": "a"})
        mem.append({"role": "user", "content": "b"})  # triggers compact
        mem.compact(_client(port))

        _Fake.REPLY = "second"
        mem.append({"role": "user", "content": "c"})  # triggers compact again
        mem.compact(_client(port))

        # Second captured request must reference the prior summary.
        body = _Fake.CAPTURED[-1]
        last_user_content = body["messages"][-1]["content"]
        assert "Previous summary:" in last_user_content
        assert "first" in last_user_content
    finally:
        srv.shutdown()


def test_system_via_append_sets_pin():
    mem = SummaryBufferMemory(max_recent_messages=5, summarize_chunk=2)
    mem.append({"role": "system", "content": "sys via append"})
    mem.append({"role": "user", "content": "hi"})
    msgs = mem.messages()
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "sys via append"


def test_compact_is_idempotent_when_under_cap():
    srv, port = _spawn()
    try:
        mem = SummaryBufferMemory(max_recent_messages=10, summarize_chunk=2)
        mem.append({"role": "user", "content": "hi"})
        # Compact 10× in a row — no LLM call, no state change.
        for _ in range(10):
            assert mem.compact(_client(port)) is False
        assert len(_Fake.CAPTURED) == 0
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_no_compact_below_cap,
        test_compact_folds_oldest_into_summary,
        test_system_pin_survives_compaction,
        test_compact_all_empties_buffer,
        test_clear_wipes_summary_and_buffer,
        test_repr_reflects_state,
        test_snapshot_roundtrip_preserves_summary,
        test_second_compact_extends_prior_summary,
        test_system_via_append_sets_pin,
        test_compact_is_idempotent_when_under_cap,
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

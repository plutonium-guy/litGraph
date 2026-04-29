"""Conversation summarization — `summarize_conversation()` standalone helper
+ `BufferMemory.summarize_and_compact()` against a fake OpenAI server."""
import http.server
import json
import threading

from litgraph.memory import BufferMemory, summarize_conversation
from litgraph.providers import OpenAIChat


# Fake OpenAI server that captures the prompt and returns a canned summary.

class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    REPLY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeOpenAI.LAST_BODY[0] = body
        canned = FakeOpenAI.REPLY[0] or "they greeted each other"
        payload = {
            "id": "x", "object": "chat.completion", "model": "gpt-x",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": canned},
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


def _spawn():
    FakeOpenAI.LAST_BODY[0] = None
    FakeOpenAI.REPLY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _client(port):
    return OpenAIChat(api_key="k", model="gpt-x",
                      base_url=f"http://127.0.0.1:{port}/v1")


def test_summarize_conversation_returns_canned_summary():
    srv, port = _spawn()
    try:
        FakeOpenAI.REPLY[0] = "user asked about X; agent answered Y"
        chat = _client(port)
        out = summarize_conversation(chat, [
            {"role": "user", "content": "what is X?"},
            {"role": "assistant", "content": "X is the answer Y"},
        ])
        assert out == "user asked about X; agent answered Y"
        # Verify the request body — transcript was built with `role: text` lines.
        sent = json.loads(FakeOpenAI.LAST_BODY[0])
        user_prompt = sent["messages"][-1]["content"]
        assert "user: what is X?" in user_prompt
        assert "assistant: X is the answer Y" in user_prompt
        # No prior summary passed → no "Previous summary:" preamble.
        assert "Previous summary:" not in user_prompt
        # Determinism: temperature pinned to 0.
        assert sent.get("temperature") == 0.0
    finally:
        srv.shutdown()


def test_summarize_conversation_extends_prior_summary():
    srv, port = _spawn()
    try:
        chat = _client(port)
        summarize_conversation(chat,
            [{"role": "user", "content": "more context"}],
            prior_summary="user prefers Rust")
        sent = json.loads(FakeOpenAI.LAST_BODY[0])
        prompt = sent["messages"][-1]["content"]
        assert "Previous summary:\nuser prefers Rust" in prompt
        assert "New conversation turns to incorporate:" in prompt
    finally:
        srv.shutdown()


def test_summarize_empty_messages_returns_prior_no_http_call():
    """Empty messages → just hand back the prior summary; never call the model."""
    # base_url unreachable: if we touched HTTP we'd error.
    chat = OpenAIChat(api_key="k", model="gpt-x", base_url="http://127.0.0.1:1/v1")
    out = summarize_conversation(chat, [], prior_summary="kept")
    assert out == "kept"


def test_buffer_memory_summarize_and_compact_replaces_old_turns():
    """Compact the oldest 4 messages → leaves the 5th + a synthetic system pin."""
    srv, port = _spawn()
    try:
        FakeOpenAI.REPLY[0] = "they discussed X and Y"
        chat = _client(port)
        mem = BufferMemory(max_messages=20)
        mem.append({"role": "user", "content": "hi"})
        mem.append({"role": "assistant", "content": "hello"})
        mem.append({"role": "user", "content": "how are you"})
        mem.append({"role": "assistant", "content": "good thanks"})
        mem.append({"role": "user", "content": "tell me about Rust"})

        mem.summarize_and_compact(chat, summarize_count=4)
        msgs = mem.messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "they discussed X and Y"
        assert msgs[1]["content"] == "tell me about Rust"
    finally:
        srv.shutdown()


def test_summarize_and_compact_noop_when_history_too_short():
    """If history < summarize_count, do nothing (no HTTP either)."""
    chat = OpenAIChat(api_key="k", model="gpt-x", base_url="http://127.0.0.1:1/v1")
    mem = BufferMemory(max_messages=10)
    mem.append({"role": "user", "content": "alone"})
    mem.summarize_and_compact(chat, summarize_count=5)
    msgs = mem.messages()
    assert len(msgs) == 1
    assert msgs[0]["content"] == "alone"


def test_summarize_and_compact_folds_prior_system_pin_into_summary():
    """If a system pin exists, the helper extends it via prior_summary so the
    new summary inherits the original instructions."""
    srv, port = _spawn()
    try:
        FakeOpenAI.REPLY[0] = "merged summary"
        chat = _client(port)
        mem = BufferMemory(max_messages=20)
        mem.append({"role": "system", "content": "be terse"})
        for i in range(5):
            mem.append({"role": "user", "content": f"u{i}"})
        mem.summarize_and_compact(chat, summarize_count=3)

        # The model received "be terse" as the prior_summary.
        sent = json.loads(FakeOpenAI.LAST_BODY[0])
        prompt = sent["messages"][-1]["content"]
        assert "Previous summary:\nbe terse" in prompt
        # New system pin = the canned summary.
        msgs = mem.messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "merged summary"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_summarize_conversation_returns_canned_summary,
        test_summarize_conversation_extends_prior_summary,
        test_summarize_empty_messages_returns_prior_no_http_call,
        test_buffer_memory_summarize_and_compact_replaces_old_turns,
        test_summarize_and_compact_noop_when_history_too_short,
        test_summarize_and_compact_folds_prior_system_pin_into_summary,
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

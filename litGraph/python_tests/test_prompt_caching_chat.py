"""PromptCachingChat — wrap a ChatModel to auto-mark messages as Anthropic
prompt-cache breakpoints. Default: cache system. Opt-in: cache long user
context + manual indices. The Anthropic provider converts Message.cache=True
into cache_control: {type: ephemeral} on the last content block."""
import http.server
import json
import threading

from litgraph.providers import AnthropicChat, OpenAIChat, PromptCachingChat


class _FakeAnthropic(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED.append(json.loads(body))
        resp = json.dumps({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


class _FakeOpenAI(http.server.BaseHTTPRequestHandler):
    """OpenAI-format fake — no cache_control expected, flag should be ignored."""
    CAPTURED: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED.append(json.loads(body))
        payload = {
            "id": "r", "model": "gpt-test", "object": "chat.completion",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn_anthropic():
    _FakeAnthropic.CAPTURED = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeAnthropic)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _spawn_openai():
    _FakeOpenAI.CAPTURED = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _anthropic(port):
    return AnthropicChat(
        api_key="k", model="claude-opus-4-7-1m",
        base_url=f"http://127.0.0.1:{port}",
    )


def _openai(port):
    return OpenAIChat(
        api_key="sk-test", model="gpt-4o-mini",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def _has_cache(block):
    return "cache_control" in block and block["cache_control"] == {"type": "ephemeral"}


def test_default_caches_system_on_anthropic():
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(_anthropic(port))
        chat.invoke([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    # system becomes a typed-block array with cache_control attached.
    assert isinstance(sent["system"], list)
    assert _has_cache(sent["system"][0])
    # user message block does NOT have cache_control.
    user_block = sent["messages"][0]["content"][-1]
    assert "cache_control" not in user_block


def test_without_system_leaves_system_alone():
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(_anthropic(port), cache_system=False)
        chat.invoke([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "hi"},
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    # With cache_system=False, system passes through as flat string (no cache_control).
    if isinstance(sent["system"], list):
        assert not _has_cache(sent["system"][0])


def test_cache_last_user_over_threshold():
    srv, port = _spawn_anthropic()
    try:
        long_ctx = "x" * 5000
        chat = PromptCachingChat(
            _anthropic(port),
            cache_system=False,
            cache_last_user_if_over=4096,
        )
        chat.invoke([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},       # short — not cached
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": long_ctx},   # long — cached
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    # Last user message's last block has cache_control.
    msgs = sent["messages"]
    last_user_idx = max(i for i, m in enumerate(msgs) if m["role"] == "user")
    last_block = msgs[last_user_idx]["content"][-1]
    assert _has_cache(last_block)


def test_short_user_not_cached():
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(
            _anthropic(port),
            cache_system=False,
            cache_last_user_if_over=4096,
        )
        chat.invoke([
            {"role": "user", "content": "short"},
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    user_block = sent["messages"][0]["content"][-1]
    assert "cache_control" not in user_block


def test_cache_indices_marks_specific_messages():
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(
            _anthropic(port),
            cache_system=False,
            cache_indices=[1],  # mark the second message (first user)
        )
        chat.invoke([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "a"},
            {"role": "user", "content": "second"},
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    msgs = sent["messages"]
    # User messages in the Anthropic payload: [first, second]. index=1 in the
    # full history points at "first" (first user msg).
    first_user = msgs[0]
    last_block = first_user["content"][-1]
    assert _has_cache(last_block)
    # Second user untouched.
    second_user = msgs[-1]
    assert "cache_control" not in second_user["content"][-1]


def test_ignored_by_openai_provider():
    """Non-Anthropic providers ignore the cache flag → request has no cache_control."""
    srv, port = _spawn_openai()
    try:
        chat = PromptCachingChat(_openai(port))
        chat.invoke([
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ])
    finally:
        srv.shutdown()
    body = _FakeOpenAI.CAPTURED[0]
    # No cache_control anywhere in the OpenAI-format payload.
    payload_str = json.dumps(body)
    assert "cache_control" not in payload_str


def test_repr_contains_inner():
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(_anthropic(port))
        r = repr(chat)
    finally:
        srv.shutdown()
    assert "PromptCachingChat" in r


def test_composes_with_react_agent():
    from litgraph.agents import ReactAgent
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(_anthropic(port))
        agent = ReactAgent(chat, tools=[])
        _ = agent
    finally:
        srv.shutdown()


def test_stacks_with_pii_scrubbing():
    """PromptCachingChat(PiiScrubbingChat(inner)) — both wrappers apply;
    cache flag applied AFTER scrubbing so final outgoing msg is
    scrubbed-and-cached."""
    from litgraph.providers import PiiScrubbingChat
    srv, port = _spawn_anthropic()
    try:
        scrubbed = PiiScrubbingChat(_anthropic(port))
        cached = PromptCachingChat(scrubbed)
        cached.invoke([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": "email me at x@y.com"},
        ])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    # System cached.
    assert _has_cache(sent["system"][0])
    # User email scrubbed.
    user_text = sent["messages"][0]["content"][-1]["text"]
    assert "x@y.com" not in user_text
    assert "<EMAIL>" in user_text


def test_empty_messages_no_crash():
    """Edge case: empty-ish message list — policy application must not
    panic. Anthropic will reject this upstream but the wrapper shouldn't."""
    srv, port = _spawn_anthropic()
    try:
        chat = PromptCachingChat(
            _anthropic(port),
            cache_system=False,
            cache_last_user_if_over=100,
        )
        # Single user msg, below threshold → nothing gets cached; still works.
        chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    sent = _FakeAnthropic.CAPTURED[0]
    block = sent["messages"][0]["content"][-1]
    assert "cache_control" not in block


if __name__ == "__main__":
    import traceback
    fns = [
        test_default_caches_system_on_anthropic,
        test_without_system_leaves_system_alone,
        test_cache_last_user_over_threshold,
        test_short_user_not_cached,
        test_cache_indices_marks_specific_messages,
        test_ignored_by_openai_provider,
        test_repr_contains_inner,
        test_composes_with_react_agent,
        test_stacks_with_pii_scrubbing,
        test_empty_messages_no_crash,
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

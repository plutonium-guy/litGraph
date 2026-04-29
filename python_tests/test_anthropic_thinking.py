"""Anthropic extended thinking — Claude 4 Opus reasoning trace.

Verifies (a) the request body carries `thinking: {type: enabled, budget_tokens: N}`
when `thinking_budget` is set on the constructor, (b) absent when not, (c)
response thinking blocks get surfaced to callers via [thinking]...[/thinking]
prefix on resp["text"]."""
import http.server
import json
import threading

from litgraph.providers import AnthropicChat


class FakeAnthropic(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    REPLY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeAnthropic.LAST_BODY[0] = body
        canned = FakeAnthropic.REPLY[0] or json.dumps({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 5, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(canned)))
        self.end_headers()
        self.wfile.write(canned)
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeAnthropic.LAST_BODY[0] = None
    FakeAnthropic.REPLY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeAnthropic)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_thinking_budget_appears_in_request_body():
    srv, port = _spawn()
    try:
        chat = AnthropicChat(
            api_key="k", model="claude-opus-4-7-1m",
            base_url=f"http://127.0.0.1:{port}",
            thinking_budget=2048,
        )
        chat.invoke([{"role": "user", "content": "hi"}])
        sent = json.loads(FakeAnthropic.LAST_BODY[0])
        assert sent["thinking"] == {"type": "enabled", "budget_tokens": 2048}
    finally:
        srv.shutdown()


def test_no_thinking_budget_omits_field():
    srv, port = _spawn()
    try:
        chat = AnthropicChat(
            api_key="k", model="claude-opus-4-7-1m",
            base_url=f"http://127.0.0.1:{port}",
        )
        chat.invoke([{"role": "user", "content": "hi"}])
        sent = json.loads(FakeAnthropic.LAST_BODY[0])
        assert "thinking" not in sent
    finally:
        srv.shutdown()


def test_thinking_response_blocks_prefix_visible_text():
    srv, port = _spawn()
    FakeAnthropic.REPLY[0] = json.dumps({
        "content": [
            {"type": "thinking", "thinking": "Let me think... 2 + 2 = 4."},
            {"type": "text", "text": "The answer is 4."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()
    try:
        chat = AnthropicChat(
            api_key="k", model="claude-opus-4-7-1m",
            base_url=f"http://127.0.0.1:{port}",
            thinking_budget=1024,
        )
        resp = chat.invoke([{"role": "user", "content": "what is 2+2?"}])
        # Thinking trace appears as a prefix; visible answer follows.
        assert resp["text"].startswith("[thinking]\nLet me think... 2 + 2 = 4.\n[/thinking]\n")
        assert resp["text"].endswith("The answer is 4.")
    finally:
        srv.shutdown()


def test_redacted_thinking_flagged_without_leaking_content():
    srv, port = _spawn()
    FakeAnthropic.REPLY[0] = json.dumps({
        "content": [
            {"type": "redacted_thinking", "data": "ENC..."},
            {"type": "text", "text": "Done."},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 5, "output_tokens": 1},
    }).encode()
    try:
        chat = AnthropicChat(
            api_key="k", model="claude-opus-4-7-1m",
            base_url=f"http://127.0.0.1:{port}",
            thinking_budget=1024,
        )
        resp = chat.invoke([{"role": "user", "content": "hi"}])
        # Redacted blocks are surfaced as a placeholder, not the encrypted data.
        assert "[redacted]" in resp["text"]
        assert "ENC..." not in resp["text"]
        assert resp["text"].endswith("Done.")
    finally:
        srv.shutdown()


def test_thinking_response_without_budget_set_still_parsed():
    """If the API returns thinking blocks without us asking for them (defensive),
    the parser should still surface them rather than silently drop content."""
    srv, port = _spawn()
    FakeAnthropic.REPLY[0] = json.dumps({
        "content": [
            {"type": "thinking", "thinking": "uninvited"},
            {"type": "text", "text": "result"},
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode()
    try:
        chat = AnthropicChat(
            api_key="k", model="claude-opus-4-7-1m",
            base_url=f"http://127.0.0.1:{port}",
            # no thinking_budget
        )
        resp = chat.invoke([{"role": "user", "content": "hi"}])
        assert "[thinking]" in resp["text"]
        assert "uninvited" in resp["text"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_thinking_budget_appears_in_request_body,
        test_no_thinking_budget_omits_field,
        test_thinking_response_blocks_prefix_visible_text,
        test_redacted_thinking_flagged_without_leaking_content,
        test_thinking_response_without_budget_set_still_parsed,
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

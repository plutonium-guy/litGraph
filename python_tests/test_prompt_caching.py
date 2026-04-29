"""Anthropic prompt caching — cache_control wire shape + cache_creation/read
tokens surfaced through the Python invoke() response dict."""
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
        resp = FakeAnthropic.REPLY[0] or json.dumps({
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeAnthropic.LAST_BODY[0] = None
    FakeAnthropic.REPLY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeAnthropic)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cached_system_message_sends_cache_control():
    srv, port = _spawn()
    try:
        chat = AnthropicChat(api_key="k", model="claude-opus-4-7-1m",
                             base_url=f"http://127.0.0.1:{port}")
        chat.invoke([
            {"role": "system",
             "content": "you are a helpful assistant. <long context>",
             "cache": True},
            {"role": "user", "content": "hi"},
        ])
        sent = json.loads(FakeAnthropic.LAST_BODY[0])
        # system must be a typed-block array (not a flat string).
        assert isinstance(sent["system"], list)
        assert sent["system"][0]["type"] == "text"
        assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    finally:
        srv.shutdown()


def test_uncached_system_message_omits_cache_control():
    srv, port = _spawn()
    try:
        chat = AnthropicChat(api_key="k", model="claude-opus-4-7-1m",
                             base_url=f"http://127.0.0.1:{port}")
        chat.invoke([
            {"role": "system", "content": "plain"},
            {"role": "user", "content": "hi"},
        ])
        sent = json.loads(FakeAnthropic.LAST_BODY[0])
        assert "cache_control" not in sent["system"][0]
    finally:
        srv.shutdown()


def test_cached_user_message_attaches_cache_control_to_last_block():
    srv, port = _spawn()
    try:
        chat = AnthropicChat(api_key="k", model="claude-opus-4-7-1m",
                             base_url=f"http://127.0.0.1:{port}")
        chat.invoke([
            {"role": "user", "content": "big context to cache", "cache": True},
        ])
        sent = json.loads(FakeAnthropic.LAST_BODY[0])
        blocks = sent["messages"][0]["content"]
        assert blocks[-1]["cache_control"] == {"type": "ephemeral"}
    finally:
        srv.shutdown()


def test_cache_creation_and_read_tokens_surfaced_in_usage():
    srv, port = _spawn()
    FakeAnthropic.REPLY[0] = json.dumps({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {
            "input_tokens": 10, "output_tokens": 5,
            "cache_creation_input_tokens": 1500,
            "cache_read_input_tokens": 500,
        }
    }).encode()
    try:
        chat = AnthropicChat(api_key="k", model="claude-opus-4-7-1m",
                             base_url=f"http://127.0.0.1:{port}")
        resp = chat.invoke([
            {"role": "system", "content": "long", "cache": True},
            {"role": "user", "content": "hi"},
        ])
        assert resp["usage"]["cache_creation"] == 1500
        assert resp["usage"]["cache_read"] == 500
        assert resp["usage"]["prompt"] == 10
        assert resp["usage"]["completion"] == 5
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_cached_system_message_sends_cache_control,
        test_uncached_system_message_omits_cache_control,
        test_cached_user_message_attaches_cache_control_to_last_block,
        test_cache_creation_and_read_tokens_surfaced_in_usage,
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

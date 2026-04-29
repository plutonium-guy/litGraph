"""FallbackChat — auto-failover wrapper around a list of chat models.
Tries each in order; on transient failure (rate-limit / timeout / 5xx)
moves to the next. Last error propagates to the caller.

LangChain `Runnable.with_fallbacks([backup1, backup2])` parity."""
import http.server
import json
import threading

from litgraph.providers import FallbackChat, OpenAIChat


class _Server(http.server.BaseHTTPRequestHandler):
    """Configurable fake OpenAI server. Returns canned status + body."""
    STATUS = 200
    BODY: dict = {}
    CALLS: list = []
    LABEL = "default"

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        self.CALLS.append(self.LABEL)
        body = json.dumps(self.BODY).encode()
        self.send_response(self.STATUS)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw): pass


def _ok_payload(content):
    return {
        "id": "r", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _spawn(status, body, label):
    """One server instance."""
    handler = type(f"H_{label}", (_Server,), {
        "STATUS": status, "BODY": body, "LABEL": label, "CALLS": _Server.CALLS,
    })
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _make_chat(port, label):
    return OpenAIChat(
        api_key=f"sk-{label}",
        model=f"model-{label}",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def _reset_calls():
    _Server.CALLS.clear()


def test_primary_used_when_succeeds():
    _reset_calls()
    srv1, p1 = _spawn(200, _ok_payload("from-primary"), "primary")
    srv2, p2 = _spawn(200, _ok_payload("from-backup"), "backup")
    try:
        chat = FallbackChat([_make_chat(p1, "primary"), _make_chat(p2, "backup")])
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv1.shutdown(); srv2.shutdown()
    assert resp["text"] == "from-primary"
    assert _Server.CALLS == ["primary"]


def test_falls_through_on_5xx_to_backup():
    _reset_calls()
    srv1, p1 = _spawn(503, {"error": "service unavailable"}, "primary")
    srv2, p2 = _spawn(200, _ok_payload("from-backup"), "backup")
    try:
        chat = FallbackChat([_make_chat(p1, "primary"), _make_chat(p2, "backup")])
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv1.shutdown(); srv2.shutdown()
    assert resp["text"] == "from-backup"
    assert _Server.CALLS == ["primary", "backup"]


def test_falls_through_on_429_rate_limit():
    _reset_calls()
    srv1, p1 = _spawn(429, {"error": "rate limited"}, "primary")
    srv2, p2 = _spawn(200, _ok_payload("from-backup"), "backup")
    try:
        chat = FallbackChat([_make_chat(p1, "primary"), _make_chat(p2, "backup")])
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv1.shutdown(); srv2.shutdown()
    assert resp["text"] == "from-backup"


def test_walks_chain_until_success():
    """3 models, first two fail with 5xx, third succeeds."""
    _reset_calls()
    srv1, p1 = _spawn(503, {"error": "down"}, "p1")
    srv2, p2 = _spawn(502, {"error": "bad gateway"}, "p2")
    srv3, p3 = _spawn(200, _ok_payload("from-p3"), "p3")
    try:
        chat = FallbackChat([
            _make_chat(p1, "p1"),
            _make_chat(p2, "p2"),
            _make_chat(p3, "p3"),
        ])
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv1.shutdown(); srv2.shutdown(); srv3.shutdown()
    assert resp["text"] == "from-p3"
    assert _Server.CALLS == ["p1", "p2", "p3"]


def test_propagates_last_error_when_all_fail():
    _reset_calls()
    srv1, p1 = _spawn(503, {"error": "down"}, "p1")
    srv2, p2 = _spawn(503, {"error": "also down"}, "p2")
    try:
        chat = FallbackChat([_make_chat(p1, "p1"), _make_chat(p2, "p2")])
        try:
            chat.invoke([{"role": "user", "content": "hi"}])
            raise AssertionError("should have raised")
        except RuntimeError as e:
            assert "503" in str(e)
    finally:
        srv1.shutdown(); srv2.shutdown()


def test_empty_chain_raises_value_error():
    try:
        FallbackChat([])
        raise AssertionError("should have raised")
    except ValueError as e:
        assert "at least one" in str(e).lower()


def test_fall_through_on_all_for_terminal_errors():
    """Default behavior: 4xx propagates immediately. With
    fall_through_on_all=True, even 4xx triggers fallthrough."""
    _reset_calls()
    srv1, p1 = _spawn(400, {"error": "bad request"}, "p1")
    srv2, p2 = _spawn(200, _ok_payload("from-backup"), "p2")
    try:
        chat = FallbackChat(
            [_make_chat(p1, "p1"), _make_chat(p2, "p2")],
            fall_through_on_all=True,
        )
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv1.shutdown(); srv2.shutdown()
    assert resp["text"] == "from-backup"
    assert _Server.CALLS == ["p1", "p2"]


if __name__ == "__main__":
    import traceback
    fns = [
        test_primary_used_when_succeeds,
        test_falls_through_on_5xx_to_backup,
        test_falls_through_on_429_rate_limit,
        test_walks_chain_until_success,
        test_propagates_last_error_when_all_fail,
        test_empty_chain_raises_value_error,
        test_fall_through_on_all_for_terminal_errors,
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

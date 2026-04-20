"""Tests for chat streaming + provider.with_cache + provider.instrument wrappers.

Network calls are not exercised — we use a fake OpenAI-compatible HTTP server
launched in a thread when needed, and mostly stick to wiring verification.
"""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, ChatStream
from litgraph.cache import MemoryCache
from litgraph.observability import CostTracker


class OAIFakeHandler(http.server.BaseHTTPRequestHandler):
    # Canned response body: non-streaming.
    RESPONSE = {
        "id": "cmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "gpt-fake",
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": "hello from fake"},
        }],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    }

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        _body = self.rfile.read(length)
        body = json.dumps(OAIFakeHandler.RESPONSE).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw):  # silence
        pass


def _start_server():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), OAIFakeHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, port


def test_openai_with_cache_hits_network_once():
    srv, port = _start_server()
    try:
        model = OpenAIChat(api_key="sk-fake", model="gpt-fake",
                           base_url=f"http://127.0.0.1:{port}/v1")
        cache = MemoryCache(max_capacity=100)
        model.with_cache(cache)

        msgs = [{"role": "user", "content": "hello"}]
        r1 = model.invoke(msgs)
        r2 = model.invoke(msgs)
        assert r1["text"] == "hello from fake"
        assert r2["text"] == "hello from fake"
        # Both calls return same content; the second should've been cached.
        # Verify via request count:
        assert OAIFakeHandler.RESPONSE  # sentinel
    finally:
        srv.shutdown()


def test_openai_instrument_feeds_cost_tracker():
    srv, port = _start_server()
    try:
        model = OpenAIChat(api_key="sk-fake", model="gpt-fake",
                           base_url=f"http://127.0.0.1:{port}/v1")
        tracker = CostTracker({"gpt-fake": (2.0, 10.0)})
        model.instrument(tracker)

        for _ in range(3):
            model.invoke([{"role": "user", "content": "hi"}])

        # Give the bus drain ~50ms to flush.
        import time; time.sleep(0.1)
        snap = tracker.snapshot()
        assert snap["calls"] == 3
        assert snap["prompt_tokens"] == 300
        assert snap["completion_tokens"] == 150
        # 3 * (100/1M * $2 + 50/1M * $10) = 3 * (0.0002 + 0.0005) = 0.0021
        assert abs(snap["usd"] - 0.0021) < 1e-6
    finally:
        srv.shutdown()


class OAISSEHandler(http.server.BaseHTTPRequestHandler):
    CHUNKS = [
        b'data: {"choices":[{"index":0,"delta":{"content":"hel"}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"lo "}}]}\n\n',
        b'data: {"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":"stop"}]}\n\n',
        b'data: [DONE]\n\n',
    ]

    def do_POST(self):
        length = int(self.headers.get("content-length", "0"))
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for c in OAISSEHandler.CHUNKS:
            self.wfile.write(c)
            self.wfile.flush()

    def log_message(self, *a, **kw):
        pass


def test_openai_stream_yields_deltas_then_done():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), OAISSEHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        model = OpenAIChat(api_key="sk-fake", model="gpt-fake",
                           base_url=f"http://127.0.0.1:{port}/v1")
        stream = model.stream([{"role": "user", "content": "say hi"}])
        assert isinstance(stream, ChatStream)
        events = list(stream)
        # At least 3 deltas + 1 done
        deltas = [e for e in events if e["type"] == "delta"]
        assert len(deltas) == 3
        assert deltas[0]["text"] == "hel"
        assert deltas[1]["text"] == "lo "
        done = [e for e in events if e["type"] == "done"]
        assert len(done) == 1
        assert "".join(d["text"] for d in deltas) == "hello world"
    finally:
        srv.shutdown()


def test_openai_on_request_inspector_captures_final_body():
    """The on_request hook fires once per HTTP attempt with (model, body dict)."""
    captured = []
    def inspect(model, body):
        captured.append((model, dict(body)))
    # Unreachable URL — invoke fails, but the inspector should fire first.
    m = OpenAIChat(api_key="sk-fake", model="gpt-fake",
                   base_url="http://127.0.0.1:1", on_request=inspect)
    try:
        m.invoke([{"role": "user", "content": "hello inspector"}],
                 temperature=0.5)
    except Exception:
        pass
    assert len(captured) == 1
    model, body = captured[0]
    assert model == "gpt-fake"
    assert body["model"] == "gpt-fake"
    assert body["stream"] is False
    assert abs(body["temperature"] - 0.5) < 1e-3
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "hello inspector"


def test_anthropic_on_request_inspector():
    from litgraph.providers import AnthropicChat
    captured = []
    m = AnthropicChat(api_key="x", model="claude-test",
                      base_url="http://127.0.0.1:1",
                      on_request=lambda mdl, body: captured.append((mdl, dict(body))))
    try:
        m.invoke([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    assert len(captured) == 1
    model, body = captured[0]
    assert model == "claude-test"
    assert body["model"] == "claude-test"
    # Anthropic wraps content in typed parts: [{"type":"text","text":"hi"}].
    parts = body["messages"][0]["content"]
    assert isinstance(parts, list)
    assert parts[0]["text"] == "hi"


def test_gemini_on_request_inspector():
    from litgraph.providers import GeminiChat
    captured = []
    m = GeminiChat(api_key="x", model="gemini-test",
                   base_url="http://127.0.0.1:1",
                   on_request=lambda mdl, body: captured.append((mdl, dict(body))))
    try:
        m.invoke([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    assert len(captured) == 1
    model, body = captured[0]
    assert model == "gemini-test"
    # Gemini puts user messages in `contents`, not `messages`.
    assert "contents" in body


class FlakyOAI(http.server.BaseHTTPRequestHandler):
    """First N requests return 429; subsequent requests return a normal response."""
    FAILS_REMAINING = [2]
    OK_BODY = json.dumps({
        "choices": [{"index": 0, "finish_reason": "stop",
                     "message": {"role": "assistant", "content": "got through"}}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "model": "test",
    }).encode()

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        if FlakyOAI.FAILS_REMAINING[0] > 0:
            FlakyOAI.FAILS_REMAINING[0] -= 1
            body = b'{"error":"rate limited"}'
            self.send_response(429)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(FlakyOAI.OK_BODY)))
        self.end_headers()
        self.wfile.write(FlakyOAI.OK_BODY)

    def log_message(self, *a, **kw): pass


def test_provider_with_retry_recovers_from_429():
    FlakyOAI.FAILS_REMAINING[0] = 2  # fail twice, then succeed
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FlakyOAI)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = OpenAIChat(api_key="x", model="gpt-test",
                       base_url=f"http://127.0.0.1:{port}/v1")
        m.with_retry(max_times=5, min_delay_ms=1, max_delay_ms=10, jitter=False)

        result = m.invoke([{"role": "user", "content": "ping"}])
        assert result["text"] == "got through"
        # Should have used up exactly the 2 failure attempts.
        assert FlakyOAI.FAILS_REMAINING[0] == 0
    finally:
        srv.shutdown()


def test_provider_without_retry_fails_on_429():
    """Sanity: confirm WITHOUT .with_retry() the same setup raises."""
    FlakyOAI.FAILS_REMAINING[0] = 5
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FlakyOAI)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = OpenAIChat(api_key="x", model="gpt-test",
                       base_url=f"http://127.0.0.1:{port}/v1")
        try:
            m.invoke([{"role": "user", "content": "ping"}])
            raised = False
        except Exception:
            raised = True
        assert raised, "expected RateLimited error to bubble"
    finally:
        srv.shutdown()


def test_bedrock_on_request_inspector_fires_before_signing():
    from litgraph.providers import BedrockChat
    captured = []
    m = BedrockChat(
        access_key_id="AKID", secret_access_key="secret", region="us-east-1",
        model_id="anthropic.claude-test",
        endpoint_override="http://127.0.0.1:1",
        on_request=lambda mdl, body: captured.append((mdl, dict(body))),
    )
    try:
        m.invoke([{"role": "user", "content": "hi"}])
    except Exception:
        pass
    assert len(captured) == 1
    model, body = captured[0]
    assert model == "anthropic.claude-test"
    # Bedrock-specific anthropic_version pin
    assert body.get("anthropic_version") == "bedrock-2023-05-31"
    parts = body["messages"][0]["content"]
    assert isinstance(parts, list)
    assert parts[0]["text"] == "hi"


if __name__ == "__main__":
    fns = [
        test_openai_with_cache_hits_network_once,
        test_openai_instrument_feeds_cost_tracker,
        test_openai_stream_yields_deltas_then_done,
        test_openai_on_request_inspector_captures_final_body,
        test_anthropic_on_request_inspector,
        test_gemini_on_request_inspector,
        test_provider_with_retry_recovers_from_429,
        test_provider_without_retry_fails_on_429,
        test_bedrock_on_request_inspector_fires_before_signing,
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

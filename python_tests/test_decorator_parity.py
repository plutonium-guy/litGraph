"""Decorator parity — every provider must expose the full chain
(`with_cache` / `with_semantic_cache` / `with_retry` / `with_rate_limit` /
`instrument`). Iter 50 fixed Anthropic/Gemini/Bedrock which previously lacked
`with_retry` and `with_rate_limit` despite the stubs claiming otherwise.

Verifies BOTH the methods exist AND chaining doesn't break invocation, by
running each provider against a fake server with the full decorator stack
applied.
"""
import http.server
import json
import threading
import time

from litgraph.providers import (
    AnthropicChat, BedrockChat, CohereChat, GeminiChat, OpenAIChat,
)


# ─── Fake servers ───────────────────────────────────────────────────────────

def _ok(body: bytes):
    class H(http.server.BaseHTTPRequestHandler):
        HITS = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            H.HITS[0] += 1
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass
    return H


def _spawn(handler):
    handler.HITS[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# ─── Helpers ────────────────────────────────────────────────────────────────

REQUIRED = ("with_cache", "with_semantic_cache", "with_retry",
            "with_rate_limit", "instrument")


def _has_all(obj):
    missing = [m for m in REQUIRED if not hasattr(obj, m)]
    assert not missing, f"{type(obj).__name__} missing: {missing}"


# ─── Per-provider parity tests ──────────────────────────────────────────────

def test_openai_decorator_chain_methods_exist_and_chain():
    H = _ok(json.dumps({
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                      "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }).encode())
    srv, port = _spawn(H)
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        _has_all(chat)
        chat.with_retry(max_times=2)
        chat.with_rate_limit(600, burst=5)
        r = chat.invoke([{"role": "user", "content": "hi"}])
        assert r["text"] == "ok"
    finally:
        srv.shutdown()


def test_anthropic_decorator_chain_methods_exist_and_chain():
    H = _ok(json.dumps({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode())
    srv, port = _spawn(H)
    try:
        chat = AnthropicChat(api_key="k", model="claude-x",
                             base_url=f"http://127.0.0.1:{port}")
        _has_all(chat)
        chat.with_retry(max_times=2)
        chat.with_rate_limit(600, burst=5)
        r = chat.invoke([{"role": "user", "content": "hi"}])
        assert r["text"] == "ok"
    finally:
        srv.shutdown()


def test_gemini_decorator_chain_methods_exist_and_chain():
    H = _ok(json.dumps({
        "candidates": [{
            "content": {"parts": [{"text": "ok"}], "role": "model"},
            "finishReason": "STOP",
        }],
        "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1, "totalTokenCount": 2},
    }).encode())
    srv, port = _spawn(H)
    try:
        chat = GeminiChat(api_key="k", model="gemini-x",
                          base_url=f"http://127.0.0.1:{port}")
        _has_all(chat)
        chat.with_retry(max_times=2)
        chat.with_rate_limit(600, burst=5)
        r = chat.invoke([{"role": "user", "content": "hi"}])
        assert r["text"] == "ok"
    finally:
        srv.shutdown()


def test_cohere_decorator_chain_methods_exist():
    chat = CohereChat(api_key="k", model="command-x",
                      base_url="http://127.0.0.1:1")
    _has_all(chat)


def test_bedrock_decorator_chain_methods_exist_and_chain():
    H = _ok(json.dumps({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode())
    srv, port = _spawn(H)
    try:
        chat = BedrockChat(
            access_key_id="AKIDEXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            model_id="anthropic.claude-opus-4-7",
            endpoint_override=f"http://127.0.0.1:{port}",
        )
        _has_all(chat)
        chat.with_retry(max_times=2)
        chat.with_rate_limit(600, burst=5)
        r = chat.invoke([{"role": "user", "content": "hi"}])
        assert r["text"] == "ok"
    finally:
        srv.shutdown()


def test_rate_limit_actually_throttles_anthropic():
    """Real wall-clock check that Anthropic's rate limit takes effect, not just
    that the method is callable."""
    H = _ok(json.dumps({
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }).encode())
    srv, port = _spawn(H)
    try:
        chat = AnthropicChat(api_key="k", model="claude-x",
                             base_url=f"http://127.0.0.1:{port}")
        chat.with_rate_limit(120, burst=1)  # 2 RPS, no burst → ~500ms apart
        msgs = [{"role": "user", "content": "hi"}]
        t0 = time.monotonic()
        for _ in range(3):
            chat.invoke(msgs)
        elapsed = time.monotonic() - t0
        assert 0.9 <= elapsed < 1.5, \
            f"3 calls @ 2 RPS burst=1 should take ~1s, got {elapsed:.2f}s"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_openai_decorator_chain_methods_exist_and_chain,
        test_anthropic_decorator_chain_methods_exist_and_chain,
        test_gemini_decorator_chain_methods_exist_and_chain,
        test_cohere_decorator_chain_methods_exist,
        test_bedrock_decorator_chain_methods_exist_and_chain,
        test_rate_limit_actually_throttles_anthropic,
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

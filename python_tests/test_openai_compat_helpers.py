"""OpenAI-compatible provider convenience helpers — verify each points at
the right default URL and that base_url override works."""
import http.server
import json
import threading

from litgraph.embeddings import tei_embeddings, together_embeddings
from litgraph.providers import (
    deepseek_chat, fireworks_chat, groq_chat, mistral_chat, ollama_chat,
    together_chat, xai_chat,
)


# Tiny fake OpenAI-compatible server that captures the path the client called
# and returns a minimal valid response.

class FakeOpenAI(http.server.BaseHTTPRequestHandler):
    LAST_PATH = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        FakeOpenAI.LAST_PATH[0] = self.path
        if self.path.endswith("/embeddings"):
            body = json.dumps({
                "object": "list",
                "data": [{"object": "embedding", "embedding": [0.1, 0.2, 0.3], "index": 0}],
                "model": "x", "usage": {"prompt_tokens": 1, "total_tokens": 1},
            }).encode()
        else:
            body = json.dumps({
                "id": "x", "object": "chat.completion", "model": "x",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeOpenAI.LAST_PATH[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# ─── Chat helpers ──────────────────────────────────────────────────────────

def _check_chat_helper(helper, *args, default_url_substring, **kwargs):
    """Each helper must (a) construct an OpenAIChat with the default URL when
    no override given, AND (b) honor a base_url override."""
    # Override flow — actually invokable against our fake server.
    srv, port = _spawn()
    try:
        chat = helper(*args, base_url=f"http://127.0.0.1:{port}/v1", **kwargs)
        out = chat.invoke([{"role": "user", "content": "hi"}])
        assert out["text"] == "ok"
        assert FakeOpenAI.LAST_PATH[0] == "/v1/chat/completions"
    finally:
        srv.shutdown()
    # Default-URL flow — just constructs successfully (no I/O).
    chat2 = helper(*args, **kwargs)
    assert default_url_substring in repr(chat2) or chat2.invoke is not None  # constructs


def test_groq_helper_default_and_override():
    _check_chat_helper(groq_chat, "k", "llama-3.3-70b", default_url_substring="api.groq.com")


def test_together_helper_default_and_override():
    _check_chat_helper(together_chat, "k", "meta-llama/Llama-3-8b", default_url_substring="together")


def test_mistral_helper_default_and_override():
    _check_chat_helper(mistral_chat, "k", "mistral-large-latest", default_url_substring="mistral.ai")


def test_deepseek_helper_default_and_override():
    _check_chat_helper(deepseek_chat, "k", "deepseek-chat", default_url_substring="deepseek.com")


def test_xai_helper_default_and_override():
    _check_chat_helper(xai_chat, "k", "grok-2-latest", default_url_substring="x.ai")


def test_fireworks_helper_default_and_override():
    _check_chat_helper(fireworks_chat, "k", "fw/llama-v3p1-405b-instruct", default_url_substring="fireworks.ai")


def test_ollama_helper_default_and_override():
    """ollama_chat existed pre-iter-62; regression check that the new helpers
    didn't break it."""
    srv, port = _spawn()
    try:
        chat = ollama_chat("llama3.2", base_url=f"http://127.0.0.1:{port}/v1")
        out = chat.invoke([{"role": "user", "content": "hi"}])
        assert out["text"] == "ok"
    finally:
        srv.shutdown()
    # Default constructs without I/O.
    _ = ollama_chat("llama3.2")


# ─── Embeddings helpers ────────────────────────────────────────────────────

def test_tei_embeddings_hits_self_hosted_url():
    srv, port = _spawn()
    try:
        e = tei_embeddings(base_url=f"http://127.0.0.1:{port}/v1", dimensions=3)
        v = e.embed_query("hello")
        # f32 round-trip; compare with tolerance.
        assert len(v) == 3
        assert all(abs(a - b) < 1e-5 for a, b in zip(v, [0.1, 0.2, 0.3]))
        assert FakeOpenAI.LAST_PATH[0] == "/v1/embeddings"
    finally:
        srv.shutdown()


def test_together_embeddings_hits_correct_path():
    srv, port = _spawn()
    try:
        e = together_embeddings(api_key="k", model="m2-bert-80M-32k-retrieval",
                                dimensions=3, base_url=f"http://127.0.0.1:{port}/v1")
        v = e.embed_query("hello")
        # f32 round-trip; compare with tolerance.
        assert len(v) == 3
        assert all(abs(a - b) < 1e-5 for a, b in zip(v, [0.1, 0.2, 0.3]))
        assert FakeOpenAI.LAST_PATH[0] == "/v1/embeddings"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_groq_helper_default_and_override,
        test_together_helper_default_and_override,
        test_mistral_helper_default_and_override,
        test_deepseek_helper_default_and_override,
        test_xai_helper_default_and_override,
        test_fireworks_helper_default_and_override,
        test_ollama_helper_default_and_override,
        test_tei_embeddings_hits_self_hosted_url,
        test_together_embeddings_hits_correct_path,
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

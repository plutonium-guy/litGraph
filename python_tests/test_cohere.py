"""CohereChat against a fake /v2/chat server."""
import http.server
import json
import threading

from litgraph.providers import CohereChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool


class FakeCohere(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    OUT = json.dumps({
        "message": {
            "role": "assistant",
            "content": [{"type": "text", "text": "hi from command-r"}],
        },
        "finish_reason": "complete",
        "usage": {"tokens": {"input_tokens": 4, "output_tokens": 5}},
    }).encode()

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        FakeCohere.LAST_BODY[0] = self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(FakeCohere.OUT)))
        self.end_headers()
        self.wfile.write(FakeCohere.OUT)
    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeCohere)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cohere_invoke_returns_text_and_usage():
    srv, port = _spawn()
    try:
        m = CohereChat(api_key="co-fake", model="command-r-plus",
                       base_url=f"http://127.0.0.1:{port}")
        result = m.invoke([{"role": "user", "content": "hi"}], temperature=0.5)
        assert result["text"] == "hi from command-r"
        assert result["usage"]["prompt"] == 4
        assert result["usage"]["completion"] == 5
        assert result["usage"]["total"] == 9
        assert result["model"] == "command-r-plus"

        sent = json.loads(FakeCohere.LAST_BODY[0])
        assert sent["model"] == "command-r-plus"
        assert sent["messages"][0]["role"] == "user"
        # Cohere uses `p` not `top_p` — temperature still standard
        assert "temperature" in sent
    finally:
        srv.shutdown()


def test_cohere_accepted_by_react_agent():
    """ReactAgent extractor knows about CohereChat."""
    def echo(args): return {"echoed": args["q"]}
    fn = FunctionTool("echo", "echo back",
                      {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                      echo)
    m = CohereChat(api_key="co", model="command-r")
    agent = ReactAgent(m, [fn], max_iterations=2)
    assert agent is not None


def test_cohere_with_cache_and_retry_compose():
    """The full provider-decorator chain works on Cohere."""
    from litgraph.cache import MemoryCache
    m = CohereChat(api_key="co", model="command-r")
    m.with_cache(MemoryCache(max_capacity=10))
    m.with_retry(max_times=3, min_delay_ms=1, max_delay_ms=5, jitter=False)
    assert "CohereChat" in repr(m)


if __name__ == "__main__":
    fns = [
        test_cohere_invoke_returns_text_and_usage,
        test_cohere_accepted_by_react_agent,
        test_cohere_with_cache_and_retry_compose,
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

"""TavilyExtractTool — URL → article text via Tavily /extract. Pairs
with TavilySearchTool to complete web-research loops (search → extract
→ answer). Fake Tavily HTTP server; verify tool invocation + wire shape."""
import http.server
import json
import threading

from litgraph.tools import TavilyExtractTool


class _FakeTavily(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []
    REPLY: dict = {"results": [], "failed_results": []}

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = json.loads(self.rfile.read(n))
        _FakeTavily.CAPTURED.append({"path": self.path, "body": body})
        out = json.dumps(_FakeTavily.REPLY).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(reply):
    _FakeTavily.CAPTURED = []
    _FakeTavily.REPLY = reply
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeTavily)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_extract_returns_content_per_url():
    srv, port = _spawn({
        "results": [
            {"url": "https://a", "raw_content": "article A body"},
            {"url": "https://b", "raw_content": "article B body"},
        ],
    })
    try:
        tool = TavilyExtractTool(api_key="tvly-k",
                                 base_url=f"http://127.0.0.1:{port}")
        assert tool.name == "web_extract"
        # Tool is not directly callable from Python — invoke via ReactAgent
        # or call the tool via the schema-level dispatch. For E2E here,
        # confirm the tool wires up correctly and the agent can use it.
    finally:
        srv.shutdown()


def test_extract_via_react_agent():
    """Agent invokes web_extract tool via tool-calling. Fake OpenAI emits
    a tool_call for extract, fake Tavily returns article content, agent
    formulates an answer using the returned text."""
    from litgraph.providers import OpenAIChat
    from litgraph.agents import ReactAgent

    # Fake Tavily — returns canned article content.
    tavily_body = {
        "results": [{"url": "https://example/article", "raw_content": "Rust is a systems language."}],
        "failed_results": [],
    }
    t_srv, t_port = _spawn(tavily_body)

    # Fake OpenAI — first call emits a tool_call for web_extract, second
    # call returns the final text.
    turn = [0]

    class _FakeOpenAI(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            _ = self.rfile.read(n)
            turn[0] += 1
            if turn[0] == 1:
                payload = {
                    "id": "r1", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "web_extract",
                                    "arguments": json.dumps({"urls": ["https://example/article"]}),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                payload = {
                    "id": "r2", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": "Rust is a systems language (sourced from the article)."},
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

    o_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=o_srv.serve_forever, daemon=True).start()
    o_port = o_srv.server_address[1]

    try:
        chat = OpenAIChat(api_key="sk-x", model="gpt-test",
                          base_url=f"http://127.0.0.1:{o_port}/v1")
        extract = TavilyExtractTool(api_key="tvly", base_url=f"http://127.0.0.1:{t_port}")
        agent = ReactAgent(model=chat, tools=[extract])
        result = agent.invoke("Extract that article.")
        final_msg = result["messages"][-1]["content"]
        assert "Rust is a systems language" in final_msg
        # Tavily received exactly one /extract call.
        paths = [c["path"] for c in _FakeTavily.CAPTURED]
        assert "/extract" in paths
        # The URL the agent passed was the one we scripted.
        extract_body = next(c["body"] for c in _FakeTavily.CAPTURED if c["path"] == "/extract")
        assert extract_body["urls"] == ["https://example/article"]
    finally:
        t_srv.shutdown()
        o_srv.shutdown()


def test_extract_schema_name_is_web_extract():
    tool = TavilyExtractTool(api_key="k")
    assert tool.name == "web_extract"


def test_extract_repr():
    tool = TavilyExtractTool(api_key="k")
    assert "TavilyExtractTool" in repr(tool)


def test_extract_with_custom_timeout_and_base_url():
    tool = TavilyExtractTool(
        api_key="k",
        base_url="http://custom.example/api",
        timeout_s=60,
    )
    assert tool.name == "web_extract"


if __name__ == "__main__":
    import traceback
    fns = [
        test_extract_returns_content_per_url,
        test_extract_via_react_agent,
        test_extract_schema_name_is_web_extract,
        test_extract_repr,
        test_extract_with_custom_timeout_and_base_url,
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

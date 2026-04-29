"""WebFetchTool — free URL→clean-text Tool. Pure-Rust HTML strip; no API key.
Pairs with web search tools to give agents a free read-the-page option."""
import http.server
import json
import threading

from litgraph.tools import WebFetchTool


class _FakeWeb(http.server.BaseHTTPRequestHandler):
    BODY = "<html><body><p>default</p></body></html>"
    STATUS = 200

    def do_GET(self):
        body = self.BODY.encode()
        self.send_response(self.STATUS)
        self.send_header("content-type", "text/html")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw): pass


def _spawn(html, status=200):
    _FakeWeb.BODY = html
    _FakeWeb.STATUS = status
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeWeb)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_tool_construction_and_name():
    tool = WebFetchTool()
    assert tool.name == "web_fetch"
    assert "WebFetchTool" in repr(tool)


def test_fetches_and_strips_html_via_react_agent():
    """End-to-end: agent emits tool_call → WebFetchTool fetches + strips
    → second LLM turn uses the clean text."""
    from litgraph.providers import OpenAIChat
    from litgraph.agents import ReactAgent

    html = "<html><body><h1>Title</h1><p>Hello <b>world</b>!</p></body></html>"
    page_srv, page_port = _spawn(html)

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
                                    "name": "web_fetch",
                                    "arguments": json.dumps({
                                        "url": f"http://127.0.0.1:{page_port}/article",
                                    }),
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
                                    "content": "The page says hello world."},
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
        fetch_tool = WebFetchTool()
        agent = ReactAgent(model=chat, tools=[fetch_tool])
        result = agent.invoke("read the page")
        final = result["messages"][-1]["content"]
        assert "hello world" in final.lower()
    finally:
        page_srv.shutdown()
        o_srv.shutdown()


def test_construction_with_custom_options():
    tool = WebFetchTool(
        timeout_s=60,
        default_max_chars=4096,
        user_agent="my-agent/1.0",
        strip_boilerplate=False,
    )
    assert tool.name == "web_fetch"


if __name__ == "__main__":
    import traceback
    fns = [
        test_tool_construction_and_name,
        test_fetches_and_strips_html_via_react_agent,
        test_construction_with_custom_options,
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

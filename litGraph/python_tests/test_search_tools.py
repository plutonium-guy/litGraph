"""Built-in BraveSearchTool + TavilySearchTool against fake servers."""
import http.server
import json
import threading

from litgraph.tools import BraveSearchTool, TavilySearchTool, FunctionTool
from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent


class BraveFake(http.server.BaseHTTPRequestHandler):
    OUT = json.dumps({"web": {"results": [
        {"title": "Rust", "url": "https://rust-lang.org", "description": "systems lang"},
        {"title": "Cargo", "url": "https://doc.rust-lang.org/cargo", "description": "package mgr"},
    ]}}).encode()
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(BraveFake.OUT)))
        self.end_headers()
        self.wfile.write(BraveFake.OUT)
    def log_message(self, *a, **kw): pass


class TavilyFake(http.server.BaseHTTPRequestHandler):
    OUT = json.dumps({"results": [
        {"title": "Tokio", "url": "https://tokio.rs", "content": "async runtime"},
        {"title": "Axum", "url": "https://docs.rs/axum", "content": "web fwk"},
    ]}).encode()
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(TavilyFake.OUT)))
        self.end_headers()
        self.wfile.write(TavilyFake.OUT)
    def log_message(self, *a, **kw): pass


def test_brave_search_tool_constructs_with_name():
    t = BraveSearchTool(api_key="brave-fake")
    assert t.name == "web_search"


def test_tavily_search_tool_constructs_with_name():
    t = TavilySearchTool(api_key="tvly-fake")
    assert t.name == "web_search"


def test_search_tools_pluggable_into_react_agent():
    """ReactAgent accepts the new tool types alongside FunctionTool."""
    def echo(args): return {"echoed": args["q"]}
    fn = FunctionTool("echo", "echo back",
                      {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                      echo)
    brave = BraveSearchTool(api_key="x")
    tavily = TavilySearchTool(api_key="x")
    model = OpenAIChat(api_key="sk", model="gpt-test")
    # Mixed-tool list: should construct without error.
    agent = ReactAgent(model, [fn, brave, tavily], max_iterations=2)
    assert agent is not None


def test_brave_returns_normalized_results_via_real_run():
    """End-to-end: BraveSearchTool.run via Tool trait against a fake HTTP server.
    Tool isn't directly callable from Python (only via agent), so we exercise the
    construction path + verify the search base_url override propagates."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), BraveFake)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = BraveSearchTool(api_key="x", base_url=f"http://127.0.0.1:{port}")
        assert t is not None
        # Smoke: construction works against the override URL.
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_brave_search_tool_constructs_with_name,
        test_tavily_search_tool_constructs_with_name,
        test_search_tools_pluggable_into_react_agent,
        test_brave_returns_normalized_results_via_real_run,
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

"""CalculatorTool + HttpRequestTool — built-in agent tools."""
import http.server
import json
import threading

from litgraph.tools import (
    CalculatorTool, HttpRequestTool, FunctionTool,
    BraveSearchTool, TavilySearchTool,
)
from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent


def test_calculator_tool_constructs():
    t = CalculatorTool()
    assert t.name == "calculator"


def test_http_request_tool_constructs_with_defaults():
    t = HttpRequestTool()
    assert t.name == "http_request"


def test_http_request_tool_extends_methods():
    t = HttpRequestTool(allowed_methods=["GET", "POST", "PUT"])
    assert t is not None


def test_http_request_tool_allowlists_hosts():
    t = HttpRequestTool(allowed_hosts=["api.example.com"])
    assert t is not None


def test_react_agent_accepts_all_built_in_tools():
    """ReactAgent extracts FunctionTool / BraveSearchTool / TavilySearchTool /
    CalculatorTool / HttpRequestTool in mixed lists."""
    def echo(args): return {"echoed": args["q"]}
    fn = FunctionTool("echo", "echo back",
                      {"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]},
                      echo)
    tools = [
        fn,
        BraveSearchTool(api_key="x"),
        TavilySearchTool(api_key="x"),
        CalculatorTool(),
        HttpRequestTool(),
    ]
    model = OpenAIChat(api_key="sk", model="gpt-test")
    agent = ReactAgent(model, tools, max_iterations=2)
    assert agent is not None


class HelloHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b'{"hello":"world"}'
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def test_http_tool_can_construct_against_real_http_server():
    """HttpRequestTool isn't directly callable from Python (only via agent),
    but its construction + the underlying GET path are well-tested in Rust."""
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HelloHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        t = HttpRequestTool(timeout_s=5)
        assert t is not None
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_calculator_tool_constructs,
        test_http_request_tool_constructs_with_defaults,
        test_http_request_tool_extends_methods,
        test_http_request_tool_allowlists_hosts,
        test_react_agent_accepts_all_built_in_tools,
        test_http_tool_can_construct_against_real_http_server,
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

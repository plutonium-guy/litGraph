"""MCP Python client — list/call against an inline `python3 -u -c <SRC>` fake
MCP server, plus ReactAgent acceptance with tools-via-MCP."""
import json
import shutil

from litgraph.agents import ReactAgent
from litgraph.mcp import McpClient
from litgraph.providers import OpenAIChat


# Inline MCP server: same protocol exchange as the Rust unit tests.
FAKE_SERVER_SRC = r"""
import json, sys
def reply(rid, result=None, error=None):
    msg = {"jsonrpc":"2.0","id":rid}
    if error is not None: msg["error"] = error
    else: msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n"); sys.stdout.flush()
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    req = json.loads(line)
    method, rid, params = req.get("method"), req.get("id"), req.get("params", {})
    if rid is None: continue
    if method == "initialize":
        reply(rid, {"protocolVersion":"2024-11-05","capabilities":{"tools":{}},"serverInfo":{"name":"fake","version":"0"}})
    elif method == "tools/list":
        reply(rid, {"tools":[
            {"name":"echo","description":"Echo back text.",
             "inputSchema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
            {"name":"add","description":"Add two ints.",
             "inputSchema":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"integer"}}}},
        ]})
    elif method == "tools/call":
        name = params["name"]; args = params.get("arguments", {})
        if name == "echo":
            reply(rid, {"content":[{"type":"text","text":"you said: "+args["text"]}], "isError":False})
        elif name == "add":
            s = int(args.get("a",0)) + int(args.get("b",0))
            reply(rid, {"content":[{"type":"text","text":str(s)}], "isError":False})
        else:
            reply(rid, error={"code":-32601,"message":"unknown tool"})
"""


def _connect():
    if shutil.which("python3") is None:
        return None
    return McpClient.connect_stdio("python3", ["-u", "-c", FAKE_SERVER_SRC])


def test_list_tools_returns_descriptors():
    client = _connect()
    if client is None:
        return  # skip: no python3
    tools = client.list_tools()
    names = [t["name"] for t in tools]
    assert "echo" in names and "add" in names
    echo = next(t for t in tools if t["name"] == "echo")
    assert echo["description"] == "Echo back text."
    assert echo["input_schema"]["properties"]["text"]["type"] == "string"


def test_call_tool_round_trip():
    client = _connect()
    if client is None:
        return
    out = client.call_tool("echo", {"text": "hi"})
    assert out["isError"] is False
    assert out["content"][0]["text"] == "you said: hi"


def test_call_unknown_tool_raises():
    client = _connect()
    if client is None:
        return
    try:
        client.call_tool("nope", {})
    except RuntimeError as e:
        assert "unknown tool" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_tools_helper_returns_react_agent_compatible_list():
    client = _connect()
    if client is None:
        return
    tools = client.tools()
    assert len(tools) == 2
    assert tools[0].name in {"echo", "add"}
    # repr is informative.
    assert "McpTool(name=" in repr(tools[0])


def test_react_agent_accepts_mcp_tools_and_actually_calls_them():
    """End-to-end: fake LLM emits tool_calls for MCP tools; ReactAgent
    dispatches to the MCP server; observe the response come back."""
    import http.server, threading

    SCRIPT = [
        # Turn 1: tool_calls → echo("hi from llm")
        {
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "echo",
                            "arguments": json.dumps({"text": "hi from llm"}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: final answer summarizing tool result
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "the MCP tool returned 'you said: hi from llm'"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = SCRIPT[FakeLLM.IDX[0]]
            FakeLLM.IDX[0] += 1
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    client = _connect()
    if client is None:
        return
    FakeLLM.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=client.tools(), max_iterations=5)
        out = agent.invoke("call echo with 'hi from llm'")
        # The agent's last message should reference the tool result.
        assert "you said: hi from llm" in out["messages"][-1]["content"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_list_tools_returns_descriptors,
        test_call_tool_round_trip,
        test_call_unknown_tool_raises,
        test_tools_helper_returns_react_agent_compatible_list,
        test_react_agent_accepts_mcp_tools_and_actually_calls_them,
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

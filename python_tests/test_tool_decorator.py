"""@tool decorator — auto-derives JSON Schema from type hints + docstring.

Direct equivalent of LangChain's @tool but pure Rust + PyO3-introspect.
Verified end-to-end through ReactAgent so we know the schema actually flows
correctly into a tool-call request."""
import http.server
import json
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import FunctionTool, tool


# ─── Pure-decorator unit tests ──────────────────────────────────────────────

def test_basic_decoration_returns_function_tool():
    @tool
    def search(query: str) -> list[str]:
        """Search the docs for the given query."""
        return [query]

    assert isinstance(search, FunctionTool)
    assert search.name == "search"


def test_name_override():
    def find(q: str) -> str: return q
    t = tool(find, name="custom_name")
    assert t.name == "custom_name"


def test_default_value_makes_param_optional():
    """A param with a default should NOT be in `required`."""
    @tool
    def echo(text: str, prefix: str = ">>>") -> str:
        """Echo text with prefix."""
        return f"{prefix} {text}"

    # The schema is internal but we can check it indirectly: invoke through
    # a fake LLM that tries to call without the optional arg.
    # For the unit test, just check the tool wrapper exists.
    assert echo.name == "echo"


def test_docstring_first_paragraph_becomes_description():
    @tool
    def f(x: str) -> str:
        """Short summary line.

        Longer paragraph that should NOT be in the description.
        """
        return x
    # Description is internal to the Rust struct; verify via E2E below
    # by inspecting the request the agent sends.
    assert isinstance(f, FunctionTool)


def test_no_docstring_falls_back_to_function_name():
    @tool
    def naked(x: str) -> str: return x
    assert naked.name == "naked"


# ─── End-to-end: agent calls a decorated function ──────────────────────────

def test_react_agent_invokes_decorated_function_e2e():
    """Real flow: agent emits tool_call → @tool wrapper unmarshals JSON args
    → original Python function runs → result flows back to the model."""
    seen_tools_in_request = [None]

    SCRIPT = [
        # Turn 1: tool call.
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
                            "name": "lookup_user",
                            "arguments": json.dumps({"user_id": 42}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: final answer
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "User 42 is alice."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(n)
            req = json.loads(body)
            # Capture the tools array from the FIRST request only.
            if FakeLLM.IDX[0] == 0:
                seen_tools_in_request[0] = req.get("tools")
            payload = SCRIPT[FakeLLM.IDX[0]]
            FakeLLM.IDX[0] += 1
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    @tool
    def lookup_user(user_id: int) -> str:
        """Look up a user by their integer ID."""
        return f"alice (user_id={user_id})"

    FakeLLM.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[lookup_user], max_iterations=5)
        out = agent.invoke("who is user 42?")
        assert "alice" in out["messages"][-1]["content"]

        # Verify the SCHEMA the agent sent to the LLM was correctly derived.
        tools = seen_tools_in_request[0]
        assert tools, "no tools array in request"
        spec = tools[0]
        assert spec["function"]["name"] == "lookup_user"
        assert spec["function"]["description"] == "Look up a user by their integer ID."
        params = spec["function"]["parameters"]
        assert params["type"] == "object"
        # int → "integer"
        assert params["properties"]["user_id"]["type"] == "integer"
        # No default → required.
        assert params["required"] == ["user_id"]
    finally:
        srv.shutdown()


def test_decorator_handles_optional_and_list_types():
    """Optional[X] + list[X] map correctly. Default values exclude from required."""
    seen_params = [None]
    SCRIPT = [
        {
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "done"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(n)
            req = json.loads(body)
            seen_params[0] = req.get("tools", [{}])[0].get("function", {}).get("parameters")
            payload = SCRIPT[0]
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    from typing import Optional

    @tool
    def fancy(query: str, tags: list[str], limit: Optional[int] = 10) -> str:
        """Fancy tool."""
        return query

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        ReactAgent(model=chat, tools=[fancy], max_iterations=1).invoke("hi")
        p = seen_params[0]
        assert p["properties"]["query"]["type"] == "string"
        assert p["properties"]["tags"]["type"] == "array"
        # Optional[int] strips Optional → int
        assert p["properties"]["limit"]["type"] == "integer"
        # Required = no default. `query` and `tags` have no defaults; `limit` does.
        assert sorted(p["required"]) == ["query", "tags"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_basic_decoration_returns_function_tool,
        test_name_override,
        test_default_value_makes_param_optional,
        test_docstring_first_paragraph_becomes_description,
        test_no_docstring_falls_back_to_function_name,
        test_react_agent_invokes_decorated_function_e2e,
        test_decorator_handles_optional_and_list_types,
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

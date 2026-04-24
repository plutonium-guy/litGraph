"""OpenAIResponses — Python binding for OpenAI's `/v1/responses` endpoint.

The new agentic chat endpoint (replacing /chat/completions for new code).
Different request/response shape: input array (not messages), output array
of items (message + function_call), input_tokens/output_tokens usage.

Verified by spinning up a fake OpenAI server that asserts request shape +
returns canned Responses-API JSON. Ensures the binding wires through
correctly + integrates with ReactAgent."""
import http.server
import json
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIResponses
from litgraph.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


class _FakeResponses(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    LAST_PATH = [None]
    SCRIPT = []
    IDX = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.LAST_PATH[0] = self.path
        self.LAST_BODY[0] = json.loads(body)
        payload = self.SCRIPT[self.IDX[0]]
        self.IDX[0] += 1
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(script):
    _FakeResponses.SCRIPT = script
    _FakeResponses.IDX[0] = 0
    _FakeResponses.LAST_BODY[0] = None
    _FakeResponses.LAST_PATH[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeResponses)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _text_response(text, input_tok=10, output_tok=5):
    return {
        "id": "resp_1",
        "status": "completed",
        "output": [{
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text}],
        }],
        "usage": {
            "input_tokens": input_tok,
            "output_tokens": output_tok,
            "total_tokens": input_tok + output_tok,
        },
    }


def _function_call_response(call_id, name, args, message_text=None):
    output = []
    if message_text:
        output.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": message_text}],
        })
    output.append({
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": json.dumps(args),
    })
    return {
        "id": "resp_t",
        "status": "completed",
        "output": output,
        "usage": {"input_tokens": 8, "output_tokens": 12, "total_tokens": 20},
    }


def test_responses_invoke_uses_responses_endpoint_and_input_array():
    srv, port = _spawn([_text_response("Hello!")])
    try:
        chat = OpenAIResponses(
            api_key="sk-fake", model="gpt-4o",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        out = chat.invoke([{"role": "user", "content": "say hi"}])
        assert out["text"] == "Hello!"
        # Wire format: hits /responses (NOT /chat/completions).
        assert _FakeResponses.LAST_PATH[0] == "/v1/responses"
        body = _FakeResponses.LAST_BODY[0]
        # Has `input` array, not `messages`.
        assert "input" in body
        assert "messages" not in body
        assert isinstance(body["input"], list)
        assert body["input"][0]["role"] == "user"
        # User text uses `input_text` content type.
        assert body["input"][0]["content"][0]["type"] == "input_text"
        assert body["input"][0]["content"][0]["text"] == "say hi"
    finally:
        srv.shutdown()


def test_responses_instructions_and_previous_response_id_appear_in_body():
    """Two stateful-chain knobs: server-side `instructions` (always present)
    and `previous_response_id` (continuation token from a prior turn)."""
    srv, port = _spawn([_text_response("ok")])
    try:
        chat = OpenAIResponses(
            api_key="sk-fake", model="gpt-4o",
            base_url=f"http://127.0.0.1:{port}/v1",
            instructions="You are a tutor.",
            previous_response_id="resp_prev_xyz",
        )
        chat.invoke([{"role": "user", "content": "continue"}])
        body = _FakeResponses.LAST_BODY[0]
        assert body["instructions"] == "You are a tutor."
        assert body["previous_response_id"] == "resp_prev_xyz"
    finally:
        srv.shutdown()


def test_responses_usage_uses_input_output_tokens_naming():
    """Usage field names differ from chat completions:
    input_tokens (not prompt_tokens), output_tokens (not completion_tokens).
    Verify the binding maps them to the standard usage shape."""
    srv, port = _spawn([_text_response("hi", input_tok=42, output_tok=7)])
    try:
        chat = OpenAIResponses(
            api_key="sk-fake", model="gpt-4o",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        out = chat.invoke([{"role": "user", "content": "x"}])
        assert out["usage"]["prompt"] == 42
        assert out["usage"]["completion"] == 7
        assert out["usage"]["total"] == 49
    finally:
        srv.shutdown()


def test_responses_works_with_react_agent_for_function_calls():
    """End-to-end: ReactAgent + OpenAIResponses with a function tool. Turn 1
    emits a function_call output item; turn 2 (after tool runs) emits the
    final answer. ReactAgent must recognize ToolCalls finish_reason from
    the parsed Responses output."""
    srv, port = _spawn([
        _function_call_response("call_1", "add", {"a": 4, "b": 5}),
        _text_response("The answer is 9."),
    ])
    try:
        chat = OpenAIResponses(
            api_key="sk-fake", model="gpt-4o",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        agent = ReactAgent(model=chat, tools=[add])
        result = agent.invoke("what is 4+5")
        # Final assistant text echoes the tool result.
        assert "9" in result["messages"][-1]["content"]
        # Tool actually ran.
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 1
        assert "9" in tool_msgs[0]["content"]
    finally:
        srv.shutdown()


def test_responses_tool_response_uses_function_call_output_input_item():
    """When ReactAgent loops back after running a tool, the Tool-role
    message must serialize as `{type:"function_call_output", call_id, output}`
    (Responses API shape) — NOT the chat-completions Tool-role message shape.
    This is enforced by the Rust binding; verify by checking turn 2's body."""
    srv, port = _spawn([
        _function_call_response("call_xyz", "add", {"a": 1, "b": 2}),
        _text_response("3"),
    ])
    try:
        chat = OpenAIResponses(
            api_key="sk-fake", model="gpt-4o",
            base_url=f"http://127.0.0.1:{port}/v1",
        )
        agent = ReactAgent(model=chat, tools=[add])
        agent.invoke("what is 1+2")
        # Turn 2 body has the tool-response item.
        body = _FakeResponses.LAST_BODY[0]
        items_with_call_output = [
            i for i in body["input"]
            if isinstance(i, dict) and i.get("type") == "function_call_output"
        ]
        assert len(items_with_call_output) == 1
        assert items_with_call_output[0]["call_id"] == "call_xyz"
        assert "3" in items_with_call_output[0]["output"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_responses_invoke_uses_responses_endpoint_and_input_array,
        test_responses_instructions_and_previous_response_id_appear_in_body,
        test_responses_usage_uses_input_output_tokens_naming,
        test_responses_works_with_react_agent_for_function_calls,
        test_responses_tool_response_uses_function_call_output_input_item,
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

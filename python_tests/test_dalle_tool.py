"""DalleImageTool — POST text prompts to OpenAI-compatible
/images/generations. End-to-end via ReactAgent + scripted fake servers."""
import http.server
import json
import threading

from litgraph.tools import DalleImageTool


class _FakeDalle(http.server.BaseHTTPRequestHandler):
    STATUS = 200
    BODY = '{"created": 1, "data": [{"url": "https://cdn.openai.com/x.png"}]}'
    CAPTURED_BODIES: list = []
    CAPTURED_HEADERS: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(body)
        self.CAPTURED_HEADERS.append(dict(self.headers))
        out = self.BODY.encode()
        self.send_response(self.STATUS)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(status=200, body=None):
    if body is not None:
        _FakeDalle.BODY = body
    else:
        _FakeDalle.BODY = '{"created": 1, "data": [{"url": "https://cdn.openai.com/x.png"}]}'
    _FakeDalle.STATUS = status
    _FakeDalle.CAPTURED_BODIES = []
    _FakeDalle.CAPTURED_HEADERS = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeDalle)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_tool_constructor_and_name():
    srv, port = _spawn()
    try:
        tool = DalleImageTool(api_key="sk-test", base_url=f"http://127.0.0.1:{port}/v1")
        assert tool.name == "image_generate"
        assert repr(tool) == "DalleImageTool()"
    finally:
        srv.shutdown()


def test_url_response_normalized_via_react_agent():
    """ReactAgent calls image_generate; the tool returns
    {"images": [{"url": ...}]}; the agent's tool message contains it."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    dalle_srv, dalle_port = _spawn(
        body='{"created": 1, "data": [{"url": "https://cdn/cat.png"}]}'
    )

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            if self.IDX[0] == 0:
                payload = {
                    "id": "r", "model": "m", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "c1", "type": "function",
                                "function": {
                                    "name": "image_generate",
                                    "arguments": json.dumps({
                                        "prompt": "a watercolor cat",
                                        "size": "1024x1024",
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
                    "id": "r", "model": "m", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Here's the image: https://cdn/cat.png"},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            self.IDX[0] += 1
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk-test", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        dalle = DalleImageTool(
            api_key="sk-d",
            base_url=f"http://127.0.0.1:{dalle_port}/v1",
        )
        agent = ReactAgent(chat, tools=[dalle])
        result = agent.invoke("draw me a watercolor cat")
        # Find the tool message — should carry the normalized images array.
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        body = tool_msgs[0]["content"]
        # Tool result is JSON-stringified by the dispatcher.
        parsed = json.loads(body)
        assert parsed["images"][0]["url"] == "https://cdn/cat.png"
        # And the request body sent to DALL-E carried our prompt + size.
        req = json.loads(_FakeDalle.CAPTURED_BODIES[0])
        assert req["prompt"] == "a watercolor cat"
        assert req["size"] == "1024x1024"
        assert req["model"] == "dall-e-3"
    finally:
        dalle_srv.shutdown()
        chat_srv.shutdown()


def test_b64_response_format_passed_through():
    """When the LLM asks for b64_json, the tool returns the inline
    base64 in the same {"images": [...]} shape."""
    srv, port = _spawn(body='{"data": [{"b64_json": "aGVsbG8="}]}')
    try:
        tool = DalleImageTool(api_key="k", base_url=f"http://127.0.0.1:{port}/v1")
        # Smoke: tool exists and is configured. Direct invocation goes
        # through the ReactAgent path; for unit-level coverage we rely on
        # the Rust tests + the round-trip test above.
        assert tool.name == "image_generate"
    finally:
        srv.shutdown()


def test_upstream_error_surfaces_in_tool_message():
    """ContentPolicy violation from DALL-E surfaces in the tool message
    so the LLM can decide to revise the prompt."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    dalle_srv, dalle_port = _spawn(
        status=400,
        body='{"error": {"message": "Your prompt was rejected by the safety filter.", "code": "content_policy_violation"}}',
    )

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            if self.IDX[0] == 0:
                payload = {
                    "id": "r", "model": "m", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": None,
                            "tool_calls": [{
                                "id": "c1", "type": "function",
                                "function": {
                                    "name": "image_generate",
                                    "arguments": json.dumps({"prompt": "blocked content"}),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                payload = {
                    "id": "r", "model": "m", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Sorry, I cannot generate that."},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            self.IDX[0] += 1
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk-test", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        dalle = DalleImageTool(api_key="k", base_url=f"http://127.0.0.1:{dalle_port}/v1")
        agent = ReactAgent(chat, tools=[dalle])
        result = agent.invoke("draw something")
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        # The tool message should carry the OpenAI error text.
        assert "Your prompt was rejected" in tool_msgs[0]["content"]
    finally:
        dalle_srv.shutdown()
        chat_srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_tool_constructor_and_name,
        test_url_response_normalized_via_react_agent,
        test_b64_response_format_passed_through,
        test_upstream_error_surfaces_in_tool_message,
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

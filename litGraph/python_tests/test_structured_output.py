"""Verify `response_format` is correctly passed through OpenAI + Gemini wire bodies +
unwrapped from Anthropic's tool-call workaround."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, GeminiChat, AnthropicChat


def test_openai_response_format_json_object():
    captured = []
    m = OpenAIChat(api_key="x", model="gpt-test", base_url="http://127.0.0.1:1",
                   on_request=lambda mdl, body: captured.append(body))
    try:
        m.invoke([{"role": "user", "content": "list 3 items"}],
                 response_format={"type": "json_object"})
    except Exception:
        pass
    assert len(captured) == 1
    assert captured[0]["response_format"] == {"type": "json_object"}


def test_openai_response_format_json_schema():
    captured = []
    schema = {
        "name": "person",
        "schema": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name", "age"],
        },
    }
    m = OpenAIChat(api_key="x", model="gpt-test", base_url="http://127.0.0.1:1",
                   on_request=lambda mdl, body: captured.append(body))
    try:
        m.invoke([{"role": "user", "content": "give me a person"}],
                 response_format={"type": "json_schema", "json_schema": schema})
    except Exception:
        pass
    assert len(captured) == 1
    rf = captured[0]["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "person"


def test_gemini_response_format_json_object_maps_to_mime_type():
    captured = []
    m = GeminiChat(api_key="x", model="gemini-test", base_url="http://127.0.0.1:1",
                   on_request=lambda mdl, body: captured.append(body))
    try:
        m.invoke([{"role": "user", "content": "list 3 items"}],
                 response_format={"type": "json_object"})
    except Exception:
        pass
    assert len(captured) == 1
    body = captured[0]
    # Gemini's generationConfig should now declare JSON output.
    assert body["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_response_format_json_schema_maps_to_response_schema():
    captured = []
    schema = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
    }
    m = GeminiChat(api_key="x", model="gemini-test", base_url="http://127.0.0.1:1",
                   on_request=lambda mdl, body: captured.append(body))
    try:
        m.invoke([{"role": "user", "content": "give me x"}],
                 response_format={"type": "json_schema",
                                  "json_schema": {"name": "obj", "schema": schema}})
    except Exception:
        pass
    assert len(captured) == 1
    body = captured[0]
    gc = body["generationConfig"]
    assert gc["responseMimeType"] == "application/json"
    assert gc["responseSchema"] == schema


class FakeAnthropicStructured(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    OUT = json.dumps({
        "content": [{
            "type": "tool_use", "id": "tu1",
            "name": "litgraph__submit_response",
            "input": {"name": "Ada", "age": 36},
        }],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }).encode()
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        FakeAnthropicStructured.LAST_BODY[0] = self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(FakeAnthropicStructured.OUT)))
        self.end_headers()
        self.wfile.write(FakeAnthropicStructured.OUT)
    def log_message(self, *a, **kw): pass


def test_anthropic_response_format_synthesizes_tool_and_unwraps_response():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeAnthropicStructured)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = AnthropicChat(api_key="x", model="claude-test",
                          base_url=f"http://127.0.0.1:{port}/v1")
        result = m.invoke(
            [{"role": "user", "content": "give me a person"}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "person",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "age": {"type": "integer"},
                        },
                        "required": ["name", "age"],
                    },
                },
            },
        )
        sent = json.loads(FakeAnthropicStructured.LAST_BODY[0])
        tools = sent["tools"]
        assert any(t["name"] == "litgraph__submit_response" for t in tools)
        assert sent["tool_choice"]["type"] == "tool"
        assert sent["tool_choice"]["name"] == "litgraph__submit_response"

        parsed = json.loads(result["text"])
        assert parsed == {"name": "Ada", "age": 36}
        # Synthesized tool was unwrapped — finish_reason must NOT be "toolcalls"
        # (else the agent loop would dispatch a fake tool).
        assert result["finish_reason"] != "toolcalls"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_openai_response_format_json_object,
        test_openai_response_format_json_schema,
        test_gemini_response_format_json_object_maps_to_mime_type,
        test_gemini_response_format_json_schema_maps_to_response_schema,
        test_anthropic_response_format_synthesizes_tool_and_unwraps_response,
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

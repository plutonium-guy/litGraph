"""StructuredChatModel — `with_structured_output(schema)` for typed JSON
responses. Direct LangChain parity.

Verifies the wire-shape (response_format injection), parse path, error path,
and Pydantic-class auto-derivation."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, with_structured_output


_LAST_BODY = [None]
_NEXT_TEXT = [""]


class _FakeLlm(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        _LAST_BODY[0] = json.loads(body)
        text = _NEXT_TEXT[0]
        payload = {
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
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


def _spawn():
    _LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


PERSON_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
    "required": ["name", "age"],
    "additionalProperties": False,
}


def test_invoke_structured_returns_parsed_dict():
    _NEXT_TEXT[0] = '{"name": "Ada", "age": 36}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, PERSON_SCHEMA, name="Person")
        out = structured.invoke_structured([{"role": "user", "content": "who?"}])
        assert out == {"name": "Ada", "age": 36}
    finally:
        srv.shutdown()


def test_invoke_injects_response_format_json_schema_into_request_body():
    _NEXT_TEXT[0] = '{"name": "x", "age": 1}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, PERSON_SCHEMA, name="Person")
        structured.invoke_structured([{"role": "user", "content": "who?"}])
        rf = _LAST_BODY[0]["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "Person"
        assert rf["json_schema"]["strict"] is True
        assert rf["json_schema"]["schema"]["type"] == "object"
        assert "name" in rf["json_schema"]["schema"]["properties"]
    finally:
        srv.shutdown()


def test_malformed_json_response_raises_with_useful_error_message():
    _NEXT_TEXT[0] = "not valid json at all"
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, PERSON_SCHEMA, name="Person")
        try:
            structured.invoke_structured([{"role": "user", "content": "who?"}])
        except RuntimeError as e:
            msg = str(e)
            assert "Person" in msg, f"schema name should appear in error: {msg}"
            assert "not valid json" in msg, f"raw response should appear: {msg}"
        else:
            raise AssertionError("expected RuntimeError on malformed JSON")
    finally:
        srv.shutdown()


def test_strict_false_disables_strict_mode():
    _NEXT_TEXT[0] = '{"name": "x", "age": 1}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, PERSON_SCHEMA, name="Person", strict=False)
        structured.invoke_structured([{"role": "user", "content": "x"}])
        rf = _LAST_BODY[0]["response_format"]
        assert rf["json_schema"]["strict"] is False
    finally:
        srv.shutdown()


def test_pydantic_model_class_auto_derives_schema():
    """Pydantic v2: pass the class directly. We call .model_json_schema() to
    extract JSON Schema. No need for the user to write the schema dict."""
    try:
        from pydantic import BaseModel
    except ImportError:
        return  # skip — pydantic optional
    class Address(BaseModel):
        street: str
        city: str
    _NEXT_TEXT[0] = '{"street": "123 Main", "city": "Springfield"}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, Address)
        out = structured.invoke_structured([{"role": "user", "content": "extract"}])
        assert out == {"street": "123 Main", "city": "Springfield"}
        # Pydantic-derived schema landed in the request.
        rf = _LAST_BODY[0]["response_format"]
        # Default name pulled from schema "title" (Pydantic sets this to class name).
        assert rf["json_schema"]["name"] == "Address"
        assert "street" in rf["json_schema"]["schema"]["properties"]
    finally:
        srv.shutdown()


def test_invoke_returns_normal_chat_response_with_validated_json_text():
    """invoke() (vs invoke_structured()) returns the standard ChatResponse
    dict — but the text is guaranteed to be valid JSON matching the schema.
    For callers that want the wrapped-model contract."""
    _NEXT_TEXT[0] = '{"name": "Ada", "age": 36}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        structured = with_structured_output(chat, PERSON_SCHEMA, name="Person")
        resp = structured.invoke([{"role": "user", "content": "who?"}])
        # Standard response shape.
        assert "text" in resp
        assert "usage" in resp
        # Re-parsing the text gives back the same dict.
        assert json.loads(resp["text"]) == {"name": "Ada", "age": 36}
    finally:
        srv.shutdown()


def test_default_schema_name_from_schema_title_or_output():
    """When no `name` arg is given, default to schema['title'] if present,
    else 'Output'."""
    _NEXT_TEXT[0] = '{"name": "x", "age": 1}'
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        # No name → no title → "Output".
        structured = with_structured_output(chat, PERSON_SCHEMA)
        structured.invoke_structured([{"role": "user", "content": "x"}])
        assert _LAST_BODY[0]["response_format"]["json_schema"]["name"] == "Output"
        # With title in schema → uses title.
        schema_with_title = {**PERSON_SCHEMA, "title": "PersonV2"}
        structured = with_structured_output(chat, schema_with_title)
        structured.invoke_structured([{"role": "user", "content": "x"}])
        assert _LAST_BODY[0]["response_format"]["json_schema"]["name"] == "PersonV2"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_invoke_structured_returns_parsed_dict,
        test_invoke_injects_response_format_json_schema_into_request_body,
        test_malformed_json_response_raises_with_useful_error_message,
        test_strict_false_disables_strict_mode,
        test_pydantic_model_class_auto_derives_schema,
        test_invoke_returns_normal_chat_response_with_validated_json_text,
        test_default_schema_name_from_schema_title_or_output,
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

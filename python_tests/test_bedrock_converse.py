"""BedrockConverseChat — AWS Bedrock Converse API, works across Llama /
Titan / Mistral / Command / Nova / Anthropic via one unified wire format.

Verified against an inline fake Bedrock HTTP server that asserts the
Converse request shape + returns canned JSON."""
import http.server
import json
import threading
from urllib.parse import urlparse

from litgraph.providers import BedrockConverseChat


_LAST_BODY = [None]
_LAST_PATH = [None]


class _FakeBedrock(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        _LAST_BODY[0] = json.loads(body)
        _LAST_PATH[0] = self.path
        payload = {
            "output": {
                "message": {
                    "role": "assistant",
                    "content": [{"text": "Hello from Llama!"}],
                }
            },
            "stopReason": "end_turn",
            "usage": {"inputTokens": 42, "outputTokens": 7, "totalTokens": 49},
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
    _LAST_PATH[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeBedrock)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_invoke_returns_parsed_text_and_usage_via_converse_wire_shape():
    srv, port = _spawn()
    try:
        chat = BedrockConverseChat(
            access_key_id="AKIAEXAMPLE",
            secret_access_key="secretEXAMPLE",
            region="us-east-1",
            model_id="meta.llama3-70b-instruct-v1:0",
            endpoint_override=f"http://127.0.0.1:{port}",
        )
        out = chat.invoke(
            [{"role": "user", "content": "hi"}],
            temperature=0.3,
            max_tokens=256,
        )
        assert out["text"] == "Hello from Llama!"
        # usage mapped from Converse's inputTokens/outputTokens/totalTokens.
        assert out["usage"]["prompt"] == 42
        assert out["usage"]["completion"] == 7
        assert out["usage"]["total"] == 49
    finally:
        srv.shutdown()


def test_request_targets_converse_endpoint_not_invoke():
    srv, port = _spawn()
    try:
        chat = BedrockConverseChat(
            access_key_id="AKIA", secret_access_key="s",
            region="us-east-1",
            model_id="amazon.nova-lite-v1:0",
            endpoint_override=f"http://127.0.0.1:{port}",
        )
        chat.invoke([{"role": "user", "content": "x"}])
        # Path must be /model/{id}/converse — NOT /invoke.
        path = urlparse(_LAST_PATH[0]).path
        assert path.endswith("/converse"), f"got: {path}"
        assert "/invoke" not in path
    finally:
        srv.shutdown()


def test_request_body_uses_converse_shape_not_anthropic_messages():
    """Converse shape: top-level `messages`, `system`, `inferenceConfig`.
    NOT `anthropic_version` / flat `max_tokens` (that's the native Anthropic
    path used by BedrockChat)."""
    srv, port = _spawn()
    try:
        chat = BedrockConverseChat(
            access_key_id="a", secret_access_key="s",
            region="us-east-1",
            model_id="mistral.mistral-large-2407-v1:0",
            endpoint_override=f"http://127.0.0.1:{port}",
        )
        chat.invoke(
            [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
            temperature=0.5,
            max_tokens=100,
        )
        body = _LAST_BODY[0]
        # Converse-specific fields present.
        assert "messages" in body
        assert "system" in body
        assert "inferenceConfig" in body
        # Anthropic-Messages-on-Bedrock fields ABSENT.
        assert "anthropic_version" not in body
        assert "max_tokens" not in body
        # System isolated to top-level array (not in messages).
        assert body["system"][0]["text"] == "be brief"
        assert len(body["messages"]) == 1
        assert body["messages"][0]["role"] == "user"
        # inferenceConfig has Converse key names.
        assert body["inferenceConfig"]["maxTokens"] == 100
        assert abs(body["inferenceConfig"]["temperature"] - 0.5) < 1e-6
    finally:
        srv.shutdown()


def test_sigv4_auth_headers_present():
    srv, port = _spawn()
    captured_headers = []

    class CapturingFake(_FakeBedrock):
        def do_POST(self):
            captured_headers.append(dict(self.headers))
            super().do_POST()

    srv2 = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CapturingFake)
    port2 = srv2.server_address[1]
    threading.Thread(target=srv2.serve_forever, daemon=True).start()
    srv.shutdown()  # drop the unused one from _spawn()
    try:
        chat = BedrockConverseChat(
            access_key_id="AKIAEXAMPLE",
            secret_access_key="secret",
            region="us-east-1",
            model_id="meta.llama3-70b-instruct-v1:0",
            endpoint_override=f"http://127.0.0.1:{port2}",
        )
        chat.invoke([{"role": "user", "content": "x"}])
        h = captured_headers[0]
        # SigV4 auth + date headers present.
        assert "authorization" in {k.lower() for k in h.keys()} or "Authorization" in h
        assert "x-amz-date" in {k.lower() for k in h.keys()} or "X-Amz-Date" in h
        auth = h.get("Authorization") or h.get("authorization")
        assert auth.startswith("AWS4-HMAC-SHA256"), f"got: {auth}"
        # Credential scope includes the region + 'bedrock'.
        assert "us-east-1/bedrock/aws4_request" in auth
    finally:
        srv2.shutdown()


if __name__ == "__main__":
    fns = [
        test_invoke_returns_parsed_text_and_usage_via_converse_wire_shape,
        test_request_targets_converse_endpoint_not_invoke,
        test_request_body_uses_converse_shape_not_anthropic_messages,
        test_sigv4_auth_headers_present,
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

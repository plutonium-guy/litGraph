"""GeminiChat.vertex — enterprise Gemini via Google Cloud Vertex AI.

Verifies that Vertex mode hits the right endpoint path + uses Bearer auth
(not `?key=`) + carries project/location in the URL. Wire format of the
request/response body is identical to AI Studio; only URL + auth differ."""
import http.server
import json
import threading
from urllib.parse import urlparse

from litgraph.providers import GeminiChat


_LAST_PATH = [None]
_LAST_AUTH = [None]
_LAST_QUERY_KEY = [None]
_LAST_BODY = [None]


class _FakeVertex(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        parsed = urlparse(self.path)
        _LAST_PATH[0] = parsed.path
        # Parse query string for `key=` (AI Studio path) — should be absent
        # under Vertex mode.
        qs = parsed.query
        key = None
        for p in qs.split("&"):
            if p.startswith("key="):
                key = p[4:]
        _LAST_QUERY_KEY[0] = key
        _LAST_AUTH[0] = self.headers.get("Authorization")
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        try:
            _LAST_BODY[0] = json.loads(body)
        except Exception:
            _LAST_BODY[0] = None
        # Vertex returns the same generateContent response shape as AI Studio.
        payload = {
            "candidates": [{
                "content": {
                    "role": "model",
                    "parts": [{"text": "Hello from Vertex!"}],
                },
                "finishReason": "STOP",
            }],
            "usageMetadata": {
                "promptTokenCount": 10,
                "candidatesTokenCount": 5,
                "totalTokenCount": 15,
            },
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_PATH[0] = None
    _LAST_AUTH[0] = None
    _LAST_QUERY_KEY[0] = None
    _LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeVertex)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_vertex_invoke_hits_project_location_scoped_path():
    """Vertex URL is /v1/projects/{P}/locations/{L}/publishers/google/models/{model}:generateContent
    — NOT /v1beta/models/{model}:generateContent that AI Studio uses."""
    srv, port = _spawn()
    try:
        chat = GeminiChat.vertex(
            project="my-project",
            location="us-central1",
            access_token="ya29.fake-token",
            model="gemini-1.5-pro",
            base_url=f"http://127.0.0.1:{port}",  # override for test
        )
        out = chat.invoke([{"role": "user", "content": "hi"}])
        assert out["text"] == "Hello from Vertex!"
    finally:
        srv.shutdown()
    path = _LAST_PATH[0]
    assert path == "/v1/projects/my-project/locations/us-central1/publishers/google/models/gemini-1.5-pro:generateContent", \
        f"wrong path: {path}"


def test_vertex_uses_bearer_auth_not_query_key():
    """AI Studio sends `?key={api_key}`; Vertex sends `Authorization: Bearer
    {access_token}` and NO query key param."""
    srv, port = _spawn()
    try:
        chat = GeminiChat.vertex(
            project="p",
            location="us-central1",
            access_token="ya29.SECRET",
            model="gemini-1.5-pro",
            base_url=f"http://127.0.0.1:{port}",
        )
        chat.invoke([{"role": "user", "content": "x"}])
    finally:
        srv.shutdown()
    assert _LAST_AUTH[0] == "Bearer ya29.SECRET"
    # No `key=` query param under Vertex.
    assert _LAST_QUERY_KEY[0] is None


def test_vertex_request_body_matches_ai_studio_shape():
    """Only URL + auth differ; body shape is identical — same
    `contents` + `system_instruction` + `generationConfig`."""
    srv, port = _spawn()
    try:
        chat = GeminiChat.vertex(
            project="p",
            location="us-central1",
            access_token="t",
            model="gemini-1.5-pro",
            base_url=f"http://127.0.0.1:{port}",
        )
        chat.invoke(
            [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "hi"},
            ],
            temperature=0.3,
            max_tokens=128,
        )
    finally:
        srv.shutdown()
    body = _LAST_BODY[0]
    assert "contents" in body
    assert "system_instruction" in body
    assert body["system_instruction"]["parts"][0]["text"] == "Be brief."
    assert body["contents"][0]["role"] == "user"


def test_ai_studio_mode_still_works_unchanged():
    """Regression check: the non-vertex path (api_key + ?key=) keeps
    working with existing code."""
    srv, port = _spawn()
    try:
        chat = GeminiChat(
            api_key="sk-ai-studio",
            model="gemini-1.5-flash",
            base_url=f"http://127.0.0.1:{port}",
        )
        chat.invoke([{"role": "user", "content": "x"}])
    finally:
        srv.shutdown()
    # AI Studio URL: /v1beta/models/{model}:generateContent
    assert _LAST_PATH[0] == "/v1beta/models/gemini-1.5-flash:generateContent"
    # No Bearer auth; `key=...` in the query string.
    assert _LAST_AUTH[0] is None
    assert _LAST_QUERY_KEY[0] == "sk-ai-studio"


def test_vertex_model_composes_with_react_agent():
    """Enterprise use case: Vertex-auth'd Gemini works as the LLM for
    ReactAgent / MultiQueryRetriever / etc — same polymorphism as
    every other chat model."""
    from litgraph.agents import ReactAgent
    srv, port = _spawn()
    try:
        chat = GeminiChat.vertex(
            project="p",
            location="us-central1",
            access_token="t",
            model="gemini-1.5-pro",
            base_url=f"http://127.0.0.1:{port}",
        )
        # No tools needed; just verifies the extract_chat_model polymorphism
        # accepts GeminiChat regardless of AI-Studio vs Vertex construction.
        agent = ReactAgent(model=chat, tools=[])
        out = agent.invoke("hello")
        assert "Hello from Vertex" in out["messages"][-1]["content"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_vertex_invoke_hits_project_location_scoped_path,
        test_vertex_uses_bearer_auth_not_query_key,
        test_vertex_request_body_matches_ai_studio_shape,
        test_ai_studio_mode_still_works_unchanged,
        test_vertex_model_composes_with_react_agent,
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

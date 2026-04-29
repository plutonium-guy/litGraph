"""Multimodal image input flows through OpenAI + Anthropic Python bindings."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, AnthropicChat


# Captures the last POSTed body for assertion.
class CapturingHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        CapturingHandler.LAST_BODY[0] = body
        # Return a minimal but valid OpenAI chat completion.
        resp = json.dumps({
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            "model": "test",
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


class CapturingAnthropicHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        CapturingAnthropicHandler.LAST_BODY[0] = body
        resp = json.dumps({
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def test_openai_image_url_serializes_to_image_url_part():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CapturingHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = OpenAIChat(api_key="x", model="gpt-test",
                       base_url=f"http://127.0.0.1:{port}/v1")
        m.invoke([{
            "role": "user",
            "content": [
                {"type": "text", "text": "What is this?"},
                {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
            ],
        }])
        sent = json.loads(CapturingHandler.LAST_BODY[0])
        msg = sent["messages"][0]
        assert msg["role"] == "user"
        # OpenAI multimodal payload: content is a list of typed parts.
        parts = msg["content"]
        assert isinstance(parts, list)
        types = [p["type"] for p in parts]
        assert "text" in types and "image_url" in types
        url_part = next(p for p in parts if p["type"] == "image_url")
        assert url_part["image_url"]["url"] == "https://example.com/cat.png"
    finally:
        srv.shutdown()


def test_openai_base64_image_serializes_to_data_url():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CapturingHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = OpenAIChat(api_key="x", model="gpt-test",
                       base_url=f"http://127.0.0.1:{port}/v1")
        m.invoke([{
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image", "image": {"media_type": "image/png", "data": "AAAA"}},
            ],
        }])
        sent = json.loads(CapturingHandler.LAST_BODY[0])
        parts = sent["messages"][0]["content"]
        # OpenAI multimodal: base64 image becomes an image_url with data: URL.
        url_part = next(p for p in parts if p["type"] == "image_url")
        assert url_part["image_url"]["url"].startswith("data:image/png;base64,")
    finally:
        srv.shutdown()


def test_anthropic_image_serializes_to_anthropic_image_block():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), CapturingAnthropicHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        m = AnthropicChat(api_key="x", model="claude-opus-4-7",
                          base_url=f"http://127.0.0.1:{port}/v1")
        m.invoke([{
            "role": "user",
            "content": [
                {"type": "text", "text": "what is this"},
                {"type": "image", "image": {"media_type": "image/jpeg", "data": "BBBB"}},
            ],
        }])
        sent = json.loads(CapturingAnthropicHandler.LAST_BODY[0])
        msg = sent["messages"][0]
        # Anthropic content blocks: text + image (source.type=base64)
        types = [b["type"] for b in msg["content"]]
        assert "text" in types and "image" in types
        img = next(b for b in msg["content"] if b["type"] == "image")
        assert img["source"]["type"] == "base64"
        assert img["source"]["media_type"] == "image/jpeg"
        assert img["source"]["data"] == "BBBB"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_openai_image_url_serializes_to_image_url_part,
        test_openai_base64_image_serializes_to_data_url,
        test_anthropic_image_serializes_to_anthropic_image_block,
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

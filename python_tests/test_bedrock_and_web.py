"""Tests for BedrockChat construction + streaming against a fake server +
WebLoader against a fake server."""
import base64
import http.server
import socket
import struct
import threading

from litgraph.providers import BedrockChat, OpenAIChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool
from litgraph.loaders import WebLoader


def test_bedrock_chat_constructs():
    m = BedrockChat(
        access_key_id="AKIDEXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        model_id="anthropic.claude-opus-4-7-v1:0",
    )
    assert "anthropic.claude-opus-4-7-v1:0" in repr(m)


def test_bedrock_chat_with_session_token():
    m = BedrockChat(
        access_key_id="AKID",
        secret_access_key="secret",
        region="us-west-2",
        model_id="anthropic.claude-3-5-sonnet-20241022-v2:0",
        session_token="STS-TOKEN",
    )
    assert m is not None


def test_bedrock_chat_accepted_by_react_agent():
    def noop(_):
        return {}
    tool = FunctionTool("noop", "no-op", {"type": "object", "properties": {}}, noop)

    m = BedrockChat(
        access_key_id="AKID",
        secret_access_key="secret",
        region="us-east-1",
        model_id="anthropic.claude-opus-4-7-v1:0",
    )
    agent = ReactAgent(m, [tool], max_iterations=3)
    assert agent is not None


class HelloHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"<html><body>hello from fake server</body></html>"
        self.send_response(200)
        self.send_header("content-type", "text/html; charset=utf-8")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def test_web_loader_fetches_url():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HelloHandler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{port}/"
        loader = WebLoader(url)
        docs = loader.load()
        assert len(docs) == 1
        assert "hello from fake server" in docs[0]["content"]
        assert docs[0]["id"] == url
        assert docs[0]["metadata"]["status_code"] == "200"
        assert "text/html" in docs[0]["metadata"]["content_type"]
    finally:
        srv.shutdown()


def _make_chunk_frame(inner_json: bytes) -> bytes:
    """Build one AWS event-stream frame wrapping a base64'd JSON event."""
    b64 = base64.b64encode(inner_json).decode()
    payload = f'{{"bytes":"{b64}"}}'.encode()
    header_name = b":event-type"
    header_value = b"chunk"
    headers = bytearray()
    headers.append(len(header_name))
    headers.extend(header_name)
    headers.append(7)  # string type
    headers.extend(struct.pack(">H", len(header_value)))
    headers.extend(header_value)
    total_len = 12 + len(headers) + len(payload) + 4
    frame = bytearray()
    frame.extend(struct.pack(">I", total_len))
    frame.extend(struct.pack(">I", len(headers)))
    frame.extend(b"\x00\x00\x00\x00")  # dummy prelude CRC
    frame.extend(headers)
    frame.extend(payload)
    frame.extend(b"\x00\x00\x00\x00")  # dummy message CRC
    assert len(frame) == total_len
    return bytes(frame)


def _start_fake_bedrock_stream(body: bytes) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]

    def serve():
        try:
            conn, _ = s.accept()
            buf = b""
            while b"\r\n\r\n" not in buf:
                d = conn.recv(8192)
                if not d: break
                buf += d
            # Drain content-length bytes if any
            cl = 0
            for line in buf.split(b"\r\n"):
                if line.lower().startswith(b"content-length:"):
                    cl = int(line.split(b":")[1].strip())
            need = buf.index(b"\r\n\r\n") + 4 + cl
            while len(buf) < need:
                d = conn.recv(8192)
                if not d: break
                buf += d
            header = (
                f"HTTP/1.1 200 OK\r\n"
                f"Content-Type: application/vnd.amazon.eventstream\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"Connection: close\r\n\r\n"
            ).encode()
            conn.sendall(header + body)
            conn.close()
        finally:
            s.close()

    threading.Thread(target=serve, daemon=True).start()
    return port


def test_bedrock_stream_against_fake_server():
    frames = b"".join([
        _make_chunk_frame(b'{"type":"message_start","message":{"usage":{"input_tokens":4}}}'),
        _make_chunk_frame(b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"py"}}'),
        _make_chunk_frame(b'{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"thon"}}'),
        _make_chunk_frame(b'{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}'),
    ])
    port = _start_fake_bedrock_stream(frames)
    m = BedrockChat(
        access_key_id="AKID",
        secret_access_key="secret",
        region="us-east-1",
        model_id="anthropic.claude-opus-4-7-v1:0",
        endpoint_override=f"http://127.0.0.1:{port}",
    )
    deltas = []
    done = None
    for ev in m.stream([{"role": "user", "content": "say python"}]):
        if ev["type"] == "delta":
            deltas.append(ev["text"])
        elif ev["type"] == "done":
            done = ev
    assert deltas == ["py", "thon"]
    assert done is not None
    assert done["text"] == "python"
    assert done["usage"]["prompt"] == 4
    assert done["usage"]["completion"] == 2


if __name__ == "__main__":
    fns = [
        test_bedrock_chat_constructs,
        test_bedrock_chat_with_session_token,
        test_bedrock_chat_accepted_by_react_agent,
        test_web_loader_fetches_url,
        test_bedrock_stream_against_fake_server,
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

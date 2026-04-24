"""TtsAudioTool — POST text to /audio/speech, write binary audio to disk.
End-to-end via ReactAgent + scripted fake servers."""
import http.server
import json
import os
import tempfile
import threading

from litgraph.tools import TtsAudioTool


class _FakeTts(http.server.BaseHTTPRequestHandler):
    STATUS = 200
    BODY = b"\xAA\xBB\xCC\xDD\xEE"  # canned "audio" bytes
    CONTENT_TYPE = "audio/mpeg"
    CAPTURED_BODIES: list = []
    CAPTURED_HEADERS: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(body)
        self.CAPTURED_HEADERS.append(dict(self.headers))
        self.send_response(self.STATUS)
        self.send_header("content-type", self.CONTENT_TYPE)
        self.send_header("content-length", str(len(self.BODY)))
        self.end_headers()
        self.wfile.write(self.BODY)

    def log_message(self, *a, **kw): pass


def _spawn(status=200, body=b"\xAA\xBB\xCC\xDD\xEE", content_type="audio/mpeg"):
    _FakeTts.STATUS = status
    _FakeTts.BODY = body
    _FakeTts.CONTENT_TYPE = content_type
    _FakeTts.CAPTURED_BODIES = []
    _FakeTts.CAPTURED_HEADERS = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeTts)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_tool_constructor_and_name():
    srv, port = _spawn()
    try:
        tool = TtsAudioTool(api_key="sk-test", base_url=f"http://127.0.0.1:{port}/v1")
        assert tool.name == "tts_speak"
        assert repr(tool) == "TtsAudioTool()"
    finally:
        srv.shutdown()


def test_writes_audio_to_disk_via_react_agent():
    """Full flow: ReactAgent → tool_call(tts_speak) → POST /audio/speech →
    binary written to disk → tool returns file metadata."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    audio_bytes = b"\x01\x02\x03\x04\x05" * 10  # 50 fake "audio" bytes
    tts_srv, tts_port = _spawn(body=audio_bytes)

    fd, audio_out = tempfile.mkstemp(suffix=".mp3", prefix="lg_tts_e2e_")
    os.close(fd)
    os.unlink(audio_out)  # we want the tool to create it

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
                                    "name": "tts_speak",
                                    "arguments": json.dumps({
                                        "text": "Hello, this is a test.",
                                        "voice": "alloy",
                                        "output_path": audio_out,
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
                        "message": {"role": "assistant", "content": f"Saved audio to {audio_out}"},
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
        tts = TtsAudioTool(api_key="k", base_url=f"http://127.0.0.1:{tts_port}/v1")
        agent = ReactAgent(chat, tools=[tts])
        result = agent.invoke(f"Say hello and save to {audio_out}")

        # File on disk matches what the fake TTS returned.
        assert os.path.exists(audio_out)
        assert open(audio_out, "rb").read() == audio_bytes

        # Tool message carries metadata.
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed["audio_path"] == audio_out
        assert parsed["format"] == "mp3"
        assert parsed["size_bytes"] == len(audio_bytes)

        # Request body sent to TTS carried our text + voice + model.
        req = json.loads(_FakeTts.CAPTURED_BODIES[0])
        assert req["input"] == "Hello, this is a test."
        assert req["voice"] == "alloy"
        assert req["model"] == "tts-1"
        assert req["response_format"] == "mp3"
    finally:
        tts_srv.shutdown()
        chat_srv.shutdown()
        if os.path.exists(audio_out):
            os.unlink(audio_out)


def test_max_text_len_enforced_pre_request():
    """Long input refused before HTTP — saves cost + avoids 4xx."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    # Build a TTS tool with tiny cap.
    class _NoCallChat(http.server.BaseHTTPRequestHandler):
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
                                    "name": "tts_speak",
                                    "arguments": json.dumps({
                                        "text": "x" * 100,  # over cap
                                        "voice": "alloy",
                                        "output_path": "/tmp/should_not_be_written.mp3",
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
                        "message": {"role": "assistant", "content": "Too long."},
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

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _NoCallChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk-test", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        tts = TtsAudioTool(
            api_key="k",
            base_url="http://127.0.0.1:1",   # would refuse if hit
            max_text_len=10,
        )
        agent = ReactAgent(chat, tools=[tts])
        result = agent.invoke("test")
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) >= 1
        # Tool error surfaces as observation containing "> cap 10".
        assert "> cap 10" in tool_msgs[0]["content"]
        # File never written.
        assert not os.path.exists("/tmp/should_not_be_written.mp3")
    finally:
        chat_srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_tool_constructor_and_name,
        test_writes_audio_to_disk_via_react_agent,
        test_max_text_len_enforced_pre_request,
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

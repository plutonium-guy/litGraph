"""GmailSendTool — send email via Gmail REST API. Bearer-token auth.
Pairs with GmailLoader (read-only) to close the read/write loop."""
import base64
import http.server
import json
import threading

from litgraph.tools import GmailSendTool


class _FakeGmail(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []
    STATUS = 200
    REPLY = '{"id": "msg-abc", "threadId": "thr-xyz"}'

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED.append({
            "path": self.path,
            "auth": self.headers.get("authorization", ""),
            "body": json.loads(body),
        })
        out = self.REPLY.encode()
        self.send_response(self.STATUS)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(status=200, reply='{"id": "msg-abc", "threadId": "thr-xyz"}'):
    _FakeGmail.CAPTURED = []
    _FakeGmail.STATUS = status
    _FakeGmail.REPLY = reply
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeGmail)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _decode_raw(req_body):
    """Decode the base64url `raw` field into the RFC 2822 message text."""
    raw = req_body["raw"]
    # urlsafe_b64decode requires padding.
    padded = raw + "=" * (-len(raw) % 4)
    return base64.urlsafe_b64decode(padded).decode("utf-8")


def test_tool_construction_and_name():
    tool = GmailSendTool(access_token="ya29.test")
    assert tool.name == "gmail_send"
    assert "GmailSendTool" in repr(tool)


def test_send_via_react_agent_end_to_end():
    """Agent emits a tool_call → Tool POSTs to fake Gmail → second LLM
    turn returns confirmation."""
    from litgraph.providers import OpenAIChat
    from litgraph.agents import ReactAgent

    g_srv, g_port = _spawn()

    turn = [0]

    class _FakeOpenAI(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            _ = self.rfile.read(n)
            turn[0] += 1
            if turn[0] == 1:
                payload = {
                    "id": "r1", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "gmail_send",
                                    "arguments": json.dumps({
                                        "to": "alice@example.com",
                                        "subject": "Status update",
                                        "body": "shipped",
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
                    "id": "r2", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant",
                                    "content": "Sent the email."},
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

    o_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=o_srv.serve_forever, daemon=True).start()
    o_port = o_srv.server_address[1]

    try:
        chat = OpenAIChat(api_key="sk-x", model="gpt-test",
                          base_url=f"http://127.0.0.1:{o_port}/v1")
        send_tool = GmailSendTool(
            access_token="ya29.fake",
            base_url=f"http://127.0.0.1:{g_port}",
            from_addr="agent@my-domain.com",
        )
        agent = ReactAgent(model=chat, tools=[send_tool])
        result = agent.invoke("email alice")
        final = result["messages"][-1]["content"]
        assert "Sent the email" in final

        # Tool was actually invoked: fake Gmail captured one POST.
        assert len(_FakeGmail.CAPTURED) == 1
        captured = _FakeGmail.CAPTURED[0]
        assert captured["auth"] == "Bearer ya29.fake"
        assert captured["path"].endswith("/messages/send")

        # Decode the RFC 2822 body and verify headers.
        raw = _decode_raw(captured["body"])
        assert "From: agent@my-domain.com\r\n" in raw
        assert "To: alice@example.com\r\n" in raw
        assert "Subject: Status update\r\n" in raw
        assert raw.endswith("\r\nshipped")
    finally:
        g_srv.shutdown()
        o_srv.shutdown()


def test_construction_with_custom_timeout():
    tool = GmailSendTool(access_token="t", timeout_s=60)
    assert tool.name == "gmail_send"


def test_construction_with_base_url_override():
    tool = GmailSendTool(access_token="t", base_url="https://example.test/gmail")
    assert tool.name == "gmail_send"


if __name__ == "__main__":
    import traceback
    fns = [
        test_tool_construction_and_name,
        test_send_via_react_agent_end_to_end,
        test_construction_with_custom_timeout,
        test_construction_with_base_url_override,
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

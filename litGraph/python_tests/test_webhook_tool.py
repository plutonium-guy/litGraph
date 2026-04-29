"""WebhookTool — POST agent messages to Slack/Discord/generic webhooks.
URL hard-coded at construction; agent only controls message text."""
import http.server
import json
import threading

from litgraph.tools import WebhookTool


class _FakeWebhook(http.server.BaseHTTPRequestHandler):
    STATUS = 200
    BODY = "ok"
    CAPTURED_BODIES: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.CAPTURED_BODIES.append(json.loads(self.rfile.read(n)))
        out = self.BODY.encode()
        self.send_response(self.STATUS)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(status=200, body="ok"):
    _FakeWebhook.STATUS = status
    _FakeWebhook.BODY = body
    _FakeWebhook.CAPTURED_BODIES = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeWebhook)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_slack_preset_default_name_is_slack_notify():
    tool = WebhookTool(url="http://x", preset="slack")
    assert tool.name == "slack_notify"


def test_custom_name_and_description_exposed():
    tool = WebhookTool(
        url="http://x",
        preset="slack",
        name="notify_oncall",
        description="P1 only.",
    )
    assert tool.name == "notify_oncall"
    assert "WebhookTool(name='notify_oncall')" in repr(tool)


def test_invalid_preset_raises_value_error():
    try:
        WebhookTool(url="http://x", preset="teams")
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "preset" in str(e).lower()


def test_slack_payload_sent_via_react_agent():
    """End-to-end: agent emits tool_call(slack_notify, message=...);
    our fake webhook receives a Slack-shape {text, username} payload."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    hook_srv, hook_port = _spawn()

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call("c1", "slack_notify", {
                    "message": "deploy done",
                    "username": "deploy-bot",
                })
            else:
                payload = _final("Notification sent.")
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
        chat = OpenAIChat(api_key="sk", model="gpt",
                          base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
        hook = WebhookTool(url=f"http://127.0.0.1:{hook_port}", preset="slack")
        agent = ReactAgent(chat, tools=[hook])
        agent.invoke("notify the team")

        assert len(_FakeWebhook.CAPTURED_BODIES) == 1
        sent = _FakeWebhook.CAPTURED_BODIES[0]
        assert sent["text"] == "deploy done"
        assert sent["username"] == "deploy-bot"
    finally:
        chat_srv.shutdown()
        hook_srv.shutdown()


def test_discord_preset_shape_differs_from_slack():
    """Discord uses `content`, not `text`."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    hook_srv, hook_port = _spawn(status=204, body="")

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call("c1", "discord_notify", {"message": "gg"})
            else:
                payload = _final("Done.")
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
        chat = OpenAIChat(api_key="sk", model="gpt",
                          base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
        hook = WebhookTool(url=f"http://127.0.0.1:{hook_port}", preset="discord")
        agent = ReactAgent(chat, tools=[hook])
        agent.invoke("ping")
        sent = _FakeWebhook.CAPTURED_BODIES[0]
        assert sent["content"] == "gg"
        assert "text" not in sent
    finally:
        chat_srv.shutdown()
        hook_srv.shutdown()


def test_generic_preset_forwards_arbitrary_json_body():
    """Generic preset: agent's `message` is the full body (JSON string)."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    hook_srv, hook_port = _spawn()

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call("c1", "webhook_post", {
                    "message": '{"event": "page", "severity": "high"}',
                })
            else:
                payload = _final("Paged.")
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
        chat = OpenAIChat(api_key="sk", model="gpt",
                          base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
        hook = WebhookTool(url=f"http://127.0.0.1:{hook_port}", preset="generic")
        agent = ReactAgent(chat, tools=[hook])
        agent.invoke("page someone")
        sent = _FakeWebhook.CAPTURED_BODIES[0]
        assert sent["event"] == "page"
        assert sent["severity"] == "high"
    finally:
        chat_srv.shutdown()
        hook_srv.shutdown()


def _tool_call(call_id, name, args):
    return {
        "id": "r", "model": "m", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _final(content):
    return {
        "id": "r", "model": "m", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    import traceback
    fns = [
        test_slack_preset_default_name_is_slack_notify,
        test_custom_name_and_description_exposed,
        test_invalid_preset_raises_value_error,
        test_slack_payload_sent_via_react_agent,
        test_discord_preset_shape_differs_from_slack,
        test_generic_preset_forwards_arbitrary_json_body,
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

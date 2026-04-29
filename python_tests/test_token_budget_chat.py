"""TokenBudgetChat — per-invocation token cap wrapper. Strict mode
errors on overflow; auto_trim mode drops oldest non-system messages."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, TokenBudgetChat


class _EchoChat(http.server.BaseHTTPRequestHandler):
    """Returns a canned reply + captures the request body for assertions."""
    CAPTURED_BODIES: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(json.loads(body))
        payload = {
            "id": "r", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
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
    _EchoChat.CAPTURED_BODIES = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _inner(port, model="gpt-4o-mini"):
    return OpenAIChat(
        api_key="sk-test", model=model,
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def _long_history(n):
    out = [{"role": "system", "content": "You are helpful."}]
    for i in range(n):
        out.append({"role": "user", "content": f"msg {i} filler text filler text"})
        out.append({"role": "assistant", "content": f"reply {i} more filler reply"})
    return out


def test_under_budget_passes_through():
    srv, port = _spawn()
    try:
        inner = _inner(port)
        chat = TokenBudgetChat(inner, max_tokens=10_000)
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert resp["text"] == "ok"
    body = _EchoChat.CAPTURED_BODIES[0]
    assert len(body["messages"]) == 1


def test_strict_mode_errors_on_overflow():
    srv, port = _spawn()
    try:
        inner = _inner(port)
        chat = TokenBudgetChat(inner, max_tokens=50)  # strict (default)
        try:
            chat.invoke(_long_history(30))
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "budget" in str(e).lower()
    finally:
        srv.shutdown()
    # Server never hit.
    assert len(_EchoChat.CAPTURED_BODIES) == 0


def test_auto_trim_drops_oldest_and_keeps_system():
    srv, port = _spawn()
    try:
        inner = _inner(port)
        chat = TokenBudgetChat(inner, max_tokens=50, auto_trim=True)
        msgs = _long_history(30)  # 1 system + 60 user/assistant = 61 total
        chat.invoke(msgs)
    finally:
        srv.shutdown()
    body = _EchoChat.CAPTURED_BODIES[0]
    sent = body["messages"]
    # Trimmed — fewer messages reached the provider than we passed in.
    assert len(sent) < len(msgs)
    # System message survived the trim.
    assert any(m["role"] == "system" for m in sent)
    # Last message survived (it was the most recent user/assistant pair).
    assert sent[-1]["content"] == msgs[-1]["content"]


def test_auto_trim_preserves_system_under_tight_cap():
    srv, port = _spawn()
    try:
        inner = _inner(port)
        chat = TokenBudgetChat(inner, max_tokens=20, auto_trim=True)
        chat.invoke(_long_history(30))
    finally:
        srv.shutdown()
    body = _EchoChat.CAPTURED_BODIES[0]
    sent = body["messages"]
    system_count = sum(1 for m in sent if m["role"] == "system")
    assert system_count == 1


def test_strict_mode_under_cap_succeeds_no_error():
    srv, port = _spawn()
    try:
        inner = _inner(port)
        chat = TokenBudgetChat(inner, max_tokens=10_000)
        chat.invoke([{"role": "user", "content": "short"}])
    finally:
        srv.shutdown()
    assert len(_EchoChat.CAPTURED_BODIES) == 1


def test_repr_contains_inner_model_name():
    srv, port = _spawn()
    try:
        inner = _inner(port, model="gpt-4o-mini")
        chat = TokenBudgetChat(inner, max_tokens=100)
        r = repr(chat)
    finally:
        srv.shutdown()
    assert "TokenBudgetChat" in r
    assert "gpt-4o-mini" in r


def test_composes_with_react_agent():
    """TokenBudgetChat must pass the extract_chat_model polymorphism
    so agents accept it directly."""
    from litgraph.agents import ReactAgent
    srv, port = _spawn()
    try:
        inner = _inner(port)
        budgeted = TokenBudgetChat(inner, max_tokens=10_000, auto_trim=True)
        # Just build the agent — no tool calls needed for this test.
        agent = ReactAgent(budgeted, tools=[])
        _ = agent
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_under_budget_passes_through,
        test_strict_mode_errors_on_overflow,
        test_auto_trim_drops_oldest_and_keeps_system,
        test_auto_trim_preserves_system_under_tight_cap,
        test_strict_mode_under_cap_succeeds_no_error,
        test_repr_contains_inner_model_name,
        test_composes_with_react_agent,
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

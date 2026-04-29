"""PiiScrubbingChat — redact PII from user messages before send to
upstream provider. Assistant/Tool history messages pass through
untouched so agent traces stay intact."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat, PiiScrubbingChat
from litgraph.evaluators import PiiScrubber


class _EchoChat(http.server.BaseHTTPRequestHandler):
    """Canned reply + captures outgoing request body."""
    CAPTURED_BODIES: list = []
    REPLY_TEXT = "ok"

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(json.loads(body))
        payload = {
            "id": "r", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": self.REPLY_TEXT},
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


def _spawn(reply="ok"):
    _EchoChat.CAPTURED_BODIES = []
    _EchoChat.REPLY_TEXT = reply
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _inner(port):
    return OpenAIChat(
        api_key="sk-test", model="gpt-4o-mini",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def _sent_content(body, role):
    for m in body["messages"]:
        if m["role"] == role:
            return m["content"]
    return None


def test_email_redacted_from_user_message():
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        chat.invoke([{"role": "user", "content": "Email me at alice@example.com please"}])
    finally:
        srv.shutdown()
    c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "user")
    assert "alice@example.com" not in c
    assert "<EMAIL>" in c


def test_ssn_and_credit_card_redacted():
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        chat.invoke([{"role": "user", "content": "SSN 123-45-6789 card 4532015112830366"}])
    finally:
        srv.shutdown()
    c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "user")
    assert "123-45-6789" not in c
    assert "4532015112830366" not in c
    assert "<SSN>" in c
    assert "<CREDIT_CARD>" in c


def test_system_message_passes_through_by_default():
    """Operator prompts are trusted — not scrubbed unless scrub_system=True."""
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        chat.invoke([
            {"role": "system", "content": "Contact admin@corp.com for support."},
            {"role": "user", "content": "hi"},
        ])
    finally:
        srv.shutdown()
    body = _EchoChat.CAPTURED_BODIES[0]
    sys_c = _sent_content(body, "system")
    assert "admin@corp.com" in sys_c
    assert "<EMAIL>" not in sys_c


def test_scrub_system_opt_in():
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port), scrub_system=True)
        chat.invoke([
            {"role": "system", "content": "Contact admin@corp.com for support."},
            {"role": "user", "content": "hi"},
        ])
    finally:
        srv.shutdown()
    sys_c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "system")
    assert "admin@corp.com" not in sys_c
    assert "<EMAIL>" in sys_c


def test_assistant_history_never_scrubbed():
    """Prior assistant turns preserve the model's exact output so
    re-sending the conversation doesn't mangle the trace."""
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        chat.invoke([
            {"role": "user", "content": "who should I email?"},
            {"role": "assistant", "content": "Email alice@example.com."},
            {"role": "user", "content": "thanks"},
        ])
    finally:
        srv.shutdown()
    asst = _sent_content(_EchoChat.CAPTURED_BODIES[0], "assistant")
    assert "alice@example.com" in asst


def test_output_pass_through_by_default():
    """Response formatting preserved unless scrub_outputs=True."""
    srv, port = _spawn(reply="Reach me at bob@corp.com")
    try:
        chat = PiiScrubbingChat(_inner(port))
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert resp["text"] == "Reach me at bob@corp.com"


def test_output_scrubbing_opt_in():
    srv, port = _spawn(reply="Reach me at bob@corp.com")
    try:
        chat = PiiScrubbingChat(_inner(port), scrub_outputs=True)
        resp = chat.invoke([{"role": "user", "content": "hi"}])
    finally:
        srv.shutdown()
    assert "bob@corp.com" not in resp["text"]
    assert "<EMAIL>" in resp["text"]


def test_custom_scrubber_with_extra_patterns():
    """Pass a caller-configured PiiScrubber — e.g. with custom labels."""
    srv, port = _spawn()
    try:
        scrubber = PiiScrubber(extra_patterns=[("EMPLOYEE_ID", r"EMP-\d{6}")])
        chat = PiiScrubbingChat(_inner(port), scrubber=scrubber)
        chat.invoke([{"role": "user", "content": "ticket for EMP-123456"}])
    finally:
        srv.shutdown()
    c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "user")
    assert "EMP-123456" not in c
    assert "<EMPLOYEE_ID>" in c


def test_no_pii_no_change():
    """Clean text stays verbatim."""
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        chat.invoke([{"role": "user", "content": "hello world"}])
    finally:
        srv.shutdown()
    c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "user")
    assert c == "hello world"


def test_repr_contains_inner():
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        r = repr(chat)
    finally:
        srv.shutdown()
    assert "PiiScrubbingChat" in r


def test_composes_with_react_agent():
    """Polymorphism — ReactAgent must accept a PiiScrubbingChat directly."""
    from litgraph.agents import ReactAgent
    srv, port = _spawn()
    try:
        chat = PiiScrubbingChat(_inner(port))
        agent = ReactAgent(chat, tools=[])
        _ = agent
    finally:
        srv.shutdown()


def test_stacks_with_token_budget():
    """PiiScrubbingChat(TokenBudgetChat(inner)) — both wrappers apply."""
    from litgraph.providers import TokenBudgetChat
    srv, port = _spawn()
    try:
        budgeted = TokenBudgetChat(_inner(port), max_tokens=10_000)
        scrubbed = PiiScrubbingChat(budgeted)
        scrubbed.invoke([{"role": "user", "content": "email foo@bar.com"}])
    finally:
        srv.shutdown()
    c = _sent_content(_EchoChat.CAPTURED_BODIES[0], "user")
    assert "foo@bar.com" not in c
    assert "<EMAIL>" in c


if __name__ == "__main__":
    import traceback
    fns = [
        test_email_redacted_from_user_message,
        test_ssn_and_credit_card_redacted,
        test_system_message_passes_through_by_default,
        test_scrub_system_opt_in,
        test_assistant_history_never_scrubbed,
        test_output_pass_through_by_default,
        test_output_scrubbing_opt_in,
        test_custom_scrubber_with_extra_patterns,
        test_no_pii_no_change,
        test_repr_contains_inner,
        test_composes_with_react_agent,
        test_stacks_with_token_budget,
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

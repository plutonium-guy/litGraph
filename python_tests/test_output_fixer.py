"""OutputFixingParser — LLM-based parse-fix wrapper. Sends raw + error +
format hint to a fixer ChatModel; returns corrected text or parsed
value. End-to-end via OpenAIChat against a scripted fake server."""
import http.server
import json
import threading

from litgraph.parsers import fix_with_llm, parse_json_with_retry
from litgraph.providers import OpenAIChat


class _ScriptedChat(http.server.BaseHTTPRequestHandler):
    """Returns canned chat completion responses in order."""
    RESPONSES: list = []
    IDX = [0]
    SEEN_BODIES: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.SEEN_BODIES.append(json.loads(body))
        content = self.RESPONSES[self.IDX[0]]
        self.IDX[0] += 1
        payload = {
            "id": "r", "model": "scripted", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
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


def _spawn(responses):
    _ScriptedChat.RESPONSES = list(responses)
    _ScriptedChat.IDX[0] = 0
    _ScriptedChat.SEEN_BODIES = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chat(port):
    return OpenAIChat(
        api_key="sk-test", model="gpt",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def test_fix_with_llm_returns_correction():
    srv, port = _spawn(['{"x": 1}'])
    try:
        out = fix_with_llm(
            raw='{"x": 1,}',
            error="trailing comma",
            instructions="valid JSON object",
            model=_chat(port),
        )
    finally:
        srv.shutdown()
    assert out == '{"x": 1}'


def test_fix_with_llm_user_prompt_carries_error_raw_instructions():
    srv, port = _spawn(["fixed"])
    try:
        fix_with_llm(
            raw="garbage",
            error="missing brace",
            instructions="valid JSON",
            model=_chat(port),
        )
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_BODIES[0]
    user_msgs = [m for m in body["messages"] if m["role"] == "user"]
    user_text = user_msgs[0]["content"]
    assert "garbage" in user_text
    assert "missing brace" in user_text
    assert "valid JSON" in user_text


def test_parse_json_with_retry_succeeds_on_first_try_no_llm_call():
    srv, port = _spawn([])  # empty — no fixes available
    try:
        out = parse_json_with_retry(
            raw='{"answer": 42}',
            model=_chat(port),
        )
    finally:
        srv.shutdown()
    assert out == {"answer": 42}
    assert len(_ScriptedChat.SEEN_BODIES) == 0


def test_parse_json_with_retry_recovers_after_one_repair():
    srv, port = _spawn(['{"answer": 42}'])
    try:
        out = parse_json_with_retry(
            raw='{"answer": 42,}',  # trailing-comma malformed
            model=_chat(port),
            max_retries=1,
        )
    finally:
        srv.shutdown()
    assert out == {"answer": 42}
    assert len(_ScriptedChat.SEEN_BODIES) == 1


def test_parse_json_with_retry_exhausts_retries_raises():
    """Fixer keeps returning garbage; final ValueError surfaces."""
    srv, port = _spawn(["still bad", "still bad", "still bad"])
    try:
        try:
            parse_json_with_retry(
                raw="initial garbage",
                model=_chat(port),
                max_retries=2,
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "expected" in str(e).lower() or "json" in str(e).lower() or len(str(e)) > 0
    finally:
        srv.shutdown()
    # 2 retries → 2 fixer calls.
    assert len(_ScriptedChat.SEEN_BODIES) == 2


def test_parse_json_with_retry_max_retries_zero_no_repair():
    srv, port = _spawn([])
    try:
        try:
            parse_json_with_retry(
                raw="malformed",
                model=_chat(port),
                max_retries=0,
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass
    finally:
        srv.shutdown()
    # No fixer calls.
    assert len(_ScriptedChat.SEEN_BODIES) == 0


def test_parse_json_with_retry_schema_hint_passed_to_fixer():
    """When schema_hint is provided, the fixer prompt should include it."""
    srv, port = _spawn(['{"x": 1}'])
    try:
        parse_json_with_retry(
            raw="garbage",
            model=_chat(port),
            schema_hint='{"x": int}',
            max_retries=1,
        )
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_BODIES[0]
    user_text = next(
        m["content"] for m in body["messages"] if m["role"] == "user"
    )
    assert '{"x": int}' in user_text


def test_parse_json_with_retry_returns_arrays_and_scalars():
    srv, port = _spawn([])
    try:
        # Array.
        arr = parse_json_with_retry(raw="[1, 2, 3]", model=_chat(port))
        assert arr == [1, 2, 3]
        # Scalar string.
        s = parse_json_with_retry(raw='"hello"', model=_chat(port))
        assert s == "hello"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_fix_with_llm_returns_correction,
        test_fix_with_llm_user_prompt_carries_error_raw_instructions,
        test_parse_json_with_retry_succeeds_on_first_try_no_llm_call,
        test_parse_json_with_retry_recovers_after_one_repair,
        test_parse_json_with_retry_exhausts_retries_raises,
        test_parse_json_with_retry_max_retries_zero_no_repair,
        test_parse_json_with_retry_schema_hint_passed_to_fixer,
        test_parse_json_with_retry_returns_arrays_and_scalars,
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

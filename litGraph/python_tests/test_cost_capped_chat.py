"""CostCappedChat — hard USD cap on cumulative spend. Pre-check refuses
further calls once total >= max_usd; the failing call never reaches the
provider."""
import http.server
import json
import threading

from litgraph.providers import CostCappedChat, OpenAIChat


class _EchoChat(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []
    # Default: tiny usage so under cap easily.
    USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    MODEL = "gpt-4o"

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED.append(json.loads(body))
        payload = {
            "id": "r", "model": _EchoChat.MODEL, "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": _EchoChat.USAGE,
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn():
    _EchoChat.CAPTURED = []
    _EchoChat.USAGE = {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    _EchoChat.MODEL = "gpt-4o"
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _EchoChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _inner(port, model="gpt-4o"):
    return OpenAIChat(api_key="k", model=model,
                      base_url=f"http://127.0.0.1:{port}/v1")


def test_passes_through_under_cap():
    srv, port = _spawn()
    try:
        chat = CostCappedChat(_inner(port), max_usd=1.00)
        r = chat.invoke([{"role": "user", "content": "hi"}])
        assert r["text"] == "ok"
        # 100 × $2.50/Mtok + 50 × $10/Mtok = $0.00025 + $0.0005 = $0.00075
        assert abs(chat.total_usd() - 0.00075) < 1e-9
        assert abs(chat.remaining_usd() - 0.99925) < 1e-9
    finally:
        srv.shutdown()


def test_rejects_once_over_cap():
    srv, port = _spawn()
    try:
        # Huge per-call usage: 1M prompt + 1M completion = $2.50 + $10.00 = $12.50.
        _EchoChat.USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                           "total_tokens": 2_000_000}
        chat = CostCappedChat(_inner(port), max_usd=1.00)
        # First call succeeds (pre-check: $0 < $1), but pushes total to $12.50.
        chat.invoke([{"role": "user", "content": "a"}])
        assert chat.total_usd() > 1.00

        # Second call: pre-check $12.50 >= $1.00 → rejected, never hits server.
        before = len(_EchoChat.CAPTURED)
        try:
            chat.invoke([{"role": "user", "content": "b"}])
            raise AssertionError("expected ValueError")
        except ValueError as e:
            assert "cost cap" in str(e).lower()
        assert len(_EchoChat.CAPTURED) == before, "rejected call must not hit provider"
    finally:
        srv.shutdown()


def test_unpriced_model_charges_zero():
    srv, port = _spawn()
    try:
        _EchoChat.MODEL = "internal-custom-model-v9"
        _EchoChat.USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                           "total_tokens": 2_000_000}
        # Tight cap: $0.01. Unpriced model adds $0 per call → always under cap.
        chat = CostCappedChat(_inner(port, model="internal-custom-model-v9"), max_usd=0.01)
        chat.invoke([{"role": "user", "content": "a"}])
        chat.invoke([{"role": "user", "content": "b"}])
        chat.invoke([{"role": "user", "content": "c"}])
        assert chat.total_usd() == 0.0
    finally:
        srv.shutdown()


def test_custom_prices_override_defaults():
    srv, port = _spawn()
    try:
        _EchoChat.MODEL = "my-custom-model"
        _EchoChat.USAGE = {"prompt_tokens": 100_000, "completion_tokens": 50_000,
                           "total_tokens": 150_000}
        chat = CostCappedChat(
            _inner(port, model="my-custom-model"),
            max_usd=10.00,
            prices={"my-custom-model": (5.00, 20.00)},
        )
        chat.invoke([{"role": "user", "content": "a"}])
        # 100_000 × $5/Mtok + 50_000 × $20/Mtok = $0.50 + $1.00 = $1.50
        assert abs(chat.total_usd() - 1.50) < 1e-9
    finally:
        srv.shutdown()


def test_reset_clears_counter():
    srv, port = _spawn()
    try:
        chat = CostCappedChat(_inner(port), max_usd=1.00)
        chat.invoke([{"role": "user", "content": "a"}])
        assert chat.total_usd() > 0
        chat.reset()
        assert chat.total_usd() == 0.0
        assert abs(chat.remaining_usd() - 1.00) < 1e-9
    finally:
        srv.shutdown()


def test_reset_reenables_calls_after_cap_hit():
    srv, port = _spawn()
    try:
        _EchoChat.USAGE = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000,
                           "total_tokens": 2_000_000}
        chat = CostCappedChat(_inner(port), max_usd=1.00)
        chat.invoke([{"role": "user", "content": "a"}])  # blows through cap
        # Second call rejected.
        try:
            chat.invoke([{"role": "user", "content": "b"}])
            raise AssertionError("should reject")
        except ValueError:
            pass
        # Reset — now calls flow again.
        chat.reset()
        chat.invoke([{"role": "user", "content": "c"}])
    finally:
        srv.shutdown()


def test_repr_shows_spent_and_remaining():
    srv, port = _spawn()
    try:
        chat = CostCappedChat(_inner(port), max_usd=1.00)
        r = repr(chat)
        assert "CostCappedChat" in r
        assert "spent=$" in r
        assert "remaining=$" in r
    finally:
        srv.shutdown()


def test_zero_cap_rejects_all():
    srv, port = _spawn()
    try:
        chat = CostCappedChat(_inner(port), max_usd=0.0)
        try:
            chat.invoke([{"role": "user", "content": "a"}])
            raise AssertionError("should reject")
        except ValueError as e:
            assert "cost cap" in str(e).lower()
        # Server never hit.
        assert len(_EchoChat.CAPTURED) == 0
    finally:
        srv.shutdown()


def test_composes_with_react_agent():
    from litgraph.agents import ReactAgent
    srv, port = _spawn()
    try:
        chat = CostCappedChat(_inner(port), max_usd=10.00)
        agent = ReactAgent(chat, tools=[])
        _ = agent
    finally:
        srv.shutdown()


def test_stacks_with_token_budget():
    from litgraph.providers import TokenBudgetChat
    srv, port = _spawn()
    try:
        budgeted = TokenBudgetChat(_inner(port), max_tokens=10_000)
        capped = CostCappedChat(budgeted, max_usd=1.00)
        capped.invoke([{"role": "user", "content": "hi"}])
        assert capped.total_usd() > 0
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_passes_through_under_cap,
        test_rejects_once_over_cap,
        test_unpriced_model_charges_zero,
        test_custom_prices_override_defaults,
        test_reset_clears_counter,
        test_reset_reenables_calls_after_cap_hit,
        test_repr_shows_spent_and_remaining,
        test_zero_cap_rejects_all,
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

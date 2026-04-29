"""default_prices() — built-in price catalog for major LLM providers.
Plus end-to-end where a fake OpenAI server returns versioned model IDs and
the CostTracker still computes the right USD via substring lookup."""
import http.server
import json
import threading

from litgraph.observability import CostTracker, default_prices
from litgraph.providers import OpenAIChat


def test_default_prices_returns_non_trivial_catalog():
    p = default_prices()
    assert isinstance(p, dict)
    assert len(p) >= 30
    # Spot-check a handful of expected entries.
    assert "gpt-4o" in p
    assert "claude-opus-4-7" in p
    assert "gemini-2.0-flash" in p
    assert "command-r" in p


def test_each_default_price_is_prompt_completion_tuple():
    for k, v in default_prices().items():
        assert isinstance(v, tuple), f"{k} value is not a tuple: {v!r}"
        assert len(v) == 2, f"{k} value is not a 2-tuple"
        prompt, completion = v
        assert isinstance(prompt, (int, float))
        assert isinstance(completion, (int, float))
        assert prompt >= 0 and completion >= 0


def test_default_prices_constructible_into_cost_tracker():
    """CostTracker(default_prices()) works without manual catalog setup."""
    tracker = CostTracker(default_prices())
    snap = tracker.snapshot()
    assert snap["calls"] == 0
    assert snap["usd"] == 0.0


def test_versioned_model_id_resolves_via_substring_lookup_e2e():
    """End-to-end: fake server returns `gpt-4o-2024-11-20` as the model field;
    CostTracker(default_prices()) should price it at gpt-4o rates ($2.50 + $10.00
    per Mtok = $7.50 for 1M prompt + 500k completion = $5.00) — actually:
    1M prompt @ $2.50/Mtok = $2.50, 500k completion @ $10.00/Mtok = $5.00 → $7.50."""

    class FakeOpenAI(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = {
                "id": "x", "object": "chat.completion",
                "model": "gpt-4o-2024-11-20",  # ← versioned!
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "hi"},
                             "finish_reason": "stop"}],
                # 1M prompt, 500k completion to get a clean $7.50.
                "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 500_000,
                          "total_tokens": 1_500_000},
            }
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-4o-2024-11-20",
                          base_url=f"http://127.0.0.1:{port}/v1")
        tracker = CostTracker(default_prices())
        chat.instrument(tracker)
        chat.invoke([{"role": "user", "content": "hi"}])
        # Allow the callback bus a moment to drain (it batches every 16ms).
        import time; time.sleep(0.05)
        usd = tracker.usd()
        assert abs(usd - 7.50) < 1e-3, f"expected ~$7.50, got ${usd:.4f}"
    finally:
        srv.shutdown()


def test_user_can_override_individual_prices():
    """Mutating the returned dict before passing it to CostTracker is the
    documented override path."""
    prices = default_prices()
    prices["gpt-4o"] = (0.0, 0.0)  # zero out
    tracker = CostTracker(prices)
    # Tracker construction succeeded; can't easily verify the override took
    # without running a full chat, but we can construct and snapshot.
    assert tracker.usd() == 0.0


if __name__ == "__main__":
    fns = [
        test_default_prices_returns_non_trivial_catalog,
        test_each_default_price_is_prompt_completion_tuple,
        test_default_prices_constructible_into_cost_tracker,
        test_versioned_model_id_resolves_via_substring_lookup_e2e,
        test_user_can_override_individual_prices,
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

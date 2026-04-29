"""SelfConsistencyChat — N-sample majority vote (Wang et al 2022).
Parallel fan-out via Rust JoinSet; picks winning answer by voter.
Cost: N× tokens. Usage summed across samples."""
import http.server
import itertools
import json
import threading
import re

from litgraph.providers import OpenAIChat, SelfConsistencyChat


class _Scripted(http.server.BaseHTTPRequestHandler):
    """Returns canned replies in sequence — one per HTTP request."""
    REPLIES: list = []
    COUNTER = None  # itertools.count

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        _ = self.rfile.read(n)
        idx = next(_Scripted.COUNTER)
        reply = _Scripted.REPLIES[idx % len(_Scripted.REPLIES)]
        payload = {
            "id": f"r-{idx}", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": reply},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(replies):
    _Scripted.REPLIES = replies
    _Scripted.COUNTER = itertools.count()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Scripted)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _inner(port):
    return OpenAIChat(api_key="k", model="gpt-test",
                      base_url=f"http://127.0.0.1:{port}/v1")


def test_picks_majority_text():
    srv, port = _spawn(["42", "41", "42", "41", "42"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=5)
        r = chat.invoke([{"role": "user", "content": "2+40?"}])
        assert r["text"] == "42"
    finally:
        srv.shutdown()


def test_summed_usage_across_samples():
    srv, port = _spawn(["a", "a", "a"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=3)
        r = chat.invoke([{"role": "user", "content": "q"}])
        # Each sample: 10 prompt + 5 completion = 15 total. N=3 → 45 total.
        assert r["usage"]["prompt"] == 30
        assert r["usage"]["completion"] == 15
        assert r["usage"]["total"] == 45
    finally:
        srv.shutdown()


def test_single_sample_passes_through():
    srv, port = _spawn(["only"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=1)
        r = chat.invoke([{"role": "user", "content": "q"}])
        assert r["text"] == "only"
        assert r["usage"]["total"] == 15  # not N-multiplied
    finally:
        srv.shutdown()


def test_tie_winner_is_one_of_the_tied_majority():
    # 2-2-1 tie between "a" and "b". Parallel execution races on the
    # shared counter — winner is whichever tied value got observed first.
    # The invariant is: winner must be in the tied majority, not "c".
    srv, port = _spawn(["a", "b", "a", "b", "c"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=5)
        r = chat.invoke([{"role": "user", "content": "q"}])
        assert r["text"] in {"a", "b"}
    finally:
        srv.shutdown()


def test_custom_voter_extracts_answer_field():
    """Reasoning chains reach 42 through different wording — raw-text
    majority never converges, but extracting the last number does."""
    srv, port = _spawn([
        "Let me think... the answer is 42.",
        "After calculation, I get 42.",
        "Maybe 17? No, 42.",
        "Is it 41? Actually, 42.",
        "I believe it is 42.",
    ])
    try:
        def last_number(r):
            nums = re.findall(r"-?\d+", r["text"])
            return nums[-1] if nums else None

        chat = SelfConsistencyChat(_inner(port), samples=5, voter=last_number)
        r = chat.invoke([{"role": "user", "content": "q"}])
        # Winner is one of the responses that ends in 42.
        assert "42" in r["text"]
    finally:
        srv.shutdown()


def test_voter_none_excludes_response_from_vote():
    """Voter returning None excludes that response from counting, but
    the winning chosen response is still one that voted."""
    srv, port = _spawn(["ONE", "skip", "ONE", "skip", "TWO"])
    try:
        def only_uppercase(r):
            text = r["text"]
            return text if text.isupper() else None

        chat = SelfConsistencyChat(_inner(port), samples=5, voter=only_uppercase)
        r = chat.invoke([{"role": "user", "content": "q"}])
        # "ONE" has 2 votes, "TWO" has 1, "skip" is excluded → winner must be ONE.
        assert r["text"] == "ONE"
    finally:
        srv.shutdown()


def test_temperature_param_overrides_caller():
    """SelfConsistencyChat should use its own sampling temperature per call,
    regardless of what the caller passes to .invoke()."""
    captured_temps = []

    class _TempCapture(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(n))
            captured_temps.append(body.get("temperature"))
            payload = {
                "id": "r", "model": "gpt-test", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": "ok"},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _TempCapture)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    try:
        chat = SelfConsistencyChat(_inner(port), samples=3, temperature=0.9)
        # Caller passes temperature=0.0 — wrapper must override with 0.9.
        chat.invoke([{"role": "user", "content": "q"}], temperature=0.0)
    finally:
        srv.shutdown()
    assert len(captured_temps) == 3
    for t in captured_temps:
        assert abs(t - 0.9) < 1e-6


def test_repr_contains_inner():
    srv, port = _spawn(["x"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=3)
        r = repr(chat)
        assert "SelfConsistencyChat" in r
    finally:
        srv.shutdown()


def test_composes_with_react_agent():
    from litgraph.agents import ReactAgent
    srv, port = _spawn(["x"])
    try:
        chat = SelfConsistencyChat(_inner(port), samples=3)
        agent = ReactAgent(chat, tools=[])
        _ = agent
    finally:
        srv.shutdown()


def test_stacks_with_cost_capped():
    """CostCapped(SelfConsistency(inner)): N-sample fan-out usage gets
    summed and charged in one billing event."""
    from litgraph.providers import CostCappedChat
    srv, port = _spawn(["42", "42", "42"])
    try:
        voter_chat = SelfConsistencyChat(_inner(port), samples=3)
        capped = CostCappedChat(
            voter_chat,
            max_usd=1.00,
            prices={"gpt-test": (2.50, 10.00)},
        )
        r = capped.invoke([{"role": "user", "content": "q"}])
        assert r["text"] == "42"
        # N=3 × 10 prompt × $2.50/Mtok + 3 × 5 completion × $10/Mtok
        # = 30 × 2.50/1M + 15 × 10/1M = $0.000075 + $0.00015 = $0.000225
        assert abs(capped.total_usd() - 0.000225) < 1e-9
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_picks_majority_text,
        test_summed_usage_across_samples,
        test_single_sample_passes_through,
        test_tie_winner_is_one_of_the_tied_majority,
        test_custom_voter_extracts_answer_field,
        test_voter_none_excludes_response_from_vote,
        test_temperature_param_overrides_caller,
        test_repr_contains_inner,
        test_composes_with_react_agent,
        test_stacks_with_cost_capped,
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

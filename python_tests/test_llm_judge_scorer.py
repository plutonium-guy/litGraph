"""LlmJudgeScorer in run_eval — wraps an LlmJudge as an eval-harness Scorer.
Fake OpenAI server returns canned judge JSON; verify wiring + per-case
explanation surfacing + composition with deterministic scorers."""
import http.server
import json
import threading

from litgraph.evaluators import run_eval
from litgraph.providers import OpenAIChat


class _FakeJudge(http.server.BaseHTTPRequestHandler):
    PAYLOADS: list = []  # one per request, cycled
    INDEX = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        _ = self.rfile.read(n)
        idx = self.INDEX[0] % len(self.PAYLOADS)
        self.INDEX[0] += 1
        # StructuredChatModel parses from `message.content` (assistant text),
        # NOT tool_calls. Emit the JSON as the assistant's text body.
        payload = self.PAYLOADS[idx]
        wire = {
            "id": f"r-{idx}", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": json.dumps(payload),
                },
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        out = json.dumps(wire).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(payloads):
    _FakeJudge.PAYLOADS = payloads
    _FakeJudge.INDEX = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeJudge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _judge_model(port):
    return OpenAIChat(
        api_key="sk-x", model="gpt-test",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def test_llm_judge_scorer_runs_through_eval():
    srv, port = _spawn([{"score": 1.0, "reasoning": "exact"}])
    try:
        report = run_eval(
            cases=[{"input": "q1", "expected": "a1"}, {"input": "q2", "expected": "a2"}],
            target=lambda q: f"answer to {q}",
            scorers=[{"name": "llm_judge", "model": _judge_model(port)}],
            max_parallel=1,  # serialize so the canned payload sequencing is deterministic
        )
        assert report["aggregate"]["means"]["llm_judge"] == 1.0
        # Per-case explanation surfaces the judge's reasoning.
        assert report["per_case"][0]["scores"]["llm_judge"]["explanation"] == "exact"
    finally:
        srv.shutdown()


def test_llm_judge_score_value_propagates():
    srv, port = _spawn([
        {"score": 0.8, "reasoning": "close"},
        {"score": 0.2, "reasoning": "wrong"},
    ])
    try:
        report = run_eval(
            cases=[{"input": "q1", "expected": "a1"}, {"input": "q2", "expected": "a2"}],
            target=lambda q: "irrelevant",
            scorers=[{"name": "llm_judge", "model": _judge_model(port)}],
            max_parallel=1,
        )
        # Mean of 0.8 + 0.2 = 0.5
        assert abs(report["aggregate"]["means"]["llm_judge"] - 0.5) < 1e-6
    finally:
        srv.shutdown()


def test_llm_judge_with_custom_scorer_name():
    srv, port = _spawn([{"score": 1.0, "reasoning": "ok"}])
    try:
        report = run_eval(
            cases=[{"input": "q", "expected": "a"}],
            target=lambda q: "x",
            scorers=[{
                "name": "llm_judge",
                "model": _judge_model(port),
                "scorer_name": "strict_judge",
            }],
            max_parallel=1,
        )
        assert "strict_judge" in report["aggregate"]["means"]
        assert "llm_judge" not in report["aggregate"]["means"]
    finally:
        srv.shutdown()


def test_llm_judge_composes_with_deterministic_scorers():
    """LangChain pattern: stack a deterministic scorer + the judge — the
    judge is sanity-checked against ExactMatch."""
    srv, port = _spawn([{"score": 1.0, "reasoning": "matches"}])
    try:
        report = run_eval(
            cases=[{"input": "q", "expected": "Paris"}],
            target=lambda q: "Paris",
            scorers=[
                {"name": "exact_match"},
                {"name": "llm_judge", "model": _judge_model(port)},
            ],
            max_parallel=1,
        )
        # Both scorers should agree on this trivially-correct case.
        assert report["aggregate"]["means"]["exact_match"] == 1.0
        assert report["aggregate"]["means"]["llm_judge"] == 1.0
    finally:
        srv.shutdown()


def test_llm_judge_with_custom_criteria():
    srv, port = _spawn([{"score": 0.6, "reasoning": "partial"}])
    try:
        report = run_eval(
            cases=[{"input": "q", "expected": "a"}],
            target=lambda q: "x",
            scorers=[{
                "name": "llm_judge",
                "model": _judge_model(port),
                "criteria": "Score 1.0 only if the prediction matches the brand voice exactly.",
            }],
            max_parallel=1,
        )
        assert abs(report["aggregate"]["means"]["llm_judge"] - 0.6) < 1e-6
    finally:
        srv.shutdown()


def test_llm_judge_missing_model_raises_value_error():
    try:
        run_eval(
            cases=[{"input": "q", "expected": "a"}],
            target=lambda q: "x",
            scorers=[{"name": "llm_judge"}],  # no model
        )
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "model" in str(e).lower()


def test_llm_judge_invalid_score_surfaces_per_case_error():
    """Judge returns score outside [0, 1] → LlmJudge raises Error::Parse →
    per-case scorer error → score=0 with explanation."""
    srv, port = _spawn([{"score": 1.5, "reasoning": "out of range"}])
    try:
        report = run_eval(
            cases=[{"input": "q", "expected": "a"}],
            target=lambda q: "x",
            scorers=[{"name": "llm_judge", "model": _judge_model(port)}],
            max_parallel=1,
        )
        entry = report["per_case"][0]["scores"]["llm_judge"]
        assert entry["score"] == 0.0
        assert "scorer error" in entry["explanation"]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_llm_judge_scorer_runs_through_eval,
        test_llm_judge_score_value_propagates,
        test_llm_judge_with_custom_scorer_name,
        test_llm_judge_composes_with_deterministic_scorers,
        test_llm_judge_with_custom_criteria,
        test_llm_judge_missing_model_raises_value_error,
        test_llm_judge_invalid_score_surfaces_per_case_error,
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

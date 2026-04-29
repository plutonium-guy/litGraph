"""LLM-judge generation eval — faithfulness + answer-relevance + correctness.

Real ragas-style RAG eval. Uses a scripted fake OpenAI server as the judge so
we can deterministically test metric aggregation, optional-correctness
behavior, and the actual prompt shape sent to the judge."""
import http.server
import json
import threading
from collections import deque

from litgraph.providers import OpenAIChat
from litgraph.retrieval import evaluate_generation


def _make_judge_server(canned_replies: list[str]):
    """Spawn a fake OpenAI-compat server that returns `canned_replies` in order
    (round-robin if exhausted). Captures the user prompt of each call."""
    queue = deque(canned_replies)
    captured_prompts: list[str] = []

    class FakeJudge(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(n)
            req = json.loads(body)
            # The user message is the last in the messages array.
            user_msg = req["messages"][-1]["content"]
            captured_prompts.append(user_msg)
            reply = queue.popleft() if queue else "0"
            payload = {
                "id": "x", "object": "chat.completion", "model": "gpt-4o-mini",
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": reply},
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeJudge)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1], captured_prompts


def test_aggregates_macro_average_for_two_metrics():
    """Two cases, two metrics each (no reference_answer → no correctness):
       case 1: faithfulness=1, relevance=1
       case 2: faithfulness=0, relevance=1
    → faithfulness_macro = 0.5, relevance_macro = 1.0, correctness_macro = None."""
    srv, port, _ = _make_judge_server(["1", "1", "0", "1"])
    try:
        judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                           base_url=f"http://127.0.0.1:{port}/v1")
        cases = [
            {"query": "q1", "answer": "a1", "contexts": ["c1"]},
            {"query": "q2", "answer": "a2", "contexts": ["c2"]},
        ]
        report = evaluate_generation(judge, cases, max_concurrency=1)
        assert report["n_cases"] == 2
        assert abs(report["faithfulness_macro"] - 0.5) < 1e-9
        assert abs(report["answer_relevance_macro"] - 1.0) < 1e-9
        assert report["correctness_macro"] is None
        # Per-case order matches input order.
        assert report["per_case"][0]["query"] == "q1"
        assert report["per_case"][1]["query"] == "q2"
    finally:
        srv.shutdown()


def test_correctness_only_when_reference_answer_present():
    """Case 1 has reference (3 calls); case 2 doesn't (2 calls)."""
    srv, port, _ = _make_judge_server(["1", "1", "1", "0", "1"])
    try:
        judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                           base_url=f"http://127.0.0.1:{port}/v1")
        cases = [
            {"query": "q1", "answer": "a1", "contexts": [],
             "reference_answer": "ref1"},
            {"query": "q2", "answer": "a2", "contexts": []},
        ]
        report = evaluate_generation(judge, cases, max_concurrency=1)
        # Correctness aggregate = 1.0 (only one reference, that case scored 1).
        assert report["correctness_macro"] == 1.0
        assert report["per_case"][0]["correctness"] == 1.0
        assert report["per_case"][1]["correctness"] is None
    finally:
        srv.shutdown()


def test_skip_correctness_overrides_dataset():
    """Even with reference_answer, skip_correctness=True suppresses it."""
    srv, port, _ = _make_judge_server(["1", "1"])
    try:
        judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                           base_url=f"http://127.0.0.1:{port}/v1")
        cases = [{"query": "q", "answer": "a", "contexts": [],
                  "reference_answer": "ref"}]
        report = evaluate_generation(
            judge, cases, max_concurrency=1, skip_correctness=True)
        assert report["correctness_macro"] is None
        assert report["per_case"][0]["correctness"] is None
    finally:
        srv.shutdown()


def test_prompt_shape_is_correct():
    """Check the actual prompts the judge receives. Faithfulness must mention
    CONTEXTS + ANSWER; relevance must mention QUERY + ANSWER; correctness
    must mention REFERENCE."""
    srv, port, captured = _make_judge_server(["1", "1", "1"])
    try:
        judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                           base_url=f"http://127.0.0.1:{port}/v1")
        cases = [{
            "query": "Where is Paris?",
            "answer": "Paris is in France.",
            "contexts": ["Paris is the capital of France."],
            "reference_answer": "France.",
        }]
        evaluate_generation(judge, cases, max_concurrency=1)
        # Three prompts in order: faithfulness, relevance, correctness.
        assert "CONTEXTS:" in captured[0]
        assert "Paris is the capital of France." in captured[0]
        assert "Paris is in France." in captured[0]

        assert "QUERY:" in captured[1]
        assert "Where is Paris?" in captured[1]

        assert "REFERENCE ANSWER:" in captured[2]
        assert "France." in captured[2]
        assert "CANDIDATE ANSWER:" in captured[2]
    finally:
        srv.shutdown()


def test_empty_cases_raises():
    judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                       base_url="http://127.0.0.1:1/v1")
    try:
        evaluate_generation(judge, [])
    except RuntimeError as e:
        assert "empty cases" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_rambling_judge_scored_zero_not_full_credit():
    """Judge that returns natural language (not 1/0/yes/no) gets 0 — we don't
    silently inflate scores by parsing semantically."""
    srv, port, _ = _make_judge_server([
        "The answer looks correct because the contexts mention France.",  # rambles
        "1",  # complies
    ])
    try:
        judge = OpenAIChat(api_key="k", model="gpt-4o-mini",
                           base_url=f"http://127.0.0.1:{port}/v1")
        report = evaluate_generation(
            judge,
            [{"query": "q", "answer": "a", "contexts": ["c"]}],
            max_concurrency=1,
        )
        # Faithfulness call got the rambling reply → 0.0.
        # Relevance call got "1" → 1.0.
        assert report["per_case"][0]["faithfulness"] == 0.0
        assert report["per_case"][0]["answer_relevance"] == 1.0
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_aggregates_macro_average_for_two_metrics,
        test_correctness_only_when_reference_answer_present,
        test_skip_correctness_overrides_dataset,
        test_prompt_shape_is_correct,
        test_empty_cases_raises,
        test_rambling_judge_scored_zero_not_full_credit,
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

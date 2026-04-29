"""LlmJudge evaluator — LLM-as-judge scoring with score+reasoning output.
End-to-end via a scripted fake OpenAI endpoint."""
import http.server
import json
import threading

from litgraph.evaluators import LlmJudge
from litgraph.providers import OpenAIChat


class _ScriptedChat(http.server.BaseHTTPRequestHandler):
    PAYLOADS: list = []   # LIFO
    SEEN_BODIES: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.SEEN_BODIES.append(json.loads(body))
        content = self.PAYLOADS.pop() if self.PAYLOADS else '{"score":0.0,"reasoning":"none"}'
        payload = {
            "id": "r", "model": "gpt-test", "object": "chat.completion",
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


def _spawn(payloads):
    _ScriptedChat.PAYLOADS = list(reversed(payloads))
    _ScriptedChat.SEEN_BODIES = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chat(port):
    return OpenAIChat(
        api_key="sk-test", model="gpt-test",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def test_judge_returns_score_and_reasoning_dict():
    srv, port = _spawn(['{"score":0.92,"reasoning":"Matches in meaning."}'])
    try:
        judge = LlmJudge(_chat(port))
        result = judge.judge(
            prediction="The capital is Paris.",
            reference="Paris is the capital of France.",
        )
    finally:
        srv.shutdown()
    assert abs(result["score"] - 0.92) < 1e-4
    assert "Matches" in result["reasoning"]


def test_judge_low_score_for_wrong_answer():
    srv, port = _spawn(['{"score":0.0,"reasoning":"London is not the capital of France."}'])
    try:
        judge = LlmJudge(_chat(port))
        result = judge.judge("London", "Paris")
    finally:
        srv.shutdown()
    assert result["score"] == 0.0
    assert "London" in result["reasoning"] or "not" in result["reasoning"].lower()


def test_custom_criteria_appears_in_prompt():
    srv, port = _spawn(['{"score":0.8,"reasoning":"ok"}'])
    try:
        judge = LlmJudge(
            _chat(port),
            criteria="Score ONLY on whether the prediction is concise.",
        )
        judge.judge("Short answer.", "Long verbose reference.")
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_BODIES[0]
    user_msg = next(m for m in body["messages"] if m["role"] == "user")
    assert "Score ONLY on whether the prediction is concise" in user_msg["content"]


def test_default_criteria_mentions_meaning_and_factual():
    srv, port = _spawn(['{"score":0.5,"reasoning":"ok"}'])
    try:
        judge = LlmJudge(_chat(port))
        judge.judge("p", "r")
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_BODIES[0]
    user_msg = next(m for m in body["messages"] if m["role"] == "user")
    assert "meaning" in user_msg["content"].lower()
    assert "factual" in user_msg["content"].lower()


def test_judge_batch_preserves_input_order():
    srv, port = _spawn([
        '{"score":0.1,"reasoning":"first"}',
        '{"score":0.5,"reasoning":"second"}',
        '{"score":0.9,"reasoning":"third"}',
    ])
    try:
        judge = LlmJudge(_chat(port))
        scores = judge.judge_batch([("a", "b"), ("c", "d"), ("e", "f")])
    finally:
        srv.shutdown()
    assert len(scores) == 3
    assert scores[0]["score"] < scores[1]["score"] < scores[2]["score"]
    assert scores[0]["reasoning"] == "first"
    assert scores[2]["reasoning"] == "third"


def test_score_out_of_range_raises_runtime_error():
    srv, port = _spawn(['{"score":1.5,"reasoning":"bug"}'])
    try:
        judge = LlmJudge(_chat(port))
        try:
            judge.judge("p", "r")
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "out of" in str(e).lower()
    finally:
        srv.shutdown()


def test_prompt_carries_reference_before_prediction():
    srv, port = _spawn(['{"score":0.5,"reasoning":"ok"}'])
    try:
        judge = LlmJudge(_chat(port))
        judge.judge(
            prediction="PREDICTION_MARKER",
            reference="REFERENCE_MARKER",
        )
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_BODIES[0]
    user_msg = next(m for m in body["messages"] if m["role"] == "user")
    content = user_msg["content"]
    ref_pos = content.find("REFERENCE_MARKER")
    pred_pos = content.find("PREDICTION_MARKER")
    assert ref_pos < pred_pos


if __name__ == "__main__":
    import traceback
    fns = [
        test_judge_returns_score_and_reasoning_dict,
        test_judge_low_score_for_wrong_answer,
        test_custom_criteria_appears_in_prompt,
        test_default_criteria_mentions_meaning_and_factual,
        test_judge_batch_preserves_input_order,
        test_score_out_of_range_raises_runtime_error,
        test_prompt_carries_reference_before_prediction,
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

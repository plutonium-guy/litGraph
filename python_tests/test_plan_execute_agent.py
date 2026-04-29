"""PlanAndExecuteAgent — two-phase planner + executor pattern."""
import http.server
import json
import threading

from litgraph.agents import PlanAndExecuteAgent
from litgraph.providers import OpenAIChat


class _ScriptedOpenAI(http.server.BaseHTTPRequestHandler):
    REPLIES: list = []
    INDEX = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        _ = self.rfile.read(n)
        idx = self.INDEX[0]
        self.INDEX[0] += 1
        text = self.REPLIES[idx % len(self.REPLIES)]
        payload = {
            "id": f"r-{idx}", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
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


def _spawn(replies):
    _ScriptedOpenAI.REPLIES = replies
    _ScriptedOpenAI.INDEX = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _model(port):
    return OpenAIChat(api_key="sk-x", model="gpt-test",
                      base_url=f"http://127.0.0.1:{port}/v1")


def test_plan_then_execute_three_steps():
    srv, port = _spawn([
        "1. Gather facts.\n2. Analyze them.\n3. Write summary.",
        "facts gathered",
        "analyzed: trend up",
        "Summary: trends are up",
    ])
    try:
        agent = PlanAndExecuteAgent(planner=_model(port))
        result = agent.invoke("Tell me about X")
        assert len(result["plan"]) == 3
        assert "Gather facts" in result["plan"][0]
        assert len(result["steps"]) == 3
        assert result["steps"][0]["output"] == "facts gathered"
        assert result["final_answer"] == "Summary: trends are up"
    finally:
        srv.shutdown()


def test_separate_planner_and_executor_models():
    """Cost-optimization pattern: cheap planner + capable executor."""
    p_srv, p_port = _spawn([
        "1. Step one\n2. Step two",
    ])

    class _ExecutorScripted(http.server.BaseHTTPRequestHandler):
        REPLIES = ["one done", "two done"]
        INDEX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            _ = self.rfile.read(n)
            idx = self.INDEX[0]
            self.INDEX[0] += 1
            payload = {
                "id": "r", "model": "gpt-test", "object": "chat.completion",
                "choices": [{"index": 0,
                             "message": {"role": "assistant",
                                         "content": self.REPLIES[idx % len(self.REPLIES)]},
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

    _ExecutorScripted.INDEX = [0]
    e_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ExecutorScripted)
    threading.Thread(target=e_srv.serve_forever, daemon=True).start()
    e_port = e_srv.server_address[1]

    try:
        planner = _model(p_port)
        executor = OpenAIChat(api_key="sk-x", model="gpt-test",
                              base_url=f"http://127.0.0.1:{e_port}/v1")
        agent = PlanAndExecuteAgent(planner=planner, executor=executor)
        result = agent.invoke("task")
        assert result["plan"] == ["Step one", "Step two"]
        assert result["steps"][0]["output"] == "one done"
        assert result["steps"][1]["output"] == "two done"
        assert result["final_answer"] == "two done"
    finally:
        p_srv.shutdown()
        e_srv.shutdown()


def test_max_steps_truncates_plan():
    srv, port = _spawn([
        "1. a\n2. b\n3. c\n4. d\n5. e",
        "x", "x",  # only 2 step executions because max_steps=2
    ])
    try:
        agent = PlanAndExecuteAgent(planner=_model(port), max_steps=2)
        result = agent.invoke("task")
        assert len(result["plan"]) == 2
        assert len(result["steps"]) == 2
    finally:
        srv.shutdown()


def test_empty_plan_returns_empty_steps_no_crash():
    srv, port = _spawn([""])
    try:
        agent = PlanAndExecuteAgent(planner=_model(port))
        result = agent.invoke("task")
        assert result["plan"] == []
        assert result["steps"] == []
        assert result["final_answer"] == ""
    finally:
        srv.shutdown()


def test_bullet_plan_falls_back_to_lines():
    srv, port = _spawn([
        "- step alpha\n- step beta",
        "alpha out", "beta out",
    ])
    try:
        agent = PlanAndExecuteAgent(planner=_model(port))
        result = agent.invoke("task")
        assert result["plan"] == ["step alpha", "step beta"]
        assert result["final_answer"] == "beta out"
    finally:
        srv.shutdown()


def test_repr():
    srv, port = _spawn(["1. only step", "out"])
    try:
        agent = PlanAndExecuteAgent(planner=_model(port))
        assert "PlanAndExecuteAgent" in repr(agent)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_plan_then_execute_three_steps,
        test_separate_planner_and_executor_models,
        test_max_steps_truncates_plan,
        test_empty_plan_returns_empty_steps_no_crash,
        test_bullet_plan_falls_back_to_lines,
        test_repr,
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

"""ReactAgent.stream() — per-turn event stream for real-time progress UIs.

Verifies the event shape + ordering contract:
  - IterationStart → LlmMessage → ToolCallStart(s) → ToolCallResult(s) → ...
  - Final or MaxIterationsReached terminates the stream
  - ToolCallStart for ALL calls is emitted before any ToolCallResult (parallel
    tools are all kicked off before any result comes back)
  - tool errors surface as is_error=True events, not as stream exceptions

This is the event-driven equivalent of `agent.invoke()` and powers the
"show me what the agent is doing" UX that LangChain's blocking invoke makes
impossible without AsyncIteratorCallbackHandler gymnastics."""
import http.server
import json
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def boom() -> str:
    """Always raises — used to verify error surfacing."""
    raise RuntimeError("kaboom")


class _FakeLLM(http.server.BaseHTTPRequestHandler):
    """Scripted OpenAI fake: replies with responses[IDX] in order."""
    RESPONSES: list = []
    IDX = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        payload = self.RESPONSES[self.IDX[0]]
        self.IDX[0] += 1
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a, **kw): pass


def _spawn_fake_llm(responses):
    _FakeLLM.RESPONSES = responses
    _FakeLLM.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _tool_call_resp(call_id, name, args):
    return {
        "id": "r", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{
                    "id": call_id, "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _multi_tool_call_resp(calls):
    return {
        "id": "r", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": json.dumps(c["args"])}}
                    for c in calls
                ],
            },
            "finish_reason": "tool_calls",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _final_resp(text):
    return {
        "id": "r", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_stream_emits_event_sequence_for_single_tool_call():
    """One tool call + final answer. Events must be in order:
    iteration_start(1), llm_message, tool_call_start, tool_call_result,
    iteration_start(2), llm_message, final."""
    srv, port = _spawn_fake_llm([
        _tool_call_resp("t1", "add", {"a": 2, "b": 3}),
        _final_resp("5"),
    ])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add])
        events = list(agent.stream("what is 2+3"))
        types = [e["type"] for e in events]
        assert types == [
            "iteration_start", "llm_message",
            "tool_call_start", "tool_call_result",
            "iteration_start", "llm_message",
            "final",
        ], f"got: {types}"
        # Spot-check payloads.
        tcs = next(e for e in events if e["type"] == "tool_call_start")
        assert tcs["call_id"] == "t1" and tcs["name"] == "add"
        tcr = next(e for e in events if e["type"] == "tool_call_result")
        assert tcr["is_error"] is False
        assert "5" in tcr["result"]
        final = events[-1]
        assert final["iterations"] == 2
        assert final["messages"][-1]["content"] == "5"
    finally:
        srv.shutdown()


def test_stream_emits_all_starts_before_any_result_for_parallel_tools():
    """Three parallel add() calls in a single assistant turn. The ordering
    contract says ALL ToolCallStart events precede ANY ToolCallResult.
    This is how consumers can render three 'pending' spinners up front."""
    srv, port = _spawn_fake_llm([
        _multi_tool_call_resp([
            {"id": "a", "name": "add", "args": {"a": 1, "b": 1}},
            {"id": "b", "name": "add", "args": {"a": 2, "b": 2}},
            {"id": "c", "name": "add", "args": {"a": 3, "b": 3}},
        ]),
        _final_resp("done"),
    ])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add])
        events = list(agent.stream("triple add"))
    finally:
        srv.shutdown()

    starts = 0
    results = 0
    for e in events:
        t = e["type"]
        if t == "tool_call_start":
            assert results == 0, "tool_call_start emitted AFTER a tool_call_result"
            starts += 1
        elif t == "tool_call_result":
            results += 1
    assert starts == 3 and results == 3


def test_stream_tool_error_reports_is_error_true_without_stopping_stream():
    """When a tool raises, the stream must surface is_error=True and keep
    going — the next LLM turn needs the error message to recover."""
    srv, port = _spawn_fake_llm([
        _tool_call_resp("t1", "boom", {}),
        _final_resp("sorry, tool failed"),
    ])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[boom])
        events = list(agent.stream("trigger boom"))
    finally:
        srv.shutdown()

    err = next(e for e in events if e["type"] == "tool_call_result")
    assert err["is_error"] is True
    assert "kaboom" in err["result"]
    # Stream continued: we got a Final event at the end.
    assert events[-1]["type"] == "final"


def test_stream_max_iterations_emits_dedicated_event_not_final():
    """Scripted LLM that always returns tool_calls → the stream caps at
    max_iterations and emits max_iterations_reached, NOT final."""
    # Script enough responses to exceed the cap.
    srv, port = _spawn_fake_llm([
        _tool_call_resp(f"t{i}", "add", {"a": 1, "b": 1}) for i in range(6)
    ])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add], max_iterations=2)
        events = list(agent.stream("loop"))
    finally:
        srv.shutdown()

    last = events[-1]
    assert last["type"] == "max_iterations_reached"
    assert last["iterations"] == 2
    assert not any(e["type"] == "final" for e in events)


def test_stream_is_iterator_not_list():
    """stream() returns an iterator — not a list — so consumers can render
    events progressively without collecting everything first."""
    srv, port = _spawn_fake_llm([_final_resp("ok")])
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        agent = ReactAgent(model=chat, tools=[add])
        s = agent.stream("hi")
        assert hasattr(s, "__iter__") and hasattr(s, "__next__")
        first = next(s)
        assert first["type"] == "iteration_start" and first["iteration"] == 1
        # Drain the rest to clean up.
        for _ in s:
            pass
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_stream_emits_event_sequence_for_single_tool_call,
        test_stream_emits_all_starts_before_any_result_for_parallel_tools,
        test_stream_tool_error_reports_is_error_true_without_stopping_stream,
        test_stream_max_iterations_emits_dedicated_event_not_final,
        test_stream_is_iterator_not_list,
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

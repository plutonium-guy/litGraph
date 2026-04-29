"""TextReActAgent.stream() — per-turn event stream for text-mode agent.

Verifies the event shape + ordering contract. Same scripted-fake-LLM
fixture as test_text_react_agent.py.

Event order: IterationStart → LlmResponse → ParsedAction →
ToolStart → ToolResult → ... → ParsedFinal → Final (or terminal:
ParseError / ToolNotFound / MaxIterations)."""
import http.server
import json
import threading

from litgraph.agents import TextReActAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import tool


@tool
def spin(x: int = 0) -> str:
    """Echo back a canned 'spun' string."""
    return f"spun({x})"


class _ScriptedLLM(http.server.BaseHTTPRequestHandler):
    RESPONSES: list = []
    IDX = [0]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        content = self.RESPONSES[self.IDX[0]]
        self.IDX[0] += 1
        payload = {
            "id": "r", "object": "chat.completion", "model": "scripted",
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
    _ScriptedLLM.RESPONSES = responses
    _ScriptedLLM.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chat(port):
    return OpenAIChat(
        api_key="sk-test", model="scripted",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def test_stream_short_circuit_emits_iteration_response_parsed_final_final():
    srv, port = _spawn(["Thought: I know\nFinal Answer: 42"])
    try:
        agent = TextReActAgent(_chat(port), tools=[])
        events = list(agent.stream("q"))
    finally:
        srv.shutdown()
    types = [e["type"] for e in events]
    assert types == ["iteration_start", "llm_response", "parsed_final", "final"]
    assert events[3]["answer"] == "42"
    assert events[3]["iterations"] == 1


def test_stream_action_then_final_full_event_sequence():
    srv, port = _spawn([
        'Action: spin\nAction Input: {"x": 7}',
        "Final Answer: ok",
    ])
    try:
        agent = TextReActAgent(_chat(port), tools=[spin])
        events = list(agent.stream("q"))
    finally:
        srv.shutdown()
    types = [e["type"] for e in events]
    assert types == [
        "iteration_start", "llm_response", "parsed_action",
        "tool_start", "tool_result",
        "iteration_start", "llm_response", "parsed_final", "final",
    ]
    # ParsedAction carries the parsed input as a real dict.
    parsed = events[2]
    assert parsed["tool"] == "spin"
    assert parsed["input"] == {"x": 7}
    # ToolResult carries the observation string.
    tr = events[4]
    assert tr["observation"] == "spun(7)"
    assert tr["is_error"] is False
    assert isinstance(tr["duration_ms"], int)


def test_stream_terminates_on_parse_error_no_final_event():
    srv, port = _spawn(["just prose, no labels at all"])
    try:
        agent = TextReActAgent(_chat(port), tools=[])
        events = list(agent.stream("q"))
    finally:
        srv.shutdown()
    last = events[-1]
    assert last["type"] == "parse_error"
    assert "raw_response" in last
    assert not any(e["type"] == "final" for e in events)


def test_stream_emits_tool_not_found_after_parsed_action():
    srv, port = _spawn(["Action: nope\nAction Input: {}"])
    try:
        agent = TextReActAgent(_chat(port), tools=[spin])
        events = list(agent.stream("q"))
    finally:
        srv.shutdown()
    types = [e["type"] for e in events]
    # ParsedAction comes BEFORE ToolNotFound so subscribers see the bad call.
    assert "parsed_action" in types
    assert types[-1] == "tool_not_found"
    last = events[-1]
    assert last["tool"] == "nope"
    assert "spin" in last["available"]


def test_stream_emits_max_iterations_when_loop_exhausts():
    srv, port = _spawn([
        "Action: spin\nAction Input: {}",
        "Action: spin\nAction Input: {}",
        "Action: spin\nAction Input: {}",
    ])
    try:
        agent = TextReActAgent(_chat(port), tools=[spin], max_iterations=2)
        events = list(agent.stream("q"))
    finally:
        srv.shutdown()
    last = events[-1]
    assert last["type"] == "max_iterations"
    assert last["iterations"] == 2


def test_stream_iterator_protocol():
    """Returns an iterator (supports `for` and StopIteration)."""
    srv, port = _spawn(["Final Answer: ok"])
    try:
        agent = TextReActAgent(_chat(port), tools=[])
        s = agent.stream("q")
        assert iter(s) is s
        events = []
        for ev in s:
            events.append(ev)
        assert events[-1]["type"] == "final"
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback

    fns = [
        test_stream_short_circuit_emits_iteration_response_parsed_final_final,
        test_stream_action_then_final_full_event_sequence,
        test_stream_terminates_on_parse_error_no_final_event,
        test_stream_emits_tool_not_found_after_parsed_action,
        test_stream_emits_max_iterations_when_loop_exhausts,
        test_stream_iterator_protocol,
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

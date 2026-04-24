"""TextReActAgent — text-mode ReAct loop for LLMs without native tool-calling.

For Ollama / vLLM / llama.cpp local models + base-completion checkpoints +
fine-tunes trained on the ReAct format. Wires iter-107's parse_react_step
and iter-108's react_format_instructions into a runnable agent loop."""
import http.server
import json
import threading

from litgraph.agents import TextReActAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import tool


@tool
def get_weather(city: str) -> str:
    """Fetch the current weather for a city."""
    return f"15C and raining in {city}"


@tool
def web_search(query: str) -> str:
    """Search the web for a query."""
    return f"results about {query}"


class _ScriptedLLM(http.server.BaseHTTPRequestHandler):
    """Scripted OpenAI Chat Completions fake. Returns canned content
    strings in order — TextReActAgent parses each one as ReAct prose."""
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
            "id": "r",
            "object": "chat.completion",
            "model": "scripted",
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
    _ScriptedLLM.SEEN_BODIES = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _make_chat(port):
    return OpenAIChat(
        api_key="sk-test",
        model="scripted",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def test_final_answer_first_turn_short_circuits():
    srv, port = _spawn(["Thought: I know this.\nFinal Answer: 42"])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[])
        result = agent.invoke("what is the answer?")
    finally:
        srv.shutdown()
    assert result["final_answer"] == "42"
    assert result["stopped_reason"] == "final_answer"
    assert result["iterations"] == 1


def test_action_then_final_answer_round_trips():
    srv, port = _spawn([
        'Thought: need weather\nAction: get_weather\nAction Input: {"city": "Paris"}',
        "Thought: got it\nFinal Answer: 15C and raining",
    ])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[get_weather])
        result = agent.invoke("weather in Paris?")
    finally:
        srv.shutdown()
    assert result["final_answer"] == "15C and raining"
    assert result["iterations"] == 2
    assert len(result["trace"]) == 2
    assert result["trace"][0]["kind"] == "action"
    assert result["trace"][0]["tool"] == "get_weather"
    assert "raining" in result["trace"][0]["observation"]
    assert result["trace"][1]["kind"] == "final"


def test_observation_fed_back_to_model_as_user_message():
    srv, port = _spawn([
        'Action: get_weather\nAction Input: {"city": "Paris"}',
        "Final Answer: done",
    ])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[get_weather])
        agent.invoke("q")
    finally:
        srv.shutdown()
    # The 2nd LLM call should have an "Observation: ..." user message.
    second_body = _ScriptedLLM.SEEN_BODIES[1]
    msgs = second_body["messages"]
    obs_msgs = [m for m in msgs if m["role"] == "user" and "Observation:" in m["content"]]
    assert len(obs_msgs) >= 1
    assert "raining" in obs_msgs[0]["content"]


def test_auto_format_instructions_injects_tool_catalog():
    srv, port = _spawn(["Final Answer: ok"])
    try:
        agent = TextReActAgent(
            _make_chat(port),
            tools=[get_weather, web_search],
        )
        agent.invoke("q")
    finally:
        srv.shutdown()
    # First request should have system messages listing both tools and the
    # ReAct grammar.
    first = _ScriptedLLM.SEEN_BODIES[0]
    sys_msgs = [m for m in first["messages"] if m["role"] == "system"]
    sys_concat = "\n".join(m["content"] for m in sys_msgs)
    assert "get_weather" in sys_concat
    assert "web_search" in sys_concat
    assert "Action Input:" in sys_concat


def test_disable_auto_format_instructions_omits_catalog():
    srv, port = _spawn(["Final Answer: ok"])
    try:
        agent = TextReActAgent(
            _make_chat(port),
            tools=[get_weather],
            system_prompt="custom system prompt",
            auto_format_instructions=False,
        )
        agent.invoke("q")
    finally:
        srv.shutdown()
    first = _ScriptedLLM.SEEN_BODIES[0]
    sys_msgs = [m for m in first["messages"] if m["role"] == "system"]
    sys_concat = "\n".join(m["content"] for m in sys_msgs)
    assert "custom system prompt" in sys_concat
    # The ReAct grammar from format_instructions should NOT have been added.
    assert "Action Input:" not in sys_concat


def test_unknown_tool_stops_with_tool_not_found():
    srv, port = _spawn(["Action: nope\nAction Input: {}"])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[])
        result = agent.invoke("q")
    finally:
        srv.shutdown()
    assert result["stopped_reason"] == "tool_not_found"
    assert result["final_answer"] is None


def test_parse_failure_stops_with_parse_error():
    srv, port = _spawn(["just some prose with no labels at all"])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[])
        result = agent.invoke("q")
    finally:
        srv.shutdown()
    assert result["stopped_reason"] == "parse_error"


def test_max_iterations_cap_stops_loop():
    srv, port = _spawn([
        "Action: web_search\nAction Input: foo",
        "Action: web_search\nAction Input: foo",
        "Action: web_search\nAction Input: foo",
        "Action: web_search\nAction Input: foo",
        "Action: web_search\nAction Input: foo",
    ])
    try:
        agent = TextReActAgent(
            _make_chat(port),
            tools=[web_search],
            max_iterations=3,
        )
        result = agent.invoke("q")
    finally:
        srv.shutdown()
    assert result["stopped_reason"] == "max_iterations"
    assert result["iterations"] == 3
    assert result["final_answer"] is None


def test_tool_error_captured_as_observation_not_fatal():
    """When the tool itself raises, the agent doesn't crash — it captures
    the error as an observation and lets the LLM decide how to react."""
    @tool
    def boom() -> str:
        """Always raises — tests error capture."""
        raise RuntimeError("kaboom from inside tool")

    srv, port = _spawn([
        "Action: boom\nAction Input: {}",
        "Final Answer: gave up after the tool failed",
    ])
    try:
        agent = TextReActAgent(_make_chat(port), tools=[boom])
        result = agent.invoke("q")
    finally:
        srv.shutdown()
    assert result["stopped_reason"] == "final_answer"
    assert result["trace"][0]["is_error"] is True
    assert "kaboom" in result["trace"][0]["observation"]


if __name__ == "__main__":
    import traceback

    fns = [
        test_final_answer_first_turn_short_circuits,
        test_action_then_final_answer_round_trips,
        test_observation_fed_back_to_model_as_user_message,
        test_auto_format_instructions_injects_tool_catalog,
        test_disable_auto_format_instructions_omits_catalog,
        test_unknown_tool_stops_with_tool_not_found,
        test_parse_failure_stops_with_parse_error,
        test_max_iterations_cap_stops_loop,
        test_tool_error_captured_as_observation_not_fatal,
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

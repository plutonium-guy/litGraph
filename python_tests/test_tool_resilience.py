"""TimeoutTool + RetryTool — tool resilience wrappers. Parallels chat-side
RetryingChatModel. Recommended composition: RetryTool(TimeoutTool(inner))."""
import time

from litgraph.tools import RetryTool, TimeoutTool, tool


@tool
def fast_tool(x: int) -> int:
    """Returns x+1 instantly."""
    return x + 1


@tool
def slow_tool(seconds: float) -> str:
    """Sleeps for `seconds` then returns."""
    time.sleep(seconds)
    return "done"


_attempts = {"count": 0}

@tool
def flaky_tool(target: int) -> str:
    """Fails the first `target` times, then succeeds."""
    _attempts["count"] += 1
    if _attempts["count"] <= target:
        raise RuntimeError("transient")
    return "ok"


def test_timeout_tool_constructed():
    """Tool constructs and reports name + repr."""
    safe = TimeoutTool(fast_tool, timeout_s=10.0)
    assert safe.name == "fast_tool"
    assert "TimeoutTool" in repr(safe)


def test_retry_tool_constructed():
    """Tool constructs and reports name + repr."""
    resilient = RetryTool(fast_tool, max_attempts=3)
    assert resilient.name == "fast_tool"
    assert "RetryTool" in repr(resilient)


def test_resilience_tools_compose_via_react_agent():
    """End-to-end: ReactAgent uses RetryTool(TimeoutTool(fast_tool))."""
    from litgraph.providers import OpenAIChat
    from litgraph.agents import ReactAgent
    import http.server
    import json
    import threading

    turn = [0]

    class _FakeOpenAI(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            _ = self.rfile.read(n)
            turn[0] += 1
            if turn[0] == 1:
                payload = {
                    "id": "r1", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "fast_tool",
                                    "arguments": json.dumps({"x": 41}),
                                },
                            }],
                        },
                        "finish_reason": "tool_calls",
                    }],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
            else:
                payload = {
                    "id": "r2", "model": "gpt-test", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "got 42"},
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

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]

    try:
        chat = OpenAIChat(api_key="sk-x", model="gpt-test",
                          base_url=f"http://127.0.0.1:{port}/v1")
        # Stack: Retry(Timeout(fast_tool)).
        wrapped = RetryTool(TimeoutTool(fast_tool, timeout_s=5.0), max_attempts=2)
        agent = ReactAgent(model=chat, tools=[wrapped])
        result = agent.invoke("call fast_tool with 41")
        final = result["messages"][-1]["content"]
        assert "got 42" in final
    finally:
        srv.shutdown()


def test_timeout_tool_wraps_native_tool():
    """Wraps a built-in Tool (not just FunctionTool)."""
    from litgraph.tools import HttpRequestTool
    http = HttpRequestTool()
    safe = TimeoutTool(http, timeout_s=2.0)
    assert safe.name == "http_request"


def test_retry_tool_wraps_native_tool():
    from litgraph.tools import HttpRequestTool
    http = HttpRequestTool()
    resilient = RetryTool(http, max_attempts=4, initial_delay_s=0.05)
    assert resilient.name == "http_request"


def test_retry_tool_with_full_config():
    """Construction with all retry config knobs."""
    r = RetryTool(
        fast_tool,
        max_attempts=5,
        initial_delay_s=0.01,
        max_delay_s=2.0,
        multiplier=3.0,
    )
    assert r.name == "fast_tool"


def test_can_chain_retry_and_timeout_in_either_order():
    """RetryTool(TimeoutTool(...)) and TimeoutTool(RetryTool(...)) both
    construct (different semantics; the recommended order is Retry-outer)."""
    inner_first = RetryTool(TimeoutTool(fast_tool, timeout_s=1.0), max_attempts=2)
    outer_first = TimeoutTool(RetryTool(fast_tool, max_attempts=2), timeout_s=1.0)
    assert inner_first.name == "fast_tool"
    assert outer_first.name == "fast_tool"


if __name__ == "__main__":
    import traceback
    fns = [
        test_timeout_tool_constructed,
        test_retry_tool_constructed,
        test_resilience_tools_compose_via_react_agent,
        test_timeout_tool_wraps_native_tool,
        test_retry_tool_wraps_native_tool,
        test_retry_tool_with_full_config,
        test_can_chain_retry_and_timeout_in_either_order,
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

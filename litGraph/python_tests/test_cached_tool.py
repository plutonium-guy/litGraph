"""CachedTool — TTL+LRU wrapper around any litgraph Tool. Identical-arg
calls within the TTL window return the cached prior result without
invoking the inner tool. End-to-end via ReactAgent."""
import http.server
import json
import threading
import time

from litgraph.tools import CachedTool, tool


def test_constructor_and_repr():
    @tool
    def echo(x: str) -> str:
        """echo back x"""
        return x

    cached = CachedTool(echo, ttl_seconds=60, max_entries=10)
    assert "CachedTool" in repr(cached)
    assert len(cached) == 0


def test_clear_drops_all_entries():
    @tool
    def echo(x: str) -> str:
        """echo"""
        return x

    cached = CachedTool(echo, ttl_seconds=60)
    # Tool isn't invoked until the agent calls it; clearing an empty
    # cache is a no-op but must not raise.
    cached.clear()
    assert len(cached) == 0


def test_dedup_via_react_agent():
    """End-to-end: agent issues two identical tool calls; the inner tool
    server only receives ONE request thanks to CachedTool dedup."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    # Counter via @tool — easy to verify call count.
    state = {"calls": 0}

    @tool
    def lookup(q: str) -> str:
        """look something up"""
        state["calls"] += 1
        return f"result for {q} (call #{state['calls']})"

    cached_lookup = CachedTool(lookup, ttl_seconds=60, max_entries=10)

    # Fake LLM: emits two identical tool_calls back-to-back, then final.
    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call_payload("call_a", "lookup", {"q": "weather"})
            elif i == 1:
                # Re-issue the SAME query — cache should serve.
                payload = _tool_call_payload("call_b", "lookup", {"q": "weather"})
            else:
                payload = {
                    "id": "r", "model": "m", "object": "chat.completion",
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": "Done."},
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

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk-test", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        agent = ReactAgent(chat, tools=[cached_lookup], max_iterations=5)
        result = agent.invoke("look up weather twice")
        # Despite TWO tool calls, the inner @tool ran only ONCE.
        assert state["calls"] == 1, f"expected 1 call, got {state['calls']}"
        assert len(cached_lookup) == 1
        # Both tool messages should have the same content.
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["content"] == tool_msgs[1]["content"]
    finally:
        chat_srv.shutdown()


def test_different_args_dont_collide():
    """Different args = different cache entries = inner called twice."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    state = {"calls": 0}

    @tool
    def lookup(q: str) -> str:
        """look something up"""
        state["calls"] += 1
        return f"result for {q}"

    cached = CachedTool(lookup, ttl_seconds=60)

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call_payload("c1", "lookup", {"q": "alpha"})
            elif i == 1:
                payload = _tool_call_payload("c2", "lookup", {"q": "beta"})
            else:
                payload = _final_payload("done")
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        agent = ReactAgent(chat, tools=[cached], max_iterations=5)
        agent.invoke("look up two things")
        assert state["calls"] == 2
        assert len(cached) == 2
    finally:
        chat_srv.shutdown()


def test_ttl_expiry_invalidates_entry():
    """After TTL passes, second call re-invokes inner."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    state = {"calls": 0}

    @tool
    def lookup(q: str) -> str:
        """look up"""
        state["calls"] += 1
        return f"result {state['calls']}"

    cached = CachedTool(lookup, ttl_seconds=1)  # 1-second TTL

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call_payload("c1", "lookup", {"q": "x"})
            elif i == 1:
                payload = _final_payload("first done")
            else:
                payload = _final_payload("second done")
            out = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(out)))
            self.end_headers()
            self.wfile.write(out)
        def log_message(self, *a, **kw): pass

    chat_srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeChat)
    threading.Thread(target=chat_srv.serve_forever, daemon=True).start()

    try:
        chat = OpenAIChat(
            api_key="sk", model="gpt",
            base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1",
        )
        agent = ReactAgent(chat, tools=[cached], max_iterations=5)
        agent.invoke("first")
        assert state["calls"] == 1

        # Wait past TTL.
        time.sleep(1.5)

        # Second invoke fires another tool call (from a fresh chat IDX).
        _FakeChat.IDX[0] = 0  # reset so first response is tool_call again
        agent.invoke("second")
        # Inner tool was called again because TTL expired.
        assert state["calls"] == 2
    finally:
        chat_srv.shutdown()


def _tool_call_payload(call_id, name, args):
    return {
        "id": "r", "model": "m", "object": "chat.completion",
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


def _final_payload(content):
    return {
        "id": "r", "model": "m", "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


if __name__ == "__main__":
    import traceback
    fns = [
        test_constructor_and_repr,
        test_clear_drops_all_entries,
        test_dedup_via_react_agent,
        test_different_args_dont_collide,
        test_ttl_expiry_invalidates_entry,
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

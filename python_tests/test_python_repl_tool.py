"""PythonReplTool — execute Python in a sandboxed subprocess. End-to-end
via ReactAgent + scripted fake LLM."""
import http.server
import json
import os
import shutil
import tempfile
import threading

from litgraph.tools import PythonReplTool


def _have_python3():
    return shutil.which("python3") is not None


def test_constructor_validates_working_dir():
    if not _have_python3():
        return
    td = tempfile.mkdtemp()
    try:
        tool = PythonReplTool(working_dir=td)
        assert tool.name == "python_repl"
        assert repr(tool) == "PythonReplTool()"
    finally:
        shutil.rmtree(td)


def test_constructor_rejects_nonexistent_dir():
    try:
        PythonReplTool(working_dir="/nonexistent-litgraph-dir-xyz")
        raise AssertionError("should have raised")
    except RuntimeError as e:
        # Rust surfaces the underlying os error.
        assert "stat" in str(e).lower() or "no such" in str(e).lower()


def test_print_round_trips_via_react_agent():
    if not _have_python3():
        return
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    td = tempfile.mkdtemp()

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call("c1", "python_repl", {
                    "code": "import math; print(math.sqrt(2))",
                })
            else:
                payload = _final("Got it.")
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
        chat = OpenAIChat(api_key="sk", model="gpt",
                          base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
        repl = PythonReplTool(working_dir=td)
        agent = ReactAgent(chat, tools=[repl])
        result = agent.invoke("compute sqrt(2)")
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        parsed = json.loads(tool_msgs[0]["content"])
        assert parsed["exit_code"] == 0
        assert "1.41421" in parsed["stdout"]
    finally:
        chat_srv.shutdown()
        shutil.rmtree(td)


def test_runaway_code_killed_by_timeout():
    """`while True: pass` must be killed by the configured timeout."""
    if not _have_python3():
        return
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    td = tempfile.mkdtemp()

    class _FakeChat(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            i = self.IDX[0]
            self.IDX[0] += 1
            if i == 0:
                payload = _tool_call("c1", "python_repl", {"code": "while True: pass"})
            else:
                payload = _final("Done")
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
        chat = OpenAIChat(api_key="sk", model="gpt",
                          base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
        # Tight 1s timeout — runaway loop should be killed promptly.
        repl = PythonReplTool(working_dir=td, timeout_s=1)
        agent = ReactAgent(chat, tools=[repl])
        import time
        started = time.monotonic()
        result = agent.invoke("loop forever")
        elapsed = time.monotonic() - started
        assert elapsed < 5
        # Tool message should report the timeout error.
        tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
        assert "timed out" in tool_msgs[0]["content"]
    finally:
        chat_srv.shutdown()
        shutil.rmtree(td)


def test_parent_secrets_are_stripped():
    """A secret-looking env var in the parent must NOT reach the child."""
    if not _have_python3():
        return
    key = "LITGRAPH_TEST_LEAK_CHECK_44"
    os.environ[key] = "should-not-leak"
    try:
        from litgraph.agents import ReactAgent
        from litgraph.providers import OpenAIChat
        td = tempfile.mkdtemp()

        code = (
            f"import os; print(os.environ.get('{key}', 'MISSING'))"
        )

        class _FakeChat(http.server.BaseHTTPRequestHandler):
            IDX = [0]
            def do_POST(self):
                n = int(self.headers.get("content-length", "0"))
                self.rfile.read(n)
                i = self.IDX[0]
                self.IDX[0] += 1
                if i == 0:
                    payload = _tool_call("c1", "python_repl", {"code": code})
                else:
                    payload = _final("ok")
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
            chat = OpenAIChat(api_key="sk", model="gpt",
                              base_url=f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")
            repl = PythonReplTool(working_dir=td)
            agent = ReactAgent(chat, tools=[repl])
            result = agent.invoke("check env")
            tool_msgs = [m for m in result["messages"] if m["role"] == "tool"]
            parsed = json.loads(tool_msgs[0]["content"])
            assert parsed["stdout"].strip() == "MISSING"
        finally:
            chat_srv.shutdown()
            shutil.rmtree(td)
    finally:
        os.environ.pop(key, None)


def _tool_call(call_id, name, args):
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


def _final(content):
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
        test_constructor_validates_working_dir,
        test_constructor_rejects_nonexistent_dir,
        test_print_round_trips_via_react_agent,
        test_runaway_code_killed_by_timeout,
        test_parent_secrets_are_stripped,
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

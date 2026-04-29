"""ShellTool — strict allowlist, no shell expansion, working-dir restriction,
timeout, output truncation. Plus E2E ReactAgent acceptance."""
import http.server
import json
import os
import shutil
import tempfile
import threading
import time

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import ShellTool


def test_empty_allowlist_rejected_at_construction():
    with tempfile.TemporaryDirectory() as wd:
        try:
            ShellTool(working_dir=wd, allowed_commands=[])
        except ValueError as e:
            assert "non-empty" in str(e)
        else:
            raise AssertionError("expected ValueError")


def test_nonexistent_working_dir_rejected_at_construction():
    try:
        ShellTool(working_dir="/this/probably/does/not/exist", allowed_commands=["echo"])
    except RuntimeError as e:
        assert "stat" in str(e) or "not a directory" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_constructor_smoke():
    with tempfile.TemporaryDirectory() as wd:
        t = ShellTool(working_dir=wd, allowed_commands=["echo", "ls"])
        assert t.name == "shell"
        assert "ShellTool(allowlist=2)" in repr(t)


def _spawn_fake_llm(script):
    """Spawn an HTTP server that returns the next entry of `script` per request."""
    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = script[FakeLLM.IDX[0]]
            FakeLLM.IDX[0] += 1
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    FakeLLM.IDX[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_react_agent_runs_shell_tool_end_to_end():
    """Fake LLM emits a tool_call to `echo hello`; ReactAgent dispatches via
    the ShellTool; the agent's final answer references the tool's stdout."""
    if shutil.which("echo") is None:
        return  # skip if echo somehow unavailable
    SCRIPT = [
        # Turn 1: tool_call → shell({"command":"echo","args":["hello world"]})
        {
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "echo", "args": ["hello world"]}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: final summary
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "the shell printed: hello world"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    srv, port = _spawn_fake_llm(SCRIPT)
    with tempfile.TemporaryDirectory() as wd:
        try:
            chat = OpenAIChat(api_key="k", model="gpt-x",
                              base_url=f"http://127.0.0.1:{port}/v1")
            agent = ReactAgent(
                model=chat,
                tools=[ShellTool(working_dir=wd, allowed_commands=["echo"])],
                max_iterations=5,
            )
            out = agent.invoke("echo hello world")
            assert "hello world" in out["messages"][-1]["content"]
        finally:
            srv.shutdown()


def test_react_agent_disallowed_command_returned_as_tool_error():
    """If the LLM tries to run a command outside the allowlist, the ShellTool
    raises Error::InvalidInput; the agent loop surfaces that as a tool message
    so the model can react. We assert the agent completes (doesn't crash) and
    the rejection text is somewhere in the trace."""
    SCRIPT = [
        # Turn 1: tool_call → forbidden `rm -rf /`
        {
            "id": "1", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant", "content": None,
                    "tool_calls": [{
                        "id": "t1",
                        "type": "function",
                        "function": {
                            "name": "shell",
                            "arguments": json.dumps({"command": "rm", "args": ["-rf", "/"]}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: model "apologizes" — content doesn't matter for the test
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "sorry, that command isn't allowed"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]
    srv, port = _spawn_fake_llm(SCRIPT)
    with tempfile.TemporaryDirectory() as wd:
        try:
            chat = OpenAIChat(api_key="k", model="gpt-x",
                              base_url=f"http://127.0.0.1:{port}/v1")
            agent = ReactAgent(
                model=chat,
                tools=[ShellTool(working_dir=wd, allowed_commands=["echo"])],
                max_iterations=5,
            )
            out = agent.invoke("delete the universe")
            # Agent completed; the tool-result message should mention the rejection.
            joined = "\n".join(m.get("content", "") or "" for m in out["messages"])
            assert "not in the allowlist" in joined
        finally:
            srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_empty_allowlist_rejected_at_construction,
        test_nonexistent_working_dir_rejected_at_construction,
        test_constructor_smoke,
        test_react_agent_runs_shell_tool_end_to_end,
        test_react_agent_disallowed_command_returned_as_tool_error,
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

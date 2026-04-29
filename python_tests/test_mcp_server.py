"""MCP server — expose litGraph @tool functions to Claude Desktop /
Cursor / Zed via stdio JSON-RPC. Tests spawn the server as a child
process and talk to it over stdin/stdout, mirroring how a real host
would invoke it."""
import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap


def _server_script(body: str) -> str:
    """Write a temp Python script that builds a server + calls serve_stdio.
    `body` MUST define a module-level `TOOLS = [...]`. Returns path; caller
    unlinks when done."""
    # Plain concat — textwrap.dedent + interpolation leads to mixed
    # indentation that Python refuses to parse.
    script = (
        "import sys\n"
        "from litgraph.tools import tool\n"
        "from litgraph.mcp import McpServer\n"
        "\n"
        + textwrap.dedent(body)
        + "\n"
        "if __name__ == '__main__':\n"
        "    McpServer(TOOLS).serve_stdio()\n"
    )
    fd, path = tempfile.mkstemp(suffix=".py", prefix="lg_mcp_server_")
    os.write(fd, script.encode())
    os.close(fd)
    return path


def _send_recv(proc, requests):
    """Send each request + read its response immediately (if any).
    Avoids pipe-buffer deadlocks that can happen when batching many
    writes before any reads on a child process."""
    responses = []
    for req in requests:
        proc.stdin.write(json.dumps(req).encode() + b"\n")
        proc.stdin.flush()
        if "id" in req:
            line = proc.stdout.readline()
            if not line:
                break
            responses.append(json.loads(line.decode()))
    return responses


def _spawn_server(script_body: str) -> tuple[subprocess.Popen, str]:
    path = _server_script(script_body)
    py = sys.executable
    proc = subprocess.Popen(
        [py, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,  # capture so we can debug if Popen child crashes
    )
    return proc, path


def _shutdown(proc, path):
    try:
        proc.stdin.close()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    finally:
        if path is not None and os.path.exists(path):
            os.unlink(path)


def test_initialize_handshake():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def noop() -> str:
            \"\"\"no op\"\"\"
            return "ok"
        TOOLS = [noop]
    """))
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        ])
        assert resp[0]["id"] == 1
        assert resp[0]["result"]["protocolVersion"] == "2024-11-05"
        assert resp[0]["result"]["serverInfo"]["name"] == "litgraph-mcp-server"
    finally:
        _shutdown(proc, path)


def test_tools_list_returns_python_decorated_tools():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b

        @tool
        def greet(name: str) -> str:
            \"\"\"Say hello.\"\"\"
            return f"hi {name}"

        TOOLS = [add, greet]
    """))
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        ])
        tools = resp[1]["result"]["tools"]
        names = {t["name"] for t in tools}
        assert "add" in names
        assert "greet" in names
        add_schema = next(t for t in tools if t["name"] == "add")
        # Schema from @tool decorator carries type hints as JSON Schema.
        assert add_schema["inputSchema"]["type"] == "object"
        assert "a" in add_schema["inputSchema"]["properties"]
        assert "Add two integers" in add_schema["description"]
    finally:
        _shutdown(proc, path)


def test_tools_call_invokes_python_function():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def add(a: int, b: int) -> int:
            \"\"\"Add two integers.\"\"\"
            return a + b
        TOOLS = [add]
    """))
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "add", "arguments": {"a": 3, "b": 4}},
            },
        ])
        result = resp[1]["result"]
        assert result["isError"] is False
        assert result["content"][0]["type"] == "text"
        # Integer result serialized as JSON string "7".
        assert result["content"][0]["text"] == "7"
    finally:
        _shutdown(proc, path)


def test_tools_call_python_exception_surfaces_as_isError_true():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def boom() -> str:
            \"\"\"Always fails.\"\"\"
            raise RuntimeError("kaboom from python")
        TOOLS = [boom]
    """))
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            {
                "jsonrpc": "2.0", "id": 2, "method": "tools/call",
                "params": {"name": "boom", "arguments": {}},
            },
        ])
        # MCP spec: tool errors are isError=true in a regular result, not
        # a JSON-RPC error envelope.
        assert "error" not in resp[1]
        assert resp[1]["result"]["isError"] is True
        assert "kaboom" in resp[1]["result"]["content"][0]["text"]
    finally:
        _shutdown(proc, path)


def test_unknown_method_returns_jsonrpc_error():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def noop() -> str:
            \"\"\"no op\"\"\"
            return "ok"
        TOOLS = [noop]
    """))
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "bogus"},
        ])
        assert resp[0]["error"]["code"] == -32601
        assert "bogus" in resp[0]["error"]["message"]
    finally:
        _shutdown(proc, path)


def test_notifications_emit_no_response():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def noop() -> str:
            \"\"\"no op\"\"\"
            return "ok"
        TOOLS = [noop]
    """))
    try:
        # Send a notification (no `id`) followed by a request. Verify
        # the notification didn't produce a response.
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 1, "method": "ping"},
        ])
        # Only 1 response, and it matches the ping.
        assert len(resp) == 1
        assert resp[0]["id"] == 1
    finally:
        _shutdown(proc, path)


def test_custom_server_name_and_version():
    proc, path = _spawn_server(textwrap.dedent("""
        @tool
        def noop() -> str:
            \"\"\"no op\"\"\"
            return "ok"
        TOOLS = [noop]
    """))
    # Rewrite the script to customize server name/version.
    with open(path, "r") as f:
        content = f.read()
    content = content.replace(
        "McpServer(TOOLS).serve_stdio()",
        'McpServer(TOOLS, server_name="my-agent", server_version="2.0.0").serve_stdio()',
    )
    with open(path, "w") as f:
        f.write(content)
    # Relaunch with the edited script.
    _shutdown(proc, None)  # don't unlink yet
    proc = subprocess.Popen(
        [sys.executable, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        resp = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        ])
        assert resp[0]["result"]["serverInfo"]["name"] == "my-agent"
        assert resp[0]["result"]["serverInfo"]["version"] == "2.0.0"
    finally:
        _shutdown(proc, path)


if __name__ == "__main__":
    import traceback
    fns = [
        test_initialize_handshake,
        test_tools_list_returns_python_decorated_tools,
        test_tools_call_invokes_python_function,
        test_tools_call_python_exception_surfaces_as_isError_true,
        test_unknown_method_returns_jsonrpc_error,
        test_notifications_emit_no_response,
        test_custom_server_name_and_version,
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

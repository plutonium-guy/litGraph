"""MCP server — resources + prompts handlers over real subprocess stdio.
Extends iter-131 tool-only tests with iter-139's resources/read +
prompts/get JSON-RPC methods."""
import json
import os
import subprocess
import sys
import tempfile


def _server_script(body: str) -> str:
    script = (
        "import sys\n"
        "from litgraph.mcp import McpServer\n"
        "\n"
        + body
        + "\n"
        "if __name__ == '__main__':\n"
        "    SERVER.serve_stdio()\n"
    )
    fd, path = tempfile.mkstemp(suffix=".py", prefix="lg_mcp_resprompt_")
    os.write(fd, script.encode())
    os.close(fd)
    return path


def _send_recv(proc, requests):
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


def _spawn(body):
    path = _server_script(body)
    proc = subprocess.Popen(
        [sys.executable, path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, path


def _shutdown(proc, path):
    try:
        proc.stdin.close()
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=2)
    os.unlink(path)


# ---- Resources ----

def test_initialize_advertises_resources_capability():
    proc, path = _spawn(
        "def _read_readme():\n"
        "    return 'hello world'\n"
        "SERVER = McpServer(\n"
        "    tools=[],\n"
        "    resources=[{\n"
        "        'uri': 'mem://readme',\n"
        "        'name': 'readme',\n"
        "        'description': 'the readme',\n"
        "        'mime_type': 'text/plain',\n"
        "        'reader': _read_readme,\n"
        "    }],\n"
        ")\n"
    )
    try:
        [init] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        ])
        caps = init["result"]["capabilities"]
        assert "resources" in caps
        assert caps["resources"]["listChanged"] is False
    finally:
        _shutdown(proc, path)


def test_resources_list_returns_registered_resources():
    proc, path = _spawn(
        "def _r1(): return 'A'\n"
        "def _r2(): return 'B'\n"
        "SERVER = McpServer(tools=[], resources=[\n"
        "    {'uri': 'mem://a', 'name': 'a', 'description': 'first',"
        "     'mime_type': 'text/plain', 'reader': _r1},\n"
        "    {'uri': 'mem://b', 'name': 'b', 'description': 'second',"
        "     'mime_type': 'text/markdown', 'reader': _r2},\n"
        "])\n"
    )
    try:
        [list_resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "resources/list"},
        ])
        arr = list_resp["result"]["resources"]
        assert len(arr) == 2
        uris = {r["uri"] for r in arr}
        assert uris == {"mem://a", "mem://b"}
        mimes = {r["uri"]: r["mimeType"] for r in arr}
        assert mimes["mem://b"] == "text/markdown"
    finally:
        _shutdown(proc, path)


def test_resources_read_invokes_python_reader():
    proc, path = _spawn(
        "def _reader():\n"
        "    return 'config body from python'\n"
        "SERVER = McpServer(tools=[], resources=[\n"
        "    {'uri': 'mem://config', 'name': 'cfg', 'description': '',"
        "     'mime_type': 'text/plain', 'reader': _reader},\n"
        "])\n"
    )
    try:
        [read_resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "resources/read",
             "params": {"uri": "mem://config"}},
        ])
        contents = read_resp["result"]["contents"]
        assert contents[0]["uri"] == "mem://config"
        assert contents[0]["text"] == "config body from python"
        assert contents[0]["mimeType"] == "text/plain"
    finally:
        _shutdown(proc, path)


def test_resources_read_unknown_uri_returns_error():
    proc, path = _spawn(
        "def _r(): return 'x'\n"
        "SERVER = McpServer(tools=[], resources=[\n"
        "    {'uri': 'mem://known', 'name': 'n', 'description': '',"
        "     'mime_type': 'text/plain', 'reader': _r},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "resources/read",
             "params": {"uri": "mem://bogus"}},
        ])
        assert resp["error"]["code"] == -32602
        assert "bogus" in resp["error"]["message"]
    finally:
        _shutdown(proc, path)


def test_resources_read_python_reader_exception_surfaces_internal_error():
    proc, path = _spawn(
        "def _flaky():\n"
        "    raise RuntimeError('disk gone')\n"
        "SERVER = McpServer(tools=[], resources=[\n"
        "    {'uri': 'mem://flaky', 'name': 'flaky', 'description': '',"
        "     'mime_type': 'text/plain', 'reader': _flaky},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "resources/read",
             "params": {"uri": "mem://flaky"}},
        ])
        assert resp["error"]["code"] == -32603
        assert "disk gone" in resp["error"]["message"]
    finally:
        _shutdown(proc, path)


# ---- Prompts ----

def test_initialize_advertises_prompts_capability():
    proc, path = _spawn(
        "def _render(args):\n"
        "    return [('user', 'hi ' + args.get('name', 'there'))]\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'greet', 'description': 'greet',\n"
        "     'arguments': [{'name': 'name', 'description': 'who', 'required': True}],\n"
        "     'renderer': _render},\n"
        "])\n"
    )
    try:
        [init] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        ])
        caps = init["result"]["capabilities"]
        assert "prompts" in caps
    finally:
        _shutdown(proc, path)


def test_prompts_list_exposes_arguments_schema():
    proc, path = _spawn(
        "def _render(args):\n"
        "    return [('user', 'ok')]\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'analyze', 'description': 'analyze something',\n"
        "     'arguments': [\n"
        "        {'name': 'target', 'description': 'what to analyze', 'required': True},\n"
        "        {'name': 'depth', 'description': 'how deep', 'required': False},\n"
        "     ],\n"
        "     'renderer': _render},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "prompts/list"},
        ])
        prompts = resp["result"]["prompts"]
        assert len(prompts) == 1
        assert prompts[0]["name"] == "analyze"
        args = prompts[0]["arguments"]
        assert {"name": "target", "description": "what to analyze", "required": True} in args
        assert {"name": "depth", "description": "how deep", "required": False} in args
    finally:
        _shutdown(proc, path)


def test_prompts_get_renders_messages_from_python_callable():
    proc, path = _spawn(
        "def _render(args):\n"
        "    who = args['who']\n"
        "    return [\n"
        "        ('system', 'You are polite.'),\n"
        "        ('user', 'Greet ' + who + ' warmly.'),\n"
        "    ]\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'greet', 'description': 'greet by name',\n"
        "     'arguments': [{'name': 'who', 'description': 'who', 'required': True}],\n"
        "     'renderer': _render},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
             "params": {"name": "greet", "arguments": {"who": "Alice"}}},
        ])
        msgs = resp["result"]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"]["type"] == "text"
        assert msgs[0]["content"]["text"] == "You are polite."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"]["text"] == "Greet Alice warmly."
    finally:
        _shutdown(proc, path)


def test_prompts_get_accepts_dict_return_shape():
    """Renderer may return either tuples or {role, text} dicts."""
    proc, path = _spawn(
        "def _render(args):\n"
        "    return [{'role': 'user', 'text': 'ping'}]\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'p', 'description': '', 'arguments': [], 'renderer': _render},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
             "params": {"name": "p"}},
        ])
        msgs = resp["result"]["messages"]
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"]["text"] == "ping"
    finally:
        _shutdown(proc, path)


def test_prompts_get_missing_required_arg_returns_error():
    proc, path = _spawn(
        "def _render(args):\n"
        "    return [('user', 'x')]\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'needs', 'description': '',\n"
        "     'arguments': [{'name': 'who', 'description': '', 'required': True}],\n"
        "     'renderer': _render},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
             "params": {"name": "needs", "arguments": {}}},
        ])
        assert resp["error"]["code"] == -32602
        assert "who" in resp["error"]["message"]
    finally:
        _shutdown(proc, path)


def test_prompts_get_python_renderer_exception_surfaces_internal_error():
    proc, path = _spawn(
        "def _fail(args):\n"
        "    raise ValueError('render exploded')\n"
        "SERVER = McpServer(tools=[], prompts=[\n"
        "    {'name': 'x', 'description': '', 'arguments': [], 'renderer': _fail},\n"
        "])\n"
    )
    try:
        [resp] = _send_recv(proc, [
            {"jsonrpc": "2.0", "id": 1, "method": "prompts/get",
             "params": {"name": "x"}},
        ])
        assert resp["error"]["code"] == -32603
        assert "render exploded" in resp["error"]["message"]
    finally:
        _shutdown(proc, path)


def test_repr_mentions_resources_and_prompts_count():
    from litgraph.mcp import McpServer
    def _r(): return "x"
    def _p(args): return [("user", "hi")]
    s = McpServer(
        tools=[],
        resources=[{"uri": "mem://a", "name": "a", "description": "",
                    "mime_type": "text/plain", "reader": _r}],
        prompts=[{"name": "p", "description": "", "arguments": [], "renderer": _p}],
    )
    r = repr(s)
    assert "resources=1" in r
    assert "prompts=1" in r


if __name__ == "__main__":
    import traceback
    fns = [
        test_initialize_advertises_resources_capability,
        test_resources_list_returns_registered_resources,
        test_resources_read_invokes_python_reader,
        test_resources_read_unknown_uri_returns_error,
        test_resources_read_python_reader_exception_surfaces_internal_error,
        test_initialize_advertises_prompts_capability,
        test_prompts_list_exposes_arguments_schema,
        test_prompts_get_renders_messages_from_python_callable,
        test_prompts_get_accepts_dict_return_shape,
        test_prompts_get_missing_required_arg_returns_error,
        test_prompts_get_python_renderer_exception_surfaces_internal_error,
        test_repr_mentions_resources_and_prompts_count,
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

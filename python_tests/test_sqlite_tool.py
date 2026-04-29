"""SqliteQueryTool — read-only SQL access with table allowlist + row caps,
plus E2E ReactAgent dispatch where a fake LLM emits a SELECT tool call."""
import http.server
import json
import sqlite3
import tempfile
import threading

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import SqliteQueryTool


def _build_db():
    """Build a small users+secrets DB; return (path, NamedTemporaryFile-ish handle)."""
    f = tempfile.NamedTemporaryFile(prefix="sqltool-", suffix=".db", delete=False)
    f.close()
    conn = sqlite3.connect(f.name)
    conn.executescript("""
        CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        CREATE TABLE secrets (id INTEGER PRIMARY KEY, value TEXT);
        INSERT INTO users (id, name, email) VALUES
            (1, 'alice', 'alice@example.com'),
            (2, 'bob',   'bob@example.com'),
            (3, 'carol', 'carol@example.com');
        INSERT INTO secrets (id, value) VALUES (1, 'topsecret');
    """)
    conn.commit()
    conn.close()
    return f.name


def test_constructor_smoke():
    p = _build_db()
    try:
        t = SqliteQueryTool(db_path=p, allowed_tables=["users"])
        assert t.name == "sqlite_query"
        assert "read_only=true" in repr(t)  # Rust-formatted, not Python's True
    finally:
        import os; os.unlink(p)


def test_construction_fails_on_missing_db():
    try:
        SqliteQueryTool(db_path="/this/does/not/exist.db", allowed_tables=["users"])
    except RuntimeError as e:
        assert "does not exist" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_react_agent_select_with_param_binding():
    """End-to-end: fake LLM emits a SELECT tool call → agent runs it →
    rows come back to the model → final answer references row content."""
    p = _build_db()
    SCRIPT = [
        # Turn 1: tool call → SELECT name FROM users WHERE id = ?
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
                            "name": "sqlite_query",
                            "arguments": json.dumps({
                                "sql": "SELECT name FROM users WHERE id = ?",
                                "params": [2]
                            }),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        # Turn 2: final answer
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "User #2 is bob."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = SCRIPT[FakeLLM.IDX[0]]
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
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        sq = SqliteQueryTool(db_path=p, allowed_tables=["users"])
        agent = ReactAgent(model=chat, tools=[sq], max_iterations=5)
        out = agent.invoke("who is user 2?")
        # Final answer references the user's name.
        assert "bob" in out["messages"][-1]["content"].lower()
    finally:
        srv.shutdown()
        import os; os.unlink(p)


def test_disallowed_table_rejection_surfaces_through_agent():
    """LLM tries to select from a table not on the allowlist; the rejection
    flows back as a tool message so the model can react."""
    p = _build_db()
    SCRIPT = [
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
                            "name": "sqlite_query",
                            "arguments": json.dumps({"sql": "SELECT value FROM secrets"}),
                        }
                    }]
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
        {
            "id": "2", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant",
                            "content": "I cannot access that table."},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        },
    ]

    class FakeLLM(http.server.BaseHTTPRequestHandler):
        IDX = [0]
        def do_POST(self):
            n = int(self.headers.get("content-length", "0"))
            self.rfile.read(n)
            payload = SCRIPT[FakeLLM.IDX[0]]
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
    port = srv.server_address[1]
    try:
        chat = OpenAIChat(api_key="k", model="gpt-x",
                          base_url=f"http://127.0.0.1:{port}/v1")
        sq = SqliteQueryTool(db_path=p, allowed_tables=["users"])
        agent = ReactAgent(model=chat, tools=[sq], max_iterations=5)
        out = agent.invoke("read the secrets table")
        # Concatenated message contents should mention the rejection text.
        joined = "\n".join(m.get("content", "") or "" for m in out["messages"])
        assert "secrets" in joined and "not in the allowlist" in joined
    finally:
        srv.shutdown()
        import os; os.unlink(p)


if __name__ == "__main__":
    fns = [
        test_constructor_smoke,
        test_construction_fails_on_missing_db,
        test_react_agent_select_with_param_binding,
        test_disallowed_table_rejection_surfaces_through_agent,
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

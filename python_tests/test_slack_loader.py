"""SlackLoader — pull channel message history via the Slack Web API.

Verified against an inline fake Slack HTTP server that captures auth + path
query strings + returns canned `conversations.history` / `conversations.replies`
JSON envelopes."""
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import SlackLoader


_LAST_AUTH = []
_LAST_PATHS = []


class _FakeSlack(http.server.BaseHTTPRequestHandler):
    def _send(self, body):
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        _LAST_AUTH.append(self.headers.get("Authorization"))
        _LAST_PATHS.append(self.path)
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path.endswith("/conversations.history"):
            # Two pages: no cursor → page 1 with next_cursor=CURSOR1;
            # cursor=CURSOR1 → page 2 with empty next_cursor.
            if q.get("cursor", [""])[0] == "CURSOR1":
                self._send({
                    "ok": True,
                    "messages": [
                        {"type": "message", "text": "third", "user": "U3", "ts": "300.0"},
                    ],
                    "response_metadata": {"next_cursor": ""},
                })
            else:
                self._send({
                    "ok": True,
                    "messages": [
                        {"type": "message", "text": "first", "user": "U1", "ts": "100.0"},
                        {"type": "message", "text": "parent msg", "user": "U2",
                         "ts": "200.0", "thread_ts": "200.0", "reply_count": 2},
                    ],
                    "response_metadata": {"next_cursor": "CURSOR1"},
                })
        elif parsed.path.endswith("/conversations.replies"):
            self._send({
                "ok": True,
                "messages": [
                    # Parent included as replies[0]; loader should drop it.
                    {"type": "message", "text": "parent msg", "user": "U2",
                     "ts": "200.0", "thread_ts": "200.0"},
                    {"type": "message", "text": "reply one", "user": "U4",
                     "ts": "201.0", "thread_ts": "200.0"},
                    {"type": "message", "text": "reply two", "user": "U5",
                     "ts": "202.0", "thread_ts": "200.0"},
                ],
            })
        else:
            self._send({"ok": False, "error": "unknown_method"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_AUTH.clear(); _LAST_PATHS.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeSlack)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_basic_loader_paginates_and_returns_one_doc_per_message():
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="xoxb-test",
            channel_id="C123",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 3
    texts = [d["content"] for d in docs]
    assert texts == ["first", "parent msg", "third"]


def test_auth_bearer_token_on_every_request():
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="xoxb-secret",
            channel_id="C1",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    assert _LAST_AUTH, "expected at least one captured Authorization header"
    for a in _LAST_AUTH:
        assert a == "Bearer xoxb-secret"


def test_metadata_includes_channel_user_ts_source_id():
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="C999",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    d0 = docs[0]
    assert d0["metadata"]["channel"] == "C999"
    assert d0["metadata"]["user"] == "U1"
    assert d0["metadata"]["ts"] == "100.0"
    assert d0["metadata"]["source"] == "slack:C999"
    # Document id = channel:ts.
    assert d0["id"] == "C999:100.0"


def test_include_threads_flattens_replies_inline_and_drops_parent_duplicate():
    """With include_threads=True, loader follows conversations.replies for
    every parent with reply_count>0 and appends replies (dropping the
    duplicate parent that Slack returns as replies[0])."""
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="C1",
            base_url=f"http://127.0.0.1:{port}",
            include_threads=True,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # 3 top-level + 2 replies = 5.
    assert len(docs) == 5
    reply_docs = [d for d in docs if "thread_ts" in d["metadata"]]
    assert [d["content"] for d in reply_docs] == ["reply one", "reply two"]
    for r in reply_docs:
        assert r["metadata"]["thread_ts"] == "200.0"


def test_top_level_parent_does_not_get_thread_ts_metadata():
    """Parent of a thread has thread_ts == ts. Loader must NOT tag it as a
    reply; only actual replies (ts != thread_ts) get the metadata field."""
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="C1",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # docs[1] is the parent (text "parent msg", ts == thread_ts).
    assert "thread_ts" not in docs[1]["metadata"]


def test_max_messages_cap_truncates_result():
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="C1",
            base_url=f"http://127.0.0.1:{port}",
            max_messages=2,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2


def test_oldest_and_latest_appear_in_query_string():
    srv, port = _spawn()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="C1",
            base_url=f"http://127.0.0.1:{port}",
            oldest="100.0",
            latest="500.0",
        )
        loader.load()
    finally:
        srv.shutdown()
    hist_path = next(p for p in _LAST_PATHS if p.startswith("/conversations.history"))
    assert "oldest=100.0" in hist_path
    assert "latest=500.0" in hist_path


def test_slack_api_error_surfaces_as_runtime_error():
    """Slack's `ok: false` envelope → RuntimeError; NOT a silent empty list."""
    class _ErrSlack(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"ok": False, "error": "channel_not_found"}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        def log_message(self, *a, **kw): pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ErrSlack)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        loader = SlackLoader(
            api_key="x",
            channel_id="bad",
            base_url=f"http://127.0.0.1:{port}",
        )
        try:
            loader.load()
        except RuntimeError as e:
            assert "channel_not_found" in str(e)
        else:
            raise AssertionError("expected RuntimeError on Slack ok=false")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_basic_loader_paginates_and_returns_one_doc_per_message,
        test_auth_bearer_token_on_every_request,
        test_metadata_includes_channel_user_ts_source_id,
        test_include_threads_flattens_replies_inline_and_drops_parent_duplicate,
        test_top_level_parent_does_not_get_thread_ts_metadata,
        test_max_messages_cap_truncates_result,
        test_oldest_and_latest_appear_in_query_string,
        test_slack_api_error_surfaces_as_runtime_error,
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

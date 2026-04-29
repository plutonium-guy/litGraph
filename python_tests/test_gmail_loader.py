"""GmailLoader — pull messages via Gmail REST API. OAuth2 Bearer auth +
multipart body extraction (prefers text/plain, falls back to stripped HTML)."""
import base64
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import GmailLoader


_LAST_PATHS = []
_LAST_AUTH = []


def _b64url(s):
    return base64.urlsafe_b64encode(s.encode()).decode().rstrip("=")


class _FakeGmail(http.server.BaseHTTPRequestHandler):
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

        # /gmail/v1/users/{user}/messages/{id}?format=full
        if "/messages/" in parsed.path:
            mid = parsed.path.rsplit("/", 1)[-1]
            self._send({
                "id": mid,
                "threadId": f"thr_{mid}",
                "labelIds": ["INBOX", "IMPORTANT"],
                "snippet": f"Preview of {mid}",
                "payload": {
                    "mimeType": "multipart/alternative",
                    "headers": [
                        {"name": "From", "value": "alice@example.com"},
                        {"name": "To", "value": "bob@example.com"},
                        {"name": "Subject", "value": f"Subject for {mid}"},
                        {"name": "Date", "value": "Mon, 1 Jan 2024 12:00:00 +0000"},
                    ],
                    "parts": [
                        {"mimeType": "text/plain",
                         "body": {"data": _b64url(f"plain body for {mid}")}},
                        {"mimeType": "text/html",
                         "body": {"data": _b64url(f"<p>html body for {mid}</p>")}},
                    ],
                },
            })
            return
        # /gmail/v1/users/{user}/messages
        if parsed.path.endswith("/messages"):
            page_token = q.get("pageToken", [None])[0]
            if page_token == "PAGE2":
                self._send({
                    "messages": [{"id": "m3", "threadId": "t3"}],
                    "resultSizeEstimate": 3,
                })
            else:
                self._send({
                    "messages": [
                        {"id": "m1", "threadId": "t1"},
                        {"id": "m2", "threadId": "t2"},
                    ],
                    "nextPageToken": "PAGE2",
                    "resultSizeEstimate": 3,
                })
            return
        self._send({"error": "unknown"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_PATHS.clear(); _LAST_AUTH.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeGmail)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_paginates_messages_and_returns_one_doc_per_message():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="ya29.test",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 3
    # Subject prefixed as markdown H1.
    assert docs[0]["content"].startswith("# Subject for m1")
    # Plain body extracted (preferred over HTML).
    assert "plain body for m1" in docs[0]["content"]


def test_metadata_captures_headers_labels_snippet_ids():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    d = docs[0]
    assert d["metadata"]["message_id"] == "m1"
    assert d["metadata"]["thread_id"] == "thr_m1"
    assert d["metadata"]["from"] == "alice@example.com"
    assert d["metadata"]["to"] == "bob@example.com"
    assert d["metadata"]["subject"] == "Subject for m1"
    assert d["metadata"]["labels"] == "INBOX,IMPORTANT"
    assert "Preview" in d["metadata"]["snippet"]
    assert d["metadata"]["source"] == "gmail:me/m1"


def test_bearer_auth_present_on_every_request():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="ya29.SECRET",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    for a in _LAST_AUTH:
        assert a == "Bearer ya29.SECRET"


def test_query_parameter_forwarded_to_list_endpoint():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="t",
            query="from:alice after:2024/01/01",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    list_path = next(p for p in _LAST_PATHS if p.endswith("/messages") or "?" in p and "/messages?" in p)
    # Find list path that has q= parameter.
    list_paths_with_q = [p for p in _LAST_PATHS if "q=" in p]
    assert len(list_paths_with_q) >= 1
    assert "alice" in list_paths_with_q[0]


def test_max_messages_cap_truncates_result():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="t",
            max_messages=1,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1


def test_include_body_false_uses_metadata_format_and_returns_snippet():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="t",
            include_body=False,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # GET messages/{id}?format=metadata used.
    assert any("format=metadata" in p for p in _LAST_PATHS)
    # Content = snippet.
    assert "Preview of" in docs[0]["content"]


def test_user_id_override_changes_endpoint_path():
    srv, port = _spawn()
    try:
        loader = GmailLoader(
            access_token="t",
            user_id="admin@corp.com",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    # URL-encoded user id ('@' → '%40', '.' preserved).
    assert any("/users/admin%40corp.com/" in p for p in _LAST_PATHS)


if __name__ == "__main__":
    fns = [
        test_paginates_messages_and_returns_one_doc_per_message,
        test_metadata_captures_headers_labels_snippet_ids,
        test_bearer_auth_present_on_every_request,
        test_query_parameter_forwarded_to_list_endpoint,
        test_max_messages_cap_truncates_result,
        test_include_body_false_uses_metadata_format_and_returns_snippet,
        test_user_id_override_changes_endpoint_path,
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

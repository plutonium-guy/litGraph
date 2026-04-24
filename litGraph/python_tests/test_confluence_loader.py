"""ConfluenceLoader — pull pages from Confluence Cloud or Server/DC via
the REST API. Both Basic (Cloud) and Bearer (DC) auth supported."""
import base64
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import ConfluenceLoader


_LAST_AUTH = []
_LAST_PATHS = []


class _FakeConfluence(http.server.BaseHTTPRequestHandler):
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
        if parsed.path.startswith("/wiki/rest/api/content/"):
            page_id = parsed.path.rsplit("/", 1)[-1]
            self._send({
                "id": page_id,
                "title": f"Direct Page {page_id}",
                "body": {"storage": {"value": f"<p>Body for {page_id}.</p>"}},
                "version": {"number": 5},
                "space": {"key": "ENG"},
            })
        elif parsed.path.startswith("/wiki/rest/api/content"):
            # Paginated space listing.
            start = int(q.get("start", ["0"])[0])
            if start == 0:
                self._send({
                    "results": [
                        {"id": "100", "title": "First",
                         "body": {"storage": {"value": "<h2>One</h2><p>first page.</p>"}},
                         "version": {"number": 2}},
                        {"id": "101", "title": "Second",
                         "body": {"storage": {"value": "<p>second page.</p>"}},
                         "version": {"number": 1}},
                    ],
                    "size": 2,
                    "_links": {"next": "/wiki/rest/api/content?start=2"},
                })
            else:
                self._send({
                    "results": [
                        {"id": "102", "title": "Third",
                         "body": {"storage": {"value": "<p>third page.</p>"}},
                         "version": {"number": 4}},
                    ],
                    "size": 1,
                    "_links": {},
                })
        else:
            self._send({"error": "unknown"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_AUTH.clear(); _LAST_PATHS.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeConfluence)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cloud_space_loader_paginates_and_returns_one_doc_per_page():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_cloud(
            base_url=f"http://127.0.0.1:{port}",
            email="ada@example.com",
            api_token="tok",
            space_key="ENG",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 3
    # HTML → plain text.
    assert "first page" in docs[0]["content"]
    assert "second page" in docs[1]["content"]
    assert "third page" in docs[2]["content"]


def test_cloud_auth_uses_basic_header_with_base64_credentials():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_cloud(
            base_url=f"http://127.0.0.1:{port}",
            email="ada@example.com",
            api_token="secret-token",
            space_key="ENG",
        )
        loader.load()
    finally:
        srv.shutdown()
    expected = "Basic " + base64.b64encode(b"ada@example.com:secret-token").decode()
    for a in _LAST_AUTH:
        assert a == expected, f"got: {a}"


def test_bearer_auth_uses_bearer_header():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_bearer(
            base_url=f"http://127.0.0.1:{port}",
            token="pat-xyz",
            space_key="ENG",
        )
        loader.load()
    finally:
        srv.shutdown()
    for a in _LAST_AUTH:
        assert a == "Bearer pat-xyz"


def test_metadata_includes_id_title_space_key_version_source():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_bearer(
            base_url=f"http://127.0.0.1:{port}",
            token="t",
            space_key="ENG",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    d0 = docs[0]
    assert d0["metadata"]["page_id"] == "100"
    assert d0["metadata"]["title"] == "First"
    assert d0["metadata"]["space_key"] == "ENG"
    # Version comes back as a string (all metadata values stringified in docs_to_pylist).
    assert d0["metadata"]["version"] == "2"
    assert d0["metadata"]["source"] == "confluence:ENG:100"
    assert d0["id"] == "100"


def test_from_pages_bearer_fetches_each_page_by_id():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_pages_bearer(
            base_url=f"http://127.0.0.1:{port}",
            token="t",
            page_ids=["200", "201"],
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    assert docs[0]["id"] == "200"
    assert docs[1]["id"] == "201"
    # space_key pulled from /content/{id}?expand=space response.
    assert docs[0]["metadata"]["space_key"] == "ENG"


def test_max_pages_caps_space_results():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_bearer(
            base_url=f"http://127.0.0.1:{port}",
            token="t",
            space_key="ENG",
            max_pages=1,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1


def test_pagination_hits_start_zero_then_start_two():
    srv, port = _spawn()
    try:
        loader = ConfluenceLoader.from_space_bearer(
            base_url=f"http://127.0.0.1:{port}",
            token="t",
            space_key="ENG",
        )
        loader.load()
    finally:
        srv.shutdown()
    # First request has start=0, second has start=2.
    starts = [p for p in _LAST_PATHS if p.startswith("/wiki/rest/api/content?")]
    assert any("start=0" in p for p in starts)
    assert any("start=2" in p for p in starts)


if __name__ == "__main__":
    fns = [
        test_cloud_space_loader_paginates_and_returns_one_doc_per_page,
        test_cloud_auth_uses_basic_header_with_base64_credentials,
        test_bearer_auth_uses_bearer_header,
        test_metadata_includes_id_title_space_key_version_source,
        test_from_pages_bearer_fetches_each_page_by_id,
        test_max_pages_caps_space_results,
        test_pagination_hits_start_zero_then_start_two,
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

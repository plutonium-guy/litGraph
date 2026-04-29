"""NotionLoader — pull pages from a Notion database OR a list of page ids
via the Notion REST API.

Verified against an inline fake Notion HTTP server that asserts auth +
version headers + dispatches canned JSON for db.query / blocks.children /
pages.get."""
import http.server
import json
import threading

from litgraph.loaders import NotionLoader


_LAST_AUTH = []
_LAST_VERSION = []
_LAST_PATHS = []


class _FakeNotion(http.server.BaseHTTPRequestHandler):
    def _capture(self):
        _LAST_AUTH.append(self.headers.get("Authorization"))
        _LAST_VERSION.append(self.headers.get("Notion-Version"))
        _LAST_PATHS.append(self.path)

    def _send(self, body):
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        self._capture()
        if "/databases/" in self.path and self.path.endswith("/query"):
            self._send({
                "results": [
                    {"id": "page-1", "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "Alpha"}]}
                    }},
                    {"id": "page-2", "properties": {
                        "Name": {"type": "title", "title": [{"plain_text": "Beta"}]}
                    }},
                ],
                "has_more": False, "next_cursor": None,
            })
        else:
            self._send({"error": "unknown post"})

    def do_GET(self):
        self._capture()
        if "/blocks/" in self.path and "/children" in self.path:
            self._send({
                "results": [
                    {"type": "heading_1",
                     "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
                    {"type": "paragraph",
                     "paragraph": {"rich_text": [{"plain_text": "Body"}]}},
                ],
                "has_more": False, "next_cursor": None,
            })
        elif self.path.startswith("/pages/"):
            self._send({
                "id": "page-direct",
                "properties": {"Name": {"type": "title", "title": [{"plain_text": "DirectTitle"}]}},
            })
        else:
            self._send({"error": "unknown get"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_AUTH.clear(); _LAST_VERSION.clear(); _LAST_PATHS.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeNotion)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_database_loader_returns_one_doc_per_page_with_title_metadata():
    srv, port = _spawn()
    try:
        loader = NotionLoader.from_database(
            api_key="secret_xyz",
            database_id="db-123",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    titles = sorted(d["metadata"]["title"] for d in docs)
    assert titles == ["Alpha", "Beta"]
    # Markdown affordances applied to blocks.
    for d in docs:
        assert "# Title" in d["content"]
        assert "Body" in d["content"]


def test_auth_and_notion_version_headers_set_on_every_request():
    srv, port = _spawn()
    try:
        loader = NotionLoader.from_database(
            api_key="secret_xyz",
            database_id="db",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    # 1 db query + 2 block fetches = 3 requests, all with auth + version.
    assert all(a == "Bearer secret_xyz" for a in _LAST_AUTH)
    assert all(v == "2022-06-28" for v in _LAST_VERSION)


def test_from_pages_loader_fetches_each_page_individually():
    srv, port = _spawn()
    try:
        loader = NotionLoader.from_pages(
            api_key="secret",
            page_ids=["p-A", "p-B"],
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    assert docs[0]["metadata"]["page_id"] == "p-A"
    assert docs[1]["metadata"]["page_id"] == "p-B"
    # Title pulled from /pages/{id}.
    assert docs[0]["metadata"]["title"] == "DirectTitle"


def test_max_pages_caps_database_results():
    srv, port = _spawn()
    try:
        loader = NotionLoader.from_database(
            api_key="x",
            database_id="db",
            base_url=f"http://127.0.0.1:{port}",
            max_pages=1,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1


def test_each_doc_has_source_metadata_with_notion_prefix():
    srv, port = _spawn()
    try:
        loader = NotionLoader.from_database(
            api_key="x",
            database_id="db",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    for d in docs:
        assert d["metadata"]["source"].startswith("notion:page-")


if __name__ == "__main__":
    fns = [
        test_database_loader_returns_one_doc_per_page_with_title_metadata,
        test_auth_and_notion_version_headers_set_on_every_request,
        test_from_pages_loader_fetches_each_page_individually,
        test_max_pages_caps_database_results,
        test_each_doc_has_source_metadata_with_notion_prefix,
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

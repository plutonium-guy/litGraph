"""S3Loader — AWS S3 bucket loader. SigV4-signed list + get, with
prefix/extensions/size/exclude filters. Tests use a ThreadingHTTPServer
fake that parses list-type=2 queries + dispatches object GETs by key."""
import http.server
import json
import threading

from litgraph.loaders import S3Loader


class _FakeS3(http.server.BaseHTTPRequestHandler):
    LIST_PAGES: list = []        # LIFO via pop()
    OBJECTS: dict = {}
    CAPTURED_PATHS: list = []

    def do_GET(self):
        self.CAPTURED_PATHS.append(self.path)
        if "list-type=2" in self.path:
            body = self.LIST_PAGES.pop() if self.LIST_PAGES else _empty_list_xml()
            out = body.encode()
            ct = "application/xml"
            status = 200
        else:
            # Path: /bucket/key(+optional query). Strip /bucket/ prefix.
            path = self.path.split("?", 1)[0]
            parts = path.split("/", 2)
            key = parts[2] if len(parts) >= 3 else ""
            # URL-decode the key.
            import urllib.parse
            key = urllib.parse.unquote(key)
            payload = self.OBJECTS.get(key)
            if payload is None:
                status = 404
                out = b"not found"
                ct = "text/plain"
            else:
                status = 200
                out = payload
                ct = "application/octet-stream"
        self.send_response(status)
        self.send_header("content-type", ct)
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _empty_list_xml():
    return """<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>testbucket</Name>
  <IsTruncated>false</IsTruncated>
</ListBucketResult>"""


def _list_xml(truncated: bool, next_token: str | None, keys: list[tuple]):
    contents = []
    for k, sz, lm, etag in keys:
        contents.append(
            f'<Contents><Key>{k}</Key><Size>{sz}</Size>'
            f'<LastModified>{lm}</LastModified><ETag>"{etag}"</ETag></Contents>'
        )
    nxt = f"<NextContinuationToken>{next_token}</NextContinuationToken>" if next_token else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
  <Name>testbucket</Name>
  <IsTruncated>{str(truncated).lower()}</IsTruncated>
  {nxt}
  {"".join(contents)}
</ListBucketResult>"""


def _spawn(list_pages, objects: dict):
    _FakeS3.LIST_PAGES = list(reversed(list_pages))
    _FakeS3.OBJECTS = objects
    _FakeS3.CAPTURED_PATHS = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeS3)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_single_page_load_returns_docs_with_metadata():
    list_xml = _list_xml(
        False, None,
        [("docs/one.txt", 11, "2026-04-01T00:00:00.000Z", "etag1")],
    )
    srv, port = _spawn(
        [list_xml],
        {"docs/one.txt": b"hello world"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA_TEST",
            secret_access_key="secret",
            region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    d = docs[0]
    assert d["content"] == "hello world"
    assert d["metadata"]["key"] == "docs/one.txt"
    # docs_to_pylist coerces non-string metadata to str-repr for uniform JSON.
    assert d["metadata"]["size"] == "11"
    assert d["metadata"]["etag"] == "etag1"
    assert d["metadata"]["source"] == "s3://testbucket/docs/one.txt"
    assert d["id"] == "s3://testbucket/docs/one.txt"


def test_extension_filter_skips_unwanted_keys():
    srv, port = _spawn(
        [_list_xml(False, None, [
            ("a.txt", 1, "", "e1"),
            ("b.png", 10, "", "e2"),
            ("c.md", 2, "", "e3"),
        ])],
        {"a.txt": b"a", "b.png": bytes([0x89, 0x50]), "c.md": b"c"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
            extensions=["txt", ".md"],
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    assert not any("/b.png" in p for p in _FakeS3.CAPTURED_PATHS if "list-type" not in p)


def test_default_excludes_filter_node_modules_and_git():
    srv, port = _spawn(
        [_list_xml(False, None, [
            ("src/app.txt", 1, "", "e"),
            ("node_modules/x.txt", 1, "", "e"),
            (".git/HEAD", 1, "", "e"),
        ])],
        {
            "src/app.txt": b"ok",
            "node_modules/x.txt": b"skip",
            ".git/HEAD": b"skip",
        },
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["key"] == "src/app.txt"


def test_size_cap_halts_fetch_of_large_keys():
    srv, port = _spawn(
        [_list_xml(False, None, [
            ("small.txt", 10, "", "e"),
            ("big.txt", 100_000_000, "", "e"),
        ])],
        {"small.txt": b"ok", "big.txt": b"x"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
            max_file_size_bytes=1024,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    # big.txt never fetched.
    assert not any("/big.txt" in p for p in _FakeS3.CAPTURED_PATHS if "list-type" not in p)


def test_prefix_appears_in_list_query():
    srv, port = _spawn(
        [_list_xml(False, None, [("docs/a.txt", 1, "", "e")])],
        {"docs/a.txt": b"a"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
            prefix="docs/",
        )
        loader.load()
    finally:
        srv.shutdown()
    list_path = next(p for p in _FakeS3.CAPTURED_PATHS if "list-type=2" in p)
    assert "prefix=docs%2F" in list_path


def test_paginates_via_continuation_token():
    p1 = _list_xml(True, "tok1", [("page1.txt", 1, "", "e1")])
    p2 = _list_xml(False, None, [("page2.txt", 1, "", "e2")])
    srv, port = _spawn(
        [p1, p2],
        {"page1.txt": b"p1", "page2.txt": b"p2"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    lists = [p for p in _FakeS3.CAPTURED_PATHS if "list-type" in p]
    assert len(lists) == 2
    assert "continuation-token=tok1" in lists[1]


def test_max_files_halts_pagination_early():
    p1 = _list_xml(True, "tok1", [
        ("a.txt", 1, "", "e1"),
        ("b.txt", 1, "", "e2"),
    ])
    srv, port = _spawn([p1], {"a.txt": b"a", "b.txt": b"b"})
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
            max_files=1,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    lists = [p for p in _FakeS3.CAPTURED_PATHS if "list-type" in p]
    assert len(lists) == 1


def test_non_utf8_objects_silently_skipped():
    srv, port = _spawn(
        [_list_xml(False, None, [
            ("binary.bin", 4, "", "e1"),
            ("text.txt", 5, "", "e2"),
        ])],
        {"binary.bin": bytes([0xFF, 0xFE, 0x00, 0x01]), "text.txt": b"hello"},
    )
    try:
        loader = S3Loader(
            access_key_id="AKIA", secret_access_key="s", region="us-east-1",
            bucket="testbucket",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["key"] == "text.txt"


def test_session_token_passes_through_for_sts_scoped_creds():
    """Temporary STS creds include a session_token → must be accepted."""
    srv, port = _spawn([_empty_list_xml()], {})
    try:
        loader = S3Loader(
            access_key_id="ASIA_STS",
            secret_access_key="s",
            region="us-east-1",
            bucket="testbucket",
            session_token="FwoGZXIvYXdzEAIaD...",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert docs == []


if __name__ == "__main__":
    import traceback
    fns = [
        test_single_page_load_returns_docs_with_metadata,
        test_extension_filter_skips_unwanted_keys,
        test_default_excludes_filter_node_modules_and_git,
        test_size_cap_halts_fetch_of_large_keys,
        test_prefix_appears_in_list_query,
        test_paginates_via_continuation_token,
        test_max_files_halts_pagination_early,
        test_non_utf8_objects_silently_skipped,
        test_session_token_passes_through_for_sts_scoped_creds,
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

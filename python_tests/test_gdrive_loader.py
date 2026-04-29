"""GoogleDriveLoader — pull files via Drive REST API. Google native formats
use /export with the right target MIME; plain textual files use alt=media;
binary files skipped by default."""
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import GoogleDriveLoader


_LAST_PATHS = []
_LAST_AUTH = []


class _FakeDrive(http.server.BaseHTTPRequestHandler):
    def _send_json(self, body):
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def _send_text(self, body):
        b = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/plain")
        self.send_header("content-length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        _LAST_AUTH.append(self.headers.get("Authorization"))
        _LAST_PATHS.append(self.path)
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)

        # /files/{id}/export?mimeType=...
        if parsed.path.endswith("/export") and "/files/" in parsed.path:
            fid = parsed.path.split("/files/")[1].split("/")[0]
            target = q.get("mimeType", ["text/plain"])[0]
            body = f"Exported {fid} as {target}"
            self._send_text(body)
            return
        # /files/{id}?alt=media
        if "/files/" in parsed.path and q.get("alt", [""])[0] == "media":
            fid = parsed.path.rsplit("/", 1)[-1]
            self._send_text(f"Raw media {fid}")
            return
        # /files?... (list)
        if parsed.path.endswith("/files"):
            if q.get("pageToken", [""])[0] == "NEXT":
                self._send_json({
                    "files": [{
                        "id": "txt1", "name": "readme.txt",
                        "mimeType": "text/plain",
                        "modifiedTime": "2024-01-05T12:00:00Z",
                        "webViewLink": "https://drive.google.com/file/d/txt1",
                    }]
                })
            else:
                self._send_json({
                    "nextPageToken": "NEXT",
                    "files": [
                        {"id": "doc1", "name": "Design Doc",
                         "mimeType": "application/vnd.google-apps.document",
                         "modifiedTime": "2024-01-01T12:00:00Z",
                         "parents": ["folder_x"],
                         "webViewLink": "https://docs.google.com/document/d/doc1"},
                        {"id": "sheet1", "name": "Budget",
                         "mimeType": "application/vnd.google-apps.spreadsheet",
                         "modifiedTime": "2024-01-02T12:00:00Z",
                         "webViewLink": "https://docs.google.com/spreadsheets/d/sheet1"},
                        {"id": "png1", "name": "chart.png",
                         "mimeType": "image/png",
                         "modifiedTime": "2024-01-03T12:00:00Z"},
                    ]
                })
            return
        self._send_json({"error": "unknown"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_PATHS.clear(); _LAST_AUTH.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeDrive)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_paginates_and_returns_one_doc_per_readable_file():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="ya29.test",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # doc1 + sheet1 + txt1 = 3 readable (png1 is binary, skipped by default).
    assert len(docs) == 3
    names = sorted(d["metadata"]["name"] for d in docs)
    assert names == ["Budget", "Design Doc", "readme.txt"]


def test_google_native_docs_exported_as_text_plain():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    # Doc export URL includes mimeType=text/plain (encoded).
    assert any(
        "/files/doc1/export" in p and "mimeType=text%2Fplain" in p
        for p in _LAST_PATHS
    )


def test_google_sheets_exported_as_text_csv():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    assert any(
        "/files/sheet1/export" in p and "mimeType=text%2Fcsv" in p
        for p in _LAST_PATHS
    )


def test_plain_files_use_alt_media_not_export():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    # txt1 uses alt=media.
    assert any("/files/txt1" in p and "alt=media" in p for p in _LAST_PATHS)
    # txt1 does NOT use export.
    assert not any("/files/txt1/export" in p for p in _LAST_PATHS)


def test_binaries_skipped_by_default():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # PNG not downloaded.
    assert not any("/files/png1" in p for p in _LAST_PATHS if "alt=media" in p or "export" in p)
    # And not in docs.
    names = [d["metadata"]["name"] for d in docs]
    assert "chart.png" not in names


def test_include_binaries_adds_metadata_only_docs_without_fetching_content():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            include_binaries=True,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    png = next(d for d in docs if d["metadata"]["name"] == "chart.png")
    # Content is just the name as H1 (metadata-only).
    assert png["content"] == "# chart.png\n"
    # No binary-content fetch happened.
    assert not any("/files/png1" in p and ("alt=media" in p or "export" in p) for p in _LAST_PATHS)


def test_metadata_captures_id_name_mime_modified_parents_web_view_link():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    doc1 = next(d for d in docs if d["metadata"]["file_id"] == "doc1")
    assert doc1["metadata"]["name"] == "Design Doc"
    assert doc1["metadata"]["mime_type"] == "application/vnd.google-apps.document"
    assert doc1["metadata"]["modified_time"] == "2024-01-01T12:00:00Z"
    assert doc1["metadata"]["parents"] == "folder_x"
    assert doc1["metadata"]["web_view_link"].startswith("https://docs")
    assert doc1["metadata"]["source"] == "gdrive:doc1"
    assert doc1["id"] == "gdrive:doc1"
    assert doc1["content"].startswith("# Design Doc")


def test_folder_id_and_query_combine_in_effective_q_parameter():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            folder_id="FOLDER_ABC",
            query="name contains 'design'",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    list_path = next(p for p in _LAST_PATHS if p.startswith("/files?"))
    assert "q=" in list_path
    # URL-encoded; decode + compare on substrings.
    from urllib.parse import unquote
    decoded = unquote(list_path)
    assert "'FOLDER_ABC' in parents" in decoded
    assert "name contains 'design'" in decoded
    assert " and " in decoded


def test_mime_types_allowlist_filters_readable_files():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="t",
            mime_types=["application/vnd.google-apps.document"],
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["name"] == "Design Doc"


def test_bearer_auth_on_list_export_and_media_requests():
    srv, port = _spawn()
    try:
        loader = GoogleDriveLoader(
            access_token="ya29.SECRET",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    for a in _LAST_AUTH:
        assert a == "Bearer ya29.SECRET"


if __name__ == "__main__":
    fns = [
        test_paginates_and_returns_one_doc_per_readable_file,
        test_google_native_docs_exported_as_text_plain,
        test_google_sheets_exported_as_text_csv,
        test_plain_files_use_alt_media_not_export,
        test_binaries_skipped_by_default,
        test_include_binaries_adds_metadata_only_docs_without_fetching_content,
        test_metadata_captures_id_name_mime_modified_parents_web_view_link,
        test_folder_id_and_query_combine_in_effective_q_parameter,
        test_mime_types_allowlist_filters_readable_files,
        test_bearer_auth_on_list_export_and_media_requests,
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

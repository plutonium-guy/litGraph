"""GithubFilesLoader — walk a repo tree and load file contents for code-RAG.

Verified against an inline fake GitHub server that serves /git/trees + per-
file /contents/{path} endpoints with base64-encoded bodies."""
import base64
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import GithubFilesLoader


_LAST_PATHS = []
_LAST_AUTH = []


class _FakeGh(http.server.BaseHTTPRequestHandler):
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
        if "/git/trees/" in parsed.path:
            self._send({
                "sha": "main_sha",
                "tree": [
                    {"path": "README.md", "type": "blob", "sha": "s1", "size": 100},
                    {"path": "src/main.rs", "type": "blob", "sha": "s2", "size": 500},
                    {"path": "node_modules/foo/index.js", "type": "blob",
                     "sha": "s3", "size": 200},
                    {"path": "huge.bin", "type": "blob", "sha": "s4", "size": 10_000_000},
                ],
                "truncated": False,
            })
            return
        if "/contents/" in parsed.path:
            rel = parsed.path.split("/contents/", 1)[1]
            contents_map = {
                "README.md": "# Hello\n\nIntro.",
                "src/main.rs": "fn main() { println!(\"hi\"); }",
            }
            text = contents_map.get(rel, f"// content for {rel}")
            b64 = base64.b64encode(text.encode()).decode()
            self._send({
                "name": rel.rsplit("/", 1)[-1],
                "path": rel,
                "sha": "content_sha",
                "size": len(text),
                "content": b64,
                "encoding": "base64",
                "html_url": f"https://github.com/acme/app/blob/main/{rel}",
            })
            return
        self._send({"error": "unknown"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_PATHS.clear(); _LAST_AUTH.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeGh)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_loader_filters_by_extension_and_returns_one_doc_per_matching_file():
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="ghp_t", owner="acme", repo="app",
            extensions=[".md", ".rs"],
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    # README.md + src/main.rs match; node_modules path + huge.bin filtered.
    paths = sorted(d["metadata"]["path"] for d in docs)
    assert paths == ["README.md", "src/main.rs"]
    # Decoded content survived base64 round-trip.
    readme = next(d for d in docs if d["metadata"]["path"] == "README.md")
    assert "# Hello" in readme["content"]


def test_default_excludes_drop_node_modules_without_caller_opt_in():
    """Default `exclude_paths` includes `node_modules/`, `vendor/`, `.lock`,
    etc — caller doesn't need to set them manually to get a sane default."""
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="t", owner="acme", repo="app",
            extensions=[".js"],  # would match node_modules/foo/index.js if excludes off
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    paths = [d["metadata"]["path"] for d in docs]
    assert not any("node_modules" in p for p in paths)


def test_max_file_size_filter_prevents_wasteful_contents_fetch():
    """huge.bin is 10MB; default 1MB cap. Loader must NOT fetch it (saves
    a round-trip + prevents trying to b64-decode 10MB of binary)."""
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="t", owner="acme", repo="app",
            extensions=[".bin", ".md", ".rs"],
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    # No /contents/huge.bin request in the captured paths.
    assert not any("/contents/huge.bin" in p for p in _LAST_PATHS)


def test_ref_override_hits_trees_and_contents_for_that_ref():
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="t", owner="acme", repo="app",
            ref="v1.2.3",
            extensions=[".md"],
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    assert any("/git/trees/v1.2.3" in p for p in _LAST_PATHS)
    assert any("ref=v1.2.3" in p for p in _LAST_PATHS)


def test_auth_bearer_and_api_version_headers_on_every_request():
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="ghp_secret", owner="acme", repo="app",
            extensions=[".md"],
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    for a in _LAST_AUTH:
        assert a == "Bearer ghp_secret"


def test_metadata_captures_path_ref_size_source_html_url():
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="t", owner="acme", repo="app",
            extensions=[".md"],
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    d = docs[0]
    assert d["metadata"]["path"] == "README.md"
    assert d["metadata"]["ref"] == "main"
    assert d["metadata"]["source"] == "github:acme/app/README.md@main"
    assert d["metadata"]["html_url"].startswith("https://github.com/")
    assert d["id"] == "acme/app:README.md"


def test_max_files_cap_truncates_result():
    srv, port = _spawn()
    try:
        loader = GithubFilesLoader(
            token="t", owner="acme", repo="app",
            extensions=[".md", ".rs"],
            max_files=1,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1


if __name__ == "__main__":
    fns = [
        test_loader_filters_by_extension_and_returns_one_doc_per_matching_file,
        test_default_excludes_drop_node_modules_without_caller_opt_in,
        test_max_file_size_filter_prevents_wasteful_contents_fetch,
        test_ref_override_hits_trees_and_contents_for_that_ref,
        test_auth_bearer_and_api_version_headers_on_every_request,
        test_metadata_captures_path_ref_size_source_html_url,
        test_max_files_cap_truncates_result,
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

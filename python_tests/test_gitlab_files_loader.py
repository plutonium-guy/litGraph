"""GitLabFilesLoader — repo file → Document via GitLab REST API.
Parallel to GithubFilesLoader for self-hosted GitLab code-RAG."""
import base64
import http.server
import json
import threading

from litgraph.loaders import GitLabFilesLoader


class _FakeGitLab(http.server.BaseHTTPRequestHandler):
    CAPTURED: list = []
    RESPONSES: list = []
    INDEX = [0]

    def do_GET(self):
        self.CAPTURED.append({
            "path": self.path,
            "private_token": self.headers.get("PRIVATE-TOKEN", ""),
            "auth": self.headers.get("Authorization", ""),
        })
        idx = self.INDEX[0]
        self.INDEX[0] += 1
        status, body = self.RESPONSES[idx] if idx < len(self.RESPONSES) else (200, "[]")
        out = body.encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(responses):
    _FakeGitLab.CAPTURED = []
    _FakeGitLab.RESPONSES = responses
    _FakeGitLab.INDEX = [0]
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeGitLab)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _tree_entry(path):
    return {"id": "abc", "name": path.split("/")[-1], "type": "blob", "path": path, "mode": "100644"}


def _file_response(path, body, size=None):
    encoded = base64.b64encode(body.encode()).decode()
    return {
        "file_name": path.split("/")[-1],
        "file_path": path,
        "size": size if size is not None else len(body),
        "encoding": "base64",
        "content": encoded,
        "blob_id": f"blob-{path}",
        "last_commit_id": "commit-abc",
        "ref": "main",
    }


def test_loads_files_from_tree():
    tree = json.dumps([_tree_entry("README.md"), _tree_entry("src/main.rs")])
    f1 = json.dumps(_file_response("README.md", "# Hello"))
    f2 = json.dumps(_file_response("src/main.rs", "fn main() {}"))
    srv, port = _spawn([(200, tree), (200, f1), (200, f2)])
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    contents = sorted(d["content"] for d in docs)
    assert contents == ["# Hello", "fn main() {}"]


def test_private_token_auth_by_default():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabFilesLoader(
            token="my-pat", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    assert _FakeGitLab.CAPTURED[0]["private_token"] == "my-pat"
    assert _FakeGitLab.CAPTURED[0]["auth"] == ""


def test_oauth_mode_uses_bearer():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabFilesLoader(
            token="oauth-tok", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            oauth=True,
        ).load()
    finally:
        srv.shutdown()
    assert _FakeGitLab.CAPTURED[0]["auth"] == "Bearer oauth-tok"


def test_extension_filter_skips_non_matching_paths():
    tree = json.dumps([
        _tree_entry("README.md"),
        _tree_entry("src/main.rs"),
        _tree_entry("config.json"),
    ])
    f1 = json.dumps(_file_response("README.md", "rd"))
    srv, port = _spawn([(200, tree), (200, f1)])
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            extensions=[".md"],
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["path"] == "README.md"


def test_default_excludes_skip_node_modules_and_lockfiles():
    tree = json.dumps([
        _tree_entry("src/main.rs"),
        _tree_entry("node_modules/foo/index.js"),
        _tree_entry("Cargo.lock"),
    ])
    f1 = json.dumps(_file_response("src/main.rs", "ok"))
    srv, port = _spawn([(200, tree), (200, f1)])
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["path"] == "src/main.rs"


def test_max_files_caps_total_loaded():
    tree = json.dumps([_tree_entry(f"f{i}.md") for i in range(5)])
    files = [(200, json.dumps(_file_response(f"f{i}.md", "x"))) for i in range(5)]
    srv, port = _spawn([(200, tree)] + files)
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            max_files=3,
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 3


def test_oversized_file_skipped_after_fetch():
    tree = json.dumps([_tree_entry("small.txt"), _tree_entry("big.txt")])
    f_small = json.dumps(_file_response("small.txt", "x"))
    f_big = json.dumps(_file_response("big.txt", "y" * 100, size=2_000_000))
    srv, port = _spawn([(200, tree), (200, f_small), (200, f_big)])
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            max_file_size_bytes=1024 * 1024,
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["path"] == "small.txt"


def test_url_encodes_filepath_with_slashes():
    """File-fetch URL must encode `/` as `%2F` (GitLab spec)."""
    tree = json.dumps([_tree_entry("src/sub/main.rs")])
    f1 = json.dumps(_file_response("src/sub/main.rs", "code"))
    srv, port = _spawn([(200, tree), (200, f1)])
    try:
        GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    second_req = _FakeGitLab.CAPTURED[1]
    assert "src%2Fsub%2Fmain.rs" in second_req["path"]


def test_metadata_carries_full_provenance():
    tree = json.dumps([_tree_entry("file.txt")])
    f1 = json.dumps(_file_response("file.txt", "hello"))
    srv, port = _spawn([(200, tree), (200, f1)])
    try:
        docs = GitLabFilesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    md = docs[0]["metadata"]
    assert md["path"] == "file.txt"
    assert md["ref"] == "main"
    assert md["blob_id"] == "blob-file.txt"
    assert str(md["size"]) == "5"
    assert md["last_commit_id"] == "commit-abc"
    assert md["source"] == "gitlab:12345/file.txt@main"


def test_http_error_on_tree_surfaces_as_runtime_error():
    srv, port = _spawn([(401, json.dumps({"message": "401 Unauthorized"}))])
    try:
        GitLabFilesLoader(
            token="bad", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "401" in str(e)
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_loads_files_from_tree,
        test_private_token_auth_by_default,
        test_oauth_mode_uses_bearer,
        test_extension_filter_skips_non_matching_paths,
        test_default_excludes_skip_node_modules_and_lockfiles,
        test_max_files_caps_total_loaded,
        test_oversized_file_skipped_after_fetch,
        test_url_encodes_filepath_with_slashes,
        test_metadata_carries_full_provenance,
        test_http_error_on_tree_surfaces_as_runtime_error,
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

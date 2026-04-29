"""GithubIssuesLoader — pull issues + PRs from a GitHub repo via REST API.

Verified against an inline fake GitHub HTTP server that asserts auth +
X-GitHub-Api-Version + serves canned /repos/{owner}/{repo}/issues JSON."""
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.loaders import GithubIssuesLoader


_LAST_AUTH = []
_LAST_VERSION = []
_LAST_PATHS = []


class _FakeGithub(http.server.BaseHTTPRequestHandler):
    def _send(self, body):
        out = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_GET(self):
        _LAST_AUTH.append(self.headers.get("Authorization"))
        _LAST_VERSION.append(self.headers.get("X-GitHub-Api-Version"))
        _LAST_PATHS.append(self.path)
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path.endswith("/comments"):
            self._send([
                {"body": "confirmed", "user": {"login": "carol"}},
                {"body": "on it", "user": {"login": "dave"}},
            ])
        elif "/issues" in parsed.path:
            page = int(q.get("page", ["1"])[0])
            if page == 1:
                self._send([
                    {
                        "number": 1, "title": "Bug: X crashes", "state": "open",
                        "body": "Repro:\n1. click\n2. boom",
                        "user": {"login": "alice"},
                        "labels": [{"name": "bug"}, {"name": "p1"}],
                        "created_at": "2025-01-01T00:00:00Z",
                        "updated_at": "2025-01-02T00:00:00Z",
                        "html_url": "https://github.com/acme/app/issues/1",
                        "comments": 2,
                    },
                    {
                        "number": 2, "title": "Feature Y", "state": "closed",
                        "body": "Implement Y for Z.",
                        "user": {"login": "bob"},
                        "labels": [{"name": "feature"}],
                        "created_at": "2025-01-03T00:00:00Z",
                        "updated_at": "2025-01-04T00:00:00Z",
                        "html_url": "https://github.com/acme/app/pull/2",
                        "comments": 0,
                        "pull_request": {"url": "..."},
                    },
                ])
            else:
                self._send([])
        else:
            self._send({"error": "unknown"})

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST_AUTH.clear(); _LAST_VERSION.clear(); _LAST_PATHS.clear()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeGithub)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_basic_loader_returns_one_doc_per_issue():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="ghp_test", owner="acme", repo="app",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    assert docs[0]["content"].startswith("# Bug: X crashes")
    assert "Repro:" in docs[0]["content"]


def test_auth_and_api_version_headers_set_on_every_request():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="ghp_secret", owner="acme", repo="app",
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    for a in _LAST_AUTH:
        assert a == "Bearer ghp_secret"
    for v in _LAST_VERSION:
        assert v == "2022-11-28"


def test_metadata_captures_issue_attrs_and_pr_flag():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="t", owner="acme", repo="app",
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    d0, d1 = docs[0], docs[1]
    # All metadata values stringified by docs_to_pylist.
    assert d0["metadata"]["number"] == "1"
    assert d0["metadata"]["title"] == "Bug: X crashes"
    assert d0["metadata"]["state"] == "open"
    assert d0["metadata"]["user"] == "alice"
    assert d0["metadata"]["labels"] == "bug,p1"
    assert d0["metadata"]["is_pull_request"] == "false"
    assert d0["metadata"]["source"] == "github:acme/app#1"
    assert d0["id"] == "acme/app#1"
    # Issue #2 is a PR.
    assert d1["metadata"]["is_pull_request"] == "true"


def test_include_comments_inlines_threads_under_separator():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="t", owner="acme", repo="app",
            include_comments=True,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert "--- comments ---" in docs[0]["content"]
    assert "[@carol]: confirmed" in docs[0]["content"]
    assert "[@dave]: on it" in docs[0]["content"]


def test_include_comments_skips_fetch_when_comments_count_zero():
    """Zero-comment issues must NOT trigger a comments round-trip —
    otherwise a repo with 1000 comment-less PRs burns 1000 pointless calls."""
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="t", owner="acme", repo="app",
            include_comments=True,
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    comment_paths = [p for p in _LAST_PATHS if "/comments" in p]
    # Only issue #1 (comments=2) triggers a fetch; #2 (comments=0) skipped.
    assert len(comment_paths) == 1
    assert "/issues/1/comments" in comment_paths[0]


def test_state_and_labels_filters_appear_in_query():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="t", owner="acme", repo="app",
            state="open", labels=["bug", "p1"],
            base_url=f"http://127.0.0.1:{port}",
        )
        loader.load()
    finally:
        srv.shutdown()
    issue_path = next(p for p in _LAST_PATHS if "/issues?" in p)
    assert "state=open" in issue_path
    assert "labels=bug,p1" in issue_path


def test_max_issues_caps_result():
    srv, port = _spawn()
    try:
        loader = GithubIssuesLoader(
            token="t", owner="acme", repo="app",
            max_issues=1,
            base_url=f"http://127.0.0.1:{port}",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1


if __name__ == "__main__":
    fns = [
        test_basic_loader_returns_one_doc_per_issue,
        test_auth_and_api_version_headers_set_on_every_request,
        test_metadata_captures_issue_attrs_and_pr_flag,
        test_include_comments_inlines_threads_under_separator,
        test_include_comments_skips_fetch_when_comments_count_zero,
        test_state_and_labels_filters_appear_in_query,
        test_max_issues_caps_result,
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

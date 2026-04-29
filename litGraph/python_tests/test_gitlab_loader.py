"""GitLabIssuesLoader — pull issues from GitLab REST API. PRIVATE-TOKEN
auth (default) or OAuth Bearer (opt-in). Self-hosted via base_url."""
import http.server
import json
import threading

from litgraph.loaders import GitLabIssuesLoader


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


def _issue(iid, title, body, state="opened"):
    return {
        "iid": iid, "id": iid * 1000, "title": title,
        "description": body, "state": state,
        "author": {"username": "alice"},
        "labels": ["bug", "p1"],
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "web_url": f"https://gitlab.com/x/y/-/issues/{iid}",
    }


def test_loads_one_page_of_issues():
    body = json.dumps([_issue(1, "first bug", "body 1"), _issue(2, "second bug", "body 2", "closed")])
    srv, port = _spawn([(200, body)])
    try:
        docs = GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
        assert len(docs) == 2
        assert "# first bug" in docs[0]["content"]
        assert docs[0]["metadata"]["title"] == "first bug"
        assert docs[1]["metadata"]["state"] == "closed"
    finally:
        srv.shutdown()


def test_private_token_header_used_by_default():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabIssuesLoader(
            token="my-pat", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    captured = _FakeGitLab.CAPTURED[0]
    assert captured["private_token"] == "my-pat"
    assert captured["auth"] == ""


def test_oauth_mode_uses_bearer_auth():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabIssuesLoader(
            token="oauth-tok", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            oauth=True,
        ).load()
    finally:
        srv.shutdown()
    captured = _FakeGitLab.CAPTURED[0]
    assert captured["auth"] == "Bearer oauth-tok"
    assert captured["private_token"] == ""


def test_state_open_translated_to_opened():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            state="open",  # GitHub-style spelling — translated
        ).load()
    finally:
        srv.shutdown()
    assert "state=opened" in _FakeGitLab.CAPTURED[0]["path"]


def test_labels_filter_appended():
    srv, port = _spawn([(200, "[]")])
    try:
        GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            labels=["bug", "p1"],
        ).load()
    finally:
        srv.shutdown()
    assert "labels=bug,p1" in _FakeGitLab.CAPTURED[0]["path"]


def test_include_notes_appends_user_comments_skips_system():
    issue_body = _issue(7, "needs reply", "the question")
    notes = [
        {"author": {"username": "bob"}, "body": "i'll look", "system": False},
        {"author": {"username": "ghost"}, "body": "added label bug", "system": True},
        {"author": {"username": "carol"}, "body": "fixed it", "system": False},
    ]
    srv, port = _spawn([
        (200, json.dumps([issue_body])),
        (200, json.dumps(notes)),
    ])
    try:
        docs = GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            include_notes=True,
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    c = docs[0]["content"]
    assert "--- notes ---" in c
    assert "[@bob]: i'll look" in c
    assert "[@carol]: fixed it" in c
    assert "added label" not in c  # system note filtered


def test_max_issues_caps_results():
    body = json.dumps([_issue(i, f"b{i}", "x") for i in range(1, 4)])
    srv, port = _spawn([(200, body)])
    try:
        docs = GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
            max_issues=2,
        ).load()
    finally:
        srv.shutdown()
    assert len(docs) == 2


def test_metadata_carries_iid_state_labels_url():
    body = json.dumps([_issue(42, "the issue", "body")])
    srv, port = _spawn([(200, body)])
    try:
        docs = GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    md = docs[0]["metadata"]
    assert str(md["iid"]) == "42"
    assert md["state"] == "opened"
    assert md["author"] == "alice"
    assert md["labels"] == "bug,p1"
    assert md["web_url"] == "https://gitlab.com/x/y/-/issues/42"
    assert md["source"] == "gitlab:12345#42"


def test_document_id_includes_project_and_iid():
    body = json.dumps([_issue(7, "x", "y")])
    srv, port = _spawn([(200, body)])
    try:
        docs = GitLabIssuesLoader(
            token="tk", project="group%2Fmyproject",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
    finally:
        srv.shutdown()
    assert docs[0]["id"] == "group%2Fmyproject#7"


def test_http_error_surfaces_with_status_and_body():
    srv, port = _spawn([(401, json.dumps({"message": "401 Unauthorized"}))])
    try:
        GitLabIssuesLoader(
            token="bad", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "401" in str(e)
        assert "Unauthorized" in str(e)
    finally:
        srv.shutdown()


def test_self_hosted_via_base_url():
    """Self-hosted GitLab — base_url points at the on-prem instance."""
    body = json.dumps([_issue(1, "a", "b")])
    srv, port = _spawn([(200, body)])
    try:
        # Fake "self-hosted" on a different port pretending to be the corp server.
        docs = GitLabIssuesLoader(
            token="tk", project="12345",
            base_url=f"http://127.0.0.1:{port}/api/v4",
        ).load()
        assert len(docs) == 1
    finally:
        srv.shutdown()


if __name__ == "__main__":
    import traceback
    fns = [
        test_loads_one_page_of_issues,
        test_private_token_header_used_by_default,
        test_oauth_mode_uses_bearer_auth,
        test_state_open_translated_to_opened,
        test_labels_filter_appended,
        test_include_notes_appends_user_comments_skips_system,
        test_max_issues_caps_results,
        test_metadata_carries_iid_state_labels_url,
        test_document_id_includes_project_and_iid,
        test_http_error_surfaces_with_status_and_body,
        test_self_hosted_via_base_url,
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

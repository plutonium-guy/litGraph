"""JiraIssuesLoader — REST v3 loader for Jira Cloud + Data Center.
Basic-auth (Cloud: email+token) OR Bearer (DC: PAT). JQL-driven.
Extracts plain text from ADF descriptions."""
import base64
import http.server
import json
import threading

from litgraph.loaders import JiraIssuesLoader


class _FakeJira(http.server.BaseHTTPRequestHandler):
    PAGES: list = []
    CAPTURED_BODIES: list = []
    CAPTURED_AUTH: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(json.loads(body))
        self.CAPTURED_AUTH.append(self.headers.get("Authorization", ""))
        page_body = self.PAGES.pop() if self.PAGES else _empty_page()
        out = page_body.encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _empty_page():
    return json.dumps({
        "issues": [], "total": 0, "startAt": 0, "maxResults": 50,
    })


def _adf_paragraph(text):
    return {
        "type": "doc", "version": 1,
        "content": [{
            "type": "paragraph",
            "content": [{"type": "text", "text": text}],
        }],
    }


def _issue(key, summary, description_text):
    return {
        "id": "10001",
        "key": key,
        "fields": {
            "summary": summary,
            "description": _adf_paragraph(description_text),
            "status": {"name": "In Progress"},
            "priority": {"name": "High"},
            "issuetype": {"name": "Bug"},
            "assignee": {"displayName": "Alice"},
            "reporter": {"displayName": "Bob"},
            "labels": ["backend", "urgent"],
            "components": [{"name": "api"}, {"name": "auth"}],
            "created": "2026-04-01T10:00:00Z",
            "updated": "2026-04-02T12:00:00Z",
        },
    }


def _page(issues, total):
    return json.dumps({
        "issues": issues, "total": total, "startAt": 0, "maxResults": 50,
    })


def _spawn(pages):
    _FakeJira.PAGES = list(reversed(pages))
    _FakeJira.CAPTURED_BODIES = []
    _FakeJira.CAPTURED_AUTH = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeJira)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cloud_auth_sends_basic_with_email_token():
    srv, port = _spawn([_page([_issue("ENG-1", "fix", "body")], 1)])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = ENG",
            email="me@co.com",
            api_token="tok123",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    auth = _FakeJira.CAPTURED_AUTH[0]
    assert auth.startswith("Basic ")
    expected = base64.b64encode(b"me@co.com:tok123").decode()
    assert auth == f"Basic {expected}"


def test_bearer_token_for_data_center():
    srv, port = _spawn([_page([_issue("DC-1", "dc", "x")], 1)])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = DC",
            bearer_token="pat_abc",
        )
        loader.load()
    finally:
        srv.shutdown()
    assert _FakeJira.CAPTURED_AUTH[0] == "Bearer pat_abc"


def test_neither_auth_raises_runtime_error():
    try:
        JiraIssuesLoader(base_url="http://x", jql="project = X")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "email" in str(e).lower() or "bearer" in str(e).lower()


def test_both_auth_types_rejected():
    try:
        JiraIssuesLoader(
            base_url="http://x", jql="project = X",
            email="e", api_token="t",
            bearer_token="bt",
        )
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "not both" in str(e).lower() or "either" in str(e).lower()


def test_request_body_carries_jql_and_fields_list():
    srv, port = _spawn([_page([_issue("ENG-1", "t", "b")], 1)])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = ENG AND status = Open",
            email="e", api_token="t",
        )
        loader.load()
    finally:
        srv.shutdown()
    body = _FakeJira.CAPTURED_BODIES[0]
    assert body["jql"] == "project = ENG AND status = Open"
    assert body["startAt"] == 0
    assert body["maxResults"] == 50
    fields = body["fields"]
    assert "summary" in fields
    assert "status" in fields
    assert "assignee" in fields


def test_content_has_markdown_h1_plus_adf_extracted_text():
    srv, port = _spawn([_page([_issue("ENG-1", "fix the auth bug", "Root cause details.")], 1)])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = ENG", email="e", api_token="t",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert docs[0]["content"].startswith("# ENG-1 fix the auth bug")
    assert "Root cause details." in docs[0]["content"]


def test_metadata_fields_complete():
    srv, port = _spawn([_page([_issue("ENG-5", "topic", "d")], 1)])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = ENG", email="e", api_token="t",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    md = docs[0]["metadata"]
    assert md["issue_key"] == "ENG-5"
    assert md["status"] == "In Progress"
    assert md["priority"] == "High"
    assert md["issuetype"] == "Bug"
    assert md["assignee"] == "Alice"
    assert md["reporter"] == "Bob"
    assert md["labels"] == "backend, urgent"
    assert md["components"] == "api, auth"
    assert md["source"] == "jira:ENG-5"
    assert docs[0]["id"] == "jira:ENG-5"
    assert md["url"].endswith("/browse/ENG-5")


def test_paginates_through_multiple_pages():
    p1 = json.dumps({
        "issues": [_issue("E-1", "t", "x"), _issue("E-2", "t", "x")],
        "total": 3, "startAt": 0, "maxResults": 2,
    })
    p2 = json.dumps({
        "issues": [_issue("E-3", "t", "x")],
        "total": 3, "startAt": 2, "maxResults": 2,
    })
    srv, port = _spawn([p1, p2])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = E", email="e", api_token="t",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    keys = [d["metadata"]["issue_key"] for d in docs]
    assert keys == ["E-1", "E-2", "E-3"]
    assert len(_FakeJira.CAPTURED_BODIES) == 2


def test_max_issues_cap_and_halt():
    srv, port = _spawn([_page(
        [_issue("E-1", "t", "x"), _issue("E-2", "t", "x"), _issue("E-3", "t", "x")],
        100,
    )])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = E", email="e", api_token="t",
            max_issues=2,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    assert len(_FakeJira.CAPTURED_BODIES) == 1


def test_empty_issues_returns_empty_list():
    srv, port = _spawn([_empty_page()])
    try:
        loader = JiraIssuesLoader(
            base_url=f"http://127.0.0.1:{port}",
            jql="project = NONE", email="e", api_token="t",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert docs == []


if __name__ == "__main__":
    import traceback
    fns = [
        test_cloud_auth_sends_basic_with_email_token,
        test_bearer_token_for_data_center,
        test_neither_auth_raises_runtime_error,
        test_both_auth_types_rejected,
        test_request_body_carries_jql_and_fields_list,
        test_content_has_markdown_h1_plus_adf_extracted_text,
        test_metadata_fields_complete,
        test_paginates_through_multiple_pages,
        test_max_issues_cap_and_halt,
        test_empty_issues_returns_empty_list,
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

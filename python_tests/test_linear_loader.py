"""LinearIssuesLoader — GraphQL-backed SaaS loader for Linear.app issues.
First loader in the framework that uses GraphQL (all others are REST)."""
import http.server
import json
import threading

from litgraph.loaders import LinearIssuesLoader


class _FakeLinear(http.server.BaseHTTPRequestHandler):
    PAGES: list = []   # consumed newest-first via pop()
    CAPTURED_BODIES: list = []
    CAPTURED_AUTH: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.CAPTURED_BODIES.append(json.loads(body))
        self.CAPTURED_AUTH.append(self.headers.get("Authorization", ""))
        payload = self.PAGES.pop() if self.PAGES else _empty_page()
        out = payload.encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _empty_page():
    return json.dumps({
        "data": {"issues": {"pageInfo": {"hasNextPage": False}, "nodes": []}}
    })


def _page(has_next: bool, cursor: str, issues: list[tuple]):
    nodes = []
    for i, (ident, title, desc) in enumerate(issues):
        nodes.append({
            "id": f"id_{i}",
            "identifier": ident,
            "title": title,
            "description": desc,
            "url": f"https://linear.app/team/issue/{ident}",
            "createdAt": "2026-04-01T00:00:00Z",
            "updatedAt": "2026-04-02T00:00:00Z",
            "state": {"name": "In Progress"},
            "team": {"name": "Engineering", "key": "ENG"},
            "labels": {"nodes": [{"name": "bug"}, {"name": "p1"}]},
        })
    return json.dumps({
        "data": {
            "issues": {
                "pageInfo": {"endCursor": cursor, "hasNextPage": has_next},
                "nodes": nodes,
            }
        }
    })


def _spawn(pages):
    # PAGES consumed via pop() = last-in first-out; reverse so the first
    # request sees the first page.
    _FakeLinear.PAGES = list(reversed(pages))
    _FakeLinear.CAPTURED_BODIES = []
    _FakeLinear.CAPTURED_AUTH = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLinear)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_loads_single_page_builds_documents():
    srv, port = _spawn([_page(False, "cursor0", [("ENG-1", "first", "body")])])
    try:
        loader = LinearIssuesLoader(
            api_key="lin_secret",
            base_url=f"http://127.0.0.1:{port}/graphql",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["metadata"]["identifier"] == "ENG-1"
    assert "# ENG-1 first" in docs[0]["content"]
    assert "body" in docs[0]["content"]
    # Linear-quirk: Authorization has the raw key, NO "Bearer".
    assert _FakeLinear.CAPTURED_AUTH[0] == "lin_secret"
    assert not _FakeLinear.CAPTURED_AUTH[0].startswith("Bearer")


def test_paginates_across_multiple_pages():
    srv, port = _spawn([
        _page(True, "c0", [("ENG-1", "a", "a")]),
        _page(True, "c1", [("ENG-2", "b", "b")]),
        _page(False, "c2", [("ENG-3", "c", "c")]),
    ])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 3
    idents = [d["metadata"]["identifier"] for d in docs]
    assert idents == ["ENG-1", "ENG-2", "ENG-3"]
    assert len(_FakeLinear.CAPTURED_BODIES) == 3


def test_max_issues_caps_result_and_halts_pagination_early():
    srv, port = _spawn([
        _page(True, "c0", [("ENG-1", "a", "a")]),
        _page(True, "c1", [("ENG-2", "b", "b")]),
        _page(True, "c2", [("ENG-3", "c", "c")]),
    ])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
            max_issues=2,
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert len(docs) == 2
    # Third page never fetched.
    assert len(_FakeLinear.CAPTURED_BODIES) == 2


def test_filters_stack_in_graphql_variables():
    srv, port = _spawn([_page(False, "x", [("ENG-1", "t", "b")])])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
            team_key="ENG",
            state_names=["Todo", "In Progress"],
            label_names=["bug"],
        )
        loader.load()
    finally:
        srv.shutdown()
    req = _FakeLinear.CAPTURED_BODIES[0]
    flt = req["variables"]["filter"]
    assert flt["team"]["key"]["eq"] == "ENG"
    assert flt["state"]["name"]["in"] == ["Todo", "In Progress"]
    assert flt["labels"]["some"]["name"]["in"] == ["bug"]


def test_graphql_errors_surface():
    srv, port = _spawn([json.dumps({"errors": [{"message": "Access denied"}]})])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
        )
        try:
            loader.load()
            raise AssertionError("expected RuntimeError")
        except RuntimeError as e:
            assert "Access denied" in str(e)
    finally:
        srv.shutdown()


def test_metadata_fields_populated():
    srv, port = _spawn([_page(False, "x", [("ENG-7", "topic", "details")])])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    md = docs[0]["metadata"]
    assert md["identifier"] == "ENG-7"
    assert md["state_name"] == "In Progress"
    assert md["team_name"] == "Engineering"
    assert md["labels"] == "bug, p1"
    assert md["source"] == "linear:ENG-7"
    assert docs[0]["id"] == "linear:ENG-7"
    assert md["url"].startswith("https://linear.app/")


def test_empty_response_returns_empty_list():
    srv, port = _spawn([_empty_page()])
    try:
        loader = LinearIssuesLoader(
            api_key="k",
            base_url=f"http://127.0.0.1:{port}/graphql",
        )
        docs = loader.load()
    finally:
        srv.shutdown()
    assert docs == []


if __name__ == "__main__":
    import traceback
    fns = [
        test_loads_single_page_builds_documents,
        test_paginates_across_multiple_pages,
        test_max_issues_caps_result_and_halts_pagination_early,
        test_filters_stack_in_graphql_variables,
        test_graphql_errors_surface,
        test_metadata_fields_populated,
        test_empty_response_returns_empty_list,
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

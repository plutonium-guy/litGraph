"""WeaviateVectorStore — REST + GraphQL v1 API. Verified against an inline
fake HTTP server that asserts request shape + returns canned responses."""
import http.server
import json
import threading

from litgraph.retrieval import WeaviateVectorStore


_LAST = {"path": None, "body": None}


class _Fake(http.server.BaseHTTPRequestHandler):
    def _read_body(self):
        n = int(self.headers.get("content-length", "0"))
        return self.rfile.read(n)

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        body = self._read_body()
        _LAST["path"] = self.path
        try:
            req = json.loads(body) if body else None
        except Exception:
            req = None
        _LAST["body"] = req
        if self.path.startswith("/v1/batch/objects"):
            objects = (req or {}).get("objects", [])
            results = [{"id": o["id"], "result": {"status": "SUCCESS"}} for o in objects]
            self._send_json(200, results)
        elif self.path.startswith("/v1/graphql"):
            self._send_json(200, {
                "data": {"Get": {"Article": [{
                    "__content": "alpha",
                    "topic": "rust",
                    "_additional": {
                        "id": "00000000-0000-0000-0000-000000000001",
                        "distance": 0.1,
                    },
                }]}}
            })
        elif self.path.startswith("/v1/schema"):
            self._send_json(200, {"class": "Article"})
        else:
            self._send_json(404, {"error": "unknown"})

    def do_DELETE(self):
        _LAST["path"] = self.path
        self.send_response(204)
        self.send_header("content-length", "0")
        self.end_headers()

    def log_message(self, *a, **kw): pass


def _spawn():
    _LAST["path"] = None
    _LAST["body"] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_weaviate_add_uses_batch_objects_endpoint():
    srv, port = _spawn()
    try:
        store = WeaviateVectorStore(f"http://127.0.0.1:{port}", "Article")
        ids = store.add(
            [{"content": "alpha"}, {"content": "beta"}],
            [[0.1, 0.2], [0.3, 0.4]],
        )
        assert len(ids) == 2
        assert _LAST["path"] == "/v1/batch/objects"
        objects = _LAST["body"]["objects"]
        assert len(objects) == 2
        assert objects[0]["class"] == "Article"
        assert objects[0]["properties"]["__content"] == "alpha"
    finally:
        srv.shutdown()


def test_weaviate_similarity_search_emits_graphql_get_with_nearvector():
    srv, port = _spawn()
    try:
        store = WeaviateVectorStore(f"http://127.0.0.1:{port}", "Article")
        hits = store.similarity_search([0.1, 0.2], k=3)
        assert len(hits) == 1
        assert hits[0]["content"] == "alpha"
        # Distance 0.1 → score 0.9.
        assert abs(float(hits[0]["score"]) - 0.9) < 1e-5
        assert _LAST["path"] == "/v1/graphql"
        q = _LAST["body"]["query"]
        assert "nearVector" in q
        assert "Article" in q
        assert "limit: 3" in q
    finally:
        srv.shutdown()


def test_weaviate_filter_dict_becomes_where_clause():
    srv, port = _spawn()
    try:
        store = WeaviateVectorStore(f"http://127.0.0.1:{port}", "Article")
        store.similarity_search([0.1, 0.2], k=3, filter={"topic": "rust"})
        q = _LAST["body"]["query"]
        assert "where:" in q
        assert "Equal" in q
        assert "topic" in q
        assert "rust" in q
    finally:
        srv.shutdown()


def test_weaviate_delete_issues_delete_per_id():
    srv, port = _spawn()
    try:
        store = WeaviateVectorStore(f"http://127.0.0.1:{port}", "Article")
        store.delete(["uuid-a", "uuid-b"])
        # Last delete URL is per-class per-id.
        assert _LAST["path"].startswith("/v1/objects/Article/")
    finally:
        srv.shutdown()


def test_weaviate_repeat_upsert_with_same_caller_id_yields_same_uuid():
    """Deterministic UUIDv5 over (class, caller-id) — repeat upserts overwrite
    rather than duplicate. This is the idempotency guarantee we promise."""
    srv, port = _spawn()
    try:
        store = WeaviateVectorStore(f"http://127.0.0.1:{port}", "Article")
        ids1 = store.add([{"content": "alpha", "id": "a"}], [[0.1, 0.2]])
        ids2 = store.add([{"content": "alpha", "id": "a"}], [[0.1, 0.2]])
        assert ids1 == ids2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_weaviate_add_uses_batch_objects_endpoint,
        test_weaviate_similarity_search_emits_graphql_get_with_nearvector,
        test_weaviate_filter_dict_becomes_where_clause,
        test_weaviate_delete_issues_delete_per_id,
        test_weaviate_repeat_upsert_with_same_caller_id_yields_same_uuid,
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

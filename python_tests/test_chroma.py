"""ChromaVectorStore against a fake Chroma v1 HTTP server.

Verifies lazy collection get-or-create + caching, add round-trip, query
with filter, delete, and acceptance by VectorRetriever."""
import http.server
import json
import threading

from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import ChromaVectorStore, VectorRetriever


class FakeChroma(http.server.BaseHTTPRequestHandler):
    """Scripted Chroma server: handles create-collection, add, query, delete.
    Caches the collection id internally; subsequent create-collection calls
    return the same id (matches `get_or_create=true` semantics)."""
    REQUESTS = []  # list of (method, path, body)
    COLLECTION_ID = "col-uuid-deadbeef"

    def _send_json(self, body):
        out = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body_bytes = self.rfile.read(n)
        body = body_bytes.decode() if body_bytes else ""
        FakeChroma.REQUESTS.append(("POST", self.path, body))
        if self.path.startswith("/api/v1/collections?"):
            self._send_json({"id": FakeChroma.COLLECTION_ID, "name": "test"})
        elif self.path.endswith("/add"):
            self._send_json({})
        elif self.path.endswith("/query"):
            req = json.loads(body)
            n_results = req.get("n_results", 1)
            # Deterministic stub response: first n_results docs.
            ids_pool = ["a", "b", "c", "d"]
            docs_pool = ["doc a", "doc b", "doc c", "doc d"]
            metas_pool = [{"k": "v0"}, {"k": "v1"}, {"k": "v2"}, {"k": "v3"}]
            dists_pool = [0.05, 0.20, 0.50, 1.00]
            self._send_json({
                "ids": [ids_pool[:n_results]],
                "documents": [docs_pool[:n_results]],
                "metadatas": [metas_pool[:n_results]],
                "distances": [dists_pool[:n_results]],
            })
        elif self.path.endswith("/delete"):
            self._send_json({})
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeChroma.REQUESTS = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeChroma)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_add_caches_collection_id_then_query_skips_create():
    srv, port = _spawn()
    try:
        store = ChromaVectorStore(url=f"http://127.0.0.1:{port}", collection="test")
        # First call → triggers ensure_collection_id (cached after).
        ids = store.add(
            [{"content": "doc a", "id": "a"}, {"content": "doc b", "id": "b"}],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        assert ids == ["a", "b"]
        # Second call → re-uses cached id (no new POST /collections?).
        hits = store.similarity_search([1.0, 0.0], k=2)
        assert len(hits) == 2
        assert hits[0]["id"] == "a"
        assert hits[0]["content"] == "doc a"
        # docs_to_pylist stringifies score; just check it's present + numeric-looking.
        assert "0.05" in str(hits[0].get("score", ""))

        # Verify exactly ONE create-collection request.
        creates = [r for r in FakeChroma.REQUESTS
                   if r[0] == "POST" and r[1].startswith("/api/v1/collections?")]
        assert len(creates) == 1, f"got {len(creates)} create calls"
    finally:
        srv.shutdown()


def test_add_with_no_explicit_id_works():
    srv, port = _spawn()
    try:
        store = ChromaVectorStore(url=f"http://127.0.0.1:{port}", collection="t")
        ids = store.add([{"content": "no id"}], [[0.5, 0.5]])
        assert len(ids) == 1
        # Should be a UUID-shaped string.
        assert len(ids[0]) >= 32
    finally:
        srv.shutdown()


def test_query_with_filter_passes_where_clause():
    srv, port = _spawn()
    try:
        store = ChromaVectorStore(url=f"http://127.0.0.1:{port}", collection="t")
        store.similarity_search([1.0, 0.0], k=1, filter={"source": "alpha"})
        # The query body should contain the where clause.
        query_req = next(r for r in FakeChroma.REQUESTS if r[1].endswith("/query"))
        body = json.loads(query_req[2])
        assert body["where"]["source"] == "alpha"
        assert body["n_results"] == 1
    finally:
        srv.shutdown()


def test_delete_sends_ids_array():
    srv, port = _spawn()
    try:
        store = ChromaVectorStore(url=f"http://127.0.0.1:{port}", collection="t")
        # Trigger collection creation first.
        store.add([{"content": "x", "id": "1"}], [[0.0]])
        store.delete(["1", "2"])
        delete_req = next(r for r in FakeChroma.REQUESTS if r[1].endswith("/delete"))
        body = json.loads(delete_req[2])
        assert body["ids"] == ["1", "2"]
    finally:
        srv.shutdown()


def test_vector_retriever_accepts_chroma_store():
    srv, port = _spawn()
    try:
        def embed(texts):
            return [[float(len(t)), 0.0] for t in texts]
        e = FunctionEmbeddings(embed, dimensions=2, name="toy")
        store = ChromaVectorStore(url=f"http://127.0.0.1:{port}", collection="t")
        store.add(
            [{"content": "doc a", "id": "a"}, {"content": "doc bb", "id": "b"}],
            embed(["doc a", "doc bb"]),
        )
        retriever = VectorRetriever(e, store)
        hits = retriever.retrieve("query", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


def test_custom_tenant_and_database_in_url():
    srv, port = _spawn()
    try:
        store = ChromaVectorStore(
            url=f"http://127.0.0.1:{port}", collection="t",
            tenant="my-tenant", database="my-db",
        )
        store.add([{"content": "x", "id": "1"}], [[0.0]])
        # Verify query string carries the tenant+db.
        create_req = next(r for r in FakeChroma.REQUESTS if r[1].startswith("/api/v1/collections?"))
        assert "tenant=my-tenant" in create_req[1]
        assert "database=my-db" in create_req[1]
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_add_caches_collection_id_then_query_skips_create,
        test_add_with_no_explicit_id_works,
        test_query_with_filter_passes_where_clause,
        test_delete_sends_ids_array,
        test_vector_retriever_accepts_chroma_store,
        test_custom_tenant_and_database_in_url,
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

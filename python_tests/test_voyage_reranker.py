"""VoyageReranker against a fake Voyage /v1/rerank server.

Verifies wire shape (Voyage uses `top_k`+`truncation`, NOT Cohere's `top_n`),
score → Document.score mapping, and acceptance by RerankingRetriever.
"""
import http.server
import json
import threading

from litgraph.retrieval import (
    VoyageReranker, RerankingRetriever, MemoryVectorStore, VectorRetriever,
)
from litgraph.embeddings import FunctionEmbeddings


class FakeVoyage(http.server.BaseHTTPRequestHandler):
    SCRIPT = []  # ordered list of `data` lists
    LAST_BODY = [None]
    LAST_PATH = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeVoyage.LAST_BODY[0] = body
        FakeVoyage.LAST_PATH[0] = self.path
        data = FakeVoyage.SCRIPT.pop(0) if FakeVoyage.SCRIPT else []
        resp = json.dumps({
            "object": "list", "data": data,
            "model": "rerank-2", "usage": {"total_tokens": 42},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)
    def log_message(self, *a, **kw): pass


def _spawn():
    FakeVoyage.LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVoyage)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_voyage_reranker_reorders_and_writes_score():
    FakeVoyage.SCRIPT = [[
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.71},
        {"index": 1, "relevance_score": 0.30},
    ]]
    srv, port = _spawn()
    try:
        r = VoyageReranker(api_key="voy-fake", model="rerank-2",
                           base_url=f"http://127.0.0.1:{port}/v1")
        docs = [
            {"content": "first", "id": "a"},
            {"content": "second", "id": "b"},
            {"content": "third", "id": "c"},
        ]
        out = r.rerank("question", docs, 3)
        assert [d["id"] for d in out] == ["c", "a", "b"]
        assert abs(out[0]["score"] - 0.95) < 1e-3
        assert FakeVoyage.LAST_PATH[0] == "/v1/rerank"
        sent = json.loads(FakeVoyage.LAST_BODY[0])
        assert sent["model"] == "rerank-2"
        assert sent["query"] == "question"
        # Voyage uses top_k + truncation, NOT Cohere's top_n.
        assert sent["top_k"] == 3
        assert sent["truncation"] is True
        assert "top_n" not in sent
    finally:
        srv.shutdown()


def test_voyage_truncation_off_propagates():
    FakeVoyage.SCRIPT = [[]]
    srv, port = _spawn()
    try:
        r = VoyageReranker(api_key="x", model="rerank-2-lite",
                           base_url=f"http://127.0.0.1:{port}/v1",
                           truncation=False)
        r.rerank("q", [{"content": "hi", "id": "x"}], 1)
        sent = json.loads(FakeVoyage.LAST_BODY[0])
        assert sent["truncation"] is False
        assert sent["model"] == "rerank-2-lite"
    finally:
        srv.shutdown()


def test_voyage_empty_docs_no_http():
    # base_url unreachable → if we made an HTTP call, this would error.
    r = VoyageReranker(api_key="x", base_url="http://127.0.0.1:1/v1")
    out = r.rerank("q", [], 5)
    assert out == []


def test_reranking_retriever_accepts_voyage_two_stage():
    """VectorRetriever first-stage → Voyage rerank → top_k returned."""
    FakeVoyage.SCRIPT = [[
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.42},
    ]]
    srv, port = _spawn()
    try:
        def embed(texts):
            vocab = ["foo", "bar", "baz"]
            return [[float(t.lower().count(w)) for w in vocab] for t in texts]
        e = FunctionEmbeddings(embed, dimensions=3, name="bow")
        store = MemoryVectorStore()
        docs = [
            {"content": "foo foo", "id": "x"},
            {"content": "bar baz baz", "id": "y"},
            {"content": "baz", "id": "z"},
        ]
        store.add(docs, embed([d["content"] for d in docs]))

        base = VectorRetriever(e, store)
        voyage = VoyageReranker(api_key="voy-fake",
                                base_url=f"http://127.0.0.1:{port}/v1")
        retriever = RerankingRetriever(base, voyage, over_fetch_k=3)
        hits = retriever.retrieve("baz", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


def test_reranking_retriever_rejects_non_reranker():
    """RerankingRetriever should TypeError on something that's not a known reranker."""
    def embed(texts):
        return [[1.0, 0.0] for _ in texts]
    e = FunctionEmbeddings(embed, dimensions=2, name="x")
    store = MemoryVectorStore()
    base = VectorRetriever(e, store)
    try:
        RerankingRetriever(base, "not a reranker", over_fetch_k=3)  # type: ignore[arg-type]
    except TypeError as exc:
        assert "reranker must be" in str(exc)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_voyage_reranker_reorders_and_writes_score,
        test_voyage_truncation_off_propagates,
        test_voyage_empty_docs_no_http,
        test_reranking_retriever_accepts_voyage_two_stage,
        test_reranking_retriever_rejects_non_reranker,
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

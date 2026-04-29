"""CohereReranker + RerankingRetriever against a fake Cohere /v2/rerank server."""
import http.server
import json
import threading

from litgraph.retrieval import (
    CohereReranker, RerankingRetriever, MemoryVectorStore, VectorRetriever,
)
from litgraph.embeddings import FunctionEmbeddings


class FakeCohere(http.server.BaseHTTPRequestHandler):
    SCRIPT = []  # ordered list of `results` lists

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        results = FakeCohere.SCRIPT.pop(0) if FakeCohere.SCRIPT else []
        body = json.dumps({"results": results}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeCohere)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cohere_reranker_reorders_by_score():
    FakeCohere.SCRIPT = [[
        {"index": 2, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.71},
        {"index": 1, "relevance_score": 0.30},
    ]]
    srv, port = _spawn()
    try:
        r = CohereReranker(api_key="co-fake", model="rerank-english-v3.0",
                           base_url=f"http://127.0.0.1:{port}")
        docs = [
            {"content": "first", "id": "a"},
            {"content": "second", "id": "b"},
            {"content": "third", "id": "c"},
        ]
        out = r.rerank("question", docs, 3)
        assert [d["id"] for d in out] == ["c", "a", "b"]
        assert abs(out[0]["score"] - 0.95) < 1e-3
    finally:
        srv.shutdown()


def test_reranking_retriever_two_stage():
    """VectorRetriever pulls candidates → Cohere reorders → top_k returned."""
    # Cohere will reorder so that doc index 1 wins out of the 3 candidates.
    FakeCohere.SCRIPT = [[
        {"index": 1, "relevance_score": 0.95},
        {"index": 0, "relevance_score": 0.42},
    ]]
    srv, port = _spawn()
    try:
        # First-stage retrieval: tiny BOW embedder + memory store
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
        cohere = CohereReranker(api_key="co-fake",
                                base_url=f"http://127.0.0.1:{port}")
        retriever = RerankingRetriever(base, cohere, over_fetch_k=3)

        hits = retriever.retrieve("baz", k=2)
        assert len(hits) == 2
        # The reranker chose index 1 first (whatever VectorRetriever had ranked
        # second in its initial pull).
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_cohere_reranker_reorders_by_score,
        test_reranking_retriever_two_stage,
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

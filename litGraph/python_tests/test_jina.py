"""Jina AI — embeddings + reranker against fake servers."""
import http.server
import json
import threading

from litgraph.embeddings import JinaEmbeddings
from litgraph.retrieval import (
    JinaReranker, MemoryVectorStore, RerankingRetriever, VectorRetriever,
)


# ─── Fake servers ───────────────────────────────────────────────────────────

class FakeJinaEmbed(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeJinaEmbed.LAST_BODY[0] = body
        req = json.loads(body)
        inputs = req["input"]
        data = [{"object":"embedding", "embedding":[float(len(t)), float(t.count(" ")), float(i), 1.0], "index": i}
                for i, t in enumerate(inputs)]
        resp = json.dumps({"object":"list", "data": data, "model": req["model"], "usage":{"total_tokens": 0}}).encode()
        self.send_response(200); self.send_header("content-type","application/json")
        self.send_header("content-length", str(len(resp))); self.end_headers(); self.wfile.write(resp)
    def log_message(self, *a, **kw): pass


class FakeJinaRerank(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeJinaRerank.LAST_BODY[0] = body
        req = json.loads(body)
        top_n = req.get("top_n", 10)
        # Static reorder: index 2 wins, then 0, then 1; honor top_n.
        all_results = [
            {"index": 2, "relevance_score": 0.95},
            {"index": 0, "relevance_score": 0.71},
            {"index": 1, "relevance_score": 0.30},
        ]
        resp = json.dumps({"results": all_results[:top_n]}).encode()
        self.send_response(200); self.send_header("content-type","application/json")
        self.send_header("content-length", str(len(resp))); self.end_headers(); self.wfile.write(resp)
    def log_message(self, *a, **kw): pass


def _spawn(handler):
    handler.LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


# ─── Embeddings ─────────────────────────────────────────────────────────────

def test_jina_embed_query_sends_retrieval_query_task():
    srv, port = _spawn(FakeJinaEmbed)
    try:
        e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3", dimensions=4,
                           base_url=f"http://127.0.0.1:{port}/v1")
        v = e.embed_query("hello world")
        assert len(v) == 4 and v[0] == 11.0
        sent = json.loads(FakeJinaEmbed.LAST_BODY[0])
        assert sent["model"] == "jina-embeddings-v3"
        assert sent["input"] == ["hello world"]
        assert sent["task"] == "retrieval.query"
    finally:
        srv.shutdown()


def test_jina_embed_documents_sends_retrieval_passage_task():
    srv, port = _spawn(FakeJinaEmbed)
    try:
        e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3", dimensions=4,
                           base_url=f"http://127.0.0.1:{port}/v1")
        out = e.embed_documents(["a", "ab", "abc"])
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
        sent = json.loads(FakeJinaEmbed.LAST_BODY[0])
        assert sent["task"] == "retrieval.passage"
    finally:
        srv.shutdown()


def test_jina_output_dimensions_truncates_and_updates_dim():
    srv, port = _spawn(FakeJinaEmbed)
    try:
        e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3",
                           dimensions=1024, base_url=f"http://127.0.0.1:{port}/v1",
                           output_dimensions=256)
        assert e.dimensions == 256
        e.embed_query("hi")
        sent = json.loads(FakeJinaEmbed.LAST_BODY[0])
        assert sent["dimensions"] == 256
    finally:
        srv.shutdown()


def test_jina_task_override_to_classification():
    srv, port = _spawn(FakeJinaEmbed)
    try:
        e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3", dimensions=4,
                           base_url=f"http://127.0.0.1:{port}/v1",
                           task_document="classification",
                           task_query="classification")
        e.embed_query("hi")
        sent = json.loads(FakeJinaEmbed.LAST_BODY[0])
        assert sent["task"] == "classification"
    finally:
        srv.shutdown()


def test_jina_embeddings_empty_returns_empty():
    e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3", dimensions=4,
                       base_url="http://127.0.0.1:1/v1")
    assert e.embed_documents([]) == []


def test_vector_retriever_accepts_jina_embeddings():
    srv, port = _spawn(FakeJinaEmbed)
    try:
        e = JinaEmbeddings(api_key="k", model="jina-embeddings-v3", dimensions=4,
                           base_url=f"http://127.0.0.1:{port}/v1")
        store = MemoryVectorStore()
        docs = [{"content":"a","id":"1"},{"content":"abcd","id":"2"}]
        store.add(docs, e.embed_documents([d["content"] for d in docs]))
        retriever = VectorRetriever(e, store)
        hits = retriever.retrieve("abc", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


# ─── Reranker ────────────────────────────────────────────────────────────────

def test_jina_reranker_reorders_and_writes_score():
    srv, port = _spawn(FakeJinaRerank)
    try:
        r = JinaReranker(api_key="k", model="jina-reranker-v2-base-multilingual",
                         base_url=f"http://127.0.0.1:{port}/v1")
        docs = [
            {"content":"first","id":"a"},
            {"content":"second","id":"b"},
            {"content":"third","id":"c"},
        ]
        out = r.rerank("q", docs, 3)
        assert [d["id"] for d in out] == ["c", "a", "b"]
        assert abs(out[0]["score"] - 0.95) < 1e-3
        sent = json.loads(FakeJinaRerank.LAST_BODY[0])
        # Jina uses Cohere-style top_n.
        assert sent["top_n"] == 3
        assert "top_k" not in sent
    finally:
        srv.shutdown()


def test_reranking_retriever_accepts_jina():
    srv, port = _spawn(FakeJinaRerank)
    try:
        # Quick stub: BOW embedder + memory store + Jina rerank.
        from litgraph.embeddings import FunctionEmbeddings
        def embed(texts):
            vocab = ["foo", "bar", "baz"]
            return [[float(t.lower().count(w)) for w in vocab] for t in texts]
        e = FunctionEmbeddings(embed, dimensions=3, name="bow")
        store = MemoryVectorStore()
        docs = [
            {"content":"foo foo","id":"x"},
            {"content":"bar baz","id":"y"},
            {"content":"baz","id":"z"},
        ]
        store.add(docs, embed([d["content"] for d in docs]))
        base = VectorRetriever(e, store)
        jina = JinaReranker(api_key="k", base_url=f"http://127.0.0.1:{port}/v1")
        retriever = RerankingRetriever(base, jina, over_fetch_k=3)
        hits = retriever.retrieve("baz", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_jina_embed_query_sends_retrieval_query_task,
        test_jina_embed_documents_sends_retrieval_passage_task,
        test_jina_output_dimensions_truncates_and_updates_dim,
        test_jina_task_override_to_classification,
        test_jina_embeddings_empty_returns_empty,
        test_vector_retriever_accepts_jina_embeddings,
        test_jina_reranker_reorders_and_writes_score,
        test_reranking_retriever_accepts_jina,
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

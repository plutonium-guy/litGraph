"""Native CohereEmbeddings against a fake /v2/embed server."""
import http.server
import json
import threading

from litgraph.embeddings import CohereEmbeddings
from litgraph.retrieval import VectorRetriever, MemoryVectorStore


class FakeCohereEmbedHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    LAST_PATH = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeCohereEmbedHandler.LAST_BODY[0] = body
        FakeCohereEmbedHandler.LAST_PATH[0] = self.path
        req = json.loads(body)
        texts = req["texts"]
        # Deterministic 4-dim vector keyed off length + index.
        rows = []
        for i, t in enumerate(texts):
            rows.append([float(len(t)), float(t.count(" ")), float(i), 1.0])
        resp = json.dumps({
            "id": "fake",
            "embeddings": {"float": rows},
            "texts": texts,
            "model": req["model"],
            "meta": {"billed_units": {"input_tokens": 1}},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeCohereEmbedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_cohere_embeddings_embed_query_uses_search_query_input_type():
    srv, port = _spawn()
    try:
        e = CohereEmbeddings(api_key="x", model="embed-english-v3.0", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}")
        v = e.embed_query("hello world")
        assert len(v) == 4
        assert v[0] == 11.0  # len("hello world")
        sent = json.loads(FakeCohereEmbedHandler.LAST_BODY[0])
        assert sent["model"] == "embed-english-v3.0"
        assert sent["texts"] == ["hello world"]
        assert sent["input_type"] == "search_query"
        assert sent["embedding_types"] == ["float"]
        assert FakeCohereEmbedHandler.LAST_PATH[0] == "/v2/embed"
    finally:
        srv.shutdown()


def test_cohere_embeddings_embed_documents_uses_search_document_input_type():
    srv, port = _spawn()
    try:
        e = CohereEmbeddings(api_key="x", model="embed-english-v3.0", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}")
        out = e.embed_documents(["a", "ab", "abc"])
        assert len(out) == 3
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
        sent = json.loads(FakeCohereEmbedHandler.LAST_BODY[0])
        assert sent["input_type"] == "search_document"
    finally:
        srv.shutdown()


def test_cohere_embeddings_input_type_override():
    srv, port = _spawn()
    try:
        e = CohereEmbeddings(api_key="x", model="embed-multilingual-v3.0", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}",
                             input_type_document="classification",
                             input_type_query="classification")
        e.embed_query("hi")
        sent = json.loads(FakeCohereEmbedHandler.LAST_BODY[0])
        assert sent["input_type"] == "classification"
    finally:
        srv.shutdown()


def test_cohere_embeddings_embed_documents_empty_returns_empty():
    e = CohereEmbeddings(api_key="x", model="embed-english-v3.0", dimensions=4,
                         base_url="http://127.0.0.1:1")
    assert e.embed_documents([]) == []


def test_vector_retriever_accepts_cohere_embeddings():
    srv, port = _spawn()
    try:
        e = CohereEmbeddings(api_key="x", model="embed-english-v3.0", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}")
        store = MemoryVectorStore()
        docs = [{"content": "a", "id": "1"}, {"content": "abcd", "id": "2"}]
        embs = e.embed_documents([d["content"] for d in docs])
        store.add(docs, embs)

        retriever = VectorRetriever(e, store)
        hits = retriever.retrieve("abc", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_cohere_embeddings_embed_query_uses_search_query_input_type,
        test_cohere_embeddings_embed_documents_uses_search_document_input_type,
        test_cohere_embeddings_input_type_override,
        test_cohere_embeddings_embed_documents_empty_returns_empty,
        test_vector_retriever_accepts_cohere_embeddings,
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

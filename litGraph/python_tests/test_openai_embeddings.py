"""Native OpenAIEmbeddings against a fake /embeddings server."""
import http.server
import json
import threading

from litgraph.embeddings import OpenAIEmbeddings, FunctionEmbeddings
from litgraph.retrieval import VectorRetriever, MemoryVectorStore


class FakeEmbedHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeEmbedHandler.LAST_BODY[0] = body
        req = json.loads(body)
        inputs = req["input"] if isinstance(req["input"], list) else [req["input"]]
        # Deterministic: return a 4-dim vector based on input length.
        data = []
        for i, t in enumerate(inputs):
            vec = [float(len(t)), float(t.count(" ")), float(i), 1.0]
            data.append({"index": i, "embedding": vec, "object": "embedding"})
        resp = json.dumps({
            "object": "list",
            "data": data,
            "model": req["model"],
            "usage": {"prompt_tokens": sum(len(t.split()) for t in inputs), "total_tokens": 0},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeEmbedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_openai_embeddings_embed_query_returns_vector():
    srv, port = _spawn()
    try:
        e = OpenAIEmbeddings(api_key="x", model="text-embedding-3-small", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
        v = e.embed_query("hello world")
        assert len(v) == 4
        # First component = len("hello world") = 11
        assert v[0] == 11.0
        # Verify wire body shape
        sent = json.loads(FakeEmbedHandler.LAST_BODY[0])
        assert sent["model"] == "text-embedding-3-small"
        assert sent["input"] == ["hello world"]
    finally:
        srv.shutdown()


def test_openai_embeddings_batch_embed_documents():
    srv, port = _spawn()
    try:
        e = OpenAIEmbeddings(api_key="x", model="text-embedding-3-small", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
        out = e.embed_documents(["a", "ab", "abc"])
        assert len(out) == 3
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
    finally:
        srv.shutdown()


def test_openai_embeddings_with_override_dimensions_sends_dimensions_field():
    srv, port = _spawn()
    try:
        e = OpenAIEmbeddings(api_key="x", model="text-embedding-3-large",
                             dimensions=1536, base_url=f"http://127.0.0.1:{port}/v1",
                             override_dimensions=512)
        # dimensions getter should reflect the override
        assert e.dimensions == 512
        e.embed_query("hi")
        sent = json.loads(FakeEmbedHandler.LAST_BODY[0])
        assert sent["dimensions"] == 512
    finally:
        srv.shutdown()


def test_vector_retriever_accepts_openai_embeddings():
    srv, port = _spawn()
    try:
        e = OpenAIEmbeddings(api_key="x", model="text-embedding-3-small", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
        store = MemoryVectorStore()
        # Manually pre-embed + add (the fake embedder returns deterministic vectors).
        docs = [{"content": "a", "id": "1"}, {"content": "abcd", "id": "2"}]
        embs = e.embed_documents([d["content"] for d in docs])
        store.add(docs, embs)

        # VectorRetriever now accepts the OpenAI embedder directly (no Python callable).
        retriever = VectorRetriever(e, store)
        hits = retriever.retrieve("abc", k=2)
        assert len(hits) == 2
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_openai_embeddings_embed_query_returns_vector,
        test_openai_embeddings_batch_embed_documents,
        test_openai_embeddings_with_override_dimensions_sends_dimensions_field,
        test_vector_retriever_accepts_openai_embeddings,
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

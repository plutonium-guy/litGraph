"""Native VoyageEmbeddings against a fake /v1/embeddings server."""
import http.server
import json
import threading

from litgraph.embeddings import VoyageEmbeddings
from litgraph.retrieval import VectorRetriever, MemoryVectorStore


class FakeVoyageHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    LAST_PATH = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeVoyageHandler.LAST_BODY[0] = body
        FakeVoyageHandler.LAST_PATH[0] = self.path
        req = json.loads(body)
        inputs = req["input"]
        data = []
        for i, t in enumerate(inputs):
            vec = [float(len(t)), float(t.count(" ")), float(i), 1.0]
            data.append({"object": "embedding", "embedding": vec, "index": i})
        resp = json.dumps({
            "object": "list",
            "data": data,
            "model": req["model"],
            "usage": {"total_tokens": sum(len(t.split()) for t in inputs)},
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeVoyageHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_voyage_embed_query_sends_query_input_type():
    srv, port = _spawn()
    try:
        e = VoyageEmbeddings(api_key="x", model="voyage-3", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
        v = e.embed_query("hello world")
        assert len(v) == 4
        assert v[0] == 11.0  # len("hello world")
        sent = json.loads(FakeVoyageHandler.LAST_BODY[0])
        assert sent["model"] == "voyage-3"
        assert sent["input"] == ["hello world"]
        assert sent["input_type"] == "query"
        assert FakeVoyageHandler.LAST_PATH[0] == "/v1/embeddings"
    finally:
        srv.shutdown()


def test_voyage_embed_documents_sends_document_input_type():
    srv, port = _spawn()
    try:
        e = VoyageEmbeddings(api_key="x", model="voyage-3", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
        out = e.embed_documents(["a", "ab", "abc"])
        assert len(out) == 3
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
        sent = json.loads(FakeVoyageHandler.LAST_BODY[0])
        assert sent["input_type"] == "document"
    finally:
        srv.shutdown()


def test_voyage_input_type_none_omits_field():
    srv, port = _spawn()
    try:
        e = VoyageEmbeddings(api_key="x", model="voyage-3-large", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1",
                             input_type_document=None,
                             input_type_query=None)
        e.embed_query("hi")
        sent = json.loads(FakeVoyageHandler.LAST_BODY[0])
        assert "input_type" not in sent  # omitted entirely → Voyage default
    finally:
        srv.shutdown()


def test_voyage_embed_documents_empty_returns_empty():
    e = VoyageEmbeddings(api_key="x", model="voyage-3", dimensions=4,
                         base_url="http://127.0.0.1:1/v1")
    assert e.embed_documents([]) == []


def test_vector_retriever_accepts_voyage_embeddings():
    srv, port = _spawn()
    try:
        e = VoyageEmbeddings(api_key="x", model="voyage-3", dimensions=4,
                             base_url=f"http://127.0.0.1:{port}/v1")
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
        test_voyage_embed_query_sends_query_input_type,
        test_voyage_embed_documents_sends_document_input_type,
        test_voyage_input_type_none_omits_field,
        test_voyage_embed_documents_empty_returns_empty,
        test_vector_retriever_accepts_voyage_embeddings,
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

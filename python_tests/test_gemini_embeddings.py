"""Native GeminiEmbeddings against a fake batchEmbedContents server."""
import http.server
import json
import threading
from urllib.parse import urlparse, parse_qs

from litgraph.embeddings import GeminiEmbeddings
from litgraph.retrieval import VectorRetriever, MemoryVectorStore


class FakeGeminiEmbedHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]
    LAST_PATH = [None]
    LAST_QUERY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeGeminiEmbedHandler.LAST_BODY[0] = body
        parsed = urlparse(self.path)
        FakeGeminiEmbedHandler.LAST_PATH[0] = parsed.path
        FakeGeminiEmbedHandler.LAST_QUERY[0] = parse_qs(parsed.query)
        req = json.loads(body)
        # batchEmbedContents body: { "requests": [ {model, content:{parts:[{text}]}, task_type?, ...}, ... ] }
        items = req["requests"]
        embeddings = []
        for i, r in enumerate(items):
            text = r["content"]["parts"][0]["text"]
            vals = [float(len(text)), float(text.count(" ")), float(i), 1.0]
            embeddings.append({"values": vals})
        resp = json.dumps({"embeddings": embeddings}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def _spawn():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeGeminiEmbedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def test_gemini_embed_query_uses_retrieval_query_task_type():
    srv, port = _spawn()
    try:
        e = GeminiEmbeddings(api_key="test-key", model="text-embedding-004",
                             dimensions=4, base_url=f"http://127.0.0.1:{port}")
        v = e.embed_query("hello world")
        assert len(v) == 4
        assert v[0] == 11.0  # len("hello world")
        sent = json.loads(FakeGeminiEmbedHandler.LAST_BODY[0])
        assert len(sent["requests"]) == 1
        req0 = sent["requests"][0]
        assert req0["model"] == "models/text-embedding-004"
        assert req0["content"]["parts"][0]["text"] == "hello world"
        assert req0["task_type"] == "RETRIEVAL_QUERY"
        # API key must be in query string, not Bearer header
        assert FakeGeminiEmbedHandler.LAST_QUERY[0]["key"] == ["test-key"]
        # Path normalizes to v1beta/models/...:batchEmbedContents
        assert FakeGeminiEmbedHandler.LAST_PATH[0] == "/v1beta/models/text-embedding-004:batchEmbedContents"
    finally:
        srv.shutdown()


def test_gemini_embed_documents_uses_retrieval_document_task_type():
    srv, port = _spawn()
    try:
        e = GeminiEmbeddings(api_key="x", model="text-embedding-004",
                             dimensions=4, base_url=f"http://127.0.0.1:{port}")
        out = e.embed_documents(["a", "ab", "abc"])
        assert len(out) == 3
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
        sent = json.loads(FakeGeminiEmbedHandler.LAST_BODY[0])
        for r in sent["requests"]:
            assert r["task_type"] == "RETRIEVAL_DOCUMENT"
    finally:
        srv.shutdown()


def test_gemini_task_type_override_to_classification():
    srv, port = _spawn()
    try:
        e = GeminiEmbeddings(api_key="x", model="text-embedding-004",
                             dimensions=4, base_url=f"http://127.0.0.1:{port}",
                             task_type_document="CLASSIFICATION",
                             task_type_query="CLASSIFICATION")
        e.embed_query("hi")
        sent = json.loads(FakeGeminiEmbedHandler.LAST_BODY[0])
        assert sent["requests"][0]["task_type"] == "CLASSIFICATION"
    finally:
        srv.shutdown()


def test_gemini_output_dimensionality_sent_per_request_and_updates_dim():
    srv, port = _spawn()
    try:
        e = GeminiEmbeddings(api_key="x", model="text-embedding-004",
                             dimensions=768, base_url=f"http://127.0.0.1:{port}",
                             output_dimensionality=256)
        # `dimensions` getter should reflect override
        assert e.dimensions == 256
        e.embed_query("hi")
        sent = json.loads(FakeGeminiEmbedHandler.LAST_BODY[0])
        assert sent["requests"][0]["output_dimensionality"] == 256
    finally:
        srv.shutdown()


def test_gemini_embed_documents_empty_returns_empty():
    e = GeminiEmbeddings(api_key="x", model="text-embedding-004", dimensions=4,
                         base_url="http://127.0.0.1:1")
    assert e.embed_documents([]) == []


def test_vector_retriever_accepts_gemini_embeddings():
    srv, port = _spawn()
    try:
        e = GeminiEmbeddings(api_key="x", model="text-embedding-004",
                             dimensions=4, base_url=f"http://127.0.0.1:{port}")
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
        test_gemini_embed_query_uses_retrieval_query_task_type,
        test_gemini_embed_documents_uses_retrieval_document_task_type,
        test_gemini_task_type_override_to_classification,
        test_gemini_output_dimensionality_sent_per_request_and_updates_dim,
        test_gemini_embed_documents_empty_returns_empty,
        test_vector_retriever_accepts_gemini_embeddings,
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

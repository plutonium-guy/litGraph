"""Native BedrockEmbeddings against a fake /model/{id}/invoke server.

Covers both wire-format dispatchers:
  - Titan: single inputText per request, parallel batching for embed_documents
  - Cohere-on-Bedrock: batched texts array, single round-trip
"""
import http.server
import json
import threading
from urllib.parse import urlparse

from litgraph.embeddings import BedrockEmbeddings
from litgraph.retrieval import VectorRetriever, MemoryVectorStore


class FakeTitanHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODIES = []
    LAST_PATHS = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeTitanHandler.LAST_BODIES.append(body)
        FakeTitanHandler.LAST_PATHS.append(urlparse(self.path).path)
        # Verify SigV4 signing happened (we don't validate, just check presence).
        assert "authorization" in {k.lower() for k in self.headers.keys()}
        req = json.loads(body)
        text = req["inputText"]
        vec = [float(len(text)), float(text.count(" ")), 0.0, 1.0]
        resp = json.dumps({"embedding": vec, "inputTextTokenCount": len(text.split())}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


class FakeCohereOnBedrockHandler(http.server.BaseHTTPRequestHandler):
    LAST_BODY = [None]

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        FakeCohereOnBedrockHandler.LAST_BODY[0] = body
        req = json.loads(body)
        texts = req["texts"]
        rows = [[float(len(t)), float(t.count(" ")), float(i), 1.0] for i, t in enumerate(texts)]
        resp = json.dumps({
            "embeddings": {"float": rows},
            "id": "fake",
            "response_type": "embeddings_by_type",
            "texts": texts,
        }).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, *a, **kw): pass


def _spawn(handler):
    if hasattr(handler, "LAST_BODIES"):
        handler.LAST_BODIES = []
    if hasattr(handler, "LAST_PATHS"):
        handler.LAST_PATHS = []
    if hasattr(handler, "LAST_BODY"):
        handler.LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _new_titan(port, **kw):
    return BedrockEmbeddings(
        access_key_id="AKIDEXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        model_id="amazon.titan-embed-text-v2:0",
        dimensions=4,
        endpoint_override=f"http://127.0.0.1:{port}",
        **kw,
    )


def _new_cohere_bedrock(port, **kw):
    return BedrockEmbeddings(
        access_key_id="AKIDEXAMPLE",
        secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
        region="us-east-1",
        model_id="cohere.embed-english-v3",
        dimensions=4,
        endpoint_override=f"http://127.0.0.1:{port}",
        **kw,
    )


def test_titan_embed_query_sends_single_input_text_with_normalize():
    srv, port = _spawn(FakeTitanHandler)
    try:
        e = _new_titan(port)
        v = e.embed_query("hello world")
        assert len(v) == 4
        assert v[0] == 11.0
        assert len(FakeTitanHandler.LAST_BODIES) == 1
        sent = json.loads(FakeTitanHandler.LAST_BODIES[0])
        assert sent["inputText"] == "hello world"
        assert sent["dimensions"] == 4
        assert sent["normalize"] is True
        # Bedrock URL path is /model/{urlencoded-modelid}/invoke
        assert FakeTitanHandler.LAST_PATHS[0] == "/model/amazon.titan-embed-text-v2%3A0/invoke"
    finally:
        srv.shutdown()


def test_titan_embed_documents_parallelizes_one_request_per_input():
    srv, port = _spawn(FakeTitanHandler)
    try:
        e = _new_titan(port, max_concurrency=4)
        out = e.embed_documents(["a", "ab", "abc", "abcd"])
        assert len(out) == 4
        # Order preserved (buffered, not buffer_unordered).
        assert [v[0] for v in out] == [1.0, 2.0, 3.0, 4.0]
        # One HTTP request per input.
        assert len(FakeTitanHandler.LAST_BODIES) == 4
    finally:
        srv.shutdown()


def test_titan_normalize_can_be_disabled():
    srv, port = _spawn(FakeTitanHandler)
    try:
        e = _new_titan(port, normalize=False)
        e.embed_query("x")
        sent = json.loads(FakeTitanHandler.LAST_BODIES[0])
        assert sent["normalize"] is False
    finally:
        srv.shutdown()


def test_cohere_on_bedrock_batches_texts_array():
    srv, port = _spawn(FakeCohereOnBedrockHandler)
    try:
        e = _new_cohere_bedrock(port)  # format auto-detected from cohere.* prefix
        out = e.embed_documents(["a", "ab", "abc"])
        assert len(out) == 3
        assert [v[0] for v in out] == [1.0, 2.0, 3.0]
        sent = json.loads(FakeCohereOnBedrockHandler.LAST_BODY[0])
        # Single batched request, not 3 separate ones.
        assert sent["texts"] == ["a", "ab", "abc"]
        assert sent["input_type"] == "search_document"
        assert sent["embedding_types"] == ["float"]
    finally:
        srv.shutdown()


def test_cohere_on_bedrock_query_uses_search_query_input_type():
    srv, port = _spawn(FakeCohereOnBedrockHandler)
    try:
        e = _new_cohere_bedrock(port)
        e.embed_query("hi")
        sent = json.loads(FakeCohereOnBedrockHandler.LAST_BODY[0])
        assert sent["input_type"] == "search_query"
    finally:
        srv.shutdown()


def test_format_override_to_titan_for_unknown_model_id():
    srv, port = _spawn(FakeTitanHandler)
    try:
        # Force titan format on a model id that wouldn't auto-detect.
        e = BedrockEmbeddings(
            access_key_id="AKIDEXAMPLE",
            secret_access_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            region="us-east-1",
            model_id="my-private.titan-clone",
            dimensions=4,
            endpoint_override=f"http://127.0.0.1:{port}",
            format="titan",
        )
        v = e.embed_query("x")
        assert len(v) == 4
        # Titan body shape sent
        sent = json.loads(FakeTitanHandler.LAST_BODIES[0])
        assert "inputText" in sent
    finally:
        srv.shutdown()


def test_format_invalid_raises():
    try:
        BedrockEmbeddings(
            access_key_id="x", secret_access_key="x", region="us-east-1",
            model_id="amazon.titan-embed-text-v2:0", dimensions=4,
            endpoint_override="http://127.0.0.1:1", format="ollama",
        )
    except ValueError as e:
        assert "format must be" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_bedrock_embeddings_empty_returns_empty():
    e = _new_titan(1)
    assert e.embed_documents([]) == []


def test_vector_retriever_accepts_bedrock_embeddings():
    srv, port = _spawn(FakeTitanHandler)
    try:
        e = _new_titan(port)
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
        test_titan_embed_query_sends_single_input_text_with_normalize,
        test_titan_embed_documents_parallelizes_one_request_per_input,
        test_titan_normalize_can_be_disabled,
        test_cohere_on_bedrock_batches_texts_array,
        test_cohere_on_bedrock_query_uses_search_query_input_type,
        test_format_override_to_titan_for_unknown_model_id,
        test_format_invalid_raises,
        test_bedrock_embeddings_empty_returns_empty,
        test_vector_retriever_accepts_bedrock_embeddings,
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

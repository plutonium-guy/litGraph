"""ContextualCompressionRetriever — wrap a base retriever with a compressor
that filters / extracts relevant content before returning to the LLM.

Reduces context window usage. Direct LangChain parity."""
import http.server
import json
import threading

from litgraph.embeddings import FunctionEmbeddings
from litgraph.providers import OpenAIChat
from litgraph.retrieval import (
    Bm25Index,
    ContextualCompressionRetriever,
    EmbeddingsFilterCompressor,
    LlmExtractCompressor,
    PipelineCompressor,
)


_SCRIPT = []
_IDX = [0]
_CALLS = [0]


class _FakeLlm(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        _CALLS[0] += 1
        payload = _SCRIPT[_IDX[0] % len(_SCRIPT)]  # cycle if exhausted
        _IDX[0] += 1
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *a, **kw): pass


def _spawn(script):
    global _SCRIPT
    _SCRIPT = script
    _IDX[0] = 0
    _CALLS[0] = 0
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _cc(text):
    return {
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_llm_extract_compressor_drops_no_output_docs_keeps_extracted_text():
    """LLM returns NO_OUTPUT for irrelevant docs (dropped) and a sentence
    for relevant docs (content rewritten to just that sentence)."""
    idx = Bm25Index()
    # Use a shared term ("language") so BM25 returns all 3 docs for the query.
    idx.add([
        {"id": "a", "content": "rust language ownership prevents data races"},
        {"id": "b", "content": "python language uses gil and dynamic typing"},
        {"id": "c", "content": "go language goroutines lightweight concurrency"},
    ])
    # 3 docs → 3 LLM calls. Script: relevant, NO_OUTPUT, relevant.
    srv, port = _spawn([
        _cc("rust language ownership prevents data races."),
        _cc("NO_OUTPUT"),
        _cc("go language goroutines lightweight concurrency."),
    ])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        comp = LlmExtractCompressor(llm=llm)
        ccr = ContextualCompressionRetriever(base=idx, compressor=comp, over_fetch_factor=1)
        hits = ccr.retrieve("language", k=3)
        assert _CALLS[0] == 3, f"one LLM call per retrieved doc, got {_CALLS[0]}"
        # Doc b dropped (NO_OUTPUT), a + c kept with rewritten content.
        assert len(hits) == 2
        contents = [h["content"] for h in hits]
        assert any("data races" in c for c in contents)
        assert any("goroutines" in c for c in contents)
        assert not any("python" in c for c in contents)
    finally:
        srv.shutdown()


def test_embeddings_filter_compressor_drops_below_threshold():
    """Embeddings-based filter: compute cosine similarity between query and
    each doc. Drop docs below threshold."""
    # Bag-of-words style: dim 0 = "rust", dim 1 = "python", dim 2 = "go".
    # Query embedding aligns with a specific axis; non-matching docs get cos = 0.
    def bow(texts):
        out = []
        for t in texts:
            tl = t.lower()
            out.append([
                1.0 if "rust" in tl else 0.0,
                1.0 if "python" in tl else 0.0,
                1.0 if "go" in tl else 0.0,
                1.0 if "shared" in tl else 0.0,
            ])
        return out
    embed = FunctionEmbeddings(bow, dimensions=4, name="bow")

    idx = Bm25Index()
    idx.add([
        {"id": "match", "content": "rust shared content"},
        {"id": "miss",  "content": "python shared content"},
    ])
    comp = EmbeddingsFilterCompressor(embeddings=embed, similarity_threshold=0.6)
    # Query "rust" → vector [1,0,0,0]. "rust shared" doc → [1,0,0,1] → cos ≈ 0.707.
    # "python shared" doc → [0,1,0,1] → cos = 0. Drop the python doc.
    ccr = ContextualCompressionRetriever(base=idx, compressor=comp, over_fetch_factor=1)
    hits = ccr.retrieve("rust", k=5)
    ids = [h["id"] for h in hits]
    assert ids == ["match"]


def test_pipeline_compressor_filters_then_extracts():
    """Real prod pattern: cheap embeddings filter first, expensive LLM
    extraction second. The LLM should only see docs that survived the filter."""
    def fn(texts):
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]
    embed = FunctionEmbeddings(fn, dimensions=4, name="len")

    idx = Bm25Index()
    idx.add([
        {"id": "match",   "content": "rs"},   # len 2
        {"id": "no_match", "content": "x" * 50}, # len 50, drops in filter
    ])
    srv, port = _spawn([
        _cc("rs is short"),  # only one LLM call expected (after filter)
    ])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        pipe = PipelineCompressor([
            EmbeddingsFilterCompressor(embeddings=embed, similarity_threshold=0.99),
            LlmExtractCompressor(llm=llm),
        ])
        ccr = ContextualCompressionRetriever(base=idx, compressor=pipe, over_fetch_factor=1)
        hits = ccr.retrieve("rs", k=5)  # query len 2, "rs" matches
        # Pipeline: filter dropped huge → LLM only ran on 1 doc.
        assert _CALLS[0] == 1, f"expected 1 LLM call after filter, got {_CALLS[0]}"
        assert len(hits) == 1
        assert "rs" in hits[0]["content"]
    finally:
        srv.shutdown()


def test_pipeline_short_circuits_when_filter_drops_everything():
    """If the first step (filter) returns [], the LLM step should NOT run."""
    def fn(texts):
        return [[float(len(t)), 1.0, 0.0, 0.0] for t in texts]
    embed = FunctionEmbeddings(fn, dimensions=4, name="len")
    idx = Bm25Index()
    idx.add([{"id": "h", "content": "x" * 100}])
    srv, port = _spawn([_cc("should not run")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        pipe = PipelineCompressor([
            EmbeddingsFilterCompressor(embeddings=embed, similarity_threshold=2.0),  # impossible
            LlmExtractCompressor(llm=llm),
        ])
        ccr = ContextualCompressionRetriever(base=idx, compressor=pipe, over_fetch_factor=1)
        hits = ccr.retrieve("any", k=5)
        assert hits == []
        assert _CALLS[0] == 0, "LLM should not run when filter drops everything"
    finally:
        srv.shutdown()


def test_ccr_over_fetches_then_truncates():
    """base.retrieve should see k * over_fetch_factor; final result ≤ k."""
    # Identity embedder so cos sim = 1 for all docs, threshold 0 keeps all.
    def fn(texts):
        return [[1.0, 0.0, 0.0, 0.0] for _ in texts]
    embed = FunctionEmbeddings(fn, dimensions=4, name="constant")
    idx = Bm25Index()
    # Shared term ("widget") so BM25 returns all docs for the query.
    docs = [{"id": f"d{i}", "content": f"widget number {i} description"} for i in range(10)]
    idx.add(docs)
    comp = EmbeddingsFilterCompressor(embeddings=embed, similarity_threshold=0.0)
    ccr = ContextualCompressionRetriever(base=idx, compressor=comp, over_fetch_factor=3)
    hits = ccr.retrieve("widget", k=2)
    assert len(hits) == 2


def test_ccr_rejects_invalid_compressor_type():
    idx = Bm25Index()
    try:
        ContextualCompressionRetriever(base=idx, compressor="not a compressor")
    except TypeError as e:
        assert "compressor" in str(e)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_llm_extract_compressor_drops_no_output_docs_keeps_extracted_text,
        test_embeddings_filter_compressor_drops_below_threshold,
        test_pipeline_compressor_filters_then_extracts,
        test_pipeline_short_circuits_when_filter_drops_everything,
        test_ccr_over_fetches_then_truncates,
        test_ccr_rejects_invalid_compressor_type,
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

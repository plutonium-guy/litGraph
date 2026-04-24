"""MultiQueryRetriever — LLM paraphrases a query into N variations, runs the
base retriever for each in parallel, dedupes, returns top-k.

Verified end-to-end through a fake OpenAI server (script the paraphrase
response) + a BM25 base retriever (deterministic, no embedding noise)."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat
from litgraph.retrieval import Bm25Index, MultiQueryRetriever


_SCRIPT = []
_IDX = [0]


class _FakeLlm(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        self.rfile.read(n)
        payload = _SCRIPT[_IDX[0]]
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
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chat_completion(text):
    return {
        "id": "x", "object": "chat.completion", "model": "m",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def test_multi_query_dedupes_results_across_paraphrases():
    """Index 4 docs in BM25; LLM paraphrases query; the 3 base retrievals
    fan out, find overlapping docs, retriever dedupes the union."""
    idx = Bm25Index()
    idx.add([
        {"id": "rust", "content": "rust ownership and borrow checker"},
        {"id": "py", "content": "python dynamic typing and gil"},
        {"id": "go", "content": "go goroutines green threads concurrency"},
        {"id": "js", "content": "javascript event loop concurrency"},
    ])
    # LLM rewrites "memory safety" into 2 paraphrases that BM25 will
    # match on different docs. Original query also runs.
    srv, port = _spawn([_chat_completion("borrow checker rust\nconcurrency model")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        mqr = MultiQueryRetriever(base=idx, llm=llm, num_variations=2)
        hits = mqr.retrieve("memory safety", k=10)
        ids = sorted(h["id"] for h in hits)
        # rust hits twice (original + "borrow checker rust"), go and js hit
        # via "concurrency model". Dedup → unique set.
        assert "rust" in ids
        # No id appears twice.
        assert len(ids) == len(set(ids))
    finally:
        srv.shutdown()


def test_multi_query_generate_queries_returns_cleaned_list():
    srv, port = _spawn([_chat_completion("1. first phrasing\n2. second phrasing\n- bulleted\n* starred")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        idx = Bm25Index()  # not used
        mqr = MultiQueryRetriever(base=idx, llm=llm, num_variations=4)
        queries = mqr.generate_queries("original")
        # Numbering and bullet prefixes stripped.
        assert queries == ["first phrasing", "second phrasing", "bulleted", "starred"]
    finally:
        srv.shutdown()


def test_multi_query_truncates_to_k():
    idx = Bm25Index()
    for i in range(20):
        idx.add([{"id": f"d{i}", "content": f"document number {i} concurrency"}])
    srv, port = _spawn([_chat_completion("concurrency\nthreads\nrace conditions")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        mqr = MultiQueryRetriever(base=idx, llm=llm, num_variations=3)
        hits = mqr.retrieve("concurrency", k=4)
        assert len(hits) == 4
    finally:
        srv.shutdown()


def test_multi_query_falls_back_to_literal_when_llm_returns_empty():
    """LLM returns empty string. With include_original=False, we should
    still run the literal query — degraded gracefully, not [] silently."""
    idx = Bm25Index()
    idx.add([{"id": "d1", "content": "literal hit on rust"}])
    srv, port = _spawn([_chat_completion("")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        mqr = MultiQueryRetriever(
            base=idx, llm=llm, num_variations=2, include_original=False,
        )
        hits = mqr.retrieve("rust", k=5)
        assert len(hits) == 1
    finally:
        srv.shutdown()


def test_multi_query_rejects_invalid_base_type():
    srv, port = _spawn([_chat_completion("a")])
    try:
        llm = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        try:
            MultiQueryRetriever(base="not a retriever", llm=llm)
        except TypeError as e:
            assert "base" in str(e)
        else:
            raise AssertionError("expected TypeError")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_multi_query_dedupes_results_across_paraphrases,
        test_multi_query_generate_queries_returns_cleaned_list,
        test_multi_query_truncates_to_k,
        test_multi_query_falls_back_to_literal_when_llm_returns_empty,
        test_multi_query_rejects_invalid_base_type,
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

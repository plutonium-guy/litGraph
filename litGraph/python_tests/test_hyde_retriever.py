"""HydeRetriever — Hypothetical Document Embeddings. LLM writes an
ideal answer; that answer's embedding drives retrieval."""
import http.server
import json
import threading

from litgraph.providers import OpenAIChat
from litgraph.retrieval import Bm25Index, HydeRetriever


class _ScriptedChat(http.server.BaseHTTPRequestHandler):
    """Returns canned content strings as chat completions in order."""
    ANSWERS: list = []
    SEEN_PROMPTS: list = []

    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        self.SEEN_PROMPTS.append(json.loads(body))
        content = self.ANSWERS.pop(0) if self.ANSWERS else "default answer"
        payload = {
            "id": "r", "model": "gpt-test", "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
        out = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a, **kw): pass


def _spawn(answers):
    _ScriptedChat.ANSWERS = list(answers)
    _ScriptedChat.SEEN_PROMPTS = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _ScriptedChat)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _chat(port):
    return OpenAIChat(
        api_key="sk-test", model="gpt-test",
        base_url=f"http://127.0.0.1:{port}/v1",
    )


def _bm25_with_docs(docs: list[dict]) -> Bm25Index:
    idx = Bm25Index()
    idx.add(docs)
    return idx


def test_hypothetical_drives_retrieval_not_raw_query():
    """LLM writes a photosynthesis-flavored hypothetical → retrieval
    surfaces the photosynthesis doc. Raw query has none of those tokens."""
    srv, port = _spawn([
        "Photosynthesis converts light energy into glucose within chloroplasts."
    ])
    try:
        # Only one doc contains words from the hypothetical answer.
        base = _bm25_with_docs([
            {"content": "chloroplasts glucose photosynthesis mitochondria", "id": "d1"},
            {"content": "banking interest rates federal reserve", "id": "d2"},
        ])
        hyde = HydeRetriever(
            base,
            llm=_chat(port),
            include_original=False,
        )
        # Raw query has no domain vocabulary — BM25 on the raw query
        # would miss. HyDE's hypothetical contains "photosynthesis" /
        # "glucose" / "chloroplasts" which match d1.
        docs = hyde.retrieve("how do plants feed themselves", k=1)
    finally:
        srv.shutdown()
    assert len(docs) == 1
    assert docs[0]["id"] == "d1"


def test_include_original_true_merges_both_retrievals():
    srv, port = _spawn(["glucose chloroplasts photosynthesis"])
    try:
        base = _bm25_with_docs([
            {"content": "chloroplasts glucose photosynthesis", "id": "hypo_doc"},
            {"content": "how plants feed themselves naturally", "id": "orig_doc"},
        ])
        hyde = HydeRetriever(base, llm=_chat(port))  # include_original defaults True
        docs = hyde.retrieve("how do plants feed themselves", k=5)
    finally:
        srv.shutdown()
    ids = {d["id"] for d in docs}
    assert "hypo_doc" in ids
    assert "orig_doc" in ids


def test_generate_hypothetical_preview_call():
    srv, port = _spawn(["Gravity is the mutual attraction between masses."])
    try:
        base = Bm25Index()
        hyde = HydeRetriever(base, llm=_chat(port))
        passage = hyde.generate_hypothetical("why do things fall down?")
    finally:
        srv.shutdown()
    assert "Gravity" in passage
    # Prompt should include the user's query.
    seen = _ScriptedChat.SEEN_PROMPTS[0]
    user = next(m for m in seen["messages"] if m["role"] == "user")
    assert "why do things fall down?" in user["content"]


def test_include_original_false_single_llm_call_single_retrieval():
    """When include_original=False, exactly ONE retrieval runs (the
    hypothetical-driven one). Verify via the retriever side-effect."""
    srv, port = _spawn(["hypothetical answer text"])
    try:
        base = _bm25_with_docs([
            {"content": "hypothetical answer text here", "id": "d1"},
        ])
        hyde = HydeRetriever(base, llm=_chat(port), include_original=False)
        docs = hyde.retrieve("anything", k=5)
    finally:
        srv.shutdown()
    # Exactly one LLM call.
    assert len(_ScriptedChat.SEEN_PROMPTS) == 1


def test_custom_system_prompt_reaches_model():
    srv, port = _spawn(["short answer"])
    try:
        base = Bm25Index()
        hyde = HydeRetriever(
            base,
            llm=_chat(port),
            system_prompt="Write a single word answer.",
        )
        hyde.generate_hypothetical("q")
    finally:
        srv.shutdown()
    seen = _ScriptedChat.SEEN_PROMPTS[0]
    sys_msg = next(m for m in seen["messages"] if m["role"] == "system")
    assert "single word" in sys_msg["content"]


def test_temperature_zero_in_request_body():
    """Hypothetical generation pins temperature=0 for cache-friendly
    deterministic passages."""
    srv, port = _spawn(["x"])
    try:
        base = Bm25Index()
        hyde = HydeRetriever(base, llm=_chat(port))
        hyde.generate_hypothetical("q")
    finally:
        srv.shutdown()
    body = _ScriptedChat.SEEN_PROMPTS[0]
    assert body["temperature"] == 0.0


def test_composes_with_bm25_in_react_agent_flow():
    """End-to-end smoke: HyDE → BM25 → returns expected top doc."""
    srv, port = _spawn(["unique_token_XYZ"])
    try:
        base = _bm25_with_docs([
            {"content": "unique_token_XYZ here is the target doc", "id": "target"},
            {"content": "irrelevant content about other things", "id": "other"},
        ])
        hyde = HydeRetriever(base, llm=_chat(port), include_original=False)
        docs = hyde.retrieve("anything that doesnt match directly", k=1)
    finally:
        srv.shutdown()
    assert docs[0]["id"] == "target"


if __name__ == "__main__":
    import traceback
    fns = [
        test_hypothetical_drives_retrieval_not_raw_query,
        test_include_original_true_merges_both_retrievals,
        test_generate_hypothetical_preview_call,
        test_include_original_false_single_llm_call_single_retrieval,
        test_custom_system_prompt_reaches_model,
        test_temperature_zero_in_request_body,
        test_composes_with_bm25_in_react_agent_flow,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

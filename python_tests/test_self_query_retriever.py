"""SelfQueryRetriever — LLM extracts metadata filter from natural-language
query, runs filtered vector search. Direct LangChain parity.

End-to-end via fake OpenAI server returning a canned JSON extraction +
MemoryVectorStore that supports metadata filtering."""
import http.server
import json
import threading

from litgraph.embeddings import FunctionEmbeddings
from litgraph.providers import OpenAIChat
from litgraph.retrieval import MemoryVectorStore, SelfQueryRetriever


_LAST_BODY = [None]
_NEXT_TEXT = [""]


class _FakeLlm(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(n)
        _LAST_BODY[0] = json.loads(body)
        text = _NEXT_TEXT[0]
        payload = {
            "id": "x", "object": "chat.completion", "model": "m",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
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


def _spawn():
    _LAST_BODY[0] = None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _FakeLlm)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, srv.server_address[1]


def _bow(texts):
    """4-dim BoW: [rust, python, go, anything-else]."""
    out = []
    for t in texts:
        tl = t.lower()
        out.append([
            1.0 if "rust" in tl else 0.0,
            1.0 if "python" in tl else 0.0,
            1.0 if "go" in tl else 0.0,
            1.0,
        ])
    return out


ATTRS = [
    {"name": "language", "description": "Programming language", "type": "string"},
    {"name": "year", "description": "Publication year", "type": "integer"},
]


def test_self_query_extract_returns_query_and_filter():
    """Just the LLM extraction step — preview without firing a search."""
    _NEXT_TEXT[0] = json.dumps({
        "query": "memory safety crates",
        "filter": {"language": "rust", "year": 2024},
    })
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        embed = FunctionEmbeddings(_bow, dimensions=4, name="bow")
        store = MemoryVectorStore()
        sqr = SelfQueryRetriever(
            embeddings=embed,
            store=store,
            llm=chat,
            document_contents="rust crate descriptions",
            attributes=ATTRS,
        )
        out = sqr.extract_query_and_filter("memory safety crates in rust from 2024")
        assert out["query"] == "memory safety crates"
        assert out["filter"]["language"] == "rust"
        assert out["filter"]["year"] == 2024
    finally:
        srv.shutdown()


def test_self_query_request_body_has_json_schema_with_attribute_props():
    """Wire-shape check: extraction request uses response_format json_schema
    with the attribute table embedded as filter properties."""
    _NEXT_TEXT[0] = json.dumps({"query": "x", "filter": {}})
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        sqr = SelfQueryRetriever(
            embeddings=FunctionEmbeddings(_bow, dimensions=4, name="b"),
            store=MemoryVectorStore(),
            llm=chat,
            document_contents="rust crates",
            attributes=ATTRS,
        )
        sqr.extract_query_and_filter("rust posts")
        body = _LAST_BODY[0]
        rf = body["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "SelfQueryExtraction"
        schema = rf["json_schema"]["schema"]
        assert schema["properties"]["query"]["type"] == "string"
        # Attribute table → filter properties.
        assert schema["properties"]["filter"]["properties"]["language"]["type"] == "string"
        assert schema["properties"]["filter"]["properties"]["year"]["type"] == "integer"
    finally:
        srv.shutdown()


def test_self_query_filters_vector_search_by_extracted_metadata():
    """Real end-to-end: index docs with `language` metadata, query
    'rust crates', LLM extracts filter={language:rust}, vector search
    returns ONLY rust docs."""
    _NEXT_TEXT[0] = json.dumps({
        "query": "crates",
        "filter": {"language": "rust"},
    })
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        embed = FunctionEmbeddings(_bow, dimensions=4, name="bow")
        store = MemoryVectorStore()
        store.add(
            [
                {"id": "r1", "content": "rust crate one", "metadata": {"language": "rust"}},
                {"id": "p1", "content": "python pkg one", "metadata": {"language": "python"}},
                {"id": "r2", "content": "rust crate two", "metadata": {"language": "rust"}},
            ],
            _bow(["rust", "python", "rust"]),
        )
        sqr = SelfQueryRetriever(
            embeddings=embed,
            store=store,
            llm=chat,
            document_contents="package descriptions",
            attributes=ATTRS,
        )
        hits = sqr.retrieve("rust crates", k=10)
        # Filter applied → only rust docs survived.
        ids = sorted(h["id"] for h in hits)
        assert ids == ["r1", "r2"]
    finally:
        srv.shutdown()


def test_self_query_empty_filter_means_no_metadata_filter_applied():
    """LLM returns empty filter → store sees no filter → all docs
    candidate (just topic similarity matters)."""
    _NEXT_TEXT[0] = json.dumps({"query": "rust", "filter": {}})
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        embed = FunctionEmbeddings(_bow, dimensions=4, name="bow")
        store = MemoryVectorStore()
        store.add(
            [
                {"id": "r1", "content": "rust", "metadata": {"language": "rust"}},
                {"id": "p1", "content": "python", "metadata": {"language": "python"}},
            ],
            _bow(["rust", "python"]),
        )
        sqr = SelfQueryRetriever(
            embeddings=embed,
            store=store,
            llm=chat,
            document_contents="packages",
            attributes=ATTRS,
        )
        hits = sqr.retrieve("rust", k=10)
        # No filter → both docs candidates (returned in similarity order).
        ids = {h["id"] for h in hits}
        assert ids == {"r1", "p1"}
    finally:
        srv.shutdown()


def test_self_query_rejects_invalid_attribute_dict():
    """Each attribute dict needs name/description/type. Missing key raises
    at construction (early failure beats runtime confusion)."""
    srv, port = _spawn()
    try:
        chat = OpenAIChat(api_key="k", model="m", base_url=f"http://127.0.0.1:{port}/v1")
        try:
            SelfQueryRetriever(
                embeddings=FunctionEmbeddings(_bow, dimensions=4, name="b"),
                store=MemoryVectorStore(),
                llm=chat,
                document_contents="x",
                attributes=[{"name": "broken"}],  # missing description + type
            )
        except ValueError as e:
            assert "description" in str(e)
        else:
            raise AssertionError("expected ValueError on incomplete attribute")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    fns = [
        test_self_query_extract_returns_query_and_filter,
        test_self_query_request_body_has_json_schema_with_attribute_props,
        test_self_query_filters_vector_search_by_extracted_metadata,
        test_self_query_empty_filter_means_no_metadata_filter_applied,
        test_self_query_rejects_invalid_attribute_dict,
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

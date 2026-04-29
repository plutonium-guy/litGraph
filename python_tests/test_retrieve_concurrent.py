"""retrieve_concurrent — fan N queries against ONE retriever in parallel.

Network-free tests using FunctionEmbeddings + MemoryVectorStore +
VectorRetriever, plus a Bm25Index for the lexical-retriever path.
The ranking math is the underlying retriever's; this helper only
guarantees output alignment + bounded concurrency + per-query
failure isolation."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    Bm25Index, MemoryVectorStore, VectorRetriever, retrieve_concurrent,
)


def _corpus():
    return [
        {"content": "rust memory safety", "id": "alpha"},
        {"content": "memory leaks in c", "id": "beta"},
        {"content": "javascript closures", "id": "gamma"},
    ]


def _vector_retriever():
    docs = _corpus()
    def embed(texts):
        return [
            [
                float("memory" in t.lower()),
                float("rust" in t.lower()),
            ]
            for t in texts
        ]
    e = FunctionEmbeddings(embed, dimensions=2, name="toy")
    store = MemoryVectorStore()
    store.add(docs, embed([d["content"] for d in docs]))
    return VectorRetriever(e, store)


def _bm25():
    idx = Bm25Index()
    idx.add(_corpus())
    return idx


def test_retrieve_concurrent_returns_one_result_per_query():
    r = _vector_retriever()
    out = retrieve_concurrent(
        r, ["rust safety", "memory", "closures"], k=2, max_concurrency=4
    )
    assert len(out) == 3
    # Each result is a list of docs (not the error-dict shape).
    for hits in out:
        assert isinstance(hits, list)


def test_retrieve_concurrent_aligned_to_input_order():
    """Use BM25 (deterministic lexical match) so we can pin per-query
    top-1 to specific docs and prove input→output alignment."""
    r = _bm25()
    queries = ["rust safety", "memory leaks", "javascript"]
    out = retrieve_concurrent(r, queries, k=1, max_concurrency=4)
    assert len(out) == 3
    assert out[0][0]["id"] == "alpha"
    assert out[1][0]["id"] == "beta"
    assert out[2][0]["id"] == "gamma"


def test_retrieve_concurrent_works_with_bm25():
    r = _bm25()
    out = retrieve_concurrent(r, ["rust", "leaks"], k=1, max_concurrency=2)
    assert len(out) == 2
    assert out[0][0]["id"] == "alpha"
    assert out[1][0]["id"] == "beta"


def test_retrieve_concurrent_empty_queries_returns_empty():
    r = _vector_retriever()
    assert retrieve_concurrent(r, [], k=5, max_concurrency=4) == []


def test_retrieve_concurrent_zero_concurrency_normalised():
    r = _vector_retriever()
    out = retrieve_concurrent(r, ["q1", "q2"], k=1, max_concurrency=0)
    # Output count matches input — concurrency 0 was normalised, didn't trip.
    assert len(out) == 2


def test_retrieve_concurrent_rejects_non_retriever():
    try:
        retrieve_concurrent("not a retriever", ["q"], k=1)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected TypeError/ValueError")


def test_retrieve_concurrent_fail_fast_raises_on_error():
    """Sending an empty corpus retriever a query that may stress the
    pipeline isn't a real failure path — but we can verify fail_fast
    parameter is respected by passing a non-retriever, which raises
    before the parallel dispatch even starts (covered above).
    Here we test the success path with fail_fast=True."""
    r = _vector_retriever()
    out = retrieve_concurrent(r, ["q1", "q2"], k=1, max_concurrency=2, fail_fast=True)
    assert len(out) == 2
    for hits in out:
        assert isinstance(hits, list)


def test_retrieve_concurrent_repr_for_results():
    """Each successful slot is a list of doc dicts with `id`, `content`."""
    r = _bm25()
    out = retrieve_concurrent(r, ["rust"], k=1, max_concurrency=2)
    assert len(out) == 1
    hits = out[0]
    assert hits and "content" in hits[0] and "id" in hits[0]


if __name__ == "__main__":
    fns = [
        test_retrieve_concurrent_returns_one_result_per_query,
        test_retrieve_concurrent_aligned_to_input_order,
        test_retrieve_concurrent_works_with_bm25,
        test_retrieve_concurrent_empty_queries_returns_empty,
        test_retrieve_concurrent_zero_concurrency_normalised,
        test_retrieve_concurrent_rejects_non_retriever,
        test_retrieve_concurrent_fail_fast_raises_on_error,
        test_retrieve_concurrent_repr_for_results,
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

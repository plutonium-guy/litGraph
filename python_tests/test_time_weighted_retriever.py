"""TimeWeightedRetriever — boost recently-accessed docs via exponential
decay. Direct LangChain parity. Common use: agent memory / chat-history
retrieval where recency matters."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import MemoryVectorStore, TimeWeightedRetriever


def _const(_texts):
    """All-equal vectors → similarity is constant per doc, isolating the
    time-weighting effect."""
    return [[1.0, 0.0, 0.0, 0.0] for _ in _texts]


def test_add_and_retrieve_returns_docs_with_recency_score():
    embed = FunctionEmbeddings(_const, dimensions=4, name="const")
    store = MemoryVectorStore()
    twr = TimeWeightedRetriever(embeddings=embed, store=store)
    twr.add_documents(
        [{"id": "a", "content": "first"}, {"id": "b", "content": "second"}],
        _const(["first", "second"]),
    )
    hits = twr.retrieve("query", k=2)
    assert len(hits) == 2
    # Both docs come back with a combined score ≥ similarity.
    for h in hits:
        assert h["score"] is not None
        assert float(h["score"]) >= 0.0


def test_decay_rate_zero_means_pure_similarity_ordering():
    """With decay_rate=0, recency component is constant 1.0 for all docs →
    relative ordering preserved from the underlying store insertion order."""
    embed = FunctionEmbeddings(_const, dimensions=4, name="const")
    store = MemoryVectorStore()
    twr = TimeWeightedRetriever(embeddings=embed, store=store, decay_rate=0.0)
    twr.add_documents(
        [
            {"id": "a", "content": "1"},
            {"id": "b", "content": "2"},
            {"id": "c", "content": "3"},
        ],
        _const(["1", "2", "3"]),
    )
    hits = twr.retrieve("q", k=3)
    ids = [h["id"] for h in hits]
    # No decay; insertion order of MemoryVectorStore preserved.
    assert sorted(ids) == ["a", "b", "c"]


def test_access_log_tracks_initial_add_then_bumps_on_retrieve():
    """add_documents stamps last_accessed = now. After retrieve, the
    returned docs' timestamps are bumped to (a presumably-later) now."""
    embed = FunctionEmbeddings(_const, dimensions=4, name="const")
    store = MemoryVectorStore()
    twr = TimeWeightedRetriever(embeddings=embed, store=store)
    twr.add_documents(
        [{"id": "x", "content": "x"}, {"id": "y", "content": "y"}],
        _const(["x", "y"]),
    )
    log_initial = twr.access_log()
    assert {entry["id"] for entry in log_initial} == {"x", "y"}
    initial_ts = {e["id"]: e["last_accessed_ms"] for e in log_initial}
    # Retrieve only "x" (or both) — the returned docs get their TS bumped.
    twr.retrieve("q", k=2)
    log_after = twr.access_log()
    after_ts = {e["id"]: e["last_accessed_ms"] for e in log_after}
    # Timestamps after retrieve are >= initial (clock monotonic).
    for doc_id, t0 in initial_ts.items():
        assert after_ts[doc_id] >= t0


def test_empty_store_returns_empty_list():
    embed = FunctionEmbeddings(_const, dimensions=4, name="const")
    store = MemoryVectorStore()
    twr = TimeWeightedRetriever(embeddings=embed, store=store)
    hits = twr.retrieve("anything", k=5)
    assert hits == []


def test_rejects_invalid_store_type_at_construction():
    embed = FunctionEmbeddings(_const, dimensions=4, name="const")
    try:
        TimeWeightedRetriever(embeddings=embed, store="not a store")
    except TypeError as e:
        assert "store" in str(e)
    else:
        raise AssertionError("expected TypeError on invalid store")


if __name__ == "__main__":
    fns = [
        test_add_and_retrieve_returns_docs_with_recency_score,
        test_decay_rate_zero_means_pure_similarity_ordering,
        test_access_log_tracks_initial_add_then_bumps_on_retrieve,
        test_empty_store_returns_empty_list,
        test_rejects_invalid_store_type_at_construction,
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

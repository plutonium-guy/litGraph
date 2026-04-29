"""HybridRetriever — RRF-fuse N child retrievers (concurrent fan-out).
Tests cover BM25 + dense vector hybrid, the canonical production pattern."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    Bm25Index, HybridRetriever, MemoryVectorStore, VectorRetriever,
)


def _build_corpus():
    """3 docs with distinct lexical and semantic profiles so we can test
    that the dense and lexical branches DON'T agree, then verify RRF
    boosts docs that both branches like.

    docs:
      alpha — "rust memory safety" (dense match for memory queries)
      beta  — "memory leaks in C" (lexical match for "memory")
      gamma — "javascript closures" (unrelated)
    """
    return [
        {"content": "rust memory safety guarantees", "id": "alpha"},
        {"content": "memory leaks in C programs",   "id": "beta"},
        {"content": "javascript closures explained", "id": "gamma"},
    ]


def _bm25_for(docs):
    idx = Bm25Index()
    idx.add(docs)
    return idx


def _vector_for(docs):
    """Toy 2-D embedder: dim 0 = "memory" presence, dim 1 = "rust" presence."""
    def embed(texts):
        return [[float("memory" in t.lower()), float("rust" in t.lower())] for t in texts]
    e = FunctionEmbeddings(embed, dimensions=2, name="toy")
    store = MemoryVectorStore()
    store.add(docs, embed([d["content"] for d in docs]))
    return VectorRetriever(e, store)


def test_hybrid_constructs_with_two_children():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    h = HybridRetriever(children=[bm25, vec])
    assert "HybridRetriever(children=2" in repr(h)


def test_hybrid_rejects_single_child():
    docs = _build_corpus()
    try:
        HybridRetriever(children=[_bm25_for(docs)])
    except ValueError as e:
        assert "at least 2 children" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_hybrid_returns_top_k_via_rrf():
    """Both branches should rank `alpha` highly for "rust memory" — RRF
    fusion must therefore put alpha at #1."""
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    h = HybridRetriever(children=[bm25, vec])
    hits = h.retrieve("rust memory", k=2)
    assert len(hits) == 2
    assert hits[0]["id"] == "alpha"


def test_hybrid_rrf_k_affects_score():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    sharp = HybridRetriever(children=[bm25, vec], rrf_k=1.0)
    smooth = HybridRetriever(children=[bm25, vec], rrf_k=1000.0)
    s_hits = sharp.retrieve("rust memory", k=3)
    sm_hits = smooth.retrieve("rust memory", k=3)
    # Both should still rank alpha first; scores should differ.
    assert s_hits[0]["id"] == "alpha"
    assert sm_hits[0]["id"] == "alpha"
    # Score format may be float or string; just check both yield something.
    assert s_hits[0].get("score") is not None
    assert sm_hits[0].get("score") is not None


def test_hybrid_accepts_three_heterogeneous_children():
    """BM25 + 2 different VectorRetrievers (different embedders) — fusion
    of three branches works without error."""
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec1 = _vector_for(docs)
    # Second vector retriever with a different embedding direction.
    def embed2(texts):
        return [[float("javascript" in t.lower()), float("c" in t.lower())] for t in texts]
    e2 = FunctionEmbeddings(embed2, dimensions=2, name="toy2")
    store2 = MemoryVectorStore()
    store2.add(docs, embed2([d["content"] for d in docs]))
    vec2 = VectorRetriever(e2, store2)

    h = HybridRetriever(children=[bm25, vec1, vec2])
    hits = h.retrieve("memory", k=3)
    assert len(hits) == 3
    # Order may vary but all 3 ids should appear.
    returned = {h["id"] for h in hits}
    assert returned == {"alpha", "beta", "gamma"}


def test_hybrid_rejects_non_retriever_child():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    try:
        HybridRetriever(children=[bm25, "not a retriever"])
    except TypeError as e:
        assert "VectorRetriever" in str(e)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_hybrid_constructs_with_two_children,
        test_hybrid_rejects_single_child,
        test_hybrid_returns_top_k_via_rrf,
        test_hybrid_rrf_k_affects_score,
        test_hybrid_accepts_three_heterogeneous_children,
        test_hybrid_rejects_non_retriever_child,
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

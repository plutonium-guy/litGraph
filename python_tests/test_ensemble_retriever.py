"""EnsembleRetriever — weighted RRF across N concurrent child retrievers.

Mirrors the HybridRetriever test corpus so the relative behaviour is
easy to compare. The point of EnsembleRetriever over HybridRetriever is
*per-child weights* — these tests check the weight knob actually moves
ranks the way the docstring promises."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    Bm25Index, EnsembleRetriever, MemoryVectorStore, VectorRetriever,
)


def _build_corpus():
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
    def embed(texts):
        return [[float("memory" in t.lower()), float("rust" in t.lower())] for t in texts]
    e = FunctionEmbeddings(embed, dimensions=2, name="toy")
    store = MemoryVectorStore()
    store.add(docs, embed([d["content"] for d in docs]))
    return VectorRetriever(e, store)


def test_ensemble_constructs_with_two_children():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    r = EnsembleRetriever(children=[bm25, vec])
    assert "EnsembleRetriever(children=2" in repr(r)


def test_ensemble_rejects_single_child():
    docs = _build_corpus()
    try:
        EnsembleRetriever(children=[_bm25_for(docs)])
    except ValueError as e:
        assert "at least 2 children" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ensemble_default_weights_match_hybrid_behaviour():
    """With weights=None, ranks should be the same as a HybridRetriever."""
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    r = EnsembleRetriever(children=[bm25, vec])
    hits = r.retrieve("rust memory", k=2)
    assert hits[0]["id"] == "alpha"


def test_ensemble_weights_length_must_match_children():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    try:
        EnsembleRetriever(children=[bm25, vec], weights=[1.0])
    except ValueError as e:
        assert "length mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ensemble_zero_weight_silences_branch():
    """A weight of 0 must drop that branch from fusion entirely."""
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    # Zero out BM25; only vector retriever's ranking matters.
    r = EnsembleRetriever(children=[bm25, vec], weights=[0.0, 1.0])
    only_vec = vec.retrieve("memory", k=3)
    fused = r.retrieve("memory", k=3)
    # Same set of returned ids as the vector-only retriever.
    assert {h["id"] for h in fused} == {h["id"] for h in only_vec}


def test_ensemble_high_weight_branch_wins_top_slot():
    """Cranking one branch's weight should pull its top result to the
    fused #1 position even when the other branch ranks something else
    higher."""
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    # 100x weight on vec should make vec's top doc dominate.
    r = EnsembleRetriever(children=[bm25, vec], weights=[1.0, 100.0])
    vec_top = vec.retrieve("rust", k=1)[0]["id"]
    hits = r.retrieve("rust", k=1)
    assert hits[0]["id"] == vec_top


def test_ensemble_accepts_hybrid_as_child():
    """An EnsembleRetriever should be able to wrap a HybridRetriever as
    one of its branches — composition is the point of having both."""
    from litgraph.retrieval import HybridRetriever
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    vec = _vector_for(docs)
    inner = HybridRetriever(children=[bm25, vec])
    bm25_b = _bm25_for(docs)
    outer = EnsembleRetriever(children=[inner, bm25_b], weights=[0.5, 0.5])
    hits = outer.retrieve("memory", k=2)
    assert len(hits) == 2


def test_ensemble_rejects_non_retriever_child():
    docs = _build_corpus()
    bm25 = _bm25_for(docs)
    try:
        EnsembleRetriever(children=[bm25, "not a retriever"])
    except TypeError as e:
        assert "VectorRetriever" in str(e)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_ensemble_constructs_with_two_children,
        test_ensemble_rejects_single_child,
        test_ensemble_default_weights_match_hybrid_behaviour,
        test_ensemble_weights_length_must_match_children,
        test_ensemble_zero_weight_silences_branch,
        test_ensemble_high_weight_branch_wins_top_slot,
        test_ensemble_accepts_hybrid_as_child,
        test_ensemble_rejects_non_retriever_child,
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

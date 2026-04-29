"""EnsembleReranker — fan out N rerankers concurrently, fuse via RRF.

Network-free tests aren't possible from Python because the registered
reranker pyclasses (CohereReranker / VoyageReranker / JinaReranker)
require API keys. We exercise the constructor + type validation + the
ensemble's composition into a RerankingRetriever wrapper here; the
actual ranking-fusion math is covered by the Rust test suite where
mock rerankers are easy to instantiate."""
from litgraph.retrieval import (
    CohereReranker, EnsembleReranker, JinaReranker, VoyageReranker,
)


def _scaffolding():
    """Three real reranker pyclass instances pointing at fake hosts.
    Construction succeeds even though the URLs are unreachable —
    we only invoke the constructor + composition tests, never .rerank."""
    return [
        CohereReranker(api_key="ck-fake", base_url="http://127.0.0.1:1"),
        VoyageReranker(api_key="vk-fake", base_url="http://127.0.0.1:1"),
        JinaReranker(api_key="jk-fake", base_url="http://127.0.0.1:1"),
    ]


def test_ensemble_reranker_constructs_with_two_children():
    rerankers = _scaffolding()
    e = EnsembleReranker([rerankers[0], rerankers[1]])
    assert "EnsembleReranker(children=2" in repr(e)


def test_ensemble_reranker_with_three_heterogeneous_children():
    rerankers = _scaffolding()
    e = EnsembleReranker(rerankers)
    assert "children=3" in repr(e)


def test_ensemble_reranker_rejects_single_child():
    rerankers = _scaffolding()
    try:
        EnsembleReranker([rerankers[0]])
    except ValueError as e:
        assert "at least 2 children" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ensemble_reranker_weights_length_must_match():
    rerankers = _scaffolding()
    try:
        EnsembleReranker([rerankers[0], rerankers[1]], weights=[1.0])
    except ValueError as e:
        assert "length mismatch" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_ensemble_reranker_accepts_per_child_weights():
    rerankers = _scaffolding()
    e = EnsembleReranker(
        [rerankers[0], rerankers[1]], weights=[0.7, 0.3]
    )
    assert "EnsembleReranker(children=2" in repr(e)


def test_ensemble_reranker_rejects_non_reranker_child():
    rerankers = _scaffolding()
    try:
        EnsembleReranker([rerankers[0], "not a reranker"])
    except TypeError as e:
        assert "CohereReranker" in str(e)
    else:
        raise AssertionError("expected TypeError")


def test_ensemble_reranker_composes_into_reranking_retriever():
    """Verify EnsembleReranker plugs into RerankingRetriever as the
    `reranker` argument — the composition contract that makes the
    primitive useful in real RAG pipelines."""
    from litgraph.embeddings import FunctionEmbeddings
    from litgraph.retrieval import (
        MemoryVectorStore, RerankingRetriever, VectorRetriever,
    )
    rerankers = _scaffolding()
    ensemble = EnsembleReranker([rerankers[0], rerankers[1]])
    # Build a trivial base retriever.
    docs = [{"content": "rust safety", "id": "a"}]
    emb = FunctionEmbeddings(
        lambda ts: [[1.0, 0.0] for _ in ts], dimensions=2, name="toy"
    )
    store = MemoryVectorStore()
    store.add(docs, [[1.0, 0.0]])
    base = VectorRetriever(emb, store)
    rr = RerankingRetriever(base, ensemble, over_fetch_k=4)
    # Just constructing it exercises the type-extraction path.
    assert rr is not None


if __name__ == "__main__":
    fns = [
        test_ensemble_reranker_constructs_with_two_children,
        test_ensemble_reranker_with_three_heterogeneous_children,
        test_ensemble_reranker_rejects_single_child,
        test_ensemble_reranker_weights_length_must_match,
        test_ensemble_reranker_accepts_per_child_weights,
        test_ensemble_reranker_rejects_non_reranker_child,
        test_ensemble_reranker_composes_into_reranking_retriever,
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

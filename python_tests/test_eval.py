"""RAG evaluation harness — recall@k / mrr@k / ndcg@k against a labeled dataset."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    MemoryVectorStore, VectorRetriever, evaluate_retrieval,
)


def _build_retriever_with_known_corpus():
    """Tiny BOW embedder + 4-doc memory store with KNOWN ranking behavior:
    cos(query "foo", doc "foo foo") > cos(query "foo", doc "foo bar") >
    cos(query "foo", doc "bar baz") so we can hand-craft an eval dataset.
    """
    vocab = ["foo", "bar", "baz"]
    def embed(texts):
        return [[float(t.lower().count(w)) for w in vocab] for t in texts]
    e = FunctionEmbeddings(embed, dimensions=3, name="bow")
    store = MemoryVectorStore()
    docs = [
        {"content": "foo foo foo", "id": "alpha"},  # purely "foo"
        {"content": "foo bar",      "id": "beta"},   # mixed foo+bar
        {"content": "bar bar bar",  "id": "gamma"},  # purely "bar"
        {"content": "baz baz baz",  "id": "delta"},  # purely "baz"
    ]
    store.add(docs, embed([d["content"] for d in docs]))
    return VectorRetriever(e, store)


def test_perfect_retrieval_yields_recall_mrr_ndcg_1():
    """Single query whose top-1 is the only relevant doc → all metrics 1.0."""
    r = _build_retriever_with_known_corpus()
    dataset = [{"query": "foo", "relevant_ids": ["alpha"]}]
    rep = evaluate_retrieval(r, dataset, k=3)
    assert rep["n_queries"] == 1
    assert rep["k"] == 3
    assert abs(rep["recall_macro"] - 1.0) < 1e-9
    assert abs(rep["mrr_macro"] - 1.0) < 1e-9
    assert abs(rep["ndcg_macro"] - 1.0) < 1e-9
    # Per-query breakdown present + top doc is alpha.
    assert rep["per_query"][0]["query"] == "foo"
    assert rep["per_query"][0]["returned_ids"][0] == "alpha"


def test_zero_match_yields_all_zero_macro_metrics():
    r = _build_retriever_with_known_corpus()
    # Relevant id doesn't even exist in the corpus → no hit possible.
    dataset = [{"query": "foo", "relevant_ids": ["nonexistent-id"]}]
    rep = evaluate_retrieval(r, dataset, k=3)
    assert rep["recall_macro"] == 0.0
    assert rep["mrr_macro"] == 0.0
    assert rep["ndcg_macro"] == 0.0


def test_macro_average_over_multiple_queries():
    """Two queries: one perfect, one missing → metrics = 0.5 each."""
    r = _build_retriever_with_known_corpus()
    dataset = [
        {"query": "foo", "relevant_ids": ["alpha"]},   # perfect hit
        {"query": "baz", "relevant_ids": ["nope"]},    # impossible
    ]
    rep = evaluate_retrieval(r, dataset, k=3)
    assert rep["n_queries"] == 2
    assert abs(rep["recall_macro"] - 0.5) < 1e-9
    assert abs(rep["mrr_macro"] - 0.5) < 1e-9
    assert abs(rep["ndcg_macro"] - 0.5) < 1e-9


def test_per_query_order_matches_dataset_input():
    r = _build_retriever_with_known_corpus()
    dataset = [
        {"query": "first", "relevant_ids": ["alpha"]},
        {"query": "second", "relevant_ids": ["beta"]},
        {"query": "third", "relevant_ids": ["gamma"]},
    ]
    rep = evaluate_retrieval(r, dataset, k=3)
    queries = [q["query"] for q in rep["per_query"]]
    assert queries == ["first", "second", "third"]


def test_empty_dataset_raises():
    r = _build_retriever_with_known_corpus()
    try:
        evaluate_retrieval(r, [], k=5)
    except RuntimeError as e:
        assert "empty dataset" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_dataset_missing_relevant_ids_raises():
    r = _build_retriever_with_known_corpus()
    try:
        evaluate_retrieval(r, [{"query": "foo"}], k=5)
    except ValueError as e:
        assert "relevant_ids" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_max_concurrency_does_not_break_metric_correctness():
    """Same dataset run with max_concurrency=1 vs 16 must yield identical metrics."""
    r = _build_retriever_with_known_corpus()
    dataset = [
        {"query": "foo", "relevant_ids": ["alpha", "beta"]},
        {"query": "bar", "relevant_ids": ["gamma"]},
        {"query": "baz", "relevant_ids": ["delta"]},
    ]
    serial = evaluate_retrieval(r, dataset, k=3, max_concurrency=1)
    parallel = evaluate_retrieval(r, dataset, k=3, max_concurrency=16)
    assert abs(serial["recall_macro"] - parallel["recall_macro"]) < 1e-9
    assert abs(serial["mrr_macro"] - parallel["mrr_macro"]) < 1e-9
    assert abs(serial["ndcg_macro"] - parallel["ndcg_macro"]) < 1e-9


if __name__ == "__main__":
    fns = [
        test_perfect_retrieval_yields_recall_mrr_ndcg_1,
        test_zero_match_yields_all_zero_macro_metrics,
        test_macro_average_over_multiple_queries,
        test_per_query_order_matches_dataset_input,
        test_empty_dataset_raises,
        test_dataset_missing_relevant_ids_raises,
        test_max_concurrency_does_not_break_metric_correctness,
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

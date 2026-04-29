"""RaceRetriever — N retrievers race for first success, losers aborted.

Distinct from EnsembleRetriever (which fuses ALL children's results
via weighted RRF). RaceRetriever optimises for latency: hedge a fast
in-memory retriever against a slow remote one, take whichever
returns first."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    Bm25Index, MemoryVectorStore, RaceRetriever, VectorRetriever,
    retrieve_concurrent,
)


def _corpus():
    return [
        {"content": "rust memory safety", "id": "alpha"},
        {"content": "memory leaks in c", "id": "beta"},
        {"content": "javascript closures", "id": "gamma"},
    ]


def _bm25():
    idx = Bm25Index()
    idx.add(_corpus())
    return idx


def _vec():
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


def test_race_retriever_constructs_with_repr():
    r = RaceRetriever([_bm25(), _vec()])
    assert "RaceRetriever(children=2)" in repr(r)


def test_race_retriever_rejects_empty_list():
    try:
        RaceRetriever([])
    except ValueError as e:
        assert "at least one" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_race_retriever_rejects_non_retriever_child():
    try:
        RaceRetriever([_bm25(), "not a retriever"])
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected TypeError/ValueError")


def test_race_retriever_returns_a_winner_result():
    """One of the inners wins; the result is non-empty for a query
    that matches both. We don't pin which (GIL-serialised Python
    callbacks make scheduling order non-deterministic)."""
    r = RaceRetriever([_bm25(), _vec()])
    hits = r.retrieve("rust memory", k=2)
    assert isinstance(hits, list)
    # Both inner retrievers can produce hits for "rust memory"; the
    # contract is that we get a valid result with content present.
    assert hits and "content" in hits[0]


def test_race_retriever_single_inner_passthrough():
    r = RaceRetriever([_bm25()])
    hits = r.retrieve("rust", k=1)
    assert hits[0]["id"] == "alpha"


def test_race_retriever_composes_with_retrieve_concurrent():
    """RaceRetriever must be acceptable as input to other retriever-
    consumers via the central extract_retriever_arc helper."""
    r = RaceRetriever([_bm25(), _vec()])
    out = retrieve_concurrent(r, ["rust", "javascript"], k=1, max_concurrency=2)
    assert len(out) == 2


def test_race_retriever_three_children():
    """Three heterogeneous retrievers — race must accept and produce
    a valid result."""
    docs = _corpus()
    bm = _bm25()
    v1 = _vec()
    bm2 = Bm25Index()
    bm2.add(docs)
    r = RaceRetriever([bm, v1, bm2])
    hits = r.retrieve("memory", k=2)
    assert isinstance(hits, list)


if __name__ == "__main__":
    fns = [
        test_race_retriever_constructs_with_repr,
        test_race_retriever_rejects_empty_list,
        test_race_retriever_rejects_non_retriever_child,
        test_race_retriever_returns_a_winner_result,
        test_race_retriever_single_inner_passthrough,
        test_race_retriever_composes_with_retrieve_concurrent,
        test_race_retriever_three_children,
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

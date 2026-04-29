"""MaxMarginalRelevanceRetriever — diversity-aware retrieval. Wraps
any base retriever, over-fetches, then MMR-selects final K balancing
relevance vs novelty."""
from litgraph.retrieval import (
    Bm25Index,
    MaxMarginalRelevanceRetriever,
)
from litgraph.embeddings import FunctionEmbeddings


def keyword_embedder(keywords):
    def embed(texts):
        out = []
        for text in texts:
            lower = text.lower()
            out.append([1.0 if kw in lower else 0.0 for kw in keywords])
        return out
    return FunctionEmbeddings(embed, len(keywords))


def _bm25(docs):
    """Build a Bm25Index from a list of (id, content) pairs."""
    idx = Bm25Index()
    idx.add([{"id": doc_id, "content": content} for doc_id, content in docs])
    return idx


def test_mmr_returns_at_most_k_results():
    base = _bm25([
        ("a", "rust borrow checker"),
        ("b", "rust borrow tip"),
        ("c", "css flexbox"),
        ("d", "python comprehension"),
    ])
    embedder = keyword_embedder(["rust", "borrow", "css", "flexbox", "python", "comprehension"])
    mmr = MaxMarginalRelevanceRetriever(
        base=base, embeddings=embedder, fetch_k=4, lambda_mult=0.5,
    )
    docs = mmr.retrieve("rust borrow", k=2)
    assert len(docs) == 2


def test_lambda_one_equals_top_k_relevance():
    """lambda=1.0 → diversity term zeroed → behaves like vanilla top-K."""
    base = _bm25([
        ("a", "rust borrow lifetime"),
        ("b", "rust borrow"),
        ("c", "rust"),
        ("d", "css"),
    ])
    embedder = keyword_embedder(["rust", "borrow", "lifetime", "css"])
    mmr = MaxMarginalRelevanceRetriever(
        base=base, embeddings=embedder, fetch_k=4, lambda_mult=1.0,
    )
    docs = mmr.retrieve("rust borrow lifetime", k=3)
    assert len(docs) == 3
    contents = [d["content"] for d in docs]
    # Top-3 by relevance: 3-hit, 2-hit, 1-hit.
    assert contents[0] == "rust borrow lifetime"


def test_low_lambda_picks_diverse_results():
    """lambda=0.0 → among candidates the base retriever returns, MMR picks
    the most-spread-apart pair. Note: BM25 base only returns docs with
    non-zero query overlap — diversity has to be among those candidates."""
    # All 4 docs share "code" so BM25 returns them all; embedder then
    # measures their diversity in a richer keyword space.
    base = _bm25([
        ("a", "rust code about lifetime"),
        ("b", "rust code about lifetime annotation"),
        ("c", "rust code about lifetime syntax"),
        ("d", "python code about closures"),
    ])
    embedder = keyword_embedder(["rust", "lifetime", "python", "closure"])
    mmr = MaxMarginalRelevanceRetriever(
        base=base, embeddings=embedder, fetch_k=4, lambda_mult=0.0,
    )
    docs = mmr.retrieve("code", k=2)
    contents = [d["content"] for d in docs]
    # Diversity-only mode picks 2 maximally-different docs; rust + python
    # cover orthogonal keyword vectors.
    assert any("python" in c for c in contents), f"expected python doc in {contents}"


def test_k_zero_returns_empty():
    base = _bm25([("a", "rust")])
    embedder = keyword_embedder(["rust"])
    mmr = MaxMarginalRelevanceRetriever(base=base, embeddings=embedder)
    assert mmr.retrieve("q", k=0) == []


def test_empty_base_returns_empty():
    base = Bm25Index()
    embedder = keyword_embedder(["x"])
    mmr = MaxMarginalRelevanceRetriever(base=base, embeddings=embedder)
    assert mmr.retrieve("q", k=5) == []


def test_repr_shows_config():
    base = _bm25([])
    embedder = keyword_embedder(["x"])
    mmr = MaxMarginalRelevanceRetriever(
        base=base, embeddings=embedder, fetch_k=15, lambda_mult=0.7,
    )
    r = repr(mmr)
    assert "MaxMarginalRelevanceRetriever" in r
    assert "fetch_k=15" in r


def test_default_lambda_and_fetch_k_match_langchain():
    base = _bm25([])
    embedder = keyword_embedder(["x"])
    mmr = MaxMarginalRelevanceRetriever(base=base, embeddings=embedder)
    r = repr(mmr)
    assert "fetch_k=20" in r
    assert "lambda_mult=0.5" in r


if __name__ == "__main__":
    import traceback
    fns = [
        test_mmr_returns_at_most_k_results,
        test_lambda_one_equals_top_k_relevance,
        test_low_lambda_picks_diverse_results,
        test_k_zero_returns_empty,
        test_empty_base_returns_empty,
        test_repr_shows_config,
        test_default_lambda_and_fetch_k_match_langchain,
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

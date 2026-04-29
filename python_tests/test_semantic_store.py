"""SemanticStore — long-term memory with cosine semantic-search recall.

Wraps any Store (today: InMemoryStore) with an Embeddings provider.
Tests use FunctionEmbeddings so they run network-free."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.store import InMemoryStore, SemanticStore


def _toy_embedder():
    """2-D embedding: dim 0 fires on 'rust', dim 1 fires on 'memory'.
    Lets tests hand-pick which items are 'near' a query."""
    def embed(texts):
        out = []
        for t in texts:
            lower = t.lower()
            v = [
                1.0 if "rust" in lower else 0.0,
                1.0 if "memory" in lower else 0.0,
            ]
            # Normalise non-zero vectors so cosine is well-defined for
            # the [0.5, 0.5] fallback class too.
            if v == [0.0, 0.0]:
                v = [0.5, 0.5]
            out.append(v)
        return out
    return FunctionEmbeddings(embed, dimensions=2, name="toy")


def _fixture():
    return SemanticStore(InMemoryStore(), _toy_embedder())


def test_semantic_put_get_strips_embedding():
    s = _fixture()
    ns = ("users", "alice")
    s.put(ns, "f:1", "rust safety", {"k": 1})
    got = s.get(ns, "f:1")
    assert got is not None
    assert got["text"] == "rust safety"
    assert got["value"] == {"k": 1}


def test_semantic_get_missing_returns_none():
    s = _fixture()
    assert s.get(("nope",), "missing") is None


def test_semantic_search_ranks_top_hit():
    s = _fixture()
    ns = ("facts",)
    s.put(ns, "a", "rust safety", {"id": "a"})
    s.put(ns, "b", "memory leaks", {"id": "b"})
    s.put(ns, "c", "javascript closures", {"id": "c"})
    hits = s.search(ns, "rust performance", k=3)
    assert len(hits) == 3
    assert hits[0]["key"] == "a"
    # Scores monotone descending.
    for prev, curr in zip(hits, hits[1:]):
        assert prev["score"] >= curr["score"]


def test_semantic_search_caps_at_k():
    s = _fixture()
    ns = ("bag",)
    for i in range(7):
        s.put(ns, f"k{i}", "rust thing", {"i": i})
    hits = s.search(ns, "rust", k=3)
    assert len(hits) == 3


def test_semantic_search_empty_namespace_is_empty():
    s = _fixture()
    assert s.search(("empty",), "anything", k=5) == []


def test_semantic_search_namespace_isolation():
    s = _fixture()
    s.put(("users", "alice"), "k", "rust", {"who": "alice"})
    s.put(("users", "bob"), "k", "rust", {"who": "bob"})
    bob_hits = s.search(("users", "bob"), "rust", k=5)
    assert len(bob_hits) == 1
    assert bob_hits[0]["value"] == {"who": "bob"}


def test_semantic_delete_removes_from_search():
    s = _fixture()
    ns = ("d",)
    s.put(ns, "a", "rust safety", None)
    s.put(ns, "b", "rust threads", None)
    assert s.delete(ns, "a") is True
    hits = s.search(ns, "rust", k=5)
    assert len(hits) == 1
    assert hits[0]["key"] == "b"


def test_semantic_search_k_zero_returns_empty():
    s = _fixture()
    s.put(("z",), "k", "rust", None)
    assert s.search(("z",), "rust", k=0) == []


def test_semantic_store_rejects_non_store():
    try:
        SemanticStore("not a store", _toy_embedder())
    except RuntimeError as e:
        assert "InMemoryStore" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


if __name__ == "__main__":
    fns = [
        test_semantic_put_get_strips_embedding,
        test_semantic_get_missing_returns_none,
        test_semantic_search_ranks_top_hit,
        test_semantic_search_caps_at_k,
        test_semantic_search_empty_namespace_is_empty,
        test_semantic_search_namespace_isolation,
        test_semantic_delete_removes_from_search,
        test_semantic_search_k_zero_returns_empty,
        test_semantic_store_rejects_non_store,
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

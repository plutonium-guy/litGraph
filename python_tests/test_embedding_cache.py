"""CachedEmbeddings — read-through cache that batches misses-only to the
inner provider. Verifies the cost-saving claim: re-embedding the same corpus
hits the cache; only new texts go to the API."""
from litgraph.cache import CachedEmbeddings, MemoryEmbeddingCache
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import MemoryVectorStore, VectorRetriever


def _counting_embedder():
    """Returns (embedder, calls_dict). `calls_dict["count"]` increments by N
    on each call where N = number of texts in the call."""
    calls = {"count": 0}
    def embed(texts):
        calls["count"] += len(texts)
        return [[float(len(t)), 1.0] for t in texts]
    e = FunctionEmbeddings(embed, dimensions=2, name="counter")
    return e, calls


def test_query_caches_on_second_call():
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)

    v1 = wrapped.embed_query("hello")
    v2 = wrapped.embed_query("hello")
    assert v1 == v2
    # Inner saw "hello" exactly once.
    assert calls["count"] == 1


def test_documents_partition_only_misses_go_to_inner():
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)

    # Warm the cache with two texts.
    wrapped.embed_documents(["alpha", "beta"])
    assert calls["count"] == 2

    # Mixed batch: 2 hits + 2 new misses.
    out = wrapped.embed_documents(["alpha", "gamma", "beta", "delta"])
    assert len(out) == 4
    # Inner only saw the 2 misses.
    assert calls["count"] == 2 + 2

    # Order preserved.
    assert out[0] == [float(len("alpha")), 1.0]
    assert out[1] == [float(len("gamma")), 1.0]
    assert out[2] == [float(len("beta")), 1.0]
    assert out[3] == [float(len("delta")), 1.0]


def test_empty_documents_no_inner_call():
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)
    out = wrapped.embed_documents([])
    assert out == []
    assert calls["count"] == 0


def test_clear_forces_re_embedding():
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)

    wrapped.embed_query("foo")
    assert calls["count"] == 1
    cache.clear()
    wrapped.embed_query("foo")
    assert calls["count"] == 2


def test_name_and_dimensions_match_inner():
    inner, _ = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)
    assert wrapped.name == "counter"
    assert wrapped.dimensions == 2


def test_vector_retriever_accepts_cached_embeddings():
    """CachedEmbeddings must drop straight into the existing retriever
    extractor — no new code path required."""
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache()
    wrapped = CachedEmbeddings(inner, cache)
    store = MemoryVectorStore()
    docs = [{"content": "alpha", "id": "a"}, {"content": "beta", "id": "b"}]
    store.add(docs, wrapped.embed_documents([d["content"] for d in docs]))
    # The above hit the inner once for both texts.
    n_after_indexing = calls["count"]
    assert n_after_indexing == 2

    retriever = VectorRetriever(wrapped, store)
    hits = retriever.retrieve("alpha", k=2)
    assert len(hits) == 2
    # The query "alpha" is already cached → inner NOT called again.
    assert calls["count"] == 2


def test_ttl_eviction_path():
    """TTL=0.1s; sleep past it; re-query → must re-call inner."""
    import time
    inner, calls = _counting_embedder()
    cache = MemoryEmbeddingCache(max_capacity=100, ttl_s=1)  # 1s TTL
    wrapped = CachedEmbeddings(inner, cache)
    wrapped.embed_query("ttl-test")
    assert calls["count"] == 1
    time.sleep(1.2)  # past TTL
    wrapped.embed_query("ttl-test")
    assert calls["count"] == 2


if __name__ == "__main__":
    fns = [
        test_query_caches_on_second_call,
        test_documents_partition_only_misses_go_to_inner,
        test_empty_documents_no_inner_call,
        test_clear_forces_re_embedding,
        test_name_and_dimensions_match_inner,
        test_vector_retriever_accepts_cached_embeddings,
        test_ttl_eviction_path,
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

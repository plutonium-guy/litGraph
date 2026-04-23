"""SqliteEmbeddingCache — durable embedding cache surviving process restarts.

The canonical production win: a nightly indexing job that runs 3x across a
week should only pay to embed the NEW docs. MemoryEmbeddingCache dies with
the process; SqliteEmbeddingCache keeps the vectors on disk."""
import os
import tempfile

from litgraph.cache import (
    CachedEmbeddings, MemoryEmbeddingCache, SqliteEmbeddingCache,
)
from litgraph.embeddings import FunctionEmbeddings


def _counting():
    """Counting embedder — same as iter 72 test helper."""
    calls = {"count": 0}
    def embed(texts):
        calls["count"] += len(texts)
        return [[float(len(t)), 1.0] for t in texts]
    return FunctionEmbeddings(embed, dimensions=2, name="counter"), calls


def test_in_memory_sqlite_behaves_like_memory_cache():
    """Functional equivalence: SqliteEmbeddingCache.in_memory() behind
    CachedEmbeddings should give the same hit behavior as MemoryEmbeddingCache."""
    e, calls = _counting()
    cache = SqliteEmbeddingCache.in_memory()
    wrapped = CachedEmbeddings(e, cache)
    wrapped.embed_query("hello")
    wrapped.embed_query("hello")
    assert calls["count"] == 1


def test_documents_partition_against_sqlite_backend():
    e, calls = _counting()
    cache = SqliteEmbeddingCache.in_memory()
    wrapped = CachedEmbeddings(e, cache)

    wrapped.embed_documents(["alpha", "beta"])
    assert calls["count"] == 2
    out = wrapped.embed_documents(["alpha", "gamma", "beta", "delta"])
    assert len(out) == 4
    # Only the two new texts went to the inner provider.
    assert calls["count"] == 4


def test_durability_across_instances_via_file():
    """Write embeddings via one cache instance; close it; open a NEW instance
    at the same path; the vectors are still there. This is the cross-process
    guarantee the sqlite backend exists for."""
    fd, path = tempfile.mkstemp(prefix="emb-", suffix=".db")
    os.close(fd)
    os.unlink(path)  # SqliteEmbeddingCache.open creates the file itself.
    try:
        # Run 1: cache 3 texts.
        e1, calls1 = _counting()
        c1 = SqliteEmbeddingCache(path)
        w1 = CachedEmbeddings(e1, c1)
        w1.embed_documents(["dog", "cat", "bird"])
        assert calls1["count"] == 3
        # Drop the wrapper + cache (simulates process exit).
        del w1, c1, e1

        # Run 2: open SAME path with a fresh embedder (counter reset).
        e2, calls2 = _counting()
        c2 = SqliteEmbeddingCache(path)
        w2 = CachedEmbeddings(e2, c2)
        # All 3 texts are cached on disk → zero inner calls.
        out = w2.embed_documents(["dog", "cat", "bird"])
        assert len(out) == 3
        assert calls2["count"] == 0, "durability broken: expected 0 inner calls, got"

        # New text: partial miss — inner sees only "fish".
        w2.embed_documents(["dog", "fish"])
        assert calls2["count"] == 1
    finally:
        try: os.unlink(path)
        except FileNotFoundError: pass
        # WAL files alongside.
        for suffix in ("-shm", "-wal"):
            try: os.unlink(path + suffix)
            except FileNotFoundError: pass


def test_clear_empties_sqlite_cache():
    fd, path = tempfile.mkstemp(prefix="emb-clear-", suffix=".db")
    os.close(fd); os.unlink(path)
    try:
        e, calls = _counting()
        cache = SqliteEmbeddingCache(path)
        wrapped = CachedEmbeddings(e, cache)
        wrapped.embed_query("x")
        assert calls["count"] == 1
        cache.clear()
        wrapped.embed_query("x")
        assert calls["count"] == 2
    finally:
        for suffix in ("", "-shm", "-wal"):
            try: os.unlink(path + suffix)
            except FileNotFoundError: pass


def test_cached_embeddings_rejects_non_cache_argument():
    e, _ = _counting()
    try:
        CachedEmbeddings(e, "not a cache")
    except TypeError as exc:
        assert "MemoryEmbeddingCache or SqliteEmbeddingCache" in str(exc)
    else:
        raise AssertionError("expected TypeError")


def test_both_backends_are_interchangeable_behind_cached_embeddings():
    """Same wrapped behavior regardless of backend — the Cache trait's job."""
    for maker in [lambda: MemoryEmbeddingCache(),
                  lambda: SqliteEmbeddingCache.in_memory()]:
        e, calls = _counting()
        wrapped = CachedEmbeddings(e, maker())
        wrapped.embed_documents(["a", "b"])
        wrapped.embed_documents(["a", "b", "c"])
        # First batch: 2 inner; second batch: 1 inner (miss). Total = 3.
        assert calls["count"] == 3


if __name__ == "__main__":
    fns = [
        test_in_memory_sqlite_behaves_like_memory_cache,
        test_documents_partition_against_sqlite_backend,
        test_durability_across_instances_via_file,
        test_clear_empties_sqlite_cache,
        test_cached_embeddings_rejects_non_cache_argument,
        test_both_backends_are_interchangeable_behind_cached_embeddings,
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

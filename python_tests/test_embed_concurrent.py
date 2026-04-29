"""embed_documents_concurrent — bounded-concurrency parallel embedder.

Uses FunctionEmbeddings (in-process, deterministic) so tests are
network-free and fast. The point is to verify the parallel-chunking
plumbing — chunk splitting, alignment, error propagation."""
from litgraph.embeddings import FunctionEmbeddings, embed_documents_concurrent


def _toy_embedder():
    """Returns a 1-D vector containing the text length. Trivial,
    deterministic, sufficient for alignment + error-path tests."""
    def embed(texts):
        return [[float(len(t))] for t in texts]
    return FunctionEmbeddings(embed, dimensions=1, name="toy-len")


def test_embed_concurrent_empty_returns_empty():
    out = embed_documents_concurrent(_toy_embedder(), [], chunk_size=4)
    assert out == []


def test_embed_concurrent_aligns_output_to_input():
    """Chunking must not reorder. Output[i] = embedding(input[i])."""
    texts = [f"text-of-len-{i:03d}-pad" for i in range(13)]
    out = embed_documents_concurrent(
        _toy_embedder(), texts, chunk_size=4, max_concurrency=4
    )
    assert len(out) == 13
    for i, v in enumerate(out):
        assert int(v[0]) == len(texts[i])


def test_embed_concurrent_chunk_size_zero_means_one_call():
    texts = ["a", "bb", "ccc"]
    out = embed_documents_concurrent(
        _toy_embedder(), texts, chunk_size=0, max_concurrency=4
    )
    assert [int(v[0]) for v in out] == [1, 2, 3]


def test_embed_concurrent_max_concurrency_zero_normalised():
    texts = ["a", "b", "c", "d"]
    out = embed_documents_concurrent(
        _toy_embedder(), texts, chunk_size=2, max_concurrency=0
    )
    assert [int(v[0]) for v in out] == [1, 1, 1, 1]


def test_embed_concurrent_propagates_failure():
    """Inner embedder that raises must surface as RuntimeError."""
    def bad_embed(_texts):
        raise RuntimeError("synthetic embed failure")
    bad = FunctionEmbeddings(bad_embed, dimensions=1, name="bad")
    try:
        embed_documents_concurrent(bad, ["a", "b"], chunk_size=1, max_concurrency=2)
    except RuntimeError as e:
        assert "synthetic embed failure" in str(e)
    else:
        raise AssertionError("expected RuntimeError")


def test_embed_concurrent_rejects_non_embedder():
    try:
        embed_documents_concurrent("not an embedder", ["a"], chunk_size=4)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_embed_concurrent_empty_returns_empty,
        test_embed_concurrent_aligns_output_to_input,
        test_embed_concurrent_chunk_size_zero_means_one_call,
        test_embed_concurrent_max_concurrency_zero_normalised,
        test_embed_concurrent_propagates_failure,
        test_embed_concurrent_rejects_non_embedder,
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

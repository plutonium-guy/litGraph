"""FallbackEmbeddings — cross-provider embedding failover. Parallels
FallbackChat for chat models. Dimensions must match across providers
(else construction raises ValueError)."""
from litgraph.embeddings import FallbackEmbeddings, FunctionEmbeddings


def _fn_embed(dim: int, name: str, fail: bool = False):
    """Build a FunctionEmbeddings that returns `dim`-dim zero vectors, or
    raises RuntimeError if `fail=True`."""
    def embed(texts):
        if fail:
            raise RuntimeError("simulated provider error")
        return [[0.0] * dim for _ in texts]

    return FunctionEmbeddings(func=embed, dimensions=dim, name=name)


def test_matching_dims_construct_ok():
    primary = _fn_embed(768, "primary")
    backup = _fn_embed(768, "backup")
    emb = FallbackEmbeddings([primary, backup])
    assert emb.dimensions == 768


def test_mismatched_dims_raise_value_error_at_construction():
    primary = _fn_embed(1536, "openai-3-small")
    backup = _fn_embed(1024, "voyage-3-large")
    try:
        FallbackEmbeddings([primary, backup])
        raise AssertionError("expected ValueError for dim mismatch")
    except ValueError as e:
        assert "dim" in str(e).lower()


def test_empty_providers_list_raises():
    try:
        FallbackEmbeddings([])
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "non-empty" in str(e).lower() or "providers" in str(e).lower()


def test_primary_success_no_backup_called():
    """Healthy primary: embed_query + embed_documents both succeed,
    backup's tracking counter never increments."""
    backup_calls = {"n": 0}

    def backup_func(texts):
        backup_calls["n"] += 1
        return [[0.0] * 512 for _ in texts]

    primary = _fn_embed(512, "primary")
    backup = FunctionEmbeddings(func=backup_func, dimensions=512, name="backup")
    emb = FallbackEmbeddings([primary, backup])

    v = emb.embed_query("hi")
    assert len(v) == 512
    vs = emb.embed_documents(["a", "b"])
    assert len(vs) == 2
    assert backup_calls["n"] == 0


def test_single_provider_fallback_works_as_identity():
    """Chain of size 1 just proxies to the sole provider."""
    only = _fn_embed(256, "only")
    emb = FallbackEmbeddings([only])
    v = emb.embed_query("hello")
    assert len(v) == 256


def test_repr_contains_dimensions():
    emb = FallbackEmbeddings([_fn_embed(1024, "a")])
    r = repr(emb)
    assert "FallbackEmbeddings" in r
    assert "1024" in r


def test_composes_as_embeddings_consumer_sees_dim_via_getter():
    """FallbackEmbeddings implements the Embeddings trait, so any
    downstream code expecting that trait works."""
    emb = FallbackEmbeddings([_fn_embed(768, "p"), _fn_embed(768, "b")])
    # Standard Embeddings interface: `.dimensions` getter.
    assert emb.dimensions == 768
    # And both embed methods return the right shape.
    assert len(emb.embed_query("x")) == 768
    assert len(emb.embed_documents(["x", "y", "z"])[0]) == 768


if __name__ == "__main__":
    import traceback
    fns = [
        test_matching_dims_construct_ok,
        test_mismatched_dims_raise_value_error_at_construction,
        test_empty_providers_list_raises,
        test_primary_success_no_backup_called,
        test_single_provider_fallback_works_as_identity,
        test_repr_contains_dimensions,
        test_composes_as_embeddings_consumer_sees_dim_via_getter,
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

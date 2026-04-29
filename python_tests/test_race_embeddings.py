"""RaceEmbeddings — race N embedders concurrently, first success wins.

Network-free tests using FunctionEmbeddings so behaviour is
deterministic. Uses small artificial delays to simulate latency
differences between providers."""
import time

from litgraph.embeddings import FunctionEmbeddings, RaceEmbeddings


def _delayed_embedder(name, dim, delay_ms, marker):
    """Returns a FunctionEmbeddings that sleeps then emits a constant
    `marker`-valued vector. Lets tests pin which provider wins by
    delay."""
    def embed(texts):
        time.sleep(delay_ms / 1000.0)
        return [[float(marker)] * dim for _ in texts]
    return FunctionEmbeddings(embed, dimensions=dim, name=name)


def test_race_embeddings_rejects_empty_list():
    try:
        RaceEmbeddings([])
    except ValueError as e:
        assert "non-empty" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_race_embeddings_rejects_dim_mismatch():
    a = _delayed_embedder("a", 4, 0, 0.0)
    b = _delayed_embedder("b", 8, 0, 0.0)
    try:
        RaceEmbeddings([a, b])
    except ValueError as e:
        assert "dim" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def _close(actual, expected, tol=1e-5):
    """f32 round-trips through Python introduces ~1e-7 noise; use a
    generous tolerance for marker-vector comparisons."""
    return all(abs(a - e) < tol for a, e in zip(actual, expected))


def test_race_embeddings_query_returns_a_valid_winner():
    """Both providers return distinguishable markers — race must return
    exactly one of them (the GIL serialises FunctionEmbeddings calls so
    we don't assert a specific winner deterministically). Contract:
    shape is right + the result matches one of the markers."""
    a = _delayed_embedder("a", 4, 0, 0.42)
    b = _delayed_embedder("b", 4, 0, 0.99)
    race = RaceEmbeddings([a, b])
    out = race.embed_query("hi")
    assert len(out) == 4
    assert _close(out, [0.42] * 4) or _close(out, [0.99] * 4)


def test_race_embeddings_documents_returns_winner_shape():
    a = _delayed_embedder("a", 4, 0, 0.5)
    b = _delayed_embedder("b", 4, 0, 0.9)
    race = RaceEmbeddings([a, b])
    out = race.embed_documents(["hi", "there"])
    assert len(out) == 2
    assert all(len(v) == 4 for v in out)
    # Single winning provider → marker consistent across the batch.
    assert _close(out[0], out[1])
    assert _close(out[0], [0.5] * 4) or _close(out[0], [0.9] * 4)


def test_race_embeddings_falls_through_failures():
    """Failing provider lets the working provider win.
    Single working provider → result must be its marker."""
    def bad(_texts):
        raise RuntimeError("bad provider")
    fail = FunctionEmbeddings(bad, dimensions=4, name="fail")
    good = _delayed_embedder("good", 4, 0, 0.7)
    race = RaceEmbeddings([fail, good])
    out = race.embed_query("q")
    assert _close(out, [0.7] * 4)


def test_race_embeddings_aggregates_when_all_fail():
    def bad_a(_texts):
        raise RuntimeError("A failed")
    def bad_b(_texts):
        raise RuntimeError("B failed")
    a = FunctionEmbeddings(bad_a, dimensions=4, name="a")
    b = FunctionEmbeddings(bad_b, dimensions=4, name="b")
    race = RaceEmbeddings([a, b])
    try:
        race.embed_query("q")
    except RuntimeError as e:
        msg = str(e)
        assert "all 2 inners failed" in msg
    else:
        raise AssertionError("expected RuntimeError")


def test_race_embeddings_dimensions_property():
    a = _delayed_embedder("a", 7, 0, 0.0)
    b = _delayed_embedder("b", 7, 0, 0.0)
    race = RaceEmbeddings([a, b])
    assert race.dimensions == 7


def test_race_embeddings_extractable_via_other_wrappers():
    """Extracting through `extract_embeddings` lets RaceEmbeddings
    plug into RetryingEmbeddings / RateLimitedEmbeddings / etc."""
    from litgraph.embeddings import RetryingEmbeddings
    a = _delayed_embedder("a", 4, 0, 0.0)
    b = _delayed_embedder("b", 4, 0, 0.0)
    race = RaceEmbeddings([a, b])
    wrapped = RetryingEmbeddings(race, max_retries=2)
    assert wrapped is not None


if __name__ == "__main__":
    fns = [
        test_race_embeddings_rejects_empty_list,
        test_race_embeddings_rejects_dim_mismatch,
        test_race_embeddings_query_returns_a_valid_winner,
        test_race_embeddings_documents_returns_winner_shape,
        test_race_embeddings_falls_through_failures,
        test_race_embeddings_aggregates_when_all_fail,
        test_race_embeddings_dimensions_property,
        test_race_embeddings_extractable_via_other_wrappers,
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

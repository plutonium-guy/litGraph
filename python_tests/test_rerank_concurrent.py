"""rerank_concurrent — fan one reranker over N (query, candidates) pairs.

Network-free tests can't actually rerank because all registered
rerankers (Cohere/Voyage/Jina) require API keys. We exercise the
input parsing + dispatch + per-pair-error-isolation paths against
unreachable hosts; the actual rerank scoring is covered by the
Rust test suite with a mock Reranker."""
from litgraph.retrieval import (
    CohereReranker, EnsembleReranker, JinaReranker, VoyageReranker,
    rerank_concurrent,
)


def _broken_cohere(label):
    return CohereReranker(api_key=f"ck-{label}", base_url="http://127.0.0.1:1")


def _doc(id_: str, content: str = "x"):
    return {"id": id_, "content": content}


def test_rerank_concurrent_returns_one_slot_per_pair():
    r = _broken_cohere("a")
    pairs = [
        {"query": "q1", "candidates": [_doc("d1"), _doc("d2")]},
        {"query": "q2", "candidates": [_doc("d3")]},
    ]
    out = rerank_concurrent(r, pairs, top_k=5, max_concurrency=2)
    assert len(out) == 2


def test_rerank_concurrent_isolates_per_pair_errors_against_unreachable():
    """Every call fails (broken host); all slots become error dicts,
    iterator doesn't raise."""
    r = _broken_cohere("a")
    pairs = [
        {"query": f"q{i}", "candidates": [_doc(f"d{i}")]} for i in range(3)
    ]
    out = rerank_concurrent(r, pairs, top_k=2, max_concurrency=4)
    assert len(out) == 3
    for slot in out:
        assert "error" in slot


def test_rerank_concurrent_fail_fast_raises_against_unreachable():
    r = _broken_cohere("a")
    pairs = [{"query": "q", "candidates": [_doc("d")]}]
    try:
        rerank_concurrent(r, pairs, top_k=1, max_concurrency=2, fail_fast=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_rerank_concurrent_empty_returns_empty():
    r = _broken_cohere("a")
    assert rerank_concurrent(r, [], top_k=5, max_concurrency=4) == []


def test_rerank_concurrent_accepts_tuple_pairs():
    r = _broken_cohere("a")
    pairs = [("q1", [_doc("d1")]), ("q2", [_doc("d2")])]
    out = rerank_concurrent(r, pairs, top_k=1, max_concurrency=2)
    assert len(out) == 2


def test_rerank_concurrent_rejects_malformed_pair():
    r = _broken_cohere("a")
    try:
        rerank_concurrent(r, ["not a tuple or dict"], top_k=1, max_concurrency=2)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError")


def test_rerank_concurrent_rejects_pair_missing_keys():
    r = _broken_cohere("a")
    pairs = [{"query": "q1"}]  # missing candidates
    try:
        rerank_concurrent(r, pairs, top_k=1, max_concurrency=2)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError on missing 'candidates'")


def test_rerank_concurrent_rejects_non_reranker():
    pairs = [{"query": "q", "candidates": [_doc("d")]}]
    try:
        rerank_concurrent("not a reranker", pairs, top_k=1)
    except (TypeError, ValueError):
        pass
    else:
        raise AssertionError("expected TypeError/ValueError")


def test_rerank_concurrent_accepts_ensemble_reranker():
    """EnsembleReranker should compose as a single reranker."""
    a = _broken_cohere("a")
    b = JinaReranker(api_key="jk", base_url="http://127.0.0.1:1")
    ensemble = EnsembleReranker([a, b])
    pairs = [{"query": "q", "candidates": [_doc("d")]}]
    out = rerank_concurrent(ensemble, pairs, top_k=1, max_concurrency=2)
    assert len(out) == 1


def test_rerank_concurrent_accepts_voyage_reranker():
    r = VoyageReranker(api_key="vk", base_url="http://127.0.0.1:1")
    pairs = [{"query": "q", "candidates": [_doc("d")]}]
    out = rerank_concurrent(r, pairs, top_k=1, max_concurrency=2)
    assert len(out) == 1


if __name__ == "__main__":
    fns = [
        test_rerank_concurrent_returns_one_slot_per_pair,
        test_rerank_concurrent_isolates_per_pair_errors_against_unreachable,
        test_rerank_concurrent_fail_fast_raises_against_unreachable,
        test_rerank_concurrent_empty_returns_empty,
        test_rerank_concurrent_accepts_tuple_pairs,
        test_rerank_concurrent_rejects_malformed_pair,
        test_rerank_concurrent_rejects_pair_missing_keys,
        test_rerank_concurrent_rejects_non_reranker,
        test_rerank_concurrent_accepts_ensemble_reranker,
        test_rerank_concurrent_accepts_voyage_reranker,
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

"""RetryingEmbeddings + RateLimitedEmbeddings — embed-side parallels of
the chat-model resilience wrappers. Completes the embedding resilience
trio started in iter 133 with FallbackEmbeddings."""
import time

from litgraph.embeddings import (
    FallbackEmbeddings,
    FunctionEmbeddings,
    RateLimitedEmbeddings,
    RetryingEmbeddings,
)


def test_retrying_exposes_inner_dim():
    raw = FunctionEmbeddings(
        func=lambda texts: [[0.0] * 512 for _ in texts],
        dimensions=512,
        name="raw",
    )
    r = RetryingEmbeddings(raw, max_retries=3)
    assert r.dimensions == 512
    # Successful path passes through unchanged.
    v = r.embed_query("hi")
    assert len(v) == 512


def test_retrying_repr_mentions_dim():
    raw = FunctionEmbeddings(
        func=lambda t: [[0.0] * 128 for _ in t], dimensions=128, name="r",
    )
    r = RetryingEmbeddings(raw, max_retries=2)
    assert "RetryingEmbeddings" in repr(r)
    assert "128" in repr(r)


def test_retrying_composes_with_fallback():
    """Retry(Fallback(list)) — stack the two wrappers. Common prod shape:
    retry each inner first, then fall over to the next on persistent fail."""
    a = FunctionEmbeddings(
        func=lambda t: [[0.1] * 256 for _ in t], dimensions=256, name="a",
    )
    b = FunctionEmbeddings(
        func=lambda t: [[0.2] * 256 for _ in t], dimensions=256, name="b",
    )
    fb = FallbackEmbeddings([a, b])
    # Now wrap the fallback in retry.
    r = RetryingEmbeddings(fb, max_retries=2, min_delay_ms=1, max_delay_ms=5)
    assert r.dimensions == 256
    r.embed_query("hi")


def test_ratelimit_steady_state_respects_rpm():
    """4 calls @ 120 RPM (2 RPS) burst=1 → first instant, then 500ms/each
    → total ~1.5s."""
    raw = FunctionEmbeddings(
        func=lambda t: [[0.0] * 64 for _ in t], dimensions=64, name="raw",
    )
    r = RateLimitedEmbeddings(raw, requests_per_minute=120, burst=1)
    start = time.monotonic()
    for _ in range(4):
        r.embed_query("hi")
    elapsed = time.monotonic() - start
    assert 1.3 <= elapsed <= 2.2, f"expected ~1.5s, got {elapsed:.2f}s"


def test_ratelimit_burst_absorbs_spike():
    """burst=5 → first 5 calls run immediately, 6th throttles."""
    raw = FunctionEmbeddings(
        func=lambda t: [[0.0] * 32 for _ in t], dimensions=32, name="raw",
    )
    r = RateLimitedEmbeddings(raw, requests_per_minute=60, burst=5)
    start = time.monotonic()
    for _ in range(5):
        r.embed_query("hi")
    burst_time = time.monotonic() - start
    assert burst_time < 0.2, f"burst should be instant, took {burst_time:.3f}s"
    # 6th call forces a wait at 1 RPS steady state.
    r.embed_query("hi")
    total = time.monotonic() - start
    assert total >= 0.9, f"6th call should wait ~1s, total={total:.2f}s"


def test_ratelimit_batch_counts_as_one_token():
    """embed_documents on 100 texts consumes 1 token (not 100). Matches
    provider billing semantics (per-call, not per-text)."""
    raw = FunctionEmbeddings(
        func=lambda t: [[0.0] * 128 for _ in t], dimensions=128, name="raw",
    )
    r = RateLimitedEmbeddings(raw, requests_per_minute=60, burst=1)
    # First batch — uses the burst token instantly.
    start = time.monotonic()
    r.embed_documents(["text_%d" % i for i in range(100)])
    first = time.monotonic() - start
    assert first < 0.1
    # Second batch — waits ~1s (1 RPS).
    r.embed_documents(["next"])
    total = time.monotonic() - start
    assert total >= 0.9


def test_ratelimit_exposes_inner_dim():
    raw = FunctionEmbeddings(
        func=lambda t: [[0.0] * 1536 for _ in t], dimensions=1536, name="raw",
    )
    r = RateLimitedEmbeddings(raw, requests_per_minute=10_000)
    assert r.dimensions == 1536


def test_stacked_retry_ratelimit_fallback_compose():
    """Full prod stack: RateLimit(Retry(Fallback([primary, backup])))."""
    a = FunctionEmbeddings(
        func=lambda t: [[0.1] * 768 for _ in t], dimensions=768, name="a",
    )
    b = FunctionEmbeddings(
        func=lambda t: [[0.2] * 768 for _ in t], dimensions=768, name="b",
    )
    fb = FallbackEmbeddings([a, b])
    retried = RetryingEmbeddings(fb, max_retries=3, min_delay_ms=1, max_delay_ms=3)
    throttled = RateLimitedEmbeddings(retried, requests_per_minute=6000, burst=10)
    # Single call exercises the whole stack.
    v = throttled.embed_query("hi")
    assert len(v) == 768


if __name__ == "__main__":
    import traceback
    fns = [
        test_retrying_exposes_inner_dim,
        test_retrying_repr_mentions_dim,
        test_retrying_composes_with_fallback,
        test_ratelimit_steady_state_respects_rpm,
        test_ratelimit_burst_absorbs_spike,
        test_ratelimit_batch_counts_as_one_token,
        test_ratelimit_exposes_inner_dim,
        test_stacked_retry_ratelimit_fallback_compose,
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

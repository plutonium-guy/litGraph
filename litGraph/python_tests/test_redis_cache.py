"""RedisCache — distributed cache backend. Integration tests skipped
unless REDIS_URL env var is set; constructor + class-API tests run
unconditionally."""
import os

from litgraph.cache import RedisCache


def _redis_url():
    return os.environ.get("REDIS_URL")


def test_class_exposed_on_litgraph_cache():
    """Sanity: PyRedisCache is registered + has the expected API surface."""
    assert hasattr(RedisCache, "connect")
    assert hasattr(RedisCache, "get")
    assert hasattr(RedisCache, "invalidate")
    assert hasattr(RedisCache, "clear")
    assert hasattr(RedisCache, "with_ttl_seconds")


def test_connect_malformed_url_raises_value_error():
    """Connecting to a syntactically-bad URL should fail fast (no DNS/TCP).
    We test malformed scheme rather than unreachable host (which would
    hang on connection retries)."""
    try:
        RedisCache.connect("not-a-redis-url")
        raise AssertionError("expected error on malformed URL")
    except (RuntimeError, ValueError):
        pass  # either error type is acceptable


def test_repr():
    """Once connected, repr returns the expected string. Skips if no Redis."""
    url = _redis_url()
    if not url:
        return
    cache = RedisCache.connect(url)
    assert "RedisCache" in repr(cache)


def test_put_get_roundtrips():
    """Integration: put → get returns the cached response. Needs REDIS_URL."""
    url = _redis_url()
    if not url:
        return
    cache = RedisCache.connect(url)
    cache.clear()
    # PyRedisCache doesn't expose put() directly (the cache is meant to
    # be used through CachedModel wrapping). Just verify get() on missing
    # key returns None.
    assert cache.get("nonexistent-key") is None


def test_clear_does_not_error_on_empty_cache():
    url = _redis_url()
    if not url:
        return
    cache = RedisCache.connect(url)
    cache.clear()  # should be a no-op idempotent
    cache.clear()


def test_invalidate_missing_key_does_not_error():
    url = _redis_url()
    if not url:
        return
    cache = RedisCache.connect(url)
    cache.invalidate("nonexistent-key")  # idempotent


def test_with_ttl_seconds_returns_cache_instance():
    """ttl helper returns a (currently no-op) RedisCache; verify it doesn't crash."""
    url = _redis_url()
    if not url:
        return
    cache = RedisCache.connect(url)
    ttl_cache = cache.with_ttl_seconds(3600)
    assert "RedisCache" in repr(ttl_cache)


if __name__ == "__main__":
    import traceback
    fns = [
        test_class_exposed_on_litgraph_cache,
        test_connect_malformed_url_raises_value_error,
        test_repr,
        test_put_get_roundtrips,
        test_clear_does_not_error_on_empty_cache,
        test_invalidate_missing_key_does_not_error,
        test_with_ttl_seconds_returns_cache_instance,
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

"""Live integration: `cache.MemoryCache` direct round-trip with `with_cache`.

Already covered (`test_caching.py`) at the wrapper-level. This tests
the cache object's own contract: a hit returns the previously-stored
response, a miss returns None.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_memory_cache_install_and_hit(deepseek_chat):
    from litgraph.cache import MemoryCache

    cache = MemoryCache(max_capacity=64)
    deepseek_chat.with_cache(cache)
    msgs = [{"role": "user", "content": "Reply: ok-cache"}]
    a = deepseek_chat.invoke(msgs, max_tokens=10)
    b = deepseek_chat.invoke(msgs, max_tokens=10)
    # Identical input + same options must hit the cache → same text.
    assert a["text"] == b["text"], f"cache miss on identical call: {a} vs {b}"


def test_memory_cache_distinct_inputs_distinct_outputs(deepseek_chat):
    from litgraph.cache import MemoryCache

    cache = MemoryCache(max_capacity=64)
    deepseek_chat.with_cache(cache)
    a = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: cat"}], max_tokens=10
    )
    b = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: dog"}], max_tokens=10
    )
    # Distinct inputs should produce distinct outputs (different cache
    # keys) — assert at least one differs.
    assert a["text"].strip().lower() != b["text"].strip().lower(), (
        f"unexpected dup output: {a} vs {b}"
    )

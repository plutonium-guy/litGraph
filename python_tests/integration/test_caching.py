"""Live integration: model.with_cache(...) hit / miss against DeepSeek.

API: `model.with_cache(cache)` mutates the model in place + returns
`None` (matches `instrument` / `with_retry` / `with_rate_limit`).
The original `model` reference now reads/writes through `cache`.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_cache_hit_returns_same_text(deepseek_chat):
    from litgraph.cache import MemoryCache

    deepseek_chat.with_cache(MemoryCache(max_capacity=64))
    msgs = [{"role": "user", "content": "Reply with: PINGPING"}]

    a = deepseek_chat.invoke(msgs, max_tokens=10, temperature=0)
    b = deepseek_chat.invoke(msgs, max_tokens=10, temperature=0)

    # Cached: text is byte-identical (full ChatResponse cached).
    assert a["text"] == b["text"]


def test_cache_miss_on_different_prompt(deepseek_chat):
    from litgraph.cache import MemoryCache

    deepseek_chat.with_cache(MemoryCache(max_capacity=64))
    a = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with: ALPHA"}],
        max_tokens=10, temperature=0,
    )
    b = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with: BETA"}],
        max_tokens=10, temperature=0,
    )
    assert a["text"] != b["text"]

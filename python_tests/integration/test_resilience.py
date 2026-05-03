"""Live integration: resilience wrappers against DeepSeek.

API note: `with_retry()` / `with_rate_limit()` mutate the model in
place + return `None`. Use the original model reference after.
Wrappers should pass the happy path through unchanged.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_retry_wrapper_passes_through_on_success(deepseek_chat):
    """Wrap with retry; happy path should return identical shape."""
    deepseek_chat.with_retry(max_times=3, min_delay_ms=200)
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    assert isinstance(out["text"], str)
    assert out["text"]


def test_rate_limit_wrapper_passes_through(deepseek_chat):
    """Wrap with rate limit; one call must still succeed."""
    deepseek_chat.with_rate_limit(requests_per_minute=120)
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    assert out["text"]


def test_combined_retry_and_rate_limit(deepseek_chat):
    """Wrappers compose: stack retry + rate-limit on the same model."""
    deepseek_chat.with_retry(max_times=2, min_delay_ms=200)
    deepseek_chat.with_rate_limit(requests_per_minute=120)
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    assert out["text"]

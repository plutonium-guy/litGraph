"""Live integration: `batch_chat` bounded-concurrency fan-out."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_batch_chat_returns_one_result_per_input(deepseek_chat):
    from litgraph.agents import batch_chat

    inputs = [
        [{"role": "user", "content": "Reply with: ONE"}],
        [{"role": "user", "content": "Reply with: TWO"}],
        [{"role": "user", "content": "Reply with: THREE"}],
    ]
    results = batch_chat(deepseek_chat, inputs, max_concurrency=3)
    assert len(results) == len(inputs)
    # Each result is either {text: ...} (success) or {error: ...} (failure).
    for r in results:
        assert isinstance(r, dict)
        assert ("text" in r) or ("error" in r)


def test_batch_chat_preserves_input_order(deepseek_chat):
    from litgraph.agents import batch_chat

    prompts = ["alpha", "beta", "gamma", "delta"]
    inputs = [
        [{"role": "user", "content": f"Reply with exactly: {p.upper()}"}]
        for p in prompts
    ]
    results = batch_chat(deepseek_chat, inputs, max_concurrency=4)
    assert len(results) == len(prompts)
    # Order: i-th result corresponds to i-th input.
    for i, p in enumerate(prompts):
        text = results[i].get("text", "")
        assert p.upper() in text, (
            f"index {i} expected {p.upper()} in {text!r}"
        )


def test_batch_chat_respects_max_concurrency(deepseek_chat):
    """Smoke: batch with concurrency=1 still completes (no deadlock)."""
    from litgraph.agents import batch_chat

    inputs = [
        [{"role": "user", "content": "ok"}],
        [{"role": "user", "content": "ok"}],
    ]
    results = batch_chat(deepseek_chat, inputs, max_concurrency=1)
    assert len(results) == 2

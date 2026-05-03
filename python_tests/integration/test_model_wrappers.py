"""Live integration: per-call wrappers (TokenBudget / CostCapped /
PiiScrubbing) against DeepSeek.

These wrappers sit between the user and the underlying chat model:
- `TokenBudgetChat` — pre-call cap on input tokens. Two modes: strict
  raises `ValueError` over budget; `auto_trim=True` drops oldest
  non-system msgs until under cap.
- `CostCappedChat` — hard USD cap on cumulative spend. Once cap hit,
  further calls fail BEFORE hitting the provider.
- `PiiScrubbingChat` — strips PII (email/phone/etc) from outgoing
  user messages before they reach the model.

Note: the Python class names DROP the `Model` suffix from the Rust
`ChatModel` trait — exposed as `TokenBudgetChat`, `CostCappedChat`,
`PiiScrubbingChat` (NOT `*Model`).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_token_budget_wrapper_passes_under_budget(deepseek_chat):
    from litgraph.providers import TokenBudgetChat

    wrapped = TokenBudgetChat(deepseek_chat, max_tokens=10_000)
    out = wrapped.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    assert out["text"]


def test_token_budget_wrapper_strict_rejects_oversize(deepseek_chat):
    """Strict mode (default `auto_trim=False`) raises on over-budget."""
    from litgraph.providers import TokenBudgetChat

    wrapped = TokenBudgetChat(deepseek_chat, max_tokens=5)  # tiny cap
    with pytest.raises(Exception) as excinfo:
        wrapped.invoke(
            [{"role": "user", "content": "x " * 200}],  # ~400 chars → ~100 tokens
            max_tokens=10,
        )
    msg = str(excinfo.value).lower()
    assert "budget" in msg or "token" in msg or "exceed" in msg, (
        f"unexpected error: {excinfo.value!r}"
    )


def test_token_budget_auto_trim_drops_oldest(deepseek_chat):
    """`auto_trim=True` keeps system + last message, drops in-between."""
    from litgraph.providers import TokenBudgetChat

    wrapped = TokenBudgetChat(deepseek_chat, max_tokens=200, auto_trim=True)
    msgs = [
        {"role": "system", "content": "Be terse."},
        *[{"role": "user", "content": "noise " * 30} for _ in range(10)],
        {"role": "user", "content": "Reply: ok"},
    ]
    out = wrapped.invoke(msgs, max_tokens=10)
    assert out["text"].strip()


def test_cost_capped_wrapper_aborts_when_ceiling_hit(deepseek_chat):
    """Tiny ceiling: once usage lands, second call should refuse."""
    from litgraph.providers import CostCappedChat

    capped = CostCappedChat(
        deepseek_chat,
        max_usd=1e-12,  # impossibly small — first call's usage trips the cap
        prices={"deepseek-chat": (0.27, 1.10)},
    )
    # First call: still allowed (cap is checked AFTER usage updates).
    capped.invoke([{"role": "user", "content": "ok"}], max_tokens=10)
    # Second call: should refuse before hitting the provider.
    with pytest.raises(Exception) as excinfo:
        capped.invoke([{"role": "user", "content": "ok"}], max_tokens=10)
    msg = str(excinfo.value).lower()
    assert "cap" in msg or "usd" in msg or "cost" in msg or "ceiling" in msg, (
        f"unexpected error: {excinfo.value!r}"
    )


def test_pii_scrubbing_wrapper_invokes_cleanly(deepseek_chat):
    """Wrap DeepSeek with PII-scrubbing. The call should complete; the
    email should NOT echo back (the model never saw it)."""
    from litgraph.providers import PiiScrubbingChat

    wrapped = PiiScrubbingChat(deepseek_chat)
    out = wrapped.invoke(
        [{"role": "user", "content": "My email is alice@example.com. Reply with: GOT IT"}],
        max_tokens=20,
    )
    assert out["text"].strip()
    # The original email shouldn't appear in the reply (model received
    # `<EMAIL>` token, not the literal address).
    assert "alice@example.com" not in out["text"], (
        f"PII leaked back from model: {out['text']!r}"
    )

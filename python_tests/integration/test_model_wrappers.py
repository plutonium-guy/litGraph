"""Live integration: per-call wrappers (TokenBudget / CostCapped /
PiiScrubbing / SelfConsistency) against DeepSeek.

These wrappers sit between the user and the underlying chat model:
- `TokenBudgetChatModel` — pre-call cap on input tokens.
- `CostCappedChatModel` — runtime ceiling on USD spent.
- `PiiScrubbingChatModel` — strips PII patterns from prompts before
  sending.
- `SelfConsistencyChatModel` — sample N responses, vote.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_token_budget_wrapper_passes_short_prompt(deepseek_chat):
    try:
        from litgraph.providers import TokenBudgetChatModel
    except ImportError:
        pytest.skip("TokenBudgetChatModel not exposed")
    wrapped = TokenBudgetChatModel(deepseek_chat, max_input_tokens=100)
    out = wrapped.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    assert out["text"]


def test_token_budget_wrapper_rejects_oversize_prompt(deepseek_chat):
    try:
        from litgraph.providers import TokenBudgetChatModel
    except ImportError:
        pytest.skip("TokenBudgetChatModel not exposed")
    wrapped = TokenBudgetChatModel(deepseek_chat, max_input_tokens=5)
    with pytest.raises(Exception) as excinfo:
        wrapped.invoke(
            [{"role": "user", "content": "x " * 200}],  # ~400 tokens easily
            max_tokens=10,
        )
    assert "budget" in str(excinfo.value).lower() or "token" in str(excinfo.value).lower()


def test_cost_capped_wrapper_aborts_when_ceiling_hit(deepseek_chat):
    try:
        from litgraph.providers import CostCappedChatModel
    except ImportError:
        pytest.skip("CostCappedChatModel not exposed")
    # Tiny ceiling: even one call should bust it (DeepSeek charges
    # ~$0.0000001 per token; ceiling=$1e-12 means trip on first usage).
    capped = CostCappedChatModel(
        deepseek_chat,
        prices={"deepseek-chat": (0.27, 1.10)},
        ceiling_usd=1e-12,
    )
    capped.invoke([{"role": "user", "content": "ok"}], max_tokens=10)
    # Second call should refuse.
    with pytest.raises(Exception):
        capped.invoke([{"role": "user", "content": "ok"}], max_tokens=10)


def test_pii_scrubbing_strips_email_before_send(deepseek_chat):
    try:
        from litgraph.providers import PiiScrubbingChatModel
    except ImportError:
        pytest.skip("PiiScrubbingChatModel not exposed")
    wrapped = PiiScrubbingChatModel(deepseek_chat)
    # We can't easily prove the scrub happened (the model's reply is
    # the only side channel), but the call should complete + return
    # text without leaking the email back.
    out = wrapped.invoke(
        [{"role": "user", "content": "My email is alice@example.com. Reply with: GOT IT"}],
        max_tokens=20,
    )
    assert out["text"]
    # Any reasonable model echoes "GOT IT" or similar; the email
    # should NOT appear (because the model never saw it).
    assert "alice@example.com" not in out["text"]

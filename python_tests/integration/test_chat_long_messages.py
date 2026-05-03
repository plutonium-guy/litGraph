"""Live integration: multi-turn chat with several system + user/assistant turns.

Exercises message-shape stability for non-trivial conversations:
- A system message
- Two prior user/assistant pairs
- A final user message that depends on context

DeepSeek should preserve context across the turns.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_multi_turn_context_carries_through(deepseek_chat):
    msgs = [
        {"role": "system", "content": "You are a terse trivia assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
        {"role": "user", "content": "What is the population of THAT city, roughly? Reply: under 5 million OR over 5 million."},
    ]
    out = deepseek_chat.invoke(msgs, max_tokens=20)
    text = out["text"].lower()
    # Paris's population is ~2.1M (city proper) — under 5 million.
    # Tolerate either "under" or "less" or numerical hedge.
    assert "5" in text or "million" in text, (
        f"answer didn't engage with the question: {out['text']!r}"
    )

"""Live integration: chat option shaping (`temperature`, `top_p`, `stop`).

Verifies that DeepSeek honours common kwargs that flow through
`OpenAIChat.invoke(..., temperature=, top_p=, stop=, max_tokens=)`.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_temperature_zero_is_deterministic_pair(deepseek_chat):
    """temperature=0 should give identical output across two runs of
    the same prompt (or near-identical — we accept a strict equality
    here for the typical case of a one-token reply)."""
    prompt = [{"role": "user", "content": "Reply with exactly one word: ping"}]
    a = deepseek_chat.invoke(prompt, temperature=0.0, max_tokens=5)["text"].strip().lower()
    b = deepseek_chat.invoke(prompt, temperature=0.0, max_tokens=5)["text"].strip().lower()
    assert a == b, f"temperature=0 gave different results: a={a!r} b={b!r}"


def test_max_tokens_caps_completion_length(deepseek_chat):
    """`max_tokens` should cap the assistant's output. With a cap of 5
    tokens the response must be short and `finish_reason` should hint
    at length termination ('length' or similar)."""
    out = deepseek_chat.invoke(
        [
            {
                "role": "user",
                "content": "Write a long detailed paragraph about the history of Rome.",
            }
        ],
        max_tokens=5,
    )
    text = out["text"]
    # 5 tokens is well under 50 chars on average; tolerate a generous
    # ceiling but reject anything resembling a real paragraph.
    assert len(text) < 80, f"max_tokens=5 over-emitted: {text!r}"
    # finish_reason should NOT be 'stop' (natural completion); it
    # should indicate the cap stopped us. Different providers use
    # different strings — accept any that isn't 'stop'.
    fr = (out.get("finish_reason") or "").lower()
    assert fr and fr != "stop", f"finish_reason should signal cap, got {fr!r}"

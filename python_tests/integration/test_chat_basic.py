"""Live integration: basic chat invoke + multi-turn against DeepSeek.

`OpenAIChat.invoke` returns a dict with this shape:

    {
        "text": str,
        "finish_reason": str,
        "usage": {"prompt": int, "completion": int, "total": int, ...},
        "model": str,
    }

We use `text` as the assistant's reply.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_invoke_returns_text(deepseek_chat):
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with exactly the word: PONG"}],
        max_tokens=10,
    )
    assert isinstance(out, dict)
    assert isinstance(out.get("text"), str)
    assert len(out["text"]) > 0
    # Tolerate punctuation / whitespace; just check the substring.
    assert "PONG" in out["text"].upper()


def test_invoke_returns_usage_block(deepseek_chat):
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Say hi."}],
        max_tokens=10,
    )
    usage = out.get("usage")
    assert isinstance(usage, dict)
    # litGraph normalises to `{prompt, completion, total}`.
    assert {"prompt", "completion", "total"} <= set(usage.keys()), (
        f"unexpected usage shape: {usage}"
    )
    assert usage["total"] >= usage["prompt"]
    assert usage["total"] >= usage["completion"]


def test_invoke_returns_model_id(deepseek_chat):
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "ping"}],
        max_tokens=10,
    )
    # DeepSeek echoes the model alias used in the request.
    assert isinstance(out.get("model"), str)
    assert "deepseek" in out["model"].lower()


def test_multi_turn_conversation(deepseek_chat):
    msgs = [
        {"role": "system", "content": "You are a terse assistant. Reply in 5 words or fewer."},
        {"role": "user", "content": "What is 2 + 2?"},
        {"role": "assistant", "content": "Four."},
        {"role": "user", "content": "And 3 + 3?"},
    ]
    out = deepseek_chat.invoke(msgs, max_tokens=10)
    text = out["text"].strip()
    # Tolerate "Six" / "6" / "6." / etc.
    assert any(token in text for token in ("6", "Six", "six")), f"got: {text!r}"


def test_finish_reason_is_stop(deepseek_chat):
    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with: ok"}],
        max_tokens=10,
    )
    fr = out.get("finish_reason")
    if fr is not None:
        assert fr in ("stop", "tool_calls", "length"), f"unexpected: {fr}"

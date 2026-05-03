"""Live integration: prompt templates + memory round-trip."""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _approx_token_count(message: dict) -> int:
    """Rough estimate (4 chars per token); good enough for memory cap."""
    return len(message.get("content", "")) // 4 + 4


def test_chat_prompt_template_from_messages_renders(deepseek_chat):
    from litgraph.prompts import ChatPromptTemplate

    # litGraph prompts use minijinja: `{{ var }}`, not `{var}`.
    tpl = ChatPromptTemplate.from_messages([
        ("system", "Reply with one word only."),
        ("user", "Capital of {{ country }}?"),
    ])
    msgs = tpl.format({"country": "France"}).to_messages()
    out = deepseek_chat.invoke(msgs, max_tokens=10)
    assert "Paris" in out["text"]


def test_chat_prompt_template_multi_var(deepseek_chat):
    from litgraph.prompts import ChatPromptTemplate

    tpl = ChatPromptTemplate.from_messages([
        ("user", "Translate the {{ language }} word for '{{ thing }}' into English. One word."),
    ])
    msgs = tpl.format({"language": "French", "thing": "cat"}).to_messages()
    out = deepseek_chat.invoke(msgs, max_tokens=10)
    assert isinstance(out["text"], str)
    assert out["text"].strip()


def test_token_buffer_memory_round_trip(deepseek_chat):
    """Add user + assistant messages to memory, then reuse them in a
    follow-up call. The model should reference the earlier turn."""
    from litgraph.memory import TokenBufferMemory

    mem = TokenBufferMemory(max_tokens=2_000, counter=_approx_token_count)
    mem.set_system({"role": "system", "content": "You are a terse trivia assistant."})
    mem.append({"role": "user", "content": "My name is Alice."})
    out1 = deepseek_chat.invoke(mem.messages(), max_tokens=20)
    mem.append({"role": "assistant", "content": out1["text"]})

    mem.append({"role": "user", "content": "What is my name? Reply with just the name."})
    out2 = deepseek_chat.invoke(mem.messages(), max_tokens=10)
    assert "Alice" in out2["text"], f"memory didn't carry: {out2['text']!r}"


def test_memory_clear_drops_history(deepseek_chat):
    from litgraph.memory import TokenBufferMemory

    mem = TokenBufferMemory(max_tokens=2_000, counter=_approx_token_count)
    mem.append({"role": "user", "content": "My favourite colour is BLUE."})
    out1 = deepseek_chat.invoke(mem.messages(), max_tokens=20)
    mem.append({"role": "assistant", "content": out1["text"]})

    mem.clear()
    mem.append({
        "role": "user",
        "content": (
            "What is my favourite colour? Reply with just the colour name. "
            "If you don't know, reply 'unknown'."
        ),
    })
    out2 = deepseek_chat.invoke(mem.messages(), max_tokens=10)
    answer = out2["text"].strip().upper().rstrip(".").rstrip()
    # After clear the model has no context. It should NOT confidently
    # say BLUE — the answer should be 'unknown' or some hedge.
    assert "BLUE" not in answer or "unknown" in out2["text"].lower(), (
        f"memory leaked across clear: {out2['text']!r}"
    )

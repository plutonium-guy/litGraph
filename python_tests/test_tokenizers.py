"""Token counting + trim_messages."""
from litgraph.tokenizers import count_tokens, count_message_tokens, trim_messages


def test_count_tokens_openai_via_tiktoken():
    n = count_tokens("gpt-4o", "hello world")
    assert 1 <= n <= 5  # tiktoken: typically 2


def test_count_tokens_falls_back_for_unknown_model():
    # 10 chars → (10+3)/4 = 3
    n = count_tokens("anthropic.claude-opus-4-7", "abcdefghij")
    assert n == 3


def test_count_message_tokens_includes_overhead():
    msgs = [{"role": "user", "content": "hi"}]
    n = count_message_tokens("gpt-4o", msgs)
    # 4 (per-message) + ≤2 (hi) + 2 (reply) = ~7
    assert 5 <= n <= 12


def test_trim_keeps_system_and_last_message():
    msgs = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "a" * 2000},
        {"role": "assistant", "content": "b" * 2000},
        {"role": "user", "content": "c" * 2000},
        {"role": "user", "content": "latest question"},
    ]
    trimmed = trim_messages("anthropic.claude-test", msgs, 200)
    # System always kept
    assert trimmed[0]["role"] == "system"
    # Last always kept
    assert trimmed[-1]["content"] == "latest question"
    # Some old were dropped
    assert len(trimmed) < len(msgs)


def test_trim_returns_unchanged_when_under_budget():
    msgs = [{"role": "user", "content": "short"}]
    out = trim_messages("gpt-4o", msgs, 1000)
    assert len(out) == 1
    assert out[0]["content"] == "short"


if __name__ == "__main__":
    fns = [
        test_count_tokens_openai_via_tiktoken,
        test_count_tokens_falls_back_for_unknown_model,
        test_count_message_tokens_includes_overhead,
        test_trim_keeps_system_and_last_message,
        test_trim_returns_unchanged_when_under_budget,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

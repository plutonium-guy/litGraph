"""LengthBasedExampleSelector — token-budget-greedy example picker for
FewShot prompts. Companion to SemanticSimilarityExampleSelector
(iter 143). No async / no embeddings — purely synchronous."""
from litgraph.prompts import (
    ChatPromptTemplate,
    FewShotChatPromptTemplate,
    LengthBasedExampleSelector,
)


POOL = [
    {"input": "short q1", "output": "short a1"},                                # ~16 chars → 4 tok
    {"input": "medium length q2", "output": "medium length a2"},                # ~32 chars → 8 tok
    {"input": "longer question 3 with extra text", "output": "longer answer here"},  # ~50+ chars
]


def char_div_4(text):
    return len(text) // 4


def test_picks_prefix_under_budget():
    sel = LengthBasedExampleSelector(POOL, max_tokens=10,
                                     fields=["input", "output"], counter=char_div_4)
    picked = sel.select()
    # First example: ~17 chars / 4 = 4 tokens (input + " " + output).
    # Second: 33 / 4 = 8. Total 4+8=12 > 10 → stop after first.
    assert len(picked) == 1
    assert picked[0]["input"] == "short q1"


def test_picks_all_when_budget_huge():
    sel = LengthBasedExampleSelector(POOL, max_tokens=10_000,
                                     fields=["input", "output"], counter=char_div_4)
    assert len(sel.select()) == 3


def test_zero_budget_returns_empty():
    sel = LengthBasedExampleSelector(POOL, max_tokens=0,
                                     fields=["input"], counter=char_div_4)
    assert sel.select() == []


def test_empty_pool_returns_empty():
    sel = LengthBasedExampleSelector([], max_tokens=100,
                                     fields=["input"], counter=char_div_4)
    assert sel.select() == []


def test_select_with_budget_overrides_per_call():
    sel = LengthBasedExampleSelector(POOL, max_tokens=10_000,
                                     fields=["input", "output"], counter=char_div_4)
    # Tighten per call.
    picked = sel.select_with_budget(max_tokens=5)
    assert len(picked) == 1


def test_default_counter_used_when_none_passed():
    """When counter=None, falls back to len/4 estimate (Rust-side)."""
    sel = LengthBasedExampleSelector(POOL, max_tokens=10,
                                     fields=["input", "output"])  # no counter
    # Should still work — uses default char/4.
    assert len(sel.select()) >= 1


def test_default_fields_are_input_and_output():
    """When fields=None, defaults to ['input', 'output']."""
    sel = LengthBasedExampleSelector(POOL, max_tokens=10_000, counter=char_div_4)
    assert len(sel.select()) == 3


def test_pool_size_reports_pool_length():
    sel = LengthBasedExampleSelector(POOL, max_tokens=100, counter=char_div_4)
    assert sel.pool_size() == 3


def test_repr_shows_pool_size_and_max_tokens():
    sel = LengthBasedExampleSelector(POOL, max_tokens=2048, counter=char_div_4)
    r = repr(sel)
    assert "LengthBasedExampleSelector" in r
    assert "pool_size=3" in r
    assert "max_tokens=2048" in r


def test_picks_feed_into_few_shot_template():
    """End-to-end: select examples → render through FewShotChatPromptTemplate."""
    sel = LengthBasedExampleSelector(POOL, max_tokens=12,
                                     fields=["input", "output"], counter=char_div_4)
    picked = sel.select()

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ input }}"),
        ("assistant", "{{ output }}"),
    ])
    input_prompt = ChatPromptTemplate.from_messages([("user", "{{ q }}")])
    few_shot = FewShotChatPromptTemplate(
        examples=picked,
        example_prompt=example_prompt,
        input_prompt=input_prompt,
        system_prefix="You are helpful.",
    )
    msgs = few_shot.format({"q": "the question"})
    # System + N×(user+assistant) + final user.
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == "the question"


def test_callback_exception_falls_back_to_char_estimate():
    """Counter that throws → we fall back to len/4 estimate (don't crash)."""
    def bad_counter(text):
        raise RuntimeError("counter explodes")

    sel = LengthBasedExampleSelector(POOL, max_tokens=10_000,
                                     fields=["input", "output"], counter=bad_counter)
    # Even though counter throws, fallback estimate works → returns examples.
    picked = sel.select()
    assert len(picked) == 3


if __name__ == "__main__":
    import traceback
    fns = [
        test_picks_prefix_under_budget,
        test_picks_all_when_budget_huge,
        test_zero_budget_returns_empty,
        test_empty_pool_returns_empty,
        test_select_with_budget_overrides_per_call,
        test_default_counter_used_when_none_passed,
        test_default_fields_are_input_and_output,
        test_pool_size_reports_pool_length,
        test_repr_shows_pool_size_and_max_tokens,
        test_picks_feed_into_few_shot_template,
        test_callback_exception_falls_back_to_char_estimate,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

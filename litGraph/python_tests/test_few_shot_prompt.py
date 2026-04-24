"""FewShotChatPromptTemplate — render N example (input,output) pairs ahead
of the user's question to teach the model a format/style.

Direct LangChain `FewShotChatMessagePromptTemplate` parity. Builds on the
iter 88 ChatPromptTemplate."""
from litgraph.prompts import ChatPromptTemplate, FewShotChatPromptTemplate


def test_few_shot_renders_examples_then_input():
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ q }}"),
        ("assistant", "{{ a }}"),
    ])
    input_prompt = ChatPromptTemplate.from_messages([("user", "{{ input }}")])
    few_shot = FewShotChatPromptTemplate(
        example_prompt=example_prompt,
        examples=[
            {"q": "hello", "a": "bonjour"},
            {"q": "thank you", "a": "merci"},
        ],
        input_prompt=input_prompt,
    )
    msgs = few_shot.format({"input": "good night"})
    # 2 examples * 2 messages each + 1 input = 5 messages.
    assert len(msgs) == 5
    assert [m["role"] for m in msgs] == ["user", "assistant", "user", "assistant", "user"]
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["content"] == "bonjour"
    assert msgs[2]["content"] == "thank you"
    assert msgs[3]["content"] == "merci"
    assert msgs[4]["content"] == "good night"


def test_few_shot_system_prefix_prepended():
    """`system_prefix` injects an initial system message before any examples."""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ q }}"), ("assistant", "{{ a }}"),
    ])
    input_prompt = ChatPromptTemplate.from_messages([("user", "{{ input }}")])
    few_shot = FewShotChatPromptTemplate(
        example_prompt=example_prompt,
        examples=[{"q": "hi", "a": "salut"}],
        input_prompt=input_prompt,
        system_prefix="You translate English to French.",
    )
    msgs = few_shot.format({"input": "bye"})
    assert len(msgs) == 4
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"] == "You translate English to French."


def test_few_shot_zero_examples_renders_only_input():
    example_prompt = ChatPromptTemplate.from_messages([("user", "{{ q }}")])
    input_prompt = ChatPromptTemplate.from_messages([("user", "{{ input }}")])
    few_shot = FewShotChatPromptTemplate(
        example_prompt=example_prompt,
        examples=[],  # empty
        input_prompt=input_prompt,
    )
    msgs = few_shot.format({"input": "alone"})
    assert len(msgs) == 1
    assert msgs[0]["content"] == "alone"


def test_few_shot_example_missing_var_errors_with_index():
    """If example index N has a missing var, the error names which example
    failed — silent rendering of empty strings would be a worse footgun."""
    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ q }}"), ("assistant", "{{ a }}"),
    ])
    input_prompt = ChatPromptTemplate.from_messages([("user", "{{ input }}")])
    few_shot = FewShotChatPromptTemplate(
        example_prompt=example_prompt,
        examples=[
            {"q": "ok", "a": "good"},
            {"q": "bad"},  # missing 'a'
        ],
        input_prompt=input_prompt,
    )
    try:
        few_shot.format({"input": "x"})
    except RuntimeError as e:
        assert "example 1" in str(e), f"expected 'example 1' in error, got: {e}"
    else:
        raise AssertionError("expected RuntimeError on missing var")


def test_strict_undefined_in_chat_template_errors_on_missing_var():
    """ChatPromptTemplate is strict-undefined globally — `{{ missing }}`
    errors instead of silently rendering empty. Iter 92 contract change."""
    tmpl = ChatPromptTemplate.from_messages([
        ("user", "Hello {{ name }}, your code is {{ code }}."),
    ])
    try:
        tmpl.format({"name": "Ada"})  # missing 'code'
    except RuntimeError as e:
        # minijinja error message includes the var name.
        assert "code" in str(e).lower() or "undefined" in str(e).lower()
    else:
        raise AssertionError("expected RuntimeError on missing template var")


if __name__ == "__main__":
    fns = [
        test_few_shot_renders_examples_then_input,
        test_few_shot_system_prefix_prepended,
        test_few_shot_zero_examples_renders_only_input,
        test_few_shot_example_missing_var_errors_with_index,
        test_strict_undefined_in_chat_template_errors_on_missing_var,
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

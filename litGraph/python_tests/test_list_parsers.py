"""List + boolean output parsers — LangChain `CommaSeparatedListOutputParser` /
`NumberedListOutputParser` / `MarkdownListOutputParser` / `BooleanOutputParser`
parity."""
from litgraph.parsers import (
    parse_boolean,
    parse_comma_list,
    parse_markdown_list,
    parse_numbered_list,
)


def _raises_value_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


def test_comma_list_basic():
    assert parse_comma_list("apple, banana, cherry") == ["apple", "banana", "cherry"]


def test_comma_list_strips_surrounding_prose():
    """Real LLM output: 'Sure! Here are 3 fruits: apple, banana, cherry. Enjoy!'"""
    text = "Here you go: apple, banana, cherry. Hope this helps!"
    assert parse_comma_list(text) == ["apple", "banana", "cherry"]


def test_comma_list_strips_quotes_around_items():
    assert parse_comma_list('"apple", "banana", "cherry"') == ["apple", "banana", "cherry"]


def test_comma_list_empty_input_returns_empty_list():
    assert parse_comma_list("") == []


def test_comma_list_picks_line_with_most_commas():
    """Multi-line text — pick the list line, skip intro prose."""
    text = "Sure, I can help!\napple, banana, cherry, date, elderberry"
    assert parse_comma_list(text) == ["apple", "banana", "cherry", "date", "elderberry"]


def test_numbered_list_dot_delim():
    assert parse_numbered_list("1. apple\n2. banana\n3. cherry") == [
        "apple",
        "banana",
        "cherry",
    ]


def test_numbered_list_mixed_delims():
    """LLMs sometimes mix `.`, `)`, `:` within one list."""
    assert parse_numbered_list("1) apple\n2: banana\n3. cherry") == [
        "apple",
        "banana",
        "cherry",
    ]


def test_numbered_list_handles_intro_and_outro_prose():
    text = (
        "Here's the list you asked for:\n"
        "1. apple\n"
        "2. banana\n"
        "\n"
        "Let me know if you need more."
    )
    assert parse_numbered_list(text) == ["apple", "banana"]


def test_numbered_list_tolerates_non_dense_numbering():
    # LLMs sometimes skip numbers (or repeat them); don't validate sequence.
    assert parse_numbered_list("1. apple\n3. banana\n5. cherry") == [
        "apple",
        "banana",
        "cherry",
    ]


def test_markdown_list_dash_bullets():
    assert parse_markdown_list("- apple\n- banana") == ["apple", "banana"]


def test_markdown_list_mixed_bullets():
    """Real LLM output mixes `-`, `*`, `+` bullets."""
    assert parse_markdown_list("- apple\n* banana\n+ cherry\n• date") == [
        "apple",
        "banana",
        "cherry",
        "date",
    ]


def test_markdown_list_with_indents_and_prose():
    text = "Sure:\n  - apple\n    - banana (nested)\n  - cherry\nDone!"
    assert parse_markdown_list(text) == ["apple", "banana (nested)", "cherry"]


def test_boolean_yes_no():
    assert parse_boolean("Yes") is True
    assert parse_boolean("NO") is False


def test_boolean_true_false():
    assert parse_boolean("true") is True
    assert parse_boolean("False") is False


def test_boolean_inside_prose():
    """First matching token wins — LLMs wrap the answer in prose."""
    assert parse_boolean("Sure, yes that's right.") is True
    assert parse_boolean("No way, that's wrong.") is False


def test_boolean_ambiguous_raises_value_error():
    assert _raises_value_error(parse_boolean, "Maybe? I'm not sure.")
    assert _raises_value_error(parse_boolean, "")


def test_boolean_does_not_match_substring_inside_word():
    """`yesterday` contains `yes` but must NOT be parsed as yes."""
    assert _raises_value_error(parse_boolean, "Yesterday was fine.")


def test_list_parsers_composable_with_chat_response_text():
    """Realistic use: feed ChatModel response text straight to parser."""
    response_text = (
        "Sure! Here are 5 fruits:\n"
        "1. apple\n"
        "2. banana\n"
        "3. cherry\n"
        "4. date\n"
        "5. elderberry\n"
        "\n"
        "Enjoy!"
    )
    items = parse_numbered_list(response_text)
    assert len(items) == 5
    assert items[0] == "apple"
    assert items[-1] == "elderberry"


if __name__ == "__main__":
    import traceback

    fns = [
        test_comma_list_basic,
        test_comma_list_strips_surrounding_prose,
        test_comma_list_strips_quotes_around_items,
        test_comma_list_empty_input_returns_empty_list,
        test_comma_list_picks_line_with_most_commas,
        test_numbered_list_dot_delim,
        test_numbered_list_mixed_delims,
        test_numbered_list_handles_intro_and_outro_prose,
        test_numbered_list_tolerates_non_dense_numbering,
        test_markdown_list_dash_bullets,
        test_markdown_list_mixed_bullets,
        test_markdown_list_with_indents_and_prose,
        test_boolean_yes_no,
        test_boolean_true_false,
        test_boolean_inside_prose,
        test_boolean_ambiguous_raises_value_error,
        test_boolean_does_not_match_substring_inside_word,
        test_list_parsers_composable_with_chat_response_text,
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

"""XML output parsers — extract `<tag>...</tag>` content from LLM responses.

Direct LangChain `XMLOutputParser` parity. Complements iter-89's
StructuredChatModel (JSON path) — Anthropic's cookbook uses XML tags
heavily so this is a common migration paper-cut."""
from litgraph.parsers import parse_xml_tags, parse_nested_xml


def test_flat_parser_extracts_named_tags():
    text = "<thinking>Let me work this out</thinking>\n<answer>42</answer>"
    m = parse_xml_tags(text, ["thinking", "answer"])
    assert m["thinking"] == "Let me work this out"
    assert m["answer"] == "42"


def test_flat_parser_missing_tag_absent_not_error():
    """LLMs often forget tags. Missing → not in dict, caller decides."""
    text = "<answer>only this</answer>"
    m = parse_xml_tags(text, ["thinking", "answer"])
    assert "thinking" not in m
    assert m["answer"] == "only this"


def test_flat_parser_decodes_builtin_entities():
    """&lt; &gt; &amp; &quot; &apos; → < > & " '"""
    text = '<answer>if x &lt; 10 &amp; y &gt; 5 say &quot;hi&quot;</answer>'
    m = parse_xml_tags(text, ["answer"])
    assert m["answer"] == 'if x < 10 & y > 5 say "hi"'


def test_flat_parser_case_insensitive_tag_match():
    text = "<Thinking>hmm</Thinking>"
    m = parse_xml_tags(text, ["thinking"])
    assert m["thinking"] == "hmm"


def test_flat_parser_trims_whitespace_around_content():
    text = "<answer>\n  42  \n</answer>"
    m = parse_xml_tags(text, ["answer"])
    assert m["answer"] == "42"


def test_flat_parser_first_occurrence_wins():
    text = "<answer>first</answer><answer>second</answer>"
    m = parse_xml_tags(text, ["answer"])
    assert m["answer"] == "first"


def test_flat_parser_handles_loose_prose_around_tags():
    """Real Anthropic responses often have prose wrapping the tags —
    e.g. 'Let me think about this.\\n<thinking>...</thinking>\\nMy answer:'"""
    text = ("Let me think step by step.\n"
            "<thinking>step 1: identify the problem</thinking>\n"
            "So my answer is:\n"
            "<answer>yes</answer>\n"
            "Hope that helps!")
    m = parse_xml_tags(text, ["thinking", "answer"])
    assert m["thinking"] == "step 1: identify the problem"
    assert m["answer"] == "yes"


def test_nested_parser_builds_tree_for_simple_leaf():
    v = parse_nested_xml("<answer>42</answer>")
    assert v["answer"] == "42"


def test_nested_parser_builds_nested_object():
    text = "<response><thinking>s1</thinking><answer>42</answer></response>"
    v = parse_nested_xml(text)
    assert v["response"]["thinking"] == "s1"
    assert v["response"]["answer"] == "42"


def test_nested_parser_repeated_tags_become_list():
    text = "<root><item>a</item><item>b</item><item>c</item></root>"
    v = parse_nested_xml(text)
    assert v["root"]["item"] == ["a", "b", "c"]


def test_nested_parser_mixed_children():
    text = "<root><item>a</item><item>b</item><name>foo</name></root>"
    v = parse_nested_xml(text)
    assert v["root"]["item"] == ["a", "b"]
    assert v["root"]["name"] == "foo"


def test_nested_parser_empty_input_returns_none():
    assert parse_nested_xml("") is None
    assert parse_nested_xml("just prose no tags") is None


def test_nested_parser_decodes_entities_in_leaf_content():
    v = parse_nested_xml("<answer>5 &lt; 10 &amp; true</answer>")
    assert v["answer"] == "5 < 10 & true"


def test_nested_parser_does_not_panic_on_stray_angle_bracket():
    """Code samples inside tags often have stray `<` that aren't tag
    openers. Parser should not crash."""
    # Don't crash, just lock the "no exception" invariant.
    _ = parse_nested_xml("<code>if (x < 10) return;</code>")


def test_xml_parsers_composable_with_chat_response_text():
    """Real use case: call LLM, get response text, parse it."""
    # Simulate: response["text"] comes from ChatModel.invoke().
    response_text = (
        "I'll reason through this.\n"
        "<thinking>The user asked for a sum. 2 + 3 = 5.</thinking>\n"
        "<answer>5</answer>"
    )
    parsed = parse_xml_tags(response_text, ["thinking", "answer"])
    assert parsed["answer"] == "5"
    # Caller can now feed `parsed["answer"]` to downstream code as a
    # strongly-typed value (cast to int, validate, etc).
    assert int(parsed["answer"]) == 5


if __name__ == "__main__":
    fns = [
        test_flat_parser_extracts_named_tags,
        test_flat_parser_missing_tag_absent_not_error,
        test_flat_parser_decodes_builtin_entities,
        test_flat_parser_case_insensitive_tag_match,
        test_flat_parser_trims_whitespace_around_content,
        test_flat_parser_first_occurrence_wins,
        test_flat_parser_handles_loose_prose_around_tags,
        test_nested_parser_builds_tree_for_simple_leaf,
        test_nested_parser_builds_nested_object,
        test_nested_parser_repeated_tags_become_list,
        test_nested_parser_mixed_children,
        test_nested_parser_empty_input_returns_none,
        test_nested_parser_decodes_entities_in_leaf_content,
        test_nested_parser_does_not_panic_on_stray_angle_bracket,
        test_xml_parsers_composable_with_chat_response_text,
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

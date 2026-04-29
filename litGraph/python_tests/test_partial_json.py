"""Partial / streaming JSON parser. Accepts partial input (unclosed
braces / quotes / brackets) and returns a best-effort Value snapshot.
For rendering progressive UIs as structured-output tokens arrive."""
from litgraph.parsers import parse_partial_json, repair_partial_json


def test_complete_object_parses_identity():
    assert parse_partial_json('{"a": 1, "b": "hi"}') == {"a": 1, "b": "hi"}


def test_unclosed_object_auto_closes():
    assert parse_partial_json('{"a": 1, "b": "hi"') == {"a": 1, "b": "hi"}


def test_unclosed_string_value_auto_closes():
    assert parse_partial_json('{"a": "hi') == {"a": "hi"}


def test_unclosed_nested_containers_close_in_reverse_order():
    assert parse_partial_json('{"o": {"i": [1, 2, 3') == {"o": {"i": [1, 2, 3]}}


def test_dangling_colon_strips_the_key():
    # {"a": → repair to {} (key-without-value dropped)
    assert parse_partial_json('{"a":') == {}


def test_dangling_comma_is_dropped():
    assert parse_partial_json('{"a": 1,') == {"a": 1}


def test_dangling_key_without_colon_stripped():
    # {"ke → close string, detect dangling key → {}
    assert parse_partial_json('{"ke') == {}


def test_arrays_auto_close():
    assert parse_partial_json('[1, 2, 3') == [1, 2, 3]


def test_array_string_element_preserved_not_stripped_as_key():
    # Bug caught in Rust tests: strip-dangling-key must NOT fire inside
    # arrays. `["hi", "wor` must complete to `["hi", "wor"]`, not `["hi"]`.
    assert parse_partial_json('["hi", "wor') == ["hi", "wor"]


def test_empty_and_whitespace_return_none():
    assert parse_partial_json("") is None
    assert parse_partial_json("    \t\n") is None
    assert parse_partial_json("just words") is None


def test_leading_prose_before_root_skipped():
    # LLM-style "Here's the JSON: {...}"
    assert parse_partial_json("Here: {\"a\": 1") == {"a": 1}


def test_escape_sequences_preserved():
    assert parse_partial_json(r'{"msg": "he said \"hi\""') == {"msg": 'he said "hi"'}


def test_trailing_backslash_in_partial_string_stripped():
    # `"hello\` → strip lone `\`, close → `"hello"`
    assert parse_partial_json(r'{"a": "hello\\') in ({"a": "hello\\"}, {"a": "hello"})


def test_brackets_inside_string_dont_affect_stack():
    assert parse_partial_json('{"x": "a}[{b') == {"x": "a}[{b"}


def test_monotonic_growth_keys_accumulate():
    """As the buffer grows, parsed keys monotonically accumulate —
    no UI flicker from disappearing fields."""
    progressions = [
        '{"name": "Bob",',
        '{"name": "Bob", "age":',
        '{"name": "Bob", "age": 30,',
        '{"name": "Bob", "age": 30, "city": "NYC"}',
    ]
    prior_keys = set()
    for p in progressions:
        v = parse_partial_json(p)
        assert v is not None, f"failed: {p!r}"
        keys = set(v.keys()) if isinstance(v, dict) else set()
        assert prior_keys.issubset(keys), (
            f"snapshot {p!r} dropped keys. prior={prior_keys} new={keys}"
        )
        prior_keys = keys


def test_repair_exposes_repaired_text():
    assert repair_partial_json('{"a": "hi') == '{"a": "hi"}'


def test_deeply_nested_closes_correctly():
    out = parse_partial_json('{"a": {"b": {"c": [1, {"d": 42')
    assert out == {"a": {"b": {"c": [1, {"d": 42}]}}}


def test_realistic_openai_structured_stream_progressive_parse():
    """Simulate OpenAI streaming `response_format=json_schema`.
    Tokens arrive character-by-character; partial parses each time."""
    full = '{"answer": "Paris", "confidence": 0.92, "reasoning": "France capital"}'
    accumulated = ""
    partial_results = []
    for ch in full:
        accumulated += ch
        v = parse_partial_json(accumulated)
        if v is not None:
            partial_results.append(v)
    # Final partial == full parse.
    import json as _json
    assert partial_results[-1] == _json.loads(full)
    # At least one intermediate snapshot had just "answer" before
    # "confidence" arrived.
    any_answer_only = any(
        set(p.keys()) == {"answer"} for p in partial_results if isinstance(p, dict)
    )
    assert any_answer_only


if __name__ == "__main__":
    import traceback
    fns = [
        test_complete_object_parses_identity,
        test_unclosed_object_auto_closes,
        test_unclosed_string_value_auto_closes,
        test_unclosed_nested_containers_close_in_reverse_order,
        test_dangling_colon_strips_the_key,
        test_dangling_comma_is_dropped,
        test_dangling_key_without_colon_stripped,
        test_arrays_auto_close,
        test_array_string_element_preserved_not_stripped_as_key,
        test_empty_and_whitespace_return_none,
        test_leading_prose_before_root_skipped,
        test_escape_sequences_preserved,
        test_trailing_backslash_in_partial_string_stripped,
        test_brackets_inside_string_dont_affect_stack,
        test_monotonic_growth_keys_accumulate,
        test_repair_exposes_repaired_text,
        test_deeply_nested_closes_correctly,
        test_realistic_openai_structured_stream_progressive_parse,
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

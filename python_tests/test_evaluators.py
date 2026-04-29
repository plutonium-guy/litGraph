"""String evaluators — exact_match / levenshtein / jaccard / regex /
json_validity / embedding_cosine. LangChain `evaluation.string_distance`
parity. Pure scoring functions for prompt experiments and regression
tests."""
from litgraph.evaluators import (
    contains_all,
    contains_any,
    embedding_cosine,
    exact_match,
    exact_match_strict,
    jaccard_similarity,
    json_validity,
    levenshtein,
    levenshtein_ratio,
    regex_match,
)


def _raises_value_error(fn, *args, **kw):
    try:
        fn(*args, **kw)
    except ValueError:
        return True
    return False


def test_exact_match_trims_and_lowercases():
    assert exact_match("  Paris  ", "paris") is True
    assert exact_match("YES", "yes") is True
    assert exact_match("Paris", "London") is False


def test_exact_match_strict_byte_equal():
    assert exact_match_strict("Paris", "Paris") is True
    assert exact_match_strict("Paris", "paris") is False
    assert exact_match_strict("Paris", "Paris ") is False


def test_levenshtein_basic_distances():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "abc") == 3
    assert levenshtein("same", "same") == 0


def test_levenshtein_ratio_in_zero_one_range():
    assert levenshtein_ratio("hello", "hello") == 1.0
    assert levenshtein_ratio("abc", "xyz") == 0.0
    r = levenshtein_ratio("kitten", "sitting")
    assert 0.0 < r < 1.0


def test_jaccard_word_order_invariant():
    assert jaccard_similarity("apple banana", "banana apple") == 1.0
    r = jaccard_similarity("apple banana", "banana cherry")
    assert abs(r - (1.0 / 3.0)) < 1e-3


def test_regex_match_basic():
    assert regex_match("year is 2026", r"\d{4}") is True
    assert regex_match("no digits here", r"\d{4}") is False


def test_regex_match_invalid_pattern_raises_value_error():
    assert _raises_value_error(regex_match, "anything", r"(unclosed")


def test_json_validity_round_trip():
    assert json_validity('{"a": 1}') is True
    assert json_validity("[1, 2, 3]") is True
    assert json_validity('"string"') is True
    assert json_validity("42") is True
    assert json_validity("not json") is False
    assert json_validity('{"a": 1') is False  # unclosed


def test_embedding_cosine_identical_one():
    v = [1.0, 2.0, 3.0]
    assert abs(embedding_cosine(v, v) - 1.0) < 1e-6


def test_embedding_cosine_orthogonal_zero():
    assert abs(embedding_cosine([1.0, 0.0], [0.0, 1.0])) < 1e-6


def test_embedding_cosine_length_mismatch_raises():
    assert _raises_value_error(embedding_cosine, [1.0, 2.0], [1.0])


def test_embedding_cosine_zero_vector_returns_zero_not_nan():
    r = embedding_cosine([0.0, 0.0], [1.0, 1.0])
    assert r == 0.0


def test_contains_all_requires_every_needle():
    assert contains_all("the quick brown fox", ["quick", "fox"]) is True
    assert contains_all("the quick brown fox", ["quick", "horse"]) is False


def test_contains_any_requires_at_least_one():
    assert contains_any("hello world", ["world", "moon"]) is True
    assert contains_any("hello world", ["moon", "stars"]) is False


def test_evaluators_compose_for_eval_score():
    """Realistic use: combine multiple evaluators into one score."""
    actual = "The capital is Paris, France."
    expected = "Paris"
    score = (
        (1.0 if contains_any(actual, [expected]) else 0.0) * 0.5
        + jaccard_similarity(actual, expected) * 0.3
        + levenshtein_ratio(actual, expected) * 0.2
    )
    assert 0.5 < score <= 1.0


if __name__ == "__main__":
    import traceback

    fns = [
        test_exact_match_trims_and_lowercases,
        test_exact_match_strict_byte_equal,
        test_levenshtein_basic_distances,
        test_levenshtein_ratio_in_zero_one_range,
        test_jaccard_word_order_invariant,
        test_regex_match_basic,
        test_regex_match_invalid_pattern_raises_value_error,
        test_json_validity_round_trip,
        test_embedding_cosine_identical_one,
        test_embedding_cosine_orthogonal_zero,
        test_embedding_cosine_length_mismatch_raises,
        test_embedding_cosine_zero_vector_returns_zero_not_nan,
        test_contains_all_requires_every_needle,
        test_contains_any_requires_at_least_one,
        test_evaluators_compose_for_eval_score,
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

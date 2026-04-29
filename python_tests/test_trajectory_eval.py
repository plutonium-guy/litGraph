"""Trajectory evaluator — score an agent's tool-call path vs a reference."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.evaluators import evaluate_trajectory  # noqa: E402


def steps(*names):
    return [{"tool": n} for n in names]


def test_string_entries_accepted_and_equivalent_to_dict():
    a = evaluate_trajectory(["a", "b"], ["a", "b"], "exact_order")
    b = evaluate_trajectory(steps("a", "b"), steps("a", "b"), "exact_order")
    assert a == b == 1.0


def test_contains_all_full_score():
    a = steps("search", "calc", "shell")
    e = steps("search", "shell")
    assert evaluate_trajectory(a, e, "contains_all") == 1.0


def test_contains_all_partial():
    a = steps("search")
    e = steps("search", "shell")
    assert evaluate_trajectory(a, e, "contains_all") == 0.5


def test_exact_order_perfect():
    a = steps("a", "b", "c")
    e = steps("a", "b", "c")
    assert evaluate_trajectory(a, e, "exact_order") == 1.0


def test_exact_order_zero_on_swap():
    a = steps("a", "c", "b")
    e = steps("a", "b", "c")
    assert evaluate_trajectory(a, e, "exact_order") == 0.0


def test_subsequence_full_credit_with_extras():
    a = steps("search", "noise", "calc", "noise2", "shell")
    e = steps("search", "calc", "shell")
    assert evaluate_trajectory(a, e, "subsequence") == 1.0


def test_subsequence_partial_credit_lcs():
    a = steps("search", "noise", "shell")
    e = steps("search", "calc", "shell")
    s = evaluate_trajectory(a, e, "subsequence")
    assert abs(s - 2 / 3) < 1e-9


def test_levenshtein_returns_float_in_range():
    a = steps("search", "shell")
    e = steps("search", "calc", "shell")
    s = evaluate_trajectory(a, e, "levenshtein")
    assert 0.0 < s < 1.0


def test_unknown_policy_raises():
    with pytest.raises(ValueError):
        evaluate_trajectory(steps("a"), steps("a"), "snorkel")


def test_input_field_does_not_break_parsing():
    a = [{"tool": "calc", "input": {"x": 1}}]
    e = [{"tool": "calc"}]
    assert evaluate_trajectory(a, e, "contains_all") == 1.0


def test_empty_lists_score_one():
    for p in ("contains_all", "exact_order", "subsequence", "levenshtein"):
        assert evaluate_trajectory([], [], p) == 1.0


def test_default_policy_is_subsequence():
    a = steps("search", "noise", "shell")
    e = steps("search", "shell")
    # Default should equal explicit subsequence call.
    assert evaluate_trajectory(a, e) == evaluate_trajectory(
        a, e, "subsequence"
    )

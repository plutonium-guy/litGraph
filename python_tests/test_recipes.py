"""Tests for `litgraph.recipes` — one-call factories.

Each recipe should be runnable with mocks (no LLM credential required)
and return the documented shape.
"""
from __future__ import annotations

import pytest

from litgraph import recipes
from litgraph.testing import MockChatModel


def test_recipes_eval_returns_per_case_and_aggregate():
    m = MockChatModel(replies=["Paris", "Berlin"])

    def predict(q: str) -> str:
        return m.invoke([{"role": "user", "content": q}])["content"]

    report = recipes.eval(predict, [
        {"input": "Capital of France?", "expected": "Paris"},
        {"input": "Capital of Germany?", "expected": "Berlin"},
    ])
    assert "per_case" in report
    assert "aggregate" in report
    assert report["aggregate"]["n_cases"] == 2
    assert report["aggregate"]["n_errors"] == 0
    assert report["aggregate"]["means"]["exact_match"] == 1.0


def test_recipes_eval_default_scorers_include_levenshtein():
    m = MockChatModel(replies=["close"])

    def predict(_q: str) -> str:
        return m.invoke([{"role": "user", "content": _q}])["content"]

    report = recipes.eval(predict, [
        {"input": "x", "expected": "close"},
    ])
    assert "exact_match" in report["aggregate"]["means"]
    assert "levenshtein" in report["aggregate"]["means"]


def test_recipes_eval_custom_scorers_passed_through():
    m = MockChatModel(replies=["bar"])

    def predict(_q: str) -> str:
        return m.invoke([{"role": "user", "content": _q}])["content"]

    report = recipes.eval(
        predict,
        [{"input": "x", "expected": "bar"}],
        scorers=[{"name": "exact_match"}],
    )
    assert list(report["aggregate"]["means"].keys()) == ["exact_match"]


def test_recipes_eval_target_exception_recorded_not_raised():
    def boom(_q: str) -> str:
        raise RuntimeError("explode")

    report = recipes.eval(boom, [
        {"input": "x", "expected": "y"},
    ])
    assert report["aggregate"]["n_errors"] == 1


def test_recipes_serve_rejects_non_graph_input():
    with pytest.raises(TypeError, match="CompiledGraph"):
        recipes.serve("not a graph")


def test_recipes_serve_renders_command_for_compileable():
    class _Stub:
        def compile(self):
            return self

    cmd = recipes.serve(_Stub(), port=9000, host="127.0.0.1")
    assert "litgraph-serve" in cmd
    assert "9000" in cmd
    assert "127.0.0.1" in cmd

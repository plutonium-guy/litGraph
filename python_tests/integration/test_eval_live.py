"""Live integration: EvalHarness with DeepSeek as the target.

Drives `litgraph.recipes.eval(...)` over a tiny golden set with
DeepSeek as the prediction function. Asserts the report comes back
with the expected shape + reasonable accuracy on trivia questions.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_eval_against_deepseek_returns_report(deepseek_chat):
    from litgraph.recipes import eval as run_eval

    def predict(question: str) -> str:
        out = deepseek_chat.invoke(
            [
                {"role": "system", "content": "Reply with exactly one word."},
                {"role": "user", "content": question},
            ],
            max_tokens=10,
        )
        # Strip period / whitespace so 'Paris.' matches 'Paris'.
        return out["text"].strip().rstrip(".").rstrip()

    cases = [
        {"input": "Capital of France?", "expected": "Paris"},
        {"input": "Capital of Germany?", "expected": "Berlin"},
        {"input": "Capital of Japan?", "expected": "Tokyo"},
    ]
    report = run_eval(predict, cases, max_parallel=3)
    assert report["aggregate"]["n_cases"] == 3
    # Should get at least one right — DeepSeek knows these capitals.
    means = report["aggregate"]["means"]
    em = means.get("exact_match", 0.0)
    assert em >= 1 / 3, f"expected ≥ 1/3 exact_match, got {em}"


def test_eval_report_shape_includes_per_case(deepseek_chat):
    from litgraph.recipes import eval as run_eval

    report = run_eval(
        lambda q: deepseek_chat.invoke([{"role": "user", "content": q}], max_tokens=10)["text"],
        [{"input": "Reply with: ok", "expected": "ok"}],
        max_parallel=1,
    )
    assert "per_case" in report
    assert len(report["per_case"]) == 1
    case = report["per_case"][0]
    assert "input" in case and "expected" in case and "scores" in case

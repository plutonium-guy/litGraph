"""Live integration: `synthesize_eval_cases` — BLOCKED on DeepSeek.

Same root cause as `LlmJudge`: this evaluator wraps the model in
`StructuredChatModel` (`response_format=json_schema`), and DeepSeek
rejects schema mode with `400 invalid_request_error`. Re-enable when
DeepSeek adds support OR when StructuredChatModel falls back to
`json_object` + post-validate.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="DeepSeek rejects response_format=json_schema (used by StructuredChatModel)")
def test_synthesize_eval_cases_produces_dicts(deepseek_chat):  # pragma: no cover
    from litgraph.evaluators import synthesize_eval_cases

    seeds = [
        {"input": "What is 2+2?", "expected": "4"},
        {"input": "What is 5*3?", "expected": "15"},
    ]
    new_cases = synthesize_eval_cases(
        seeds,
        model=deepseek_chat,
        target_count=4,
        criteria="single arithmetic question with integer answer",
    )
    assert isinstance(new_cases, list)
    assert len(new_cases) >= 1


@pytest.mark.skip(reason="DeepSeek rejects response_format=json_schema (used by StructuredChatModel)")
def test_synthesize_eval_cases_no_criteria(deepseek_chat):  # pragma: no cover
    from litgraph.evaluators import synthesize_eval_cases

    seeds = [{"input": "Capital of France?", "expected": "Paris"}]
    new_cases = synthesize_eval_cases(seeds, model=deepseek_chat, target_count=2)
    assert isinstance(new_cases, list)

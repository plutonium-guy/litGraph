"""Live integration: `synthesize_eval_cases`.

Same capability requirement as `LlmJudge` -- the evaluator wraps the
model in `StructuredChatModel` (`response_format=json_schema`). Gated
on `_capabilities.SUPPORTS_JSON_SCHEMA` so it runs wherever schema
mode is available.
"""
from __future__ import annotations

import pytest

from ._capabilities import NO_JSON_SCHEMA_REASON, SUPPORTS_JSON_SCHEMA


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not SUPPORTS_JSON_SCHEMA, reason=NO_JSON_SCHEMA_REASON)
def test_synthesize_eval_cases_produces_dicts(deepseek_chat):
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


@pytest.mark.skipif(not SUPPORTS_JSON_SCHEMA, reason=NO_JSON_SCHEMA_REASON)
def test_synthesize_eval_cases_no_criteria(deepseek_chat):
    from litgraph.evaluators import synthesize_eval_cases

    seeds = [{"input": "Capital of France?", "expected": "Paris"}]
    new_cases = synthesize_eval_cases(seeds, model=deepseek_chat, target_count=2)
    assert isinstance(new_cases, list)

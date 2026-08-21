"""Live integration: LLM-as-judge.

`LlmJudge` wraps the model in `StructuredChatModel`, which sets
`response_format=json_schema` (strict schema mode).

Not every OpenAI-compatible endpoint implements schema mode --
DeepSeek answers `400 "This response_format type is unavailable now"`
and only supports the loose `json_object` variant. These tests are
gated on `_capabilities.SUPPORTS_JSON_SCHEMA` and run against any
endpoint that does support it (Ollama, vLLM, OpenAI, LM Studio).
"""
from __future__ import annotations

import pytest

from ._capabilities import NO_JSON_SCHEMA_REASON, SUPPORTS_JSON_SCHEMA


pytestmark = pytest.mark.integration


@pytest.mark.skipif(not SUPPORTS_JSON_SCHEMA, reason=NO_JSON_SCHEMA_REASON)
def test_llm_judge_scores_match(deepseek_chat):
    from litgraph.evaluators import LlmJudge

    judge = LlmJudge(deepseek_chat)
    res = judge.judge(
        prediction="The capital of France is Paris.",
        reference="Paris is the capital of France.",
    )
    assert isinstance(res, dict)
    assert 0.0 <= float(res["score"]) <= 1.0


@pytest.mark.skipif(not SUPPORTS_JSON_SCHEMA, reason=NO_JSON_SCHEMA_REASON)
def test_llm_judge_batch(deepseek_chat):
    from litgraph.evaluators import LlmJudge

    judge = LlmJudge(deepseek_chat)
    pairs = [("Paris", "Paris is the capital of France.")]
    results = judge.judge_batch(pairs)
    assert len(results) == 1

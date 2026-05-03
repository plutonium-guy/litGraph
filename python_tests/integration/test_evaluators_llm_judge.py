"""Live integration: LLM-as-judge — currently BLOCKED on DeepSeek.

`LlmJudge` wraps the model in `StructuredChatModel` which sets
`response_format=json_schema`. DeepSeek's chat-completions endpoint
returns:

    400 Bad Request: "This response_format type is unavailable now"

DeepSeek today only supports `response_format=json_object` (the loose
"emit valid JSON" mode), not the strict `json_schema` variant.

Re-enable these tests when:
- DeepSeek adds `json_schema` support, OR
- We add a fallback in `StructuredChatModel` that downgrades to
  `json_object` + post-validation when the provider rejects schema mode.

Until then the LlmJudge live path runs against OpenAI / Anthropic
(see CONDITIONALLY-TESTABLE table in INTEGRATION_TESTS.md).
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="DeepSeek rejects response_format=json_schema (used by StructuredChatModel)")
def test_llm_judge_scores_match(deepseek_chat):  # pragma: no cover
    from litgraph.evaluators import LlmJudge

    judge = LlmJudge(deepseek_chat)
    res = judge.judge(
        prediction="The capital of France is Paris.",
        reference="Paris is the capital of France.",
    )
    assert isinstance(res, dict)
    assert 0.0 <= float(res["score"]) <= 1.0


@pytest.mark.skip(reason="DeepSeek rejects response_format=json_schema (used by StructuredChatModel)")
def test_llm_judge_batch(deepseek_chat):  # pragma: no cover
    from litgraph.evaluators import LlmJudge

    judge = LlmJudge(deepseek_chat)
    pairs = [("Paris", "Paris is the capital of France.")]
    results = judge.judge_batch(pairs)
    assert len(results) == 1

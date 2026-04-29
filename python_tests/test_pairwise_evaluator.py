"""PairwiseEvaluator — LLM A/B comparison."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.evaluators import PairwiseEvaluator  # noqa: E402
from litgraph.providers import OpenAIChat  # noqa: E402


def test_construct_with_model():
    model = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    e = PairwiseEvaluator(model)
    assert e is not None


def test_construct_with_custom_criteria():
    model = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    e = PairwiseEvaluator(model, criteria="prefer shorter answers")
    assert e is not None

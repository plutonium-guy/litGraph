"""Live integration: string evaluators against real DeepSeek output.

These evaluators are pure-Python utility functions — no LLM call. We
exercise them on outputs the model just produced, mirroring how an
eval loop combines a generation step with deterministic checks.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_exact_match_on_one_word_reply(deepseek_chat):
    from litgraph.evaluators import exact_match

    out = deepseek_chat.invoke(
        [
            {"role": "system", "content": "Reply with exactly one word."},
            {"role": "user", "content": "Capital of France?"},
        ],
        max_tokens=10,
    )
    text = out["text"].strip().rstrip(".")
    # exact_match trims + case-folds, so "paris" vs "Paris" passes.
    assert exact_match(text, "Paris"), f"exact_match failed: {text!r}"


def test_contains_all_and_any_on_factual_reply(deepseek_chat):
    from litgraph.evaluators import contains_any

    out = deepseek_chat.invoke(
        [
            {"role": "user", "content": "Name 3 famous landmarks in Paris. Reply as a comma-separated list."},
        ],
        max_tokens=40,
    )
    text = out["text"]
    # The Eiffel Tower is almost certainly in the list.
    assert contains_any(text, ["Eiffel", "eiffel"]), f"no Eiffel: {text!r}"
    # Be lax on the other landmarks — accept any of several.
    assert contains_any(
        text, ["Louvre", "Notre", "Arc", "Triomphe", "Sacré", "louvre", "notre"]
    ), f"no second landmark: {text!r}"


def test_jaccard_similarity_paraphrase(deepseek_chat):
    from litgraph.evaluators import jaccard_similarity

    out = deepseek_chat.invoke(
        [
            {"role": "user", "content": "Paraphrase this in one sentence: 'The cat sat on the mat.'"},
        ],
        max_tokens=30,
    )
    paraphrase = out["text"].strip()
    score = jaccard_similarity(paraphrase, "The cat sat on the mat.")
    # Paraphrases should overlap on content words but not be identical.
    assert 0.0 <= score <= 1.0
    assert score > 0.1, f"unexpectedly low jaccard: {score} for {paraphrase!r}"


def test_json_validity_on_json_object_mode(deepseek_chat):
    from litgraph.evaluators import json_validity

    out = deepseek_chat.invoke(
        [
            {"role": "system", "content": "Reply with valid json: {capital: string}"},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        max_tokens=30,
        response_format={"type": "json_object"},
    )
    text = out["text"]
    assert json_validity(text), f"json_validity rejected: {text!r}"

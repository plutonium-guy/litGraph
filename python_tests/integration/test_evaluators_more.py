"""Live integration: more evaluator coverage on real DeepSeek output.

- `regex_match`, `levenshtein_ratio`, `contains_all` — string variants
- `PiiScrubber` — masks PII before sending to the model + re-asserts
  the masked-version is still answerable
- `evaluate_trajectory` — pure-Python score over an agent's tool-call
  trajectory
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_regex_match_on_one_word_reply(deepseek_chat):
    from litgraph.evaluators import regex_match

    out = deepseek_chat.invoke(
        [
            {"role": "system", "content": "Reply with one word only."},
            {"role": "user", "content": "Capital of Germany?"},
        ],
        max_tokens=10,
    )
    text = out["text"].strip().rstrip(".")
    # Tolerate surrounding whitespace / punctuation via flexible regex.
    assert regex_match(text, r"^Berlin$"), f"regex_match failed: {text!r}"


def test_levenshtein_ratio_paraphrase_close(deepseek_chat):
    from litgraph.evaluators import levenshtein_ratio

    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with exactly: hello world"}],
        max_tokens=10,
    )
    text = out["text"].strip().lower()
    ratio = levenshtein_ratio(text, "hello world")
    # Should be very close to 1.0 (case + trim already normalised).
    assert ratio >= 0.85, f"levenshtein_ratio low: {ratio} for {text!r}"


def test_contains_all_on_factual_list(deepseek_chat):
    from litgraph.evaluators import contains_all

    out = deepseek_chat.invoke(
        [
            {
                "role": "user",
                "content": "Reply with this exact text: 'red green blue yellow'.",
            }
        ],
        max_tokens=15,
    )
    text = out["text"]
    assert contains_all(text, ["red", "green", "blue"]), (
        f"contains_all missed: {text!r}"
    )


def test_pii_scrubber_round_trip(deepseek_chat):
    """Scrub PII from a user-style message before sending. The masked
    text should still be coherent enough for DeepSeek to respond."""
    from litgraph.evaluators import PiiScrubber

    raw = "My email is alice@example.com and my phone is 555-867-5309. Reply: ok"
    scrubber = PiiScrubber()
    r = scrubber.scrub(raw)
    assert "<EMAIL>" in r["scrubbed"]
    assert "<PHONE>" in r["scrubbed"]
    assert any(rep["kind"] == "EMAIL" for rep in r["replacements"])

    out = deepseek_chat.invoke(
        [{"role": "user", "content": r["scrubbed"]}],
        max_tokens=10,
    )
    assert out["text"].strip(), "model returned nothing on scrubbed input"


def test_evaluate_trajectory_subsequence_match(deepseek_chat):
    """`evaluate_trajectory` is pure-Python — score a fake trajectory
    so the function's contract is exercised inside the integration suite."""
    from litgraph.evaluators import evaluate_trajectory

    actual = ["read_file", "search", "summarise", "write_file"]
    expected = ["read_file", "summarise", "write_file"]
    score = evaluate_trajectory(actual, expected, policy="subsequence")
    # All expected steps appear in order in actual → perfect score.
    assert isinstance(score, (int, float))
    assert float(score) >= 0.9, f"subsequence score low: {score}"

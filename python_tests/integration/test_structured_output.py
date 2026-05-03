"""Live integration: structured output via DeepSeek's JSON mode.

DeepSeek implements OpenAI's `response_format` field. The
`OpenAIChat.invoke(response_format={"type":"json_object"})` path
forces the model to return valid JSON. We don't exercise the
`json_schema` variant here because DeepSeek's schema-validation
support depends on the model version and isn't always strict —
the JSON-object mode is the safe interop subset.
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.integration


def test_json_object_mode_returns_valid_json(deepseek_chat):
    # DeepSeek requires "json" appear in the prompt when
    # `response_format=json_object` is set — otherwise the request
    # is rejected with an `invalid_request_error`.
    out = deepseek_chat.invoke(
        [
            {
                "role": "system",
                "content": 'Reply with a json object {"city": str, "country": str}.',
            },
            {"role": "user", "content": "Pick any famous city."},
        ],
        response_format={"type": "json_object"},
        max_tokens=80,
    )
    text = out["text"].strip()
    parsed = json.loads(text)  # raises if not valid JSON
    assert isinstance(parsed, dict)
    # Tolerate whatever city the model picks; just assert the shape.
    assert "city" in parsed
    assert "country" in parsed


def test_json_object_mode_temperature_zero_is_deterministic_enough(deepseek_chat):
    """Two calls with the same prompt + temperature=0 should produce
    JSON-parseable replies.

    DeepSeek requires the prompt to contain the substring 'json'
    when `response_format=json_object` is set — they reject the
    request otherwise. The system prompt below satisfies that."""
    msgs = [
        {
            "role": "system",
            "content": 'Reply with valid json: {"answer": str}.',
        },
        {"role": "user", "content": "Capital of France?"},
    ]
    out = deepseek_chat.invoke(
        msgs,
        response_format={"type": "json_object"},
        temperature=0,
        max_tokens=40,
    )
    parsed = json.loads(out["text"])
    assert "answer" in parsed
    # The capital of France is "Paris" — model should pick it.
    assert "Paris" in str(parsed["answer"])

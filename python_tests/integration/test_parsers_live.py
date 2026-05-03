"""Live integration: `parse_json_with_retry` + `fix_with_llm` against DeepSeek.

These parsers ask a fixer model to repair malformed output. Verify the
fix-loop actually works against a real provider (not just local repair
heuristics).
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.integration


def test_parse_json_with_retry_repairs_trailing_comma(deepseek_chat):
    from litgraph.parsers import parse_json_with_retry

    raw = '{"name": "alice", "age": 30,}'  # trailing comma — invalid JSON
    data = parse_json_with_retry(
        raw=raw,
        model=deepseek_chat,
        schema_hint="{name: str, age: int}",
        max_retries=2,
    )
    assert isinstance(data, dict)
    assert data.get("name") == "alice"
    assert int(data.get("age", 0)) == 30


def test_fix_with_llm_repairs_string(deepseek_chat):
    from litgraph.parsers import fix_with_llm

    raw = '{"x": 1,}'
    fixed = fix_with_llm(
        raw=raw,
        error="trailing comma — invalid JSON",
        instructions="Return a valid JSON object only. No prose, no fences.",
        model=deepseek_chat,
    )
    # The fixer returns a string; we just verify it parses.
    text = fixed if isinstance(fixed, str) else fixed.get("text", "")
    # Models sometimes wrap in code fences — tolerate.
    text = text.strip().strip("`").strip()
    if text.startswith("json"):
        text = text[4:].strip()
    parsed = json.loads(text)
    assert parsed.get("x") == 1

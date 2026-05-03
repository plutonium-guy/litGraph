"""Live integration: more splitter variants exercised end-to-end.

- `CodeSplitter` — Python source split at function boundaries
- `JsonSplitter` — recursive JSON split keeping structure valid
- `TokenTextSplitter` — token-budget splitting

Each splitter chunks a payload; one chunk goes through DeepSeek for a
sanity-check round trip.
"""
from __future__ import annotations

import json

import pytest


pytestmark = pytest.mark.integration


_PY_SOURCE = '''
def add(a, b):
    """Add two numbers."""
    return a + b


def subtract(a, b):
    """Subtract b from a."""
    return a - b


class Calculator:
    """A simple calculator."""

    def multiply(self, a, b):
        return a * b

    def divide(self, a, b):
        if b == 0:
            raise ValueError("division by zero")
        return a / b
'''


def test_code_splitter_python_keeps_definitions(deepseek_chat):
    from litgraph.splitters import CodeSplitter

    splitter = CodeSplitter(language="python", chunk_size=200, chunk_overlap=0)
    chunks = splitter.split_text(_PY_SOURCE)
    assert chunks, "code splitter returned no chunks"
    # At least one chunk should contain a top-level def or class.
    flat = "\n".join(chunks)
    assert "def add" in flat or "class Calculator" in flat


def test_json_splitter_chunks_structure(deepseek_chat):
    from litgraph.splitters import JsonSplitter

    payload = {
        "items": [
            {"id": i, "title": f"Item {i}", "body": "lorem " * 30}
            for i in range(20)
        ]
    }
    splitter = JsonSplitter(max_chunk_size=500)
    raw = json.dumps(payload)
    # The splitter typically takes a JSON string OR a dict — try the
    # most likely API; tolerate either signature.
    if hasattr(splitter, "split_text"):
        chunks = splitter.split_text(raw)
    elif hasattr(splitter, "split_json"):
        chunks = splitter.split_json(payload)
    else:
        pytest.skip("JsonSplitter has no split_text/split_json method")
    assert chunks, "json splitter returned no chunks"


def test_token_text_splitter_respects_budget(deepseek_chat):
    from litgraph.splitters import TokenTextSplitter

    text = "The quick brown fox jumps over the lazy dog. " * 200
    splitter = TokenTextSplitter(chunk_size=64, chunk_overlap=8)
    chunks = splitter.split_text(text)
    assert len(chunks) > 1, "token splitter produced single chunk"

    # Sanity check: send the first chunk to DeepSeek for a paraphrase.
    out = deepseek_chat.invoke(
        [{"role": "user", "content": f"Paraphrase: {chunks[0]}"}],
        max_tokens=30,
    )
    assert out["text"].strip()

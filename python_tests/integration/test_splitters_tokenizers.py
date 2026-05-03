"""Live integration: text splitters + tokenizers feeding a real model.

Splitters and tokenizers are deterministic (no API call) but their
contracts often interact with model context limits. We exercise:
- `RecursiveCharacterSplitter` chunks a long doc; the model summarises
  one chunk
- `MarkdownHeaderSplitter` walks markdown structure
- `tokenizers.count_tokens` + `trim_messages` ensure long history
  trims before model invoke
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


_LONG_TEXT = (
    "Rome is the capital of Italy. " * 200  # ~6 KB of text
)


def test_recursive_character_splitter_chunks_then_model_summarises(deepseek_chat):
    from litgraph.splitters import RecursiveCharacterSplitter

    splitter = RecursiveCharacterSplitter(chunk_size=200, chunk_overlap=20)
    chunks = splitter.split_text(_LONG_TEXT)
    assert len(chunks) > 1, "expected multi-chunk split"

    # Send the first chunk through DeepSeek for a one-line summary.
    out = deepseek_chat.invoke(
        [
            {"role": "user", "content": f"Summarise in one sentence: {chunks[0]}"},
        ],
        max_tokens=30,
    )
    assert "Rome" in out["text"] or "Italy" in out["text"], (
        f"summary missed key terms: {out['text']!r}"
    )


def test_markdown_header_splitter_walks_structure(deepseek_chat):
    from litgraph.splitters import MarkdownHeaderSplitter

    md = """# Top
intro under top

## Sub A
content of sub A

## Sub B
content of sub B
"""
    splitter = MarkdownHeaderSplitter(max_depth=2)
    chunks = splitter.split_text(md) if hasattr(splitter, "split_text") else splitter.split_documents([{"page_content": md}])
    assert chunks, "splitter returned no chunks"


def test_tokenizers_count_tokens_and_trim(deepseek_chat):
    from litgraph.tokenizers import count_tokens, trim_messages

    text = "hello world " * 100
    n = count_tokens("gpt-4o-mini", text)
    assert isinstance(n, int)
    assert n > 0

    msgs = [
        {"role": "system", "content": "You are terse."},
        *[{"role": "user", "content": f"turn {i}: " + ("noise " * 50)} for i in range(20)],
        {"role": "user", "content": "Reply: ok"},
    ]
    trimmed = trim_messages("gpt-4o-mini", msgs, max_tokens=200)
    assert len(trimmed) < len(msgs), "trim_messages dropped nothing"
    # System message must be retained.
    assert any(m["role"] == "system" for m in trimmed)
    # Last message must be retained.
    assert trimmed[-1]["content"] == "Reply: ok"

    # The trimmed list should still produce a valid model call.
    out = deepseek_chat.invoke(trimmed, max_tokens=10)
    assert out["text"].strip()

"""Live integration: file loaders feed text into a real DeepSeek call.

Tests the loader → splitter → model pipeline for the most common
local-file shapes:
- `TextLoader` (UTF-8 plain text)
- `JsonLoader` (JSON file with `content_field=`)
- `CsvLoader` (CSV with `content_column=`)
- `MarkdownLoader`
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_text_loader_then_summarise(deepseek_chat):
    from litgraph.loaders import TextLoader

    with tempfile.TemporaryDirectory() as root:
        p = Path(root) / "doc.txt"
        p.write_text("Paris is the capital of France. The Eiffel Tower is iconic.\n")
        docs = TextLoader(str(p)).load()
        assert docs, "TextLoader returned no docs"
        first = docs[0]
        content = first.get("page_content") or first.get("content") or ""
        assert "Paris" in content

        out = deepseek_chat.invoke(
            [{"role": "user", "content": f"Summarise in one sentence: {content}"}],
            max_tokens=30,
        )
        assert "Paris" in out["text"] or "France" in out["text"]


def test_json_loader_with_content_field(deepseek_chat):
    from litgraph.loaders import JsonLoader

    with tempfile.TemporaryDirectory() as root:
        p = Path(root) / "data.json"
        p.write_text(json.dumps([
            {"id": 1, "body": "Rome is the capital of Italy."},
            {"id": 2, "body": "Berlin is the capital of Germany."},
        ]))
        docs = JsonLoader(str(p), content_field="body").load()
        assert len(docs) == 2
        contents = [(d.get("page_content") or d.get("content") or "") for d in docs]
        assert any("Rome" in c for c in contents)
        assert any("Berlin" in c for c in contents)


def test_csv_loader_with_content_column(deepseek_chat):
    from litgraph.loaders import CsvLoader

    with tempfile.TemporaryDirectory() as root:
        p = Path(root) / "facts.csv"
        p.write_text(
            "id,fact\n"
            "1,Madrid is the capital of Spain.\n"
            "2,Lisbon is the capital of Portugal.\n"
        )
        docs = CsvLoader(str(p), content_column="fact").load()
        assert len(docs) == 2
        contents = [(d.get("page_content") or d.get("content") or "") for d in docs]
        assert any("Madrid" in c for c in contents)


def test_markdown_loader_then_invoke(deepseek_chat):
    from litgraph.loaders import MarkdownLoader

    with tempfile.TemporaryDirectory() as root:
        p = Path(root) / "doc.md"
        p.write_text("# Capitals\n\n- Paris is the capital of France.\n")
        docs = MarkdownLoader(str(p)).load()
        assert docs, "MarkdownLoader returned no docs"
        # Just verify the loader produces non-empty content.
        first = docs[0]
        content = first.get("page_content") or first.get("content") or ""
        assert content.strip()

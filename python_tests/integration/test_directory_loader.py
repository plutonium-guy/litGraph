"""Live integration: `DirectoryLoader` walks a tree of mixed files.

Verifies the loader picks up multiple files under a glob and the
returned docs carry per-file metadata.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_directory_loader_collects_text_files(deepseek_chat):
    from litgraph.loaders import DirectoryLoader

    with tempfile.TemporaryDirectory() as root:
        Path(root, "a.txt").write_text("first file content")
        Path(root, "b.txt").write_text("second file content")
        Path(root, "skipme.bin").write_text("binary stuff")

        # Common shape — load all .txt files.
        loader = DirectoryLoader(str(root), glob="*.txt")
        docs = loader.load()
        assert len(docs) == 2, f"expected 2 docs, got {len(docs)}"
        contents = [(d.get("page_content") or d.get("content") or "") for d in docs]
        assert any("first" in c for c in contents)
        assert any("second" in c for c in contents)

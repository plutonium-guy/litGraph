"""Live integration: `ReadFileTool` + `WriteFileTool` + `ListDirectoryTool`
driven through a DeepSeek `ReactAgent`.

All three tools are sandboxed against a `sandbox_root` — paths are
resolved against the root and `..` / absolute escapes are rejected.
We seed a small text file, ask the agent to read it back, and verify
the answer threads through.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def test_react_agent_reads_seed_file(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import ListDirectoryTool, ReadFileTool

    with tempfile.TemporaryDirectory() as root:
        # Seed the sandbox with one file.
        Path(root, "secret.txt").write_text("MAGIC-WORD-7429\n", encoding="utf-8")

        agent = ReactAgent(
            deepseek_chat,
            [ReadFileTool(sandbox_root=root), ListDirectoryTool(sandbox_root=root)],
            system_prompt=(
                "You can list files and read them. The user will ask about a "
                "file in the sandbox; use the tools to find and read it. "
                "Reply with just the contents."
            ),
            max_iterations=5,
        )
        state = agent.invoke("Read secret.txt and reply with its content.")
        msgs = state["messages"]
        final = msgs[-1]
        text = final.get("content", "") if isinstance(final, dict) else str(final)
        if isinstance(text, list):
            text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
        assert "MAGIC-WORD-7429" in (text or ""), (
            f"agent did not surface file content: {final!r}"
        )


def test_react_agent_writes_then_reads(deepseek_chat):
    from litgraph.agents import ReactAgent
    from litgraph.tools import ReadFileTool, WriteFileTool

    with tempfile.TemporaryDirectory() as root:
        agent = ReactAgent(
            deepseek_chat,
            [
                WriteFileTool(sandbox_root=root),
                ReadFileTool(sandbox_root=root),
            ],
            system_prompt=(
                "You have write_file and read_file tools. The user will ask "
                "you to write a file then read it back. Be terse."
            ),
            max_iterations=6,
        )
        state = agent.invoke(
            "Write the text 'hello-litgraph' to a file named note.txt, "
            "then read note.txt and reply with its content."
        )
        msgs = state["messages"]
        # Best-effort: verify the file was actually written by checking
        # the sandbox post-run (the model may or may not echo the
        # content faithfully).
        assert (Path(root) / "note.txt").exists(), (
            "WriteFileTool didn't actually write the file"
        )
        assert "hello-litgraph" in (Path(root) / "note.txt").read_text(), (
            "file content didn't match what was requested"
        )

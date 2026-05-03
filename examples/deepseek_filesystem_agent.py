"""Sandboxed filesystem agent: read + write files inside a temp root.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_filesystem_agent.py

The agent gets two tools: `read_file` and `write_file`, both rooted in
a tempdir sandbox. Path traversal (`..`, absolute paths) is rejected.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import ReadFileTool, WriteFileTool


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    with tempfile.TemporaryDirectory() as root:
        agent = ReactAgent(
            chat,
            [
                WriteFileTool(sandbox_root=root),
                ReadFileTool(sandbox_root=root),
            ],
            system_prompt=(
                "You can write and read files in the sandbox. Be terse."
            ),
            max_iterations=6,
        )

        state = agent.invoke(
            "Write 'hello-litgraph' to note.txt, then read it back."
        )
        print("Final agent message:")
        print(state["messages"][-1])
        print()
        print(f"Sandbox content of note.txt:")
        print((Path(root) / "note.txt").read_text())


if __name__ == "__main__":
    main()

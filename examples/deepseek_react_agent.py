"""DeepSeek + a ReactAgent + one tool, end-to-end.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_react_agent.py
"""
from __future__ import annotations

import os

from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool


def add(a: int, b: int) -> dict:
    """Tool implementation. FunctionTool unpacks the JSON args
    against the declared schema → kwargs into this function."""
    return {"sum": int(a) + int(b)}


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    add_tool = FunctionTool(
        "add",
        "Add two integers. Returns {sum: int}.",
        {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
        add,
    )

    agent = ReactAgent(
        chat,
        [add_tool],
        system_prompt="Be terse. Use the `add` tool for arithmetic.",
        max_iterations=4,
    )

    state = agent.invoke("What is 17 + 25?")
    final = state["messages"][-1]
    text = final.get("content") if isinstance(final, dict) else str(final)
    print(f"Answer: {text}")
    print(f"Total messages in trace: {len(state['messages'])}")


if __name__ == "__main__":
    main()

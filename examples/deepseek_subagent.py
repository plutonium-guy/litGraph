"""SubagentTool: parent ReactAgent delegates to a child specialist.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_subagent.py

The parent agent has ONE tool — a math sub-agent. When the user asks
arithmetic, the parent invokes the sub-agent, which has its own
internal tools and prompt. Mirrors the LangChain `deepagents` `task`
primitive.
"""
from __future__ import annotations

import os

from litgraph.agents import ReactAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import FunctionTool, SubagentTool


def _add(a, b):
    return {"sum": int(a) + int(b)}


def main() -> None:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise SystemExit("Set DEEPSEEK_API_KEY in your environment.")

    chat = OpenAIChat(
        api_key=api_key,
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1",
    )

    add_tool = FunctionTool(
        "add",
        "Add two integers.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        _add,
    )

    math_subagent = ReactAgent(
        chat,
        [add_tool],
        system_prompt="You are a math specialist. Use the add tool. Reply with just the number.",
        max_iterations=4,
    )

    parent = ReactAgent(
        chat,
        [SubagentTool(
            "math_subagent",
            "Delegates an arithmetic question to a math-specialist sub-agent.",
            math_subagent,
        )],
        system_prompt=(
            "You delegate math questions to the math_subagent tool. "
            "Pass the user's question through. Be terse."
        ),
        max_iterations=4,
    )

    state = parent.invoke("What is 17 + 25? Use the math_subagent.")
    msgs = state["messages"]
    print(msgs[-1])


if __name__ == "__main__":
    main()

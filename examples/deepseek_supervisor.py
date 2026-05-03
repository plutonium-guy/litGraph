"""SupervisorAgent routing example with DeepSeek.

Run:
    export DEEPSEEK_API_KEY=sk-...
    python examples/deepseek_supervisor.py

The supervisor sees the user message, picks a worker, and hands off.
Workers are themselves `ReactAgent`s with their own tools/system prompts.
"""
from __future__ import annotations

import os

from litgraph.agents import ReactAgent, SupervisorAgent
from litgraph.providers import OpenAIChat
from litgraph.tools import FunctionTool


def _add(a, b):
    return {"result": int(a) + int(b)}


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

    math_worker = ReactAgent(
        chat,
        [add_tool],
        system_prompt="You are the math expert. Use the add tool for arithmetic.",
        max_iterations=4,
    )
    chitchat_worker = ReactAgent(
        chat,
        [],
        system_prompt="You are the chitchat agent. Be friendly and brief.",
        max_iterations=2,
    )

    sup = SupervisorAgent(
        chat,
        {"math": math_worker, "chitchat": chitchat_worker},
        system_prompt=(
            "You delegate to workers. Route arithmetic to 'math', "
            "everything else to 'chitchat'."
        ),
        max_hops=4,
    )

    print("Workers:", sup.worker_names())
    print()

    for q in ["What is 17 + 25?", "Hi, how are you?"]:
        print(f">>> {q}")
        out = sup.invoke(q)
        print(out)
        print()


if __name__ == "__main__":
    main()

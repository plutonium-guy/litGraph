"""Run a complete tool-calling agent offline with deterministic replies.

Run:  pixi run python examples/scripted_agent.py
"""
from litgraph import create_agent
from litgraph.testing import ScriptedChatModel
from litgraph.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


model = ScriptedChatModel(
    [
        {
            "tool_calls": [
                {
                    "id": "add-1",
                    "name": "add",
                    "arguments": {"a": 17, "b": 25},
                }
            ]
        },
        "17 + 25 = 42",
    ]
)
harness = create_agent(model, tools=[add], instructions="Use tools for arithmetic.")
result = harness.run("What is 17 + 25?")
print(result.output)
print(f"model calls: {model.call_count}")

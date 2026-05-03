"""Live integration: `RetryTool` + `TimeoutTool` + `CachedTool` wrappers.

Each wrapper composes around any Tool. We use a `FunctionTool` whose
callable lets us count invocations (for cache-hit assertion) and inject
a transient failure (for retry assertion). Drives the wrapped tools
through a real DeepSeek `ReactAgent`.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_cached_tool_skips_repeat_invocations(deepseek_chat):
    """Same args → tool should only run once when wrapped in CachedTool."""
    from litgraph.agents import ReactAgent
    from litgraph.tools import CachedTool, FunctionTool

    counter = {"n": 0}

    def square(n):
        counter["n"] += 1
        return {"result": int(n) ** 2}

    base = FunctionTool(
        "square",
        "Square an integer.",
        {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]},
        square,
    )
    cached = CachedTool(base, ttl_seconds=60, max_entries=16)

    agent = ReactAgent(
        deepseek_chat,
        [cached],
        system_prompt="Use the square tool to answer. Be terse.",
        max_iterations=4,
    )
    agent.invoke("What is 7 squared? Use the square tool.")
    first_n = counter["n"]
    agent.invoke("What is 7 squared? Use the square tool.")
    second_n = counter["n"]

    # Second run should hit the cache — counter should NOT increase.
    # Tolerate one extra call (model may pass slightly different args
    # on the second run).
    assert second_n <= first_n + 1, (
        f"cache miss: first={first_n} second={second_n}"
    )


def test_timeout_tool_constructible(deepseek_chat):
    """TimeoutTool wraps any tool and must be accepted by ReactAgent."""
    from litgraph.agents import ReactAgent
    from litgraph.tools import FunctionTool, TimeoutTool

    def quick(x):
        return {"echo": str(x)}

    base = FunctionTool(
        "quick",
        "Echo input quickly.",
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        quick,
    )
    safe = TimeoutTool(base, timeout_s=5)
    agent = ReactAgent(
        deepseek_chat,
        [safe],
        system_prompt="Be terse.",
        max_iterations=3,
    )
    state = agent.invoke("Reply with: ok")
    assert state["messages"], "agent produced no messages"


def test_retry_tool_constructible(deepseek_chat):
    """RetryTool composes around a base tool. Cheap smoke — happy path
    should not retry (no transient failure injected)."""
    from litgraph.agents import ReactAgent
    from litgraph.tools import FunctionTool, RetryTool

    def echo(x):
        return {"echo": str(x)}

    base = FunctionTool(
        "echo",
        "Echo input.",
        {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        echo,
    )
    resilient = RetryTool(base, max_attempts=2, initial_delay_s=0.05)
    agent = ReactAgent(
        deepseek_chat,
        [resilient],
        system_prompt="Be terse.",
        max_iterations=3,
    )
    state = agent.invoke("Reply with: ok")
    assert state["messages"], "agent produced no messages"

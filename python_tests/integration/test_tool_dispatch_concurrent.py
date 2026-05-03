"""Live integration: `tool_dispatch_concurrent` fan-out.

This API runs N tool calls in parallel (no model in the loop). The model
isn't strictly required, but we still gate on DeepSeek so the suite
runs as one block. Verifies the public Python surface for batch tool
dispatch outside the React agent loop.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _add(a, b):
    return {"result": int(a) + int(b)}


def _add_tool():
    from litgraph.tools import FunctionTool

    return FunctionTool(
        "add",
        "Add two integers.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        _add,
    )


def test_tool_dispatch_concurrent_returns_aligned_results(deepseek_chat):
    from litgraph.agents import tool_dispatch_concurrent

    tools = [_add_tool()]
    calls = [
        {"name": "add", "args": {"a": 1, "b": 2}},
        {"name": "add", "args": {"a": 10, "b": 20}},
        {"name": "add", "args": {"a": 100, "b": 200}},
    ]
    results = tool_dispatch_concurrent(tools, calls, max_concurrency=3)
    assert len(results) == 3
    # Results may be JSON strings or dicts depending on the dispatcher.
    import json

    parsed = []
    for r in results:
        if isinstance(r, str):
            parsed.append(json.loads(r))
        else:
            parsed.append(r)
    assert parsed[0]["result"] == 3
    assert parsed[1]["result"] == 30
    assert parsed[2]["result"] == 300


def test_tool_dispatch_concurrent_unknown_tool_per_call_failure(deepseek_chat):
    """fail_fast=False (default): unknown-tool calls land an error slot
    rather than tanking the whole batch."""
    from litgraph.agents import tool_dispatch_concurrent

    tools = [_add_tool()]
    calls = [
        {"name": "add", "args": {"a": 1, "b": 2}},
        {"name": "nonexistent", "args": {}},
    ]
    results = tool_dispatch_concurrent(tools, calls, max_concurrency=2, fail_fast=False)
    assert len(results) == 2
    # First call should succeed; second should be an error sentinel.
    import json

    first = results[0]
    if isinstance(first, str):
        first = json.loads(first)
    assert first.get("result") == 3

    second = results[1]
    if isinstance(second, str):
        try:
            second = json.loads(second)
        except json.JSONDecodeError:
            second = {"error": second}
    assert "error" in second or "Error" in str(second)

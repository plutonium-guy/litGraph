"""Live integration: `litgraph.tool_hooks` middleware patterns.

Note (iter 355): the native `ReactAgent`'s tool-extraction is
strict — it accepts only registered Rust-backed `Tool` types
(FunctionTool, ShellTool, …). Python-side `HookedTool` wrappers
DON'T pass the type check, so wrapping tools with
`litgraph.tool_hooks.wrap_tools` doesn't compose with `ReactAgent`
out of the box. The Rust-side `ToolMiddleware` chain (wired in
iter 348-350) is the working substitute for the agent loop.

This test file therefore covers the standalone behaviour of the
Python hooks (when used outside a native ReactAgent loop) — see
`python_tests/test_tool_hooks.py` for the unit tests, and
`MIDDLEWARE.md` for the decision matrix.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_python_hooks_compose_with_real_call(deepseek_chat):
    """We exercise the hooks directly: wrap a function-tool, drive
    it manually with input derived from a real DeepSeek call.
    Proves the Python hook chain fires end-to-end."""
    from litgraph.testing import MockTool
    from litgraph.tool_hooks import (
        BeforeToolHook, AfterToolHook, ToolBudget, wrap_tool,
    )

    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply with: 7"}],
        max_tokens=10,
    )
    user_value = out["text"].strip()

    seen_before: list = []
    seen_after: list = []
    inner = MockTool("noop", returns={"echo": user_value})
    wrapped = wrap_tool(
        inner,
        before=BeforeToolHook(lambda name, args: seen_before.append(name)),
        after=AfterToolHook(lambda name, args, r: seen_after.append((name, r))),
        budget=ToolBudget(max_calls_per_turn=3),
    )

    out_val = wrapped.invoke({"q": user_value})
    assert seen_before == ["noop"]
    assert seen_after and seen_after[0][0] == "noop"
    assert out_val["echo"] == user_value


def test_react_agent_with_hooked_tool_via_to_function_tool(deepseek_chat):
    """`HookedTool.to_function_tool()` (added iter 376) returns a
    native `FunctionTool` whose callable fires the before/after/budget
    hooks before delegating to the inner tool. ReactAgent accepts the
    resulting native tool, and the hooks observe every dispatch."""
    from litgraph.agents import ReactAgent
    from litgraph.tools import FunctionTool
    from litgraph.tool_hooks import (
        AfterToolHook,
        BeforeToolHook,
        ToolBudget,
        wrap_tool,
    )

    seen_before: list = []
    seen_after: list = []

    def _add(a, b):
        return {"sum": int(a) + int(b)}

    schema = {
        "type": "object",
        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
        "required": ["a", "b"],
    }
    inner = FunctionTool("add", "Add two integers.", schema, _add)
    hooked = wrap_tool(
        inner,
        before=BeforeToolHook(lambda name, args: (seen_before.append((name, dict(args))), args)[1]),
        after=AfterToolHook(lambda name, args, r: (seen_after.append((name, r)), r)[1]),
        budget=ToolBudget(max_calls_per_turn=4),
    )

    # `to_function_tool(callable, schema=, description=)` — native
    # FunctionTool's body and schema aren't reachable from Python, so
    # caller passes them explicitly. Hooks fire on every dispatch
    # inside the agent loop.
    agent = ReactAgent(
        deepseek_chat,
        [hooked.to_function_tool(_add, schema=schema, description="Add two integers.")],
        system_prompt="Use the add tool for the arithmetic. Be terse.",
        max_iterations=4,
    )
    state = agent.invoke("What is 17 + 25? Use the add tool.")
    msgs = state["messages"]
    final = msgs[-1]
    text = final.get("content", "") if isinstance(final, dict) else str(final)
    if isinstance(text, list):
        text = " ".join(p.get("text", "") for p in text if isinstance(p, dict))
    assert "42" in (text or ""), f"agent answer wrong: {final!r}"
    assert seen_before, "before-hook never fired inside the agent loop"
    assert seen_after, "after-hook never fired inside the agent loop"
    assert seen_before[0][0] == "add"
    # The after-hook should see the inner tool's actual result.
    assert seen_after[0][1] == {"sum": 42}

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


@pytest.mark.skip(reason=(
    "ReactAgent rejects Python-side HookedTool wrappers — its "
    "`extract_tools` only accepts Rust-backed Tool types. Use the "
    "Rust ToolMiddlewareChain (litgraph_agents::middleware) for "
    "ReactAgent integration. See MIDDLEWARE.md."
))
def test_react_agent_with_python_wrapped_tools_blocked():
    pass

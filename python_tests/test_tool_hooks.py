"""Tests for litgraph.tool_hooks — before/after hooks + budget."""
from __future__ import annotations

import pytest

from litgraph.testing import MockTool
from litgraph.tool_hooks import (
    BeforeToolHook,
    AfterToolHook,
    ToolBudget,
    ToolBudgetExceeded,
    HookedTool,
    wrap_tool,
    wrap_tools,
)


def test_before_hook_observes_call():
    seen: list = []
    hk = BeforeToolHook(lambda name, args: seen.append((name, dict(args))))
    t = wrap_tool(MockTool("noop", returns=1), before=hk)
    t.invoke({"a": 1})
    assert seen == [("noop", {"a": 1})]


def test_before_hook_can_mutate_args():
    hk = BeforeToolHook(lambda name, args: {**args, "z": 9})
    inner = MockTool("captures", returns=None)
    wrap_tool(inner, before=hk).invoke({"a": 1})
    assert inner.calls == [{"a": 1, "z": 9}]


def test_after_hook_observes_result():
    seen: list = []
    hk = AfterToolHook(lambda name, args, result: seen.append((name, result)))
    t = wrap_tool(MockTool("ret", returns={"k": 1}), after=hk)
    t.invoke({})
    assert seen == [("ret", {"k": 1})]


def test_after_hook_can_replace_result():
    hk = AfterToolHook(lambda name, args, result: {"replaced": True})
    out = wrap_tool(MockTool("orig", returns={"orig": True}), after=hk).invoke({})
    assert out == {"replaced": True}


def test_after_hook_returning_none_keeps_result():
    hk = AfterToolHook(lambda name, args, result: None)
    out = wrap_tool(MockTool("x", returns=42), after=hk).invoke({})
    assert out == 42


def test_budget_caps_calls_per_turn():
    budget = ToolBudget(max_calls_per_turn=2)
    t = wrap_tool(MockTool("x", returns=1), budget=budget)
    t.invoke({})
    t.invoke({})
    with pytest.raises(ToolBudgetExceeded):
        t.invoke({})


def test_budget_resets():
    budget = ToolBudget(max_calls_per_turn=1)
    t = wrap_tool(MockTool("x", returns=1), budget=budget)
    t.invoke({})
    with pytest.raises(ToolBudgetExceeded):
        t.invoke({})
    budget.reset()
    t.invoke({})  # should not raise


def test_budget_zero_rejected():
    with pytest.raises(ValueError):
        ToolBudget(max_calls_per_turn=0)


def test_wrap_tools_shares_budget():
    """All wrappers from a single wrap_tools share the same budget."""
    budget = ToolBudget(max_calls_per_turn=2)
    a = MockTool("a", returns=1)
    b = MockTool("b", returns=2)
    wrapped_a, wrapped_b = wrap_tools([a, b], budget=budget)
    wrapped_a.invoke({})
    wrapped_b.invoke({})
    with pytest.raises(ToolBudgetExceeded):
        wrapped_a.invoke({})


def test_hooked_tool_mirrors_metadata():
    inner = MockTool("foo", returns=1, description="docs", schema={"type": "object"})
    h = HookedTool(inner)
    assert h.name == "foo"
    assert h.description == "docs"
    assert h.schema == {"type": "object"}


def test_hooked_tool_run_and_call_aliases_invoke():
    t = wrap_tool(MockTool("x", returns=42))
    assert t.invoke({}) == 42
    assert t.run({}) == 42
    assert t({}) == 42


def test_before_hook_returning_non_mapping_keeps_args():
    """If callback returns None or junk, args pass through unchanged."""
    hk = BeforeToolHook(lambda name, args: None)
    inner = MockTool("x", returns=None)
    wrap_tool(inner, before=hk).invoke({"a": 1, "b": 2})
    assert inner.calls == [{"a": 1, "b": 2}]

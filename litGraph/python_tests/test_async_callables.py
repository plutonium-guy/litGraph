"""Async Python callables in StateGraph nodes + FunctionTool — coroutines are
detected by `__await__` and awaited via pyo3-async-runtimes."""

import asyncio

from litgraph.graph import StateGraph, START, END
from litgraph.tools import FunctionTool


async def _async_inc(state):
    """Async node: bumps `n` by 1 after a real await point."""
    await asyncio.sleep(0)
    return {"n": state["n"] + 1}


def test_state_graph_async_node():
    g = StateGraph()
    g.add_node("inc", _async_inc)
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    compiled = g.compile()
    out = compiled.invoke({"n": 5})
    assert out["n"] == 6


def test_state_graph_mixed_sync_async_nodes():
    """Sync and async nodes coexist in one graph."""
    async def double_async(s):
        await asyncio.sleep(0)
        return {"n": s["n"] * 2}

    def add_one(s):
        return {"n": s["n"] + 1}

    g = StateGraph()
    g.add_node("double", double_async)
    g.add_node("inc", add_one)
    g.add_edge(START, "double")
    g.add_edge("double", "inc")
    g.add_edge("inc", END)
    compiled = g.compile()
    assert compiled.invoke({"n": 7})["n"] == 15  # 7*2 + 1


def test_state_graph_parallel_async_branches():
    """Two async branches run concurrently; the reducer concatenates results."""
    async def branch_a(_s):
        await asyncio.sleep(0)
        return {"items": ["a"]}

    async def branch_b(_s):
        await asyncio.sleep(0)
        return {"items": ["b"]}

    def join(_s):
        return {}

    g = StateGraph()
    g.add_node("a", branch_a)
    g.add_node("b", branch_b)
    g.add_node("join", join)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge("a", "join")
    g.add_edge("b", "join")
    g.add_edge("join", END)
    compiled = g.compile()
    out = compiled.invoke({"items": []})
    assert sorted(out["items"]) == ["a", "b"]


def test_function_tool_async_callable():
    """Async function tools are awaited, not blocked on."""
    async def add_async(args):
        await asyncio.sleep(0)
        return {"sum": args["a"] + args["b"]}

    t = FunctionTool(
        "add",
        "Add two integers (async).",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        add_async,
    )
    # We can't invoke a Tool directly from Python (it's only used inside
    # ReactAgent), but verifying the FunctionTool constructs without error
    # confirms the wrapping path. The await path is exercised by the agent
    # tests in test_streaming_and_tools.py via the same Tool::run code.
    assert t.name == "add"


if __name__ == "__main__":
    fns = [
        test_state_graph_async_node,
        test_state_graph_mixed_sync_async_nodes,
        test_state_graph_parallel_async_branches,
        test_function_tool_async_callable,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

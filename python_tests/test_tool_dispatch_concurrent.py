"""tool_dispatch_concurrent — bounded-concurrency parallel tool dispatch
outside the React agent loop. Useful for Plan-and-Execute orchestrators,
custom batch dispatchers, eval harnesses.

Tests use FunctionTool (in-process Python callable) so they're
deterministic and network-free."""
from litgraph.agents import tool_dispatch_concurrent
from litgraph.tools import tool


def _echo_tool(name):
    """Build a FunctionTool with the given name, echoing its args."""
    def fn(i: int = 0, x: int = 0, v: str = ""):
        return {"tool": name, "args": {"i": i, "x": x, "v": v}}
    fn.__name__ = name
    fn.__doc__ = f"echo tool {name}"
    return tool(fn)


def _failing_tool():
    def fail(_unused: str = ""):
        """always errors"""
        raise RuntimeError("synthetic tool failure")
    return tool(fail)


def test_tool_dispatch_returns_one_result_per_call():
    tools = [_echo_tool("a"), _echo_tool("b")]
    calls = [
        {"name": "a", "args": {"i": 0}},
        {"name": "b", "args": {"i": 1}},
        {"name": "a", "args": {"i": 2}},
    ]
    out = tool_dispatch_concurrent(tools, calls, max_concurrency=2)
    assert len(out) == 3


def test_tool_dispatch_aligned_to_input():
    tools = [_echo_tool("a"), _echo_tool("b"), _echo_tool("c")]
    calls = [
        ("a", {"i": 0}),
        ("b", {"i": 1}),
        ("c", {"i": 2}),
        ("a", {"i": 3}),
    ]
    out = tool_dispatch_concurrent(tools, calls, max_concurrency=4)
    assert [r["tool"] for r in out] == ["a", "b", "c", "a"]
    for i, r in enumerate(out):
        assert r["args"]["i"] == i


def test_tool_dispatch_unknown_tool_isolated():
    tools = [_echo_tool("a")]
    calls = [
        {"name": "a", "args": {}},
        {"name": "missing", "args": {}},
        {"name": "a", "args": {}},
    ]
    out = tool_dispatch_concurrent(tools, calls, max_concurrency=4)
    assert "tool" in out[0]
    assert "error" in out[1]
    assert "unknown tool" in out[1]["error"]
    assert "tool" in out[2]


def test_tool_dispatch_per_tool_failure_isolated():
    tools = [_echo_tool("ok"), _failing_tool()]
    calls = [
        {"name": "ok", "args": {}},
        {"name": "fail", "args": {}},
        {"name": "ok", "args": {}},
    ]
    out = tool_dispatch_concurrent(tools, calls, max_concurrency=4)
    assert "tool" in out[0]
    assert "error" in out[1]
    assert "tool" in out[2]


def test_tool_dispatch_fail_fast_raises():
    tools = [_echo_tool("ok")]
    calls = [{"name": "missing", "args": {}}]
    try:
        tool_dispatch_concurrent(tools, calls, max_concurrency=4, fail_fast=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected RuntimeError")


def test_tool_dispatch_empty_calls_returns_empty():
    tools = [_echo_tool("a")]
    assert tool_dispatch_concurrent(tools, [], max_concurrency=4) == []


def test_tool_dispatch_accepts_tuple_calls():
    tools = [_echo_tool("a")]
    out = tool_dispatch_concurrent(
        tools, [("a", {"x": 1}), ("a", {"x": 2})], max_concurrency=2
    )
    assert len(out) == 2
    assert out[0]["args"]["x"] == 1


def test_tool_dispatch_rejects_malformed_call():
    tools = [_echo_tool("a")]
    try:
        tool_dispatch_concurrent(tools, ["not a tuple or dict"], max_concurrency=2)
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError")


def test_tool_dispatch_fail_fast_succeeds_when_all_ok():
    tools = [_echo_tool("a"), _echo_tool("b")]
    calls = [
        {"name": "a", "args": {}},
        {"name": "b", "args": {}},
    ]
    out = tool_dispatch_concurrent(tools, calls, max_concurrency=4, fail_fast=True)
    assert len(out) == 2


if __name__ == "__main__":
    fns = [
        test_tool_dispatch_returns_one_result_per_call,
        test_tool_dispatch_aligned_to_input,
        test_tool_dispatch_unknown_tool_isolated,
        test_tool_dispatch_per_tool_failure_isolated,
        test_tool_dispatch_fail_fast_raises,
        test_tool_dispatch_empty_calls_returns_empty,
        test_tool_dispatch_accepts_tuple_calls,
        test_tool_dispatch_rejects_malformed_call,
        test_tool_dispatch_fail_fast_succeeds_when_all_ok,
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

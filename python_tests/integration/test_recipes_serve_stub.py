"""Live integration: `recipes.serve` returns the shell command string.

Per docstring, `serve` is a stub today — it doesn't spawn the binary,
just returns the command-line string a caller could run via subprocess.
We exercise the surface so future regressions surface as test failures
once the binary actually launches. The DeepSeek model is in the
fixture but isn't called — this test gates on the same env var so the
suite stays one cohesive block.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_recipes_serve_returns_command_string(deepseek_chat):
    from litgraph.graph import END, START, StateGraph
    from litgraph.recipes import serve

    g = StateGraph()
    g.add_node("noop", lambda s: s)
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    compiled = g.compile()

    cmd = serve(compiled, port=9999, host="127.0.0.1")
    assert isinstance(cmd, str)
    assert cmd.strip(), "serve returned empty string"
    # The command should reference the binary name and the port the
    # caller picked.
    assert "litgraph-serve" in cmd or "serve" in cmd, f"unexpected cmd: {cmd!r}"
    assert "9999" in cmd, f"port not surfaced: {cmd!r}"


def test_recipes_serve_accepts_uncompiled_state_graph(deepseek_chat):
    """`serve` should also accept a `StateGraph` (uncompiled) — common
    pattern when handing off to the binary, which can compile lazily."""
    from litgraph.graph import StateGraph
    from litgraph.recipes import serve

    g = StateGraph()
    g.add_node("noop", lambda s: s)
    cmd = serve(g, port=8080, host="0.0.0.0")
    assert "8080" in cmd


def test_recipes_serve_rejects_non_graph(deepseek_chat):
    """Anything that isn't a graph should raise a clear TypeError."""
    from litgraph.recipes import serve

    with pytest.raises(TypeError):
        serve(42)

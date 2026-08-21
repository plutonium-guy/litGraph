"""Live integration: `recipes.serve` dispatches on the shape of its input.

Since iter 384 `serve` actually spawns the axum server. It has two
branches:

- **ChatModel** — spawns `litgraph-serve` in-process and returns a
  `ServeHandle`.
- **StateGraph / CompiledGraph** — deliberately deferred, raises
  `NotImplementedError`. Graph-shaped serving needs per-thread state
  and checkpoint coordination the chat endpoints don't expose.

Anything else is a `TypeError`.

The model fixture is used by the ChatModel case; the graph cases gate on
the same env var so the suite stays one cohesive block.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def _linear_graph():
    from litgraph.graph import END, START, StateGraph

    g = StateGraph()
    g.add_node("noop", lambda s: s)
    g.add_edge(START, "noop")
    g.add_edge("noop", END)
    return g


def test_recipes_serve_defers_compiled_graph(deepseek_chat):
    """A `CompiledGraph` has `.invoke` and `.stream` and — being already
    compiled — no `.compile`. It must still be recognised as a graph and
    hit the documented deferral, not be misrouted into `spawn_chat`."""
    from litgraph.recipes import serve

    compiled = _linear_graph().compile()
    with pytest.raises(NotImplementedError, match="not implemented yet"):
        serve(compiled, port=9999, host="127.0.0.1")


def test_recipes_serve_defers_uncompiled_state_graph(deepseek_chat):
    from litgraph.recipes import serve

    with pytest.raises(NotImplementedError, match="not implemented yet"):
        serve(_linear_graph(), port=8080, host="0.0.0.0")


def test_recipes_serve_rejects_non_graph(deepseek_chat):
    """Anything that isn't a graph or a chat model raises a clear TypeError."""
    from litgraph.recipes import serve

    with pytest.raises(TypeError):
        serve(42)


def test_recipes_serve_spawns_chat_model(deepseek_chat):
    """The branch that IS implemented: a ChatModel gets a live server."""
    import json
    import urllib.request

    from litgraph.recipes import serve

    handle = serve(deepseek_chat, port=0, host="127.0.0.1")
    try:
        url = handle.url()
        assert url.startswith("http://127.0.0.1:")
        health = json.loads(
            urllib.request.urlopen(f"{url}/health", timeout=10).read().decode()
        )
        assert health["status"] == "ok"
        info = json.loads(
            urllib.request.urlopen(f"{url}/info", timeout=10).read().decode()
        )
        assert "/invoke" in info["endpoints"]
    finally:
        handle.shutdown()

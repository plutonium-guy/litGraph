"""Graph visualisation — Mermaid + ASCII renderers on StateGraph + CompiledGraph."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.graph import END, START, StateGraph  # noqa: E402


def _linear():
    g = StateGraph()
    g.add_node("a", lambda s: {"n": s.get("n", 0) + 1})
    g.add_node("b", lambda s: {"n": s.get("n", 0) + 1})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    return g


def test_mermaid_starts_with_graph_td():
    g = _linear()
    out = g.to_mermaid()
    assert out.startswith("graph TD")


def test_mermaid_includes_node_labels():
    g = _linear()
    out = g.to_mermaid()
    assert "[a]" in out
    assert "[b]" in out


def test_mermaid_includes_static_edges():
    g = _linear()
    out = g.to_mermaid()
    assert "a --> b" in out


def test_mermaid_renders_start_and_end():
    g = _linear()
    out = g.to_mermaid()
    assert "Start" in out
    assert "End" in out


def test_ascii_lists_nodes_and_edges():
    g = _linear()
    out = g.to_ascii()
    assert "litgraph StateGraph" in out
    assert "a -> b" in out
    assert "Start -> a" in out
    assert "b -> End" in out


def test_compiled_graph_can_also_render():
    g = _linear()
    pre = g.to_mermaid()
    compiled = g.compile()
    post = compiled.to_mermaid()
    # Compiling is just take()+wrap — topology unchanged.
    assert post == pre


def test_render_after_compile_on_state_graph_errors():
    g = _linear()
    g.compile()  # consumes inner
    with pytest.raises(RuntimeError):
        g.to_mermaid()


def test_conditional_edges_render_diamond():
    g = StateGraph()
    g.add_node("router", lambda s: s)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: [END])
    out = g.to_mermaid()
    assert "{?}" in out


def test_render_is_deterministic():
    a = _linear().to_mermaid()
    b = _linear().to_mermaid()
    assert a == b

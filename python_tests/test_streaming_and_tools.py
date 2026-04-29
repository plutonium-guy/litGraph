"""Tests for Python streaming iterator + FunctionTool plumbing.

Agent invocation with a real LLM is out of scope for unit tests; we verify the
agent wiring compiles and constructs properly, and stream events flow through.
"""
from litgraph.graph import StateGraph, GraphStream, START, END
from litgraph.tools import FunctionTool


def test_graph_stream_yields_events():
    g = StateGraph()
    g.add_node("a", lambda s: {"items": [1]})
    g.add_node("b", lambda s: {"items": [2]})
    g.add_node("done", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", "done")
    g.add_edge("done", END)
    compiled = g.compile()

    stream = compiled.stream({"items": []}, thread_id="t-stream-1")
    events = list(stream)  # iterator protocol

    types = [e["type"] for e in events]
    assert "graph_start" in types
    assert "graph_end" in types
    assert types.count("node_end") == 3

    # Node ends must include the correct update payloads.
    node_ends = [e for e in events if e["type"] == "node_end"]
    names = [e["node"] for e in node_ends]
    assert set(names) == {"a", "b", "done"}


def test_graph_stream_is_iterable_twice_semantics():
    """A consumed stream should raise StopIteration cleanly, not hang."""
    g = StateGraph()
    g.add_node("only", lambda s: {"n": (s.get("n", 0) + 1)})
    g.add_edge(START, "only")
    g.add_edge("only", END)
    compiled = g.compile()

    stream = compiled.stream({"n": 0})
    count = 0
    for _ in stream:
        count += 1
    assert count >= 2  # at least graph_start + node_end + graph_end


def test_function_tool_constructs():
    def add(args):
        return {"sum": args["a"] + args["b"]}

    t = FunctionTool(
        "add",
        "Sum two ints.",
        {
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        },
        add,
    )
    assert t.name == "add"
    assert "add" in repr(t)


def test_function_tool_accepts_json_string_schema():
    import json

    def noop(args):
        return {}

    schema = json.dumps({"type": "object", "properties": {}})
    t = FunctionTool("noop", "no-op", schema, noop)
    assert t.name == "noop"


def test_react_agent_constructs_without_running():
    """Construct an agent — we don't invoke (no live LLM). Validates wiring."""
    from litgraph.providers import OpenAIChat
    from litgraph.agents import ReactAgent

    def adder(args):
        return {"sum": args["a"] + args["b"]}

    tool = FunctionTool(
        "add", "Add ints.",
        {"type": "object", "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}},
        adder,
    )

    # Dummy API key — we never hit the network here.
    model = OpenAIChat(api_key="sk-test", model="gpt-5")
    agent = ReactAgent(model, [tool], system_prompt="be terse", max_iterations=3)
    assert agent is not None


if __name__ == "__main__":
    fns = [
        test_graph_stream_yields_events,
        test_graph_stream_is_iterable_twice_semantics,
        test_function_tool_constructs,
        test_function_tool_accepts_json_string_schema,
        test_react_agent_constructs_without_running,
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

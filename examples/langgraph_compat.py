"""Reference compat shim for users porting from LangGraph.

Drop this file into your project (or copy the imports below) and migrate
LangGraph code with minimal renames. Not bundled with the litgraph wheel — it
is intentionally tiny (single file) so you can read what it does and adapt.

Mapping:
    langgraph.graph.StateGraph                     → litgraph.graph.StateGraph
    langgraph.graph.END                            → litgraph.graph.END
    langgraph.checkpoint.memory.MemorySaver        → MemorySaver (alias below; not needed — default is in-memory)
    langgraph.prebuilt.create_react_agent          → create_react_agent (factory below)

Things LangGraph users will run into:

- LangGraph's `StateGraph(MessagesState)` typed-state syntax is not yet
  expressible from Python (Rust nodes use `serde_json::Value`). For typed
  state, write your nodes as Rust crates importing `litgraph-graph`.
- `Send(node, state)` fan-out emit-from-node — use `NodeOutput::send()` from
  Rust; from Python, the same effect happens by adding multiple static edges
  to the same target.
- `Command(goto=..., update=...)` — return a dict from a node and add the
  conditional-edge router. Or use the Rust `NodeOutput::goto(..)` API.
"""

from litgraph.graph import StateGraph, START, END  # re-export for parity
from litgraph.agents import ReactAgent as _ReactAgent


class MemorySaver:
    """No-op alias. litGraph's CompiledGraph uses an in-memory checkpointer
    automatically; pass `thread_id=...` to `compiled.invoke()` to enable HITL
    interrupt + resume. No need to construct a saver yourself."""
    def __init__(self):
        pass


def create_react_agent(model, tools, *, system_prompt=None, max_iterations=10, **_kwargs):
    """LangGraph factory parity. Returns a litgraph.agents.ReactAgent.

    Unsupported LangGraph kwargs are silently ignored (most have no analog —
    e.g. `interrupt_before` for tools, which we do at the StateGraph level).
    """
    return _ReactAgent(
        model,
        tools,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
    )


__all__ = ["StateGraph", "START", "END", "MemorySaver", "create_react_agent"]

"""Schema-aware StateGraph overlay.

This module wraps the native `litgraph.litgraph.StateGraph` /
`CompiledGraph` with one added kwarg — `state_schema` — that auto-
coerces inputs and outputs against a Pydantic v1/v2 model, a
`@dataclass`, or a `TypedDict`. When `state_schema=None` (the default)
behaviour is identical to the bare native API.

Before this overlay landed, callers had to opt into coercion explicitly
via `coerce_one` / `coerce_stream` on every call site. The opt-in path
still works for callers who don't want to declare the schema at
construction time, but most users now write:

    from pydantic import BaseModel
    from litgraph import StateGraph

    class State(BaseModel):
        step: str
        value: int

    g = StateGraph(state_schema=State)
    g.add_node("inc", lambda s: {"value": s["value"] + 1})
    g.set_entry("inc")
    g.add_edge("inc", END)
    compiled = g.compile()
    final: State = compiled.invoke(State(step="init", value=0))

…and IDE autocomplete on `final.value` works without a separate
`coerce_one` call.

Implementation notes:

* `invoke` / `resume` dump non-dict inputs (Pydantic instances,
  dataclasses, TypedDicts) to plain `dict` before sending them across
  the PyO3 boundary — the native side only understands `dict[str, Any]`
  state.
* The native call returns a `dict`; we run that through
  `coerce_one(result, state_schema)` so the caller gets a typed
  instance. `coerce_one` already handles Pydantic / dataclass /
  TypedDict / generic constructors.
* `stream` returns an iterator of `dict` graph-events whose shape
  varies by event type. We don't auto-coerce those events here — they
  carry a per-node `update` payload, not the full state, and the
  schema rarely matches the partial. Callers who want typed stream
  state can still pipe the events through `coerce_stream` or accumulate
  via the reducer themselves.
* Time-travel / visualization methods pass through unmodified — those
  surface checkpoint metadata, not state instances.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Optional, Type

from .coerce import coerce_one

# Native module — imported lazily so the overlay still imports cleanly
# in dev environments where the wheel hasn't been built. The native
# pyclasses live under the `graph` sub-namespace of the native module
# (`litgraph.litgraph.graph`), not the top of the wheel.
try:
    from . import litgraph as _native  # type: ignore[attr-defined]
    _native_graph = _native.graph
    _NativeStateGraph = _native_graph.StateGraph
    _NativeCompiledGraph = _native_graph.CompiledGraph
except (ImportError, AttributeError):  # pragma: no cover — only in dev w/o build
    _native = None
    _NativeStateGraph = None
    _NativeCompiledGraph = None


def _dump_state(state: Any) -> dict:
    """Convert a typed state instance to the plain dict the native layer expects.

    Accepts:
    - `dict` (passed through)
    - Pydantic v2 BaseModel (uses `.model_dump()`)
    - Pydantic v1 BaseModel (uses `.dict()`)
    - `@dataclass` instance (uses `dataclasses.asdict`)
    - Anything else with `.dict()` returning a dict

    Raises `TypeError` for unknown shapes — surfaces the mismatch
    early instead of letting the PyO3 boundary fail with a less
    helpful "expected dict" error.
    """
    if isinstance(state, dict):
        return state
    if hasattr(state, "model_dump"):
        return state.model_dump()
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return dataclasses.asdict(state)
    if hasattr(state, "dict") and callable(state.dict):  # Pydantic v1 fallback
        return state.dict()
    raise TypeError(
        f"StateGraph state must be a dict or a Pydantic / dataclass / "
        f"TypedDict-compatible instance; got {type(state).__name__}"
    )


class StateGraph:
    """Schema-aware drop-in for the native ``litgraph.litgraph.StateGraph``.

    Parameters
    ----------
    state_schema:
        Optional Pydantic BaseModel, dataclass, or TypedDict. When
        provided, `compile().invoke(...)` returns an instance of this
        class instead of a raw dict, and `invoke` accepts an instance
        as input (auto-dumped to dict). `None` (default) preserves
        the legacy dict-in / dict-out behaviour.
    max_parallel:
        Forwarded to the native StateGraph. Default 16.
    recursion_limit:
        Forwarded to the native StateGraph. Default 25.
    """

    def __init__(
        self,
        state_schema: Optional[Type[Any]] = None,
        max_parallel: int = 16,
        recursion_limit: int = 25,
    ) -> None:
        if _NativeStateGraph is None:
            raise RuntimeError(
                "litgraph native module not built; run `maturin develop` "
                "to enable StateGraph"
            )
        self._inner = _NativeStateGraph(
            max_parallel=max_parallel,
            recursion_limit=recursion_limit,
        )
        self._state_schema = state_schema

    # ----- delegation to native ------------------------------------------------

    def add_node(self, name, func):
        return self._inner.add_node(name, func)

    def add_edge(self, from_, to):
        return self._inner.add_edge(from_, to)

    def add_conditional_edges(self, from_, router):
        return self._inner.add_conditional_edges(from_, router)

    def add_subgraph(self, name, sub):
        # `sub` may be the schema-aware CompiledGraph wrapper or the raw
        # native one; unwrap when needed.
        inner_sub = sub._inner if isinstance(sub, CompiledGraph) else sub
        return self._inner.add_subgraph(name, inner_sub)

    def interrupt_before(self, node):
        return self._inner.interrupt_before(node)

    def interrupt_after(self, node):
        return self._inner.interrupt_after(node)

    def set_entry(self, node):
        return self._inner.set_entry(node)

    def to_mermaid(self):
        return self._inner.to_mermaid()

    def to_ascii(self):
        return self._inner.to_ascii()

    def compile(self) -> "CompiledGraph":
        return CompiledGraph(self._inner.compile(), self._state_schema)


class CompiledGraph:
    """Schema-aware wrapper around the native ``CompiledGraph``.

    Constructed by ``StateGraph.compile()``; do not instantiate directly.
    Carries a reference to the original state schema (if any) so that
    `invoke` / `resume` results are auto-coerced.
    """

    def __init__(self, inner, state_schema: Optional[Type[Any]]) -> None:
        self._inner = inner
        self._state_schema = state_schema

    def invoke(self, state: Any, thread_id: Optional[str] = None):
        initial = _dump_state(state) if self._state_schema else state
        result = self._inner.invoke(initial, thread_id)
        if self._state_schema is not None:
            return coerce_one(result, self._state_schema)
        return result

    def resume(self, thread_id: str, update: Any = None):
        if update is None:
            upd = None
        elif self._state_schema is not None and not isinstance(update, dict):
            upd = _dump_state(update)
        else:
            upd = update
        result = self._inner.resume(thread_id, upd)
        if self._state_schema is not None:
            return coerce_one(result, self._state_schema)
        return result

    def stream(self, state: Any, thread_id: Optional[str] = None):
        # GraphStream events carry per-node `update` deltas, not full
        # state instances. We pass them through unchanged — schema
        # coercion at this layer would mis-fire on partial updates.
        # Callers wanting typed stream state still use `coerce_stream`.
        initial = _dump_state(state) if self._state_schema else state
        return self._inner.stream(initial, thread_id)

    # ----- time-travel + visualization passthrough ----------------------------

    def state_history(self, thread_id):
        return self._inner.state_history(thread_id)

    def rewind_to(self, thread_id, step):
        return self._inner.rewind_to(thread_id, step)

    def fork_at(self, thread_id, step, new_thread_id):
        return self._inner.fork_at(thread_id, step, new_thread_id)

    def clear_thread(self, thread_id):
        return self._inner.clear_thread(thread_id)

    def to_mermaid(self):
        return self._inner.to_mermaid()

    def to_ascii(self):
        return self._inner.to_ascii()

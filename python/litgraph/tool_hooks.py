"""before_tool / after_tool hooks + per-turn tool-call budget.

Wraps any tool list (or single tool) with cross-cutting concerns
without touching the agent loop. Pure Python; works with the native
`ReactAgent`, `SupervisorAgent`, `PlanAndExecuteAgent`, or any
custom dispatcher that calls `tool.invoke(args)`.

Example:

    from litgraph.tool_hooks import (
        BeforeToolHook, AfterToolHook, ToolBudget, wrap_tools,
    )

    def audit(name, args):
        print(f"calling {name}({args})")

    def redact(name, args, result):
        if isinstance(result, dict):
            result.pop("api_key", None)
        return result

    tools = wrap_tools(
        [my_search_tool, my_sql_tool],
        before=BeforeToolHook(audit),
        after=AfterToolHook(redact),
        budget=ToolBudget(max_calls_per_turn=5),
    )

    agent = ReactAgent(model, tools, system_prompt="...")
"""
from __future__ import annotations

from typing import Any, Callable, Iterable, Mapping


__all__ = [
    "BeforeToolHook",
    "AfterToolHook",
    "ToolBudget",
    "ToolBudgetExceeded",
    "wrap_tools",
    "wrap_tool",
    "HookedTool",
]


# ---- Hook primitives ----


class BeforeToolHook:
    """Callable invoked before every tool dispatch.

    The callback receives `(tool_name, args)` and may return None
    (proceed unchanged), a new args dict (mutate inputs), or raise to
    abort. Useful for: audit logging, schema-tightening, redaction.
    """

    def __init__(self, callback: Callable[[str, Mapping[str, Any]], Any]) -> None:
        self.callback = callback

    def __call__(self, tool_name: str, args: Mapping[str, Any]) -> Mapping[str, Any]:
        out = self.callback(tool_name, args)
        return out if isinstance(out, Mapping) else args


class AfterToolHook:
    """Callable invoked after every tool dispatch.

    The callback receives `(tool_name, args, result)` and may return
    None (proceed unchanged) or a replacement result. Useful for:
    PII scrubbing, response shape enforcement, cost tracking.
    """

    def __init__(self, callback: Callable[[str, Mapping[str, Any], Any], Any]) -> None:
        self.callback = callback

    def __call__(self, tool_name: str, args: Mapping[str, Any], result: Any) -> Any:
        out = self.callback(tool_name, args, result)
        return out if out is not None else result


# ---- Budget ----


class ToolBudgetExceeded(RuntimeError):
    """Raised when a turn would exceed the configured tool-call cap."""


class ToolBudget:
    """Cap on total tool invocations within one agent turn.

    Mirrors `CostCappedChatModel` ($-cap) but counts calls instead.
    Reset between turns by calling `reset()`; the agent doesn't reset
    automatically — wrap inside a turn-scoped context if you need that.
    """

    def __init__(self, max_calls_per_turn: int) -> None:
        if max_calls_per_turn <= 0:
            raise ValueError("max_calls_per_turn must be positive")
        self.max = max_calls_per_turn
        self.calls = 0

    def reset(self) -> None:
        self.calls = 0

    def consume(self, tool_name: str) -> None:
        if self.calls >= self.max:
            raise ToolBudgetExceeded(
                f"tool budget exceeded: {self.calls}/{self.max} "
                f"(tried to call {tool_name!r})"
            )
        self.calls += 1


# ---- HookedTool ----


class HookedTool:
    """Wrap a single tool. Forwards `invoke` / `run` / `__call__` to
    the underlying tool, with optional before/after hooks + budget
    consumed once per call.

    Preserves `name`, `description`, and `schema` so the agent loop
    sees the same JSON Schema.
    """

    def __init__(
        self,
        tool: Any,
        *,
        before: BeforeToolHook | None = None,
        after: AfterToolHook | None = None,
        budget: ToolBudget | None = None,
    ) -> None:
        self._tool = tool
        self._before = before
        self._after = after
        self._budget = budget
        # Mirror the underlying tool's metadata so existing agents
        # (which read these attrs) treat the wrapper as a drop-in.
        self.name = getattr(tool, "name", type(tool).__name__)
        self.description = getattr(tool, "description", "")
        self.schema = getattr(tool, "schema", None)

    def invoke(self, args: Mapping[str, Any]) -> Any:
        if self._budget is not None:
            self._budget.consume(self.name)
        munged = self._before(self.name, args) if self._before else args
        result = self._tool.invoke(dict(munged))
        if self._after is not None:
            result = self._after(self.name, dict(munged), result)
        return result

    def run(self, args: Mapping[str, Any]) -> Any:
        return self.invoke(args)

    def __call__(self, args: Mapping[str, Any]) -> Any:
        return self.invoke(args)

    def to_function_tool(
        self,
        python_callable: Any,
        *,
        schema: Mapping[str, Any] | None = None,
        description: str | None = None,
    ) -> Any:
        """Return a native Rust `FunctionTool` whose callable fires the
        Python before/after/budget hooks around `python_callable`. Use
        this to plug a `HookedTool` into the native `ReactAgent` loop,
        which only accepts registered Rust tool types via
        `extract_tools`.

        Why explicit `python_callable` + `schema` args? The wrapped
        inner `tool` may be a native `FunctionTool`, which does NOT
        expose `.invoke()`, `.schema`, or `.description` on the Python
        surface — those live in Rust. To bridge into a freshly-built
        `FunctionTool` for the agent loop, the caller passes the
        underlying function and JSON schema directly.

        The new tool's `name` defaults to this `HookedTool`'s name.
        Hooks fire on every dispatch identically to a free-standing
        `wrapped.invoke(...)` call.

        ```python
        from litgraph.tools import FunctionTool
        from litgraph.tool_hooks import wrap_tool, BeforeToolHook

        def add(a, b):
            return {"sum": a + b}

        schema = {"type":"object","properties":{"a":{"type":"integer"},
                  "b":{"type":"integer"}},"required":["a","b"]}
        inner = FunctionTool("add", "add two ints", schema, add)
        hooked = wrap_tool(inner, before=BeforeToolHook(...))
        agent_tool = hooked.to_function_tool(add, schema=schema,
                                              description="add two ints")
        ReactAgent(model, [agent_tool], ...)
        ```
        """
        from litgraph.tools import FunctionTool  # late import — avoid cycle

        before = self._before
        after = self._after
        budget = self._budget
        name = self.name
        effective_schema = schema or self.schema or {
            "type": "object",
            "properties": {},
        }
        effective_desc = description or self.description or ""

        def _hooked(**kwargs: Any) -> Any:
            if budget is not None:
                budget.consume(name)
            args = kwargs
            if before is not None:
                got = before(name, args)
                if got is not None:
                    args = got
            result = python_callable(**args)
            if after is not None:
                result = after(name, args, result)
            return result

        return FunctionTool(name, effective_desc, effective_schema, _hooked)


def wrap_tool(
    tool: Any,
    *,
    before: BeforeToolHook | None = None,
    after: AfterToolHook | None = None,
    budget: ToolBudget | None = None,
) -> HookedTool:
    """Wrap a single tool. See `HookedTool` for the contract."""
    return HookedTool(tool, before=before, after=after, budget=budget)


def wrap_tools(
    tools: Iterable[Any],
    *,
    before: BeforeToolHook | None = None,
    after: AfterToolHook | None = None,
    budget: ToolBudget | None = None,
) -> list[HookedTool]:
    """Wrap every tool in `tools` with the same hooks/budget. The
    budget is *shared* across the returned wrappers — each call to
    any tool decrements the same counter. Useful for "no agent turn
    spends more than N tool calls total"."""
    return [wrap_tool(t, before=before, after=after, budget=budget) for t in tools]

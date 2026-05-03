"""before_tool / after_tool hooks + tool-call budget cap."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping


class BeforeToolHook:
    callback: Callable[[str, Mapping[str, Any]], Any]
    def __init__(self, callback: Callable[[str, Mapping[str, Any]], Any]) -> None: ...
    def __call__(self, tool_name: str, args: Mapping[str, Any]) -> Mapping[str, Any]: ...


class AfterToolHook:
    callback: Callable[[str, Mapping[str, Any], Any], Any]
    def __init__(self, callback: Callable[[str, Mapping[str, Any], Any], Any]) -> None: ...
    def __call__(self, tool_name: str, args: Mapping[str, Any], result: Any) -> Any: ...


class ToolBudgetExceeded(RuntimeError): ...


class ToolBudget:
    max: int
    calls: int
    def __init__(self, max_calls_per_turn: int) -> None: ...
    def reset(self) -> None: ...
    def consume(self, tool_name: str) -> None: ...


class HookedTool:
    name: str
    description: str
    schema: Any | None
    def __init__(
        self,
        tool: Any,
        *,
        before: BeforeToolHook | None = ...,
        after: AfterToolHook | None = ...,
        budget: ToolBudget | None = ...,
    ) -> None: ...
    def invoke(self, args: Mapping[str, Any]) -> Any: ...
    def run(self, args: Mapping[str, Any]) -> Any: ...
    def __call__(self, args: Mapping[str, Any]) -> Any: ...


def wrap_tool(
    tool: Any,
    *,
    before: BeforeToolHook | None = ...,
    after: AfterToolHook | None = ...,
    budget: ToolBudget | None = ...,
) -> HookedTool: ...


def wrap_tools(
    tools: Iterable[Any],
    *,
    before: BeforeToolHook | None = ...,
    after: AfterToolHook | None = ...,
    budget: ToolBudget | None = ...,
) -> list[HookedTool]: ...

"""LangChain-shape compat shims."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping


class AgentExecutor:
    verbose: bool
    def __init__(
        self,
        agent: Any = ...,
        tools: Iterable[Any] | None = ...,
        verbose: bool = False,
        max_iterations: int = 10,
        **kwargs: Any,
    ) -> None: ...
    @classmethod
    def from_agent_and_tools(
        cls,
        agent: Any,
        tools: Iterable[Any],
        **kwargs: Any,
    ) -> "AgentExecutor": ...
    def invoke(self, input: Mapping[str, Any] | str) -> Any: ...
    def __call__(self, input: Any) -> Any: ...


class RunnableLambda:
    name: str
    def __init__(self, func: Callable[[Any], Any]) -> None: ...
    def invoke(self, input: Any) -> Any: ...
    def __call__(self, input: Any) -> Any: ...
    def __or__(self, other: Any) -> Any: ...


class RunnableParallel:
    def __init__(
        self, branches: Mapping[str, Any] | None = ..., **kw_branches: Any
    ) -> None: ...
    def invoke(self, input: Any) -> dict[str, Any]: ...
    def __call__(self, input: Any) -> dict[str, Any]: ...


class RunnableBranch:
    def __init__(self, *branches: Any, default: Any = ...) -> None: ...
    def invoke(self, input: Any) -> Any: ...


class RunnablePassthrough:
    def invoke(self, input: Any) -> Any: ...
    def __call__(self, input: Any) -> Any: ...

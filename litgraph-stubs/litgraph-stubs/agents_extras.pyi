"""Multi-agent helpers: Swarm-style handoffs + BigTool selection."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping


class Handoff:
    target: str
    payload: Mapping[str, Any] | None
    def __init__(self, target: str, payload: Mapping[str, Any] | None = ...) -> None: ...


class SwarmAgent:
    agents: dict[str, Any]
    entry: str
    max_handoffs: int
    on_handoff: Callable[[str, str, Mapping[str, Any] | None], None] | None
    def __init__(
        self,
        agents: Mapping[str, Any],
        entry: str,
        max_handoffs: int = 5,
        on_handoff: Callable[[str, str, Mapping[str, Any] | None], None] | None = ...,
    ) -> None: ...
    def invoke(self, user_input: str | Iterable[Mapping[str, Any]]) -> dict[str, Any]: ...


class BigToolAgent:
    k: int
    def __init__(
        self,
        agent_factory: Callable[[list[Any]], Any],
        tools: Iterable[Any],
        embeddings: Any,
        k: int = 8,
    ) -> None: ...
    def invoke(self, user_input: str) -> Any: ...

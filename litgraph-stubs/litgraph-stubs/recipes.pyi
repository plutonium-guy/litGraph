"""One-call factories for the most common litGraph patterns.

Designed for AI coding assistants — see AGENT_DX.md §13."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Mapping


def eval(
    target: Callable[[str], str],
    cases: Iterable[Mapping[str, Any]],
    scorers: Iterable[Mapping[str, Any]] | None = ...,
    max_parallel: int = 4,
) -> Any: ...


def serve(graph: Any, port: int = 8080, host: str = "0.0.0.0") -> str: ...

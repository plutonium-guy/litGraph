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


def rag(
    documents: Iterable[Mapping[str, Any]] | None = ...,
    *,
    model: Any = ...,
    embeddings: Any = ...,
    store: Any = ...,
    retriever_k: int = 5,
    system_prompt: str | None = ...,
) -> Any: ...


def multi_agent(
    workers: Mapping[str, Any],
    *,
    supervisor_model: Any,
    system_prompt: str | None = ...,
) -> Any: ...


def summarize(
    text: str,
    *,
    model: Any,
    chunk_size: int = 4000,
    chunk_overlap: int = 200,
    map_prompt: str | None = ...,
    reduce_prompt: str | None = ...,
) -> dict[str, Any]: ...

"""FAISS / niche vector-store adapters."""
from __future__ import annotations
from typing import Any, Iterable, Mapping


class FaissVectorStore:
    dim: int
    def __init__(
        self,
        dim: int,
        index_factory: str = "Flat",
        normalize: bool = True,
    ) -> None: ...
    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = ...,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: Iterable[str]) -> None: ...
    def __len__(self) -> int: ...
    def is_empty(self) -> bool: ...

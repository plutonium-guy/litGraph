"""Optional local-Python embedding adapters (lazy imports)."""
from __future__ import annotations
from typing import Any, Iterable


class SentenceTransformersEmbeddings:
    dim: int
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ) -> None: ...
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class HuggingFaceInferenceEmbeddings:
    dim: int | None
    def __init__(
        self,
        model: str,
        token: str | None = ...,
        endpoint_url: str | None = ...,
    ) -> None: ...
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class NimEmbeddings:
    dim: int | None
    def __init__(
        self,
        model: str,
        api_key: str | None = ...,
        base_url: str = "https://integrate.api.nvidia.com/v1",
    ) -> None: ...
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


def from_env(name: str, /, **overrides: Any) -> Any: ...

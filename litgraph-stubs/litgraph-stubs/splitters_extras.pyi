"""NLTK / SpaCy sentence splitters (lazy imports)."""
from __future__ import annotations
from typing import Any, Iterable


class NltkSentenceSplitter:
    language: str
    chunk_size: int | None
    chunk_overlap: int
    def __init__(
        self,
        language: str = "english",
        chunk_size: int | None = ...,
        chunk_overlap: int = 0,
    ) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]: ...


class SpacySentenceSplitter:
    model_name: str
    chunk_size: int | None
    chunk_overlap: int
    def __init__(
        self,
        model_name: str = "en_core_web_sm",
        chunk_size: int | None = ...,
        chunk_overlap: int = 0,
    ) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, docs: Iterable[dict[str, Any]]) -> list[dict[str, Any]]: ...

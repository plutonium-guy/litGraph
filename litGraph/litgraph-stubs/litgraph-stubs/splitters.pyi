from typing import Any

from .embeddings import (
    BedrockEmbeddings,
    CohereEmbeddings,
    FunctionEmbeddings,
    GeminiEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
)

class RecursiveCharacterSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]: ...
    @staticmethod
    def for_language(
        language: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> "RecursiveCharacterSplitter": ...

class MarkdownHeaderSplitter:
    def __init__(self, max_depth: int = 3, strip_headers: bool = False) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

class SemanticChunker:
    def __init__(
        self,
        embeddings: (
            FunctionEmbeddings
            | OpenAIEmbeddings
            | CohereEmbeddings
            | VoyageEmbeddings
            | GeminiEmbeddings
            | BedrockEmbeddings
        ),
        buffer_size: int = 1,
        breakpoint_percentile: float = 95.0,
        min_sentences_per_chunk: int = 1,
    ) -> None: ...
    def split_text(self, text: str) -> list[str]: ...
    def split_documents(self, docs: list[dict[str, Any]]) -> list[dict[str, Any]]: ...

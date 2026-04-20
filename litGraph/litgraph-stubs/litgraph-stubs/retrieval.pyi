from typing import Any

from .embeddings import (
    BedrockEmbeddings,
    CohereEmbeddings,
    FunctionEmbeddings,
    GeminiEmbeddings,
    JinaEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
)

class Bm25Index:
    def __init__(self) -> None: ...
    def add(self, docs: list[dict[str, Any]]) -> None: ...
    def search(self, query: str, k: int) -> list[dict[str, Any]]: ...
    def __len__(self) -> int: ...

class MemoryVectorStore:
    def __init__(self) -> None: ...
    def add(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def __len__(self) -> int: ...

class HnswVectorStore:
    def __init__(self, ef_search: int = 64, ef_construction: int = 200) -> None: ...
    def add(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: list[str]) -> None: ...
    def __len__(self) -> int: ...

class QdrantVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        api_key: str | None = None,
        vector_name: str | None = None,
        timeout_s: int = 30,
    ) -> None: ...
    def ensure_collection(self, dim: int, distance: str = "Cosine") -> None: ...
    def add(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: list[str]) -> None: ...

class PgVectorStore:
    @staticmethod
    def connect(dsn: str, table: str, dim: int) -> "PgVectorStore": ...
    def add(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: list[str]) -> None: ...

class ChromaVectorStore:
    def __init__(
        self,
        url: str,
        collection: str,
        tenant: str = "default_tenant",
        database: str = "default_database",
        timeout_s: int = 30,
    ) -> None: ...
    def add(
        self,
        docs: list[dict[str, Any]],
        embeddings: list[list[float]],
    ) -> list[str]: ...
    def similarity_search(
        self,
        query_embedding: list[float],
        k: int = 4,
        filter: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]: ...
    def delete(self, ids: list[str]) -> None: ...

class VectorRetriever:
    def __init__(
        self,
        embeddings: (
            FunctionEmbeddings
            | OpenAIEmbeddings
            | CohereEmbeddings
            | VoyageEmbeddings
            | GeminiEmbeddings
            | BedrockEmbeddings
            | JinaEmbeddings
        ),
        store: MemoryVectorStore | HnswVectorStore | QdrantVectorStore | PgVectorStore | ChromaVectorStore,
    ) -> None: ...
    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]: ...

class CohereReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-english-v3.0",
        base_url: str | None = None,
        timeout_s: int = 60,
    ) -> None: ...
    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]: ...

class VoyageReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "rerank-2",
        base_url: str | None = None,
        timeout_s: int = 60,
        truncation: bool = True,
    ) -> None: ...
    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]: ...

class JinaReranker:
    def __init__(
        self,
        api_key: str,
        model: str = "jina-reranker-v2-base-multilingual",
        base_url: str | None = None,
        timeout_s: int = 60,
    ) -> None: ...
    def rerank(
        self,
        query: str,
        docs: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]: ...

class RerankingRetriever:
    def __init__(
        self,
        base: VectorRetriever,
        reranker: CohereReranker | VoyageReranker | JinaReranker,
        over_fetch_k: int | None = None,
    ) -> None: ...
    def retrieve(self, query: str, k: int) -> list[dict[str, Any]]: ...

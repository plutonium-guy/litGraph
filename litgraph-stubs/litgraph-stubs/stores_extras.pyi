"""FAISS / Milvus / Redis-search / Neo4j / Mongo Atlas adapters."""
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


class MilvusVectorStore:
    collection_name: str
    dim: int
    def __init__(
        self,
        collection_name: str,
        dim: int,
        uri: str = "http://localhost:19530",
        token: str | None = ...,
        consistency: str = "Bounded",
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


class RedisSearchVectorStore:
    index_name: str
    dim: int
    prefix: str
    def __init__(
        self,
        index_name: str,
        dim: int,
        url: str = "redis://localhost:6379/0",
        prefix: str = "litgraph:doc:",
        metadata_fields: Iterable[tuple[str, str]] = ...,
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


class Neo4jVectorStore:
    index_name: str
    dim: int
    database: str
    node_label: str
    def __init__(
        self,
        index_name: str,
        dim: int,
        uri: str = "bolt://localhost:7687",
        username: str | None = ...,
        password: str | None = ...,
        database: str = "neo4j",
        node_label: str = "Document",
        similarity: str = "cosine",
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
    def close(self) -> None: ...


class PineconeVectorStore:
    index_name: str
    dim: int
    namespace: str
    def __init__(
        self,
        index_name: str,
        dim: int,
        api_key: str | None = ...,
        namespace: str = "",
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


class LanceDBVectorStore:
    table_name: str
    dim: int
    def __init__(
        self,
        table_name: str,
        dim: int,
        uri: str = "./lance.db",
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


class CassandraVectorStore:
    dim: int
    table_name: str
    def __init__(
        self,
        table_name: str,
        dim: int,
        keyspace: str = "litgraph",
        contact_points: tuple[str, ...] = ...,
        port: int = 9042,
        username: str | None = ...,
        password: str | None = ...,
        astra_bundle: str | None = ...,
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
    def close(self) -> None: ...


class MongoAtlasVectorStore:
    collection_name: str
    dim: int
    index_name: str
    embedding_field: str
    text_field: str
    def __init__(
        self,
        collection_name: str,
        dim: int,
        uri: str | None = ...,
        database: str = "litgraph",
        index_name: str = "vector_index",
        embedding_field: str = "embedding",
        text_field: str = "page_content",
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

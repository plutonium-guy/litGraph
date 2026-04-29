from .embeddings import (
    BedrockEmbeddings,
    CohereEmbeddings,
    FunctionEmbeddings,
    GeminiEmbeddings,
    JinaEmbeddings,
    OpenAIEmbeddings,
    VoyageEmbeddings,
)

AnyEmbeddings = (
    FunctionEmbeddings
    | OpenAIEmbeddings
    | CohereEmbeddings
    | VoyageEmbeddings
    | GeminiEmbeddings
    | BedrockEmbeddings
    | JinaEmbeddings
)

class MemoryCache:
    def __init__(self, max_capacity: int = 10_000) -> None: ...
    def clear(self) -> None: ...

class SqliteCache:
    def __init__(self, path: str) -> None: ...
    @staticmethod
    def in_memory() -> "SqliteCache": ...
    def clear(self) -> None: ...

class SemanticCache:
    def __init__(
        self,
        embeddings: FunctionEmbeddings,
        threshold: float = 0.95,
        max_entries: int = 10_000,
    ) -> None: ...
    def __len__(self) -> int: ...
    def clear(self) -> None: ...

class MemoryEmbeddingCache:
    def __init__(self, max_capacity: int = 100_000, ttl_s: int | None = None) -> None: ...
    def clear(self) -> None: ...

class SqliteEmbeddingCache:
    def __init__(self, path: str) -> None: ...
    @staticmethod
    def in_memory() -> "SqliteEmbeddingCache": ...
    def clear(self) -> None: ...

class CachedEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        embeddings: AnyEmbeddings,
        cache: MemoryEmbeddingCache | SqliteEmbeddingCache,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

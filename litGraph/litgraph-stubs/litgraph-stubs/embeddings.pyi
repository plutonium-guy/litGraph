from typing import Callable

class FunctionEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        func: Callable[[list[str]], list[list[float]]],
        dimensions: int,
        name: str = "custom",
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...

class OpenAIEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        timeout_s: int = 120,
        override_dimensions: int | None = None,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class CohereEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        timeout_s: int = 120,
        input_type_document: str | None = None,
        input_type_query: str | None = None,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class VoyageEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        timeout_s: int = 120,
        input_type_document: str | None = "document",
        input_type_query: str | None = "query",
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class GeminiEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        timeout_s: int = 120,
        task_type_document: str | None = "RETRIEVAL_DOCUMENT",
        task_type_query: str | None = "RETRIEVAL_QUERY",
        output_dimensionality: int | None = None,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class BedrockEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        model_id: str,
        dimensions: int,
        session_token: str | None = None,
        timeout_s: int = 120,
        endpoint_override: str | None = None,
        format: str | None = None,
        max_concurrency: int = 8,
        normalize: bool = True,
        cohere_input_type_document: str | None = None,
        cohere_input_type_query: str | None = None,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

class JinaEmbeddings:
    name: str
    dimensions: int
    def __init__(
        self,
        api_key: str,
        model: str,
        dimensions: int,
        base_url: str | None = None,
        timeout_s: int = 120,
        task_document: str | None = "retrieval.passage",
        task_query: str | None = "retrieval.query",
        output_dimensions: int | None = None,
    ) -> None: ...
    def embed_query(self, text: str) -> list[float]: ...
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

def tei_embeddings(
    base_url: str,
    dimensions: int,
    model: str = "tei",
    api_key: str = "",
    timeout_s: int = 120,
) -> OpenAIEmbeddings: ...

def together_embeddings(
    api_key: str,
    model: str,
    dimensions: int,
    base_url: str | None = None,
    timeout_s: int = 120,
) -> OpenAIEmbeddings: ...

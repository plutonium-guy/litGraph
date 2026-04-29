from typing import Any, Iterator

from .cache import MemoryCache, SqliteCache, SemanticCache
from .observability import CostTracker

class ChatStream:
    def __iter__(self) -> Iterator[dict[str, Any]]: ...
    def __next__(self) -> dict[str, Any]: ...

class OpenAIChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: int = 120,
        on_request: Any | None = None,
    ) -> None: ...
    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatStream: ...
    def with_cache(self, cache: MemoryCache | SqliteCache) -> None: ...
    def with_semantic_cache(self, cache: SemanticCache) -> None: ...
    def with_retry(
        self,
        max_times: int = 5,
        min_delay_ms: int = 200,
        max_delay_ms: int = 30_000,
        factor: float = 2.0,
        jitter: bool = True,
    ) -> None: ...
    def with_rate_limit(self, requests_per_minute: int, burst: int | None = None) -> None: ...
    def instrument(self, tracker: CostTracker) -> None: ...

class AnthropicChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: int = 120,
        max_tokens: int = 4096,
        on_request: Any | None = None,
        thinking_budget: int | None = None,
    ) -> None: ...
    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatStream: ...
    def with_cache(self, cache: MemoryCache | SqliteCache) -> None: ...
    def with_semantic_cache(self, cache: SemanticCache) -> None: ...
    def with_retry(
        self,
        max_times: int = 5,
        min_delay_ms: int = 200,
        max_delay_ms: int = 30_000,
        factor: float = 2.0,
        jitter: bool = True,
    ) -> None: ...
    def with_rate_limit(self, requests_per_minute: int, burst: int | None = None) -> None: ...
    def instrument(self, tracker: CostTracker) -> None: ...

class GeminiChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: int = 120,
        on_request: Any | None = None,
    ) -> None: ...
    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatStream: ...
    def with_cache(self, cache: MemoryCache | SqliteCache) -> None: ...
    def with_semantic_cache(self, cache: SemanticCache) -> None: ...
    def with_retry(
        self,
        max_times: int = 5,
        min_delay_ms: int = 200,
        max_delay_ms: int = 30_000,
        factor: float = 2.0,
        jitter: bool = True,
    ) -> None: ...
    def with_rate_limit(self, requests_per_minute: int, burst: int | None = None) -> None: ...
    def instrument(self, tracker: CostTracker) -> None: ...

class CohereChat:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str | None = None,
        timeout_s: int = 120,
        on_request: Any | None = None,
    ) -> None: ...
    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatStream: ...
    def with_cache(self, cache: MemoryCache | SqliteCache) -> None: ...
    def with_semantic_cache(self, cache: SemanticCache) -> None: ...
    def with_retry(
        self,
        max_times: int = 5,
        min_delay_ms: int = 200,
        max_delay_ms: int = 30_000,
        factor: float = 2.0,
        jitter: bool = True,
    ) -> None: ...
    def with_rate_limit(self, requests_per_minute: int, burst: int | None = None) -> None: ...
    def instrument(self, tracker: CostTracker) -> None: ...

def ollama_chat(
    model: str,
    base_url: str | None = None,
    timeout_s: int = 120,
) -> OpenAIChat: ...

def groq_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...
def together_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...
def mistral_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...
def deepseek_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...
def xai_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...
def fireworks_chat(
    api_key: str, model: str, base_url: str | None = None, timeout_s: int = 120,
) -> OpenAIChat: ...

class BedrockChat:
    def __init__(
        self,
        access_key_id: str,
        secret_access_key: str,
        region: str,
        model_id: str,
        session_token: str | None = None,
        timeout_s: int = 120,
        max_tokens: int = 4096,
        endpoint_override: str | None = None,
        on_request: Any | None = None,
    ) -> None: ...
    def invoke(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]: ...
    def stream(
        self,
        messages: list[dict[str, Any]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatStream: ...
    def with_cache(self, cache: MemoryCache | SqliteCache) -> None: ...
    def with_semantic_cache(self, cache: SemanticCache) -> None: ...
    def with_retry(
        self,
        max_times: int = 5,
        min_delay_ms: int = 200,
        max_delay_ms: int = 30_000,
        factor: float = 2.0,
        jitter: bool = True,
    ) -> None: ...
    def with_rate_limit(self, requests_per_minute: int, burst: int | None = None) -> None: ...
    def instrument(self, tracker: CostTracker) -> None: ...

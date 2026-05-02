"""Pure-Python mocks for unit tests against litGraph public APIs."""
from __future__ import annotations
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


class MockChatModel:
    calls: list[list[Mapping[str, Any]]]

    def __init__(
        self,
        replies: Sequence[str | Mapping[str, Any]] | None = ...,
        usage: Mapping[str, int] | None = ...,
        on_invoke: Callable[[list[Mapping[str, Any]]], None] | None = ...,
    ) -> None: ...

    def invoke(self, messages: list[Mapping[str, Any]], **_: Any) -> dict[str, Any]: ...
    def stream(self, messages: list[Mapping[str, Any]], **_: Any) -> Iterator[Any]: ...
    def with_structured_output(self, schema: Any) -> "MockChatModel": ...


class MockEmbeddings:
    dim: int
    calls: list[list[str]]

    def __init__(self, dim: int = 8) -> None: ...
    def embed(self, texts: Iterable[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


class MockTool:
    name: str
    description: str
    schema: dict[str, Any]
    calls: list[Mapping[str, Any]]

    def __init__(
        self,
        name: str,
        returns: Any = ...,
        side_effect: Callable[[Mapping[str, Any]], Any] | None = ...,
        description: str = ...,
        schema: Mapping[str, Any] | None = ...,
    ) -> None: ...

    def invoke(self, args: Mapping[str, Any]) -> Any: ...
    def run(self, args: Mapping[str, Any]) -> Any: ...
    def __call__(self, args: Mapping[str, Any]) -> Any: ...

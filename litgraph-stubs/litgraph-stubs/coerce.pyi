"""Type stubs for `litgraph.coerce` — Pydantic / dataclass / TypedDict
coercion of stream chunks."""

from typing import Any, AsyncIterable, AsyncIterator, Type, TypeVar

T = TypeVar("T")


def coerce_one(chunk: Any, cls: Type[T]) -> T: ...


def coerce_stream(
    stream: AsyncIterable[Any], cls: Type[T]
) -> AsyncIterator[T]: ...

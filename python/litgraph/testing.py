"""Mocks for deterministic unit tests against litGraph public APIs.

The framework's own test suite uses a Rust `ScriptedModel`; this module
exposes Python equivalents so user code can write tests without an
LLM credential or network.

Example:

    from litgraph.testing import MockChatModel, MockEmbeddings, MockTool

    m = MockChatModel(replies=["hello", "world"])
    out = m.invoke([{"role": "user", "content": "hi"}])
    assert out["content"] == "hello"

    emb = MockEmbeddings(dim=8)
    v = emb.embed(["one", "two"])  # deterministic — same text → same vec

    add = MockTool("add", returns={"sum": 42})
    assert add.invoke({"a": 1, "b": 2}) == {"sum": 42}
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Iterable, Mapping, Sequence


__all__ = [
    "MockChatModel",
    "MockEmbeddings",
    "MockTool",
]


class MockChatModel:
    """Scripted-reply chat model. `invoke` returns the next item from
    `replies`, cycling once exhausted. Records every call into
    `self.calls` so tests can assert on prompt content.

    Args:
        replies: list of strings (returned as `content`) or dicts
            (returned as the full message). Cycled.
        usage: optional dict to include in every response (`prompt`,
            `completion`, `total` token counts).
        on_invoke: optional callback fired on every call with the
            messages — useful for spy-style assertions.

    Provides `.invoke()`, `.stream()`, and `.with_structured_output(T)`
    so it drops into anywhere `ChatModel` is expected.
    """

    def __init__(
        self,
        replies: Sequence[str | Mapping[str, Any]] | None = None,
        usage: Mapping[str, int] | None = None,
        on_invoke: Callable[[list[Mapping[str, Any]]], None] | None = None,
    ) -> None:
        self._replies: list[Any] = list(replies) if replies else ["mock reply"]
        self._idx = 0
        self._usage = dict(usage) if usage else {"prompt": 0, "completion": 0, "total": 0}
        self._on_invoke = on_invoke
        self.calls: list[list[Mapping[str, Any]]] = []

    def _next(self) -> Any:
        item = self._replies[self._idx % len(self._replies)]
        self._idx += 1
        return item

    def invoke(self, messages: list[Mapping[str, Any]], **_: Any) -> dict[str, Any]:
        self.calls.append(list(messages))
        if self._on_invoke is not None:
            self._on_invoke(list(messages))
        item = self._next()
        if isinstance(item, str):
            return {
                "role": "assistant",
                "content": item,
                "tool_calls": [],
                "usage": dict(self._usage),
            }
        return dict(item)

    def stream(self, messages: list[Mapping[str, Any]], **_: Any):
        """Iterator that yields one `text` event per word of the next
        scripted reply, then a `finish` event. Mirrors the shape of
        `ChatStream` — a `for ev in stream(...)` loop sees `ev.kind`,
        `ev.text`, etc.
        """
        self.calls.append(list(messages))
        item = self._next()
        text = item if isinstance(item, str) else item.get("content", "")
        for word in text.split():
            yield _Event("text", text=word + " ")
        yield _Event("finish", text="", finish_reason="stop")

    def with_structured_output(self, schema: Any) -> "MockChatModel":
        """Return self — scripted replies are assumed to already match
        the schema. Tests can pass dict replies for richer payloads.
        """
        return self


class _Event:
    """Minimal stream-event stub matching `ChatStreamEvent` duck shape."""

    __slots__ = ("kind", "text", "finish_reason")

    def __init__(self, kind: str, text: str = "", finish_reason: str | None = None):
        self.kind = kind
        self.text = text
        self.finish_reason = finish_reason

    def __repr__(self) -> str:
        return f"<Event kind={self.kind!r} text={self.text!r}>"


class MockEmbeddings:
    """Deterministic, hash-based embeddings. Same input text → same
    vector across runs and processes. Useful for retrieval tests
    where you need vector-space behaviour without a real embedding
    provider.

    Args:
        dim: embedding dimensionality. Default 8 (small enough that
            assertions stay readable, large enough that distinct
            short strings rarely collide).

    Vectors are L2-normalised so cosine similarity == dot product.
    """

    def __init__(self, dim: int = 8) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self.calls: list[list[str]] = []

    def _vec(self, text: str) -> list[float]:
        # `shake_256` supports arbitrary digest length, so any `dim`
        # works. Map each byte (0–255) to [-1, 1).
        h = hashlib.shake_256(text.encode("utf-8")).digest(self.dim)
        out = [((b / 127.5) - 1.0) for b in h]
        # L2-normalise so cosine sim == dot product (handy for tests).
        norm = math.sqrt(sum(x * x for x in out)) or 1.0
        return [x / norm for x in out]

    def embed(self, texts: Iterable[str]) -> list[list[float]]:
        ts = list(texts)
        self.calls.append(ts)
        return [self._vec(t) for t in ts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


class MockTool:
    """Tool stub that returns a fixed value (or calls a callback).
    Records every invocation for spy-style asserts.

    Args:
        name: tool name (matches `Tool.name`).
        returns: fixed value returned by every `invoke` call.
        side_effect: optional callable; if set, called with `args` and
            its return value is used instead of `returns`.
        description: tool description (defaults to a sane stub).
        schema: JSON Schema for the tool's input. Defaults to an open
            object that accepts anything.

    Example:

        add = MockTool("add", returns={"sum": 3})
        add.invoke({"a": 1, "b": 2})    # → {"sum": 3}
        assert add.calls == [{"a": 1, "b": 2}]
    """

    def __init__(
        self,
        name: str,
        returns: Any = None,
        side_effect: Callable[[Mapping[str, Any]], Any] | None = None,
        description: str = "",
        schema: Mapping[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description or f"Mock tool {name}"
        # `schema` exposed both for direct assertion and for adapters
        # that read tool JSON-Schema for prompt construction.
        self.schema: dict[str, Any] = dict(schema) if schema else {"type": "object", "additionalProperties": True}
        self._returns = returns
        self._side_effect = side_effect
        self.calls: list[Mapping[str, Any]] = []

    def invoke(self, args: Mapping[str, Any]) -> Any:
        self.calls.append(dict(args))
        if self._side_effect is not None:
            return self._side_effect(dict(args))
        return self._returns

    # Aliases for the various tool-protocol shapes in the wild.
    def run(self, args: Mapping[str, Any]) -> Any:
        return self.invoke(args)

    def __call__(self, args: Mapping[str, Any]) -> Any:
        return self.invoke(args)

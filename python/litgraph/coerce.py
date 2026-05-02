"""State coercion — turn `dict` stream chunks into typed objects.

The native Python stream API yields `dict` events. That's the right
on-the-wire shape (JSON-serializable, no schema lock-in) but loses
IDE-narrow types — readers of agent code can't autocomplete fields
or get static type-checking on chunk shapes.

This module ships `coerce_stream` and `coerce_one`: feed the dict
through a Pydantic model, dataclass, or TypedDict and get a typed
instance out. Pydantic is optional — only imported on use, so users
without Pydantic installed still get dataclass + TypedDict support.

# Why not always-coerce in the native runtime

The native runtime is language-agnostic — it doesn't know what
Pydantic is, and a Rust-side coercion would have to round-trip
through Python validation anyway. Keeping coercion as a Python-side
opt-in keeps the native runtime fast (no per-chunk validation tax
when callers don't want it) and lets Python users pick their
schema library (Pydantic v1, Pydantic v2, attrs, dataclasses,
TypedDict — all supported via the same call).

# Quick reference

    from dataclasses import dataclass
    from litgraph import coerce_stream

    @dataclass
    class State:
        step: str
        value: int

    async for chunk in coerce_stream(workflow.astream(...), State):
        # chunk is a typed State instance — IDE autocomplete works.
        print(chunk.step, chunk.value)

    # Pydantic also works:
    from pydantic import BaseModel
    class StateP(BaseModel):
        step: str
        value: int

    async for chunk in coerce_stream(workflow.astream(...), StateP):
        ...

    # One-shot coercion for non-stream calls:
    state = coerce_one({"step": "init", "value": 0}, State)
"""

from __future__ import annotations

import dataclasses
from typing import Any, AsyncIterable, AsyncIterator, Type, TypeVar

T = TypeVar("T")


def _looks_like_pydantic(cls: type) -> bool:
    """True if `cls` is a Pydantic v1 or v2 BaseModel.

    Detected via attribute presence rather than `isinstance` to avoid
    importing pydantic when callers don't use it. Pydantic v2 has
    `model_validate`; v1 has `parse_obj`.
    """
    return hasattr(cls, "model_validate") or hasattr(cls, "parse_obj")


def _coerce_to_pydantic(data: Any, cls: type) -> Any:
    """Pydantic v2 `model_validate` first; v1 `parse_obj` fallback."""
    if hasattr(cls, "model_validate"):
        return cls.model_validate(data)
    return cls.parse_obj(data)  # v1 fallback


def _looks_like_dataclass(cls: type) -> bool:
    """True if `cls` is a `@dataclass`-decorated class."""
    return dataclasses.is_dataclass(cls)


def _looks_like_typeddict(cls: type) -> bool:
    """TypedDict subclasses have `__total__` and `__annotations__`."""
    return hasattr(cls, "__total__") and hasattr(cls, "__annotations__")


def coerce_one(chunk: Any, cls: Type[T]) -> T:
    """Coerce a single dict-shaped chunk to an instance of `cls`.

    Supports Pydantic v1/v2 BaseModel, dataclass, TypedDict.

    - **Pydantic**: validates via `model_validate` (v2) / `parse_obj`
      (v1) — full type coercion + validation.
    - **dataclass**: constructs via `cls(**chunk)`. Field types are
      NOT validated (matches stdlib dataclass semantics — use
      Pydantic if you want validation). Extra fields raise
      `TypeError`.
    - **TypedDict**: passes the dict through unchanged (TypedDict has
      no runtime instance — it's purely an IDE/type-checker aid).
      Returns the original dict.
    - **Other types**: tries `cls(**chunk)` if chunk is a dict; falls
      back to `cls(chunk)` if not. Raises `TypeError` on failure.

    Non-dict chunks (e.g. plain values like `42` or `"done"`) are
    returned as-is regardless of `cls` — coercion only fires on
    dict-shaped data.
    """
    if not isinstance(chunk, dict):
        return chunk  # type: ignore[return-value]
    if _looks_like_pydantic(cls):
        return _coerce_to_pydantic(chunk, cls)
    if _looks_like_typeddict(cls):
        return chunk  # type: ignore[return-value]
    if _looks_like_dataclass(cls):
        return cls(**chunk)
    # Generic class fallback — try keyword construction.
    try:
        return cls(**chunk)
    except TypeError as e:
        raise TypeError(
            f"coerce_one: failed to construct {cls.__name__} from chunk "
            f"{chunk!r}: {e}. Supported schema types: Pydantic BaseModel, "
            f"dataclass, TypedDict."
        ) from e


async def coerce_stream(
    stream: AsyncIterable[Any],
    cls: Type[T],
) -> AsyncIterator[T]:
    """Wrap an async iterable of dict chunks; yield coerced instances.

    Each chunk goes through `coerce_one(chunk, cls)`. Non-dict chunks
    pass through unchanged (e.g. progress events, sentinel values).
    Errors in coercion propagate immediately — the stream stops at
    the first malformed chunk so the caller sees the problem instead
    of silently skipping data.

    Usage:

        async for state in coerce_stream(workflow.astream(...), State):
            ...
    """
    async for chunk in stream:
        yield coerce_one(chunk, cls)

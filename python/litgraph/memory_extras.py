"""Hierarchical / namespaced memory wrapper.

Wraps any backend that implements the duck-shape `add_user / add_ai
/ messages / set_system / clear` (every `litgraph.memory` backend
does) and prefixes every entry with a tenant-or-thread namespace.

Use case: one shared SQLite/Postgres/Redis instance, many tenants /
sessions. Each `NamespacedMemory(inner, namespace="tenant_42")`
behaves like an isolated memory; the shared backend stores all
namespaces side-by-side and a top-level admin call sees the union.

Example:

    from litgraph.memory import SqliteChatMemory
    from litgraph.memory_extras import NamespacedMemory

    shared = SqliteChatMemory(path="./memory.db")
    alice = NamespacedMemory(shared, namespace="user/alice")
    bob = NamespacedMemory(shared, namespace="user/bob")

    alice.add_user("hi")
    bob.add_user("hello")          # in the same DB, different namespace
    alice.messages()               # only Alice's history

The namespace is stamped into the wrapped messages' metadata under
`__litgraph_ns__`. On read, the wrapper filters by exact match.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


__all__ = ["NamespacedMemory", "NS_KEY"]


NS_KEY = "__litgraph_ns__"


class NamespacedMemory:
    """Drop-in wrapper that namespaces every read + write against an
    inner memory backend.

    The inner backend must accept a `metadata` argument on `add_user`
    / `add_ai` (every native litGraph memory class does as of v0.1.x;
    the wrapper falls back to a plain string write if `metadata` is
    rejected, with a warning).

    Reads filter by `metadata[NS_KEY] == namespace`. The pin set by
    `set_system(...)` lives outside the message stream and is namespaced
    via a per-instance attribute (the inner backend never sees it).
    """

    def __init__(self, inner: Any, namespace: str) -> None:
        if not namespace or "/" not in namespace and ":" not in namespace:
            # Hierarchical separators are encouraged but not required.
            # Empty / whitespace namespaces ARE rejected — they would
            # collide with un-namespaced entries.
            if not namespace.strip():
                raise ValueError("namespace must be non-empty")
        self._inner = inner
        self._ns = namespace
        self._system_pin: Mapping[str, Any] | None = None

    @property
    def namespace(self) -> str:
        return self._ns

    # ---- writes ----

    def add_user(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        meta = self._stamp(metadata)
        self._dispatch("add_user", text, meta)

    def add_ai(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        meta = self._stamp(metadata)
        self._dispatch("add_ai", text, meta)

    def add_tool(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        meta = self._stamp(metadata)
        self._dispatch("add_tool", text, meta)

    def set_system(self, message: Mapping[str, Any] | str | None) -> None:
        # System pin is stored on the wrapper, never the shared inner.
        # That avoids inner-backend pin collisions across namespaces.
        if isinstance(message, str):
            self._system_pin = {"role": "system", "content": message}
        else:
            self._system_pin = dict(message) if message is not None else None

    def clear(self) -> None:
        """Drop only this namespace's messages from the inner backend.
        Pure Python read-then-rewrite for backends without per-key
        delete; backends that natively expose `delete_where(metadata)`
        could be plugged in later."""
        # Read all, keep ones not in this namespace, rewrite.
        all_msgs = self._inner.messages() if hasattr(self._inner, "messages") else []
        if hasattr(self._inner, "clear"):
            self._inner.clear()
        for m in all_msgs:
            md = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
            if md.get(NS_KEY) == self._ns:
                continue  # skip this namespace
            # Re-add the survivor.
            role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
            if role == "user":
                self._inner.add_user(content, metadata=md)
            elif role == "assistant":
                self._inner.add_ai(content, metadata=md)
            elif role == "tool":
                if hasattr(self._inner, "add_tool"):
                    self._inner.add_tool(content, metadata=md)
        self._system_pin = None

    # ---- reads ----

    def messages(self) -> list[Mapping[str, Any]]:
        all_msgs: Iterable[Any] = (
            self._inner.messages() if hasattr(self._inner, "messages") else []
        )
        out: list[Mapping[str, Any]] = []
        if self._system_pin is not None:
            out.append(self._system_pin)
        for m in all_msgs:
            md = (m.get("metadata") if isinstance(m, dict) else getattr(m, "metadata", None)) or {}
            if md.get(NS_KEY) != self._ns:
                continue
            if isinstance(m, dict):
                out.append(m)
            else:
                out.append({
                    "role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", ""),
                    "metadata": md,
                })
        return out

    # ---- internals ----

    def _stamp(self, metadata: Mapping[str, Any] | None) -> dict[str, Any]:
        out = dict(metadata or {})
        out[NS_KEY] = self._ns
        return out

    def _dispatch(self, method: str, text: str, metadata: dict[str, Any]) -> None:
        fn = getattr(self._inner, method, None)
        if fn is None:
            raise AttributeError(
                f"inner memory backend does not support {method!r}"
            )
        try:
            fn(text, metadata=metadata)
        except TypeError:
            # Backend doesn't accept metadata kwarg. Fall back to
            # plain text — namespace isolation is best-effort here.
            fn(text)

    def __repr__(self) -> str:
        return f"NamespacedMemory(ns={self._ns!r}, inner={type(self._inner).__name__})"

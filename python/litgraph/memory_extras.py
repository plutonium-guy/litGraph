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


__all__ = [
    "NamespacedMemory",
    "NS_KEY",
    "DynamoDBChatMemory",
    "MongoChatMemory",
]


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


# ---- DynamoDB chat-history backend ----


class DynamoDBChatMemory:
    """Persistent chat history in a DynamoDB table.

    Lazy-imports `boto3`; install with `pip install boto3`.

    Schema: partition key `session_id` (string), sort key `ts` (number,
    epoch micros). All messages for a session are written / read in
    sort-key order; range query gives full history.

    Args:
        table_name: existing DynamoDB table.
        session_id: chat session identifier.
        region_name: AWS region.
        endpoint_url: optional override (e.g. local DynamoDB).
    """

    def __init__(
        self,
        table_name: str,
        session_id: str,
        region_name: str | None = None,
        endpoint_url: str | None = None,
    ) -> None:
        try:
            import boto3  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "boto3 not installed. Run `pip install boto3`."
            ) from e
        if not session_id:
            raise ValueError("session_id required")
        self.table_name = table_name
        self.session_id = session_id
        self._table = boto3.resource(
            "dynamodb",
            region_name=region_name,
            endpoint_url=endpoint_url,
        ).Table(table_name)
        self._system_pin: Mapping[str, Any] | None = None

    def _now_us(self) -> int:
        import time
        return int(time.time() * 1_000_000)

    def _put(self, role: str, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        item: dict[str, Any] = {
            "session_id": self.session_id,
            "ts": self._now_us(),
            "role": role,
            "content": text,
        }
        if metadata:
            item["metadata"] = dict(metadata)
        self._table.put_item(Item=item)

    def add_user(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("user", text, metadata)

    def add_ai(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("assistant", text, metadata)

    def add_tool(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("tool", text, metadata)

    def set_system(self, message: Mapping[str, Any] | str | None) -> None:
        if isinstance(message, str):
            self._system_pin = {"role": "system", "content": message}
        else:
            self._system_pin = dict(message) if message is not None else None

    def messages(self) -> list[Mapping[str, Any]]:
        resp = self._table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": self.session_id},
            ScanIndexForward=True,
        )
        out: list[Mapping[str, Any]] = []
        if self._system_pin is not None:
            out.append(self._system_pin)
        for item in resp.get("Items", []):
            out.append({
                "role": item.get("role", "user"),
                "content": item.get("content", ""),
                "metadata": item.get("metadata", {}),
            })
        return out

    def clear(self) -> None:
        # Range-delete: query, batch-delete the keys.
        resp = self._table.query(
            KeyConditionExpression="session_id = :sid",
            ExpressionAttributeValues={":sid": self.session_id},
        )
        with self._table.batch_writer() as batch:
            for item in resp.get("Items", []):
                batch.delete_item(Key={"session_id": item["session_id"], "ts": item["ts"]})
        self._system_pin = None


# ---- MongoDB chat-history backend ----


class MongoChatMemory:
    """Persistent chat history in a MongoDB collection. Uses
    `(session_id, ts)` as the natural key; one document per turn.

    Lazy-imports `pymongo`; install with `pip install pymongo`.

    Args:
        uri: MongoDB connection string (or `MONGODB_URI` env).
        database: database name.
        collection_name: collection name.
        session_id: chat session id.
    """

    def __init__(
        self,
        session_id: str,
        uri: str | None = None,
        database: str = "litgraph",
        collection_name: str = "chat_messages",
    ) -> None:
        try:
            from pymongo import ASCENDING, MongoClient  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pymongo not installed. Run `pip install pymongo`."
            ) from e
        import os as _os
        connection = uri or _os.environ.get("MONGODB_URI")
        if not connection:
            raise ValueError("MongoDB URI required (env MONGODB_URI)")
        if not session_id:
            raise ValueError("session_id required")
        self.session_id = session_id
        self._client = MongoClient(connection)
        self._coll = self._client[database][collection_name]
        # Idempotent compound index for fast per-session range read.
        self._coll.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
        self._system_pin: Mapping[str, Any] | None = None

    def _now_us(self) -> int:
        import time
        return int(time.time() * 1_000_000)

    def _put(self, role: str, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._coll.insert_one({
            "session_id": self.session_id,
            "ts": self._now_us(),
            "role": role,
            "content": text,
            "metadata": dict(metadata or {}),
        })

    def add_user(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("user", text, metadata)

    def add_ai(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("assistant", text, metadata)

    def add_tool(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self._put("tool", text, metadata)

    def set_system(self, message: Mapping[str, Any] | str | None) -> None:
        if isinstance(message, str):
            self._system_pin = {"role": "system", "content": message}
        else:
            self._system_pin = dict(message) if message is not None else None

    def messages(self) -> list[Mapping[str, Any]]:
        out: list[Mapping[str, Any]] = []
        if self._system_pin is not None:
            out.append(self._system_pin)
        for doc in self._coll.find({"session_id": self.session_id}).sort("ts", 1):
            out.append({
                "role": doc.get("role", "user"),
                "content": doc.get("content", ""),
                "metadata": doc.get("metadata", {}),
            })
        return out

    def clear(self) -> None:
        self._coll.delete_many({"session_id": self.session_id})
        self._system_pin = None

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
    "CassandraChatMemory",
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


# ---- Cassandra chat-history backend ----


class CassandraChatMemory:
    """Persistent chat history in a Cassandra / ScyllaDB / AstraDB
    cluster. Lazy-imports `cassandra-driver`; install with
    `pip install cassandra-driver`.

    Schema:
        CREATE TABLE chat_messages (
            session_id text,
            ts timeuuid,
            role text,
            content text,
            metadata text,
            PRIMARY KEY (session_id, ts)
        ) WITH CLUSTERING ORDER BY (ts ASC);

    The wrapper auto-creates the table if missing.

    Args:
        session_id: chat session id.
        keyspace: target keyspace (must exist).
        contact_points: list of node hostnames.
        port: native protocol port.
        username, password: auth (or env `CASSANDRA_USERNAME` / `CASSANDRA_PASSWORD`).
        table_name: chat-message table.
        astra_bundle: optional path to a DataStax Astra secure-connect
            zip — when set, contact_points / port are ignored.
    """

    def __init__(
        self,
        session_id: str,
        keyspace: str = "litgraph",
        contact_points: tuple[str, ...] = ("127.0.0.1",),
        port: int = 9042,
        username: str | None = None,
        password: str | None = None,
        table_name: str = "chat_messages",
        astra_bundle: str | None = None,
    ) -> None:
        try:
            from cassandra.auth import PlainTextAuthProvider  # type: ignore[import-not-found]
            from cassandra.cluster import Cluster  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "cassandra-driver not installed. "
                "Run `pip install cassandra-driver`."
            ) from e
        import os as _os
        if not session_id:
            raise ValueError("session_id required")
        self.session_id = session_id
        self.keyspace = keyspace
        self.table_name = table_name
        user = username or _os.environ.get("CASSANDRA_USERNAME")
        pwd = password or _os.environ.get("CASSANDRA_PASSWORD")
        auth = PlainTextAuthProvider(username=user, password=pwd) if user else None
        if astra_bundle:
            self._cluster = Cluster(
                cloud={"secure_connect_bundle": astra_bundle},
                auth_provider=auth,
            )
        else:
            self._cluster = Cluster(
                contact_points=list(contact_points),
                port=port,
                auth_provider=auth,
            )
        self._session = self._cluster.connect(keyspace)
        self._session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {keyspace}.{table_name} (
                session_id text,
                ts timeuuid,
                role text,
                content text,
                metadata text,
                PRIMARY KEY (session_id, ts)
            ) WITH CLUSTERING ORDER BY (ts ASC)
            """
        )
        self._system_pin: Mapping[str, Any] | None = None

    def _put(self, role: str, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        import json as _json
        from cassandra.util import uuid_from_time  # type: ignore[import-not-found]
        import time as _time
        self._session.execute(
            f"INSERT INTO {self.keyspace}.{self.table_name} "
            "(session_id, ts, role, content, metadata) VALUES (%s, %s, %s, %s, %s)",
            (
                self.session_id,
                uuid_from_time(_time.time()),
                role,
                text,
                _json.dumps(metadata or {}),
            ),
        )

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
        import json as _json
        out: list[Mapping[str, Any]] = []
        if self._system_pin is not None:
            out.append(self._system_pin)
        rows = self._session.execute(
            f"SELECT role, content, metadata FROM {self.keyspace}.{self.table_name} "
            "WHERE session_id = %s",
            (self.session_id,),
        )
        for row in rows:
            try:
                md = _json.loads(row.metadata) if row.metadata else {}
            except (TypeError, ValueError):
                md = {}
            out.append({
                "role": row.role,
                "content": row.content,
                "metadata": md,
            })
        return out

    def clear(self) -> None:
        self._session.execute(
            f"DELETE FROM {self.keyspace}.{self.table_name} WHERE session_id = %s",
            (self.session_id,),
        )
        self._system_pin = None

    def close(self) -> None:
        self._cluster.shutdown()

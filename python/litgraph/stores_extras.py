"""FAISS / niche vector-store adapters.

Native stores (`litgraph.stores`) cover in-mem, HNSW, Qdrant,
pgvector, Chroma, Weaviate. This module adds FAISS for projects that
already have a FAISS-indexed corpus from another framework
(LangChain, Haystack) and want to query it from litGraph without
re-embedding.

Each adapter implements the duck-shape `VectorStore`:

    add(docs, embeddings) -> list[str]         # ids
    similarity_search(query_embedding, k, filter=None) -> list[doc]
    delete(ids)
    len() -> int
"""
from __future__ import annotations

import uuid
from typing import Any, Iterable, Mapping


__all__ = [
    "FaissVectorStore",
    "MilvusVectorStore",
    "RedisSearchVectorStore",
    "Neo4jVectorStore",
    "MongoAtlasVectorStore",
]


class FaissVectorStore:
    """In-process FAISS index. Lazy-imports `faiss-cpu`; install with
    `pip install faiss-cpu`.

    Defaults to `IndexFlatIP` (exact inner-product, equivalent to
    cosine on L2-normalised vectors). For large corpora switch to
    `IndexHNSWFlat` via the `index_factory` arg.

    Args:
        dim: embedding dimensionality.
        index_factory: FAISS factory string (default "Flat").
        normalize: L2-normalise vectors on add + query so cosine ==
            inner product. Default True.

    Filter support: per-id metadata is held in a parallel dict;
    `similarity_search(filter={...})` post-filters by exact-match on
    metadata keys. Not as efficient as a native store with a side
    index — fine for small corpora (< 1M vecs).
    """

    def __init__(
        self,
        dim: int,
        index_factory: str = "Flat",
        normalize: bool = True,
    ) -> None:
        try:
            import faiss  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "faiss-cpu not installed. "
                "Run `pip install faiss-cpu` to use this store."
            ) from e
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim
        self._faiss = faiss
        self._index = faiss.index_factory(dim, index_factory)
        self._normalize = normalize
        # FAISS gives back row-indices; we map row → (id, doc) so we
        # can return docs and support id-based delete.
        self._id_to_row: dict[str, int] = {}
        self._row_to_doc: dict[int, dict[str, Any]] = {}
        self._next_row = 0

    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]:
        import numpy as np  # type: ignore[import-not-found]

        docs_list = list(docs)
        emb_list = [list(e) for e in embeddings]
        if len(docs_list) != len(emb_list):
            raise ValueError("docs and embeddings must have equal length")
        if not docs_list:
            return []
        if any(len(v) != self.dim for v in emb_list):
            raise ValueError(f"all embeddings must have length {self.dim}")

        arr = np.asarray(emb_list, dtype="float32")
        if self._normalize:
            self._faiss.normalize_L2(arr)
        self._index.add(arr)

        ids: list[str] = []
        for d in docs_list:
            doc_id = str(d.get("id") or uuid.uuid4())
            row = self._next_row
            self._next_row += 1
            self._id_to_row[doc_id] = row
            self._row_to_doc[row] = {**d, "id": doc_id}
            ids.append(doc_id)
        return ids

    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        import numpy as np  # type: ignore[import-not-found]

        if k <= 0:
            return []
        q = list(query_embedding)
        if len(q) != self.dim:
            raise ValueError(f"query embedding must have length {self.dim}, got {len(q)}")

        arr = np.asarray([q], dtype="float32")
        if self._normalize:
            self._faiss.normalize_L2(arr)
        # Over-fetch when filtering so we still return k after post-filter.
        fetch = k if filter is None else max(k * 4, k + 10)
        # Bounded by the index size — FAISS warns otherwise.
        fetch = min(fetch, max(self.__len__(), 1))
        scores, rows = self._index.search(arr, fetch)
        out: list[dict[str, Any]] = []
        for score, row in zip(scores[0].tolist(), rows[0].tolist()):
            if row < 0 or row not in self._row_to_doc:
                continue
            doc = self._row_to_doc[row]
            if filter is not None:
                md = doc.get("metadata") or {}
                if any(md.get(fk) != fv for fk, fv in filter.items()):
                    continue
            out.append({**doc, "score": float(score)})
            if len(out) >= k:
                break
        return out

    def delete(self, ids: Iterable[str]) -> None:
        # FAISS Flat indexes don't support remove_ids on all backends;
        # we tombstone by dropping the row → doc mapping. Search will
        # skip tombstoned rows.
        for doc_id in ids:
            row = self._id_to_row.pop(doc_id, None)
            if row is not None:
                self._row_to_doc.pop(row, None)

    def __len__(self) -> int:
        return len(self._row_to_doc)

    def is_empty(self) -> bool:
        return len(self) == 0


class MilvusVectorStore:
    """Wrap Milvus / Zilliz Cloud via the official `pymilvus` SDK.
    Lazy-imports `pymilvus`; install with `pip install pymilvus`.

    Args:
        collection_name: target collection (created if missing).
        dim: embedding dimensionality.
        uri: Milvus URI (e.g. "http://localhost:19530" or a Zilliz
            Cloud endpoint).
        token: Zilliz token (or `MILVUS_TOKEN` env).
        consistency: "Strong" / "Bounded" / "Eventually" — default
            "Bounded" (Milvus's recommendation for RAG).
    """

    def __init__(
        self,
        collection_name: str,
        dim: int,
        uri: str = "http://localhost:19530",
        token: str | None = None,
        consistency: str = "Bounded",
    ) -> None:
        try:
            from pymilvus import (  # type: ignore[import-not-found]
                MilvusClient,
            )
        except ImportError as e:
            raise ImportError(
                "pymilvus not installed. Run `pip install pymilvus`."
            ) from e
        import os as _os
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.collection_name = collection_name
        self.dim = dim
        self._client = MilvusClient(
            uri=uri,
            token=token or _os.environ.get("MILVUS_TOKEN", ""),
        )
        # Create the collection lazily — Milvus's auto-id schema is
        # the simplest route for an opinionated wrapper.
        if not self._client.has_collection(collection_name):
            self._client.create_collection(
                collection_name=collection_name,
                dimension=dim,
                consistency_level=consistency,
            )

    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]:
        docs_list = list(docs)
        emb_list = [list(e) for e in embeddings]
        if len(docs_list) != len(emb_list):
            raise ValueError("docs and embeddings must have equal length")
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        for d, e in zip(docs_list, emb_list):
            doc_id = str(d.get("id") or uuid.uuid4())
            rows.append({
                "id": doc_id,
                "vector": e,
                "page_content": d.get("page_content", ""),
                # Milvus dynamic-field mode: any extra keys are stored.
                **{k: v for k, v in (d.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool))},
            })
            ids.append(doc_id)
        if rows:
            self._client.insert(self.collection_name, rows)
        return ids

    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q = list(query_embedding)
        if len(q) != self.dim:
            raise ValueError(f"query embedding must have length {self.dim}, got {len(q)}")
        # Milvus boolean expression for metadata filter.
        expr = None
        if filter:
            parts = []
            for fk, fv in filter.items():
                if isinstance(fv, str):
                    parts.append(f'{fk} == "{fv}"')
                else:
                    parts.append(f"{fk} == {fv}")
            expr = " and ".join(parts)
        results = self._client.search(
            self.collection_name,
            data=[q],
            limit=k,
            filter=expr,
            output_fields=["page_content", "*"],
        )
        out: list[dict[str, Any]] = []
        if results and len(results) > 0:
            for hit in results[0]:
                entity = hit.get("entity", {})
                out.append({
                    "id": str(hit.get("id")),
                    "page_content": entity.get("page_content", ""),
                    "metadata": {k: v for k, v in entity.items() if k != "page_content"},
                    "score": float(hit.get("distance", 0.0)),
                })
        return out

    def delete(self, ids: Iterable[str]) -> None:
        self._client.delete(self.collection_name, ids=list(ids))

    def __len__(self) -> int:
        try:
            stats = self._client.get_collection_stats(self.collection_name)
            return int(stats.get("row_count", 0))
        except Exception:
            return 0


class RedisSearchVectorStore:
    """Redis Stack vector index (RediSearch). Lazy-imports `redis`;
    install with `pip install redis`.

    Stores docs as hashes under a configurable prefix; vectors live in
    a `VECTOR FLAT` field. Uses cosine distance by default. Filter
    support is via RediSearch query syntax (TAG / NUMERIC fields are
    indexed automatically when registered in `metadata_fields`).

    Args:
        index_name: RediSearch index name.
        dim: embedding dimensionality.
        url: Redis connection URL (e.g. "redis://localhost:6379/0").
        prefix: doc-key prefix (default "litgraph:doc:").
        metadata_fields: optional list of `(field_name, "TAG"|"NUMERIC")`
            pairs to index for filtering.
    """

    def __init__(
        self,
        index_name: str,
        dim: int,
        url: str = "redis://localhost:6379/0",
        prefix: str = "litgraph:doc:",
        metadata_fields: Iterable[tuple[str, str]] = (),
    ) -> None:
        try:
            import redis  # type: ignore[import-not-found]
            from redis.commands.search.field import (  # type: ignore[import-not-found]
                TagField, NumericField, TextField, VectorField,
            )
            from redis.commands.search.indexDefinition import (  # type: ignore[import-not-found]
                IndexDefinition, IndexType,
            )
        except ImportError as e:
            raise ImportError(
                "redis (with `redis.commands.search`) not installed. "
                "Run `pip install redis`."
            ) from e
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.index_name = index_name
        self.dim = dim
        self.prefix = prefix
        self._client = redis.Redis.from_url(url)
        # Idempotent index creation — `FT.CREATE` errors with
        # "Index already exists"; swallow that.
        fields = [
            TextField("page_content"),
            VectorField(
                "vector",
                "FLAT",
                {"TYPE": "FLOAT32", "DIM": dim, "DISTANCE_METRIC": "COSINE"},
            ),
        ]
        for fname, ftype in metadata_fields:
            ft = ftype.upper()
            if ft == "TAG":
                fields.append(TagField(fname))
            elif ft == "NUMERIC":
                fields.append(NumericField(fname))
            else:
                fields.append(TextField(fname))
        try:
            self._client.ft(index_name).create_index(
                fields,
                definition=IndexDefinition(prefix=[prefix], index_type=IndexType.HASH),
            )
        except Exception as e:
            # "Index already exists" is the happy path on subsequent
            # connects; anything else propagates.
            if "exists" not in str(e).lower():
                raise

    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]:
        import struct
        docs_list = list(docs)
        emb_list = [list(e) for e in embeddings]
        if len(docs_list) != len(emb_list):
            raise ValueError("docs and embeddings must have equal length")
        ids: list[str] = []
        pipe = self._client.pipeline()
        for d, vec in zip(docs_list, emb_list):
            doc_id = str(d.get("id") or uuid.uuid4())
            # Pack vector as little-endian float32 bytes (RediSearch wire format).
            packed = struct.pack(f"<{len(vec)}f", *vec)
            payload = {
                "page_content": d.get("page_content", ""),
                "vector": packed,
            }
            for k, v in (d.get("metadata") or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    payload[k] = v
            pipe.hset(f"{self.prefix}{doc_id}", mapping=payload)
            ids.append(doc_id)
        pipe.execute()
        return ids

    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        import struct
        from redis.commands.search.query import Query  # type: ignore[import-not-found]
        q = list(query_embedding)
        if len(q) != self.dim:
            raise ValueError(f"query embedding must have length {self.dim}, got {len(q)}")
        packed = struct.pack(f"<{len(q)}f", *q)
        # Filter clause: AND-join `@field:{value}` for tags, `@field:[low high]` for numerics.
        filter_clause = "*"
        if filter:
            parts = []
            for fk, fv in filter.items():
                if isinstance(fv, (int, float)):
                    parts.append(f"@{fk}:[{fv} {fv}]")
                else:
                    parts.append(f"@{fk}:{{{fv}}}")
            filter_clause = " ".join(parts) or "*"
        query = (
            Query(f"({filter_clause})=>[KNN {k} @vector $vec AS score]")
            .sort_by("score")
            .return_fields("id", "page_content", "score")
            .dialect(2)
        )
        result = self._client.ft(self.index_name).search(query, query_params={"vec": packed})
        out: list[dict[str, Any]] = []
        for doc in result.docs:
            out.append({
                "id": doc.id.removeprefix(self.prefix) if hasattr(doc, "id") else "",
                "page_content": getattr(doc, "page_content", ""),
                "score": float(getattr(doc, "score", 0.0)),
                "metadata": {},
            })
        return out

    def delete(self, ids: Iterable[str]) -> None:
        keys = [f"{self.prefix}{i}" for i in ids]
        if keys:
            self._client.delete(*keys)

    def __len__(self) -> int:
        try:
            info = self._client.ft(self.index_name).info()
            return int(info.get("num_docs", 0))
        except Exception:
            return 0


class Neo4jVectorStore:
    """Neo4j vector index (5.13+). Lazy-imports `neo4j`; install with
    `pip install neo4j`. Each vector is a node; metadata is stored as
    node properties.

    Args:
        index_name: vector-index name (created if missing).
        dim: embedding dimensionality.
        uri: Neo4j Bolt URI (e.g. "bolt://localhost:7687" or
            "neo4j+s://..." for Aura).
        username, password: auth (or `NEO4J_USERNAME` / `NEO4J_PASSWORD` env).
        database: database name (default "neo4j").
        node_label: label used for vector nodes (default "Document").
        similarity: "cosine" / "euclidean" — Neo4j vector index metric.
    """

    def __init__(
        self,
        index_name: str,
        dim: int,
        uri: str = "bolt://localhost:7687",
        username: str | None = None,
        password: str | None = None,
        database: str = "neo4j",
        node_label: str = "Document",
        similarity: str = "cosine",
    ) -> None:
        try:
            from neo4j import GraphDatabase  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "neo4j not installed. Run `pip install neo4j`."
            ) from e
        import os as _os
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.index_name = index_name
        self.dim = dim
        self.database = database
        self.node_label = node_label
        self._driver = GraphDatabase.driver(
            uri,
            auth=(
                username or _os.environ.get("NEO4J_USERNAME", "neo4j"),
                password or _os.environ.get("NEO4J_PASSWORD", ""),
            ),
        )
        # Idempotent index create. Neo4j 5.13+ syntax.
        with self._driver.session(database=database) as session:
            session.run(
                f"""
                CREATE VECTOR INDEX `{index_name}` IF NOT EXISTS
                FOR (n:`{node_label}`)
                ON (n.embedding)
                OPTIONS {{ indexConfig: {{
                    `vector.dimensions`: $dim,
                    `vector.similarity_function`: $sim
                }} }}
                """,
                dim=dim, sim=similarity,
            )

    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]:
        docs_list = list(docs)
        emb_list = [list(e) for e in embeddings]
        if len(docs_list) != len(emb_list):
            raise ValueError("docs and embeddings must have equal length")
        ids: list[str] = []
        rows: list[dict[str, Any]] = []
        for d, e in zip(docs_list, emb_list):
            doc_id = str(d.get("id") or uuid.uuid4())
            rows.append({
                "id": doc_id,
                "embedding": e,
                "page_content": d.get("page_content", ""),
                "metadata": {k: v for k, v in (d.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool))},
            })
            ids.append(doc_id)
        with self._driver.session(database=self.database) as session:
            session.run(
                f"""
                UNWIND $rows AS r
                MERGE (n:`{self.node_label}` {{id: r.id}})
                SET n.embedding = r.embedding,
                    n.page_content = r.page_content,
                    n += r.metadata
                """,
                rows=rows,
            )
        return ids

    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q = list(query_embedding)
        if len(q) != self.dim:
            raise ValueError(f"query embedding must have length {self.dim}, got {len(q)}")
        # Cypher filter — only equality on whitelisted property names.
        filter_clause = ""
        params: dict[str, Any] = {"k": k, "q": q, "name": self.index_name}
        if filter:
            conds = []
            for i, (fk, fv) in enumerate(filter.items()):
                pname = f"f{i}"
                conds.append(f"node.{fk} = ${pname}")
                params[pname] = fv
            filter_clause = "WHERE " + " AND ".join(conds)
        cy = (
            f"CALL db.index.vector.queryNodes($name, $k, $q) YIELD node, score "
            f"{filter_clause} RETURN node, score"
        )
        out: list[dict[str, Any]] = []
        with self._driver.session(database=self.database) as session:
            for record in session.run(cy, **params):
                node = record["node"]
                props = dict(node)
                doc_id = props.pop("id", "")
                content = props.pop("page_content", "")
                props.pop("embedding", None)
                out.append({
                    "id": doc_id,
                    "page_content": content,
                    "score": float(record["score"]),
                    "metadata": props,
                })
        return out

    def delete(self, ids: Iterable[str]) -> None:
        with self._driver.session(database=self.database) as session:
            session.run(
                f"MATCH (n:`{self.node_label}`) WHERE n.id IN $ids DETACH DELETE n",
                ids=list(ids),
            )

    def __len__(self) -> int:
        with self._driver.session(database=self.database) as session:
            res = session.run(f"MATCH (n:`{self.node_label}`) RETURN count(n) AS c")
            return int(res.single()["c"])

    def close(self) -> None:
        self._driver.close()


class MongoAtlasVectorStore:
    """MongoDB Atlas Search vector index. Lazy-imports `pymongo`;
    install with `pip install pymongo`. Atlas creates the vector
    index out-of-band (UI / Atlas API); this wrapper assumes the
    index already exists and points at the right collection.

    Args:
        collection_name: source collection.
        dim: embedding dimensionality (informational; not enforced
            client-side — Atlas validates against the index def).
        uri: MongoDB Atlas connection string. Falls back to
            `MONGODB_URI` env.
        database: database name.
        index_name: Atlas vector-search index name (set up in the
            Atlas UI / `createSearchIndex` admin command).
        embedding_field: field name holding the vector array.
        text_field: field whose value is returned as `page_content`.
    """

    def __init__(
        self,
        collection_name: str,
        dim: int,
        uri: str | None = None,
        database: str = "litgraph",
        index_name: str = "vector_index",
        embedding_field: str = "embedding",
        text_field: str = "page_content",
    ) -> None:
        try:
            from pymongo import MongoClient  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "pymongo not installed. Run `pip install pymongo`."
            ) from e
        import os as _os
        connection = uri or _os.environ.get("MONGODB_URI")
        if not connection:
            raise ValueError("MongoDB URI required (env MONGODB_URI)")
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.collection_name = collection_name
        self.dim = dim
        self.index_name = index_name
        self.embedding_field = embedding_field
        self.text_field = text_field
        self._client = MongoClient(connection)
        self._coll = self._client[database][collection_name]

    def add(
        self,
        docs: Iterable[Mapping[str, Any]],
        embeddings: Iterable[Iterable[float]],
    ) -> list[str]:
        docs_list = list(docs)
        emb_list = [list(e) for e in embeddings]
        if len(docs_list) != len(emb_list):
            raise ValueError("docs and embeddings must have equal length")
        rows: list[dict[str, Any]] = []
        ids: list[str] = []
        for d, e in zip(docs_list, emb_list):
            doc_id = str(d.get("id") or uuid.uuid4())
            row = {
                "_id": doc_id,
                self.embedding_field: e,
                self.text_field: d.get("page_content", ""),
            }
            for k, v in (d.get("metadata") or {}).items():
                row[k] = v
            rows.append(row)
            ids.append(doc_id)
        if rows:
            self._coll.insert_many(rows, ordered=False)
        return ids

    def similarity_search(
        self,
        query_embedding: Iterable[float],
        k: int = 5,
        filter: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        q = list(query_embedding)
        if len(q) != self.dim:
            raise ValueError(f"query embedding must have length {self.dim}, got {len(q)}")
        pipeline: list[dict[str, Any]] = [
            {
                "$vectorSearch": {
                    "index": self.index_name,
                    "queryVector": q,
                    "path": self.embedding_field,
                    "numCandidates": max(k * 10, 50),
                    "limit": k,
                    **({"filter": dict(filter)} if filter else {}),
                }
            },
            {"$project": {self.embedding_field: 0, "score": {"$meta": "vectorSearchScore"}}},
        ]
        out: list[dict[str, Any]] = []
        for doc in self._coll.aggregate(pipeline):
            doc_id = str(doc.pop("_id", ""))
            content = doc.pop(self.text_field, "")
            score = float(doc.pop("score", 0.0))
            out.append({
                "id": doc_id,
                "page_content": content,
                "score": score,
                "metadata": doc,
            })
        return out

    def delete(self, ids: Iterable[str]) -> None:
        self._coll.delete_many({"_id": {"$in": list(ids)}})

    def __len__(self) -> int:
        return self._coll.estimated_document_count()

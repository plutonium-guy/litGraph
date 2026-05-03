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


__all__ = ["FaissVectorStore"]


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

"""Optional cache backends that wrap third-party Python libs.

Native caches in `litgraph.cache`: `MemoryCache`, `SqliteCache`,
`SemanticCache`. This module is the escape hatch for backends whose
Python ecosystem is mature enough that re-implementing them isn't
worth it: GPTCache.

Each adapter implements `get(key) -> str | None` and `put(key, value)
-> None` so it slots into `CachedChatModel` / `CachedEmbeddings`.
"""
from __future__ import annotations

from typing import Any


__all__ = ["GPTCacheAdapter"]


class GPTCacheAdapter:
    """Wrap `gptcache.Cache` so it speaks the litGraph cache protocol.
    Lazy imports `gptcache`; install with `pip install gptcache`.

    The adapter is a thin shim — GPTCache's own similarity backends
    (FAISS, Chroma, etc.) and embedding configs are passed through
    unchanged. See <https://github.com/zilliztech/GPTCache>.

    Example:

        from gptcache import Cache
        from gptcache.adapter.api import init_similar_cache
        from litgraph.cache_extras import GPTCacheAdapter
        from litgraph.cache import CachedChatModel

        gc = Cache()
        init_similar_cache(cache_obj=gc)
        cache = GPTCacheAdapter(gc)
        m = CachedChatModel(real_model, cache=cache)
    """

    def __init__(self, gptcache_obj: Any) -> None:
        try:
            import gptcache  # type: ignore[import-not-found]  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "gptcache not installed. "
                "Run `pip install gptcache` to use this adapter."
            ) from e
        self._cache = gptcache_obj

    def get(self, key: str) -> str | None:
        try:
            from gptcache.adapter.api import get as _gc_get  # type: ignore[import-not-found]
            return _gc_get(key, cache_obj=self._cache)
        except Exception:
            return None

    def put(self, key: str, value: str) -> None:
        try:
            from gptcache.adapter.api import put as _gc_put  # type: ignore[import-not-found]
            _gc_put(key, value, cache_obj=self._cache)
        except Exception:
            # Silent failure on the cache layer — never let cache
            # writes break the request path.
            pass

    def clear(self) -> None:
        try:
            self._cache.flush()
        except (AttributeError, TypeError):
            pass

//! Response cache for `ChatModel` invocations.
//!
//! The cache is keyed by a blake3 hash of `(model, serialized messages, serialized options)`.
//! Hash is deterministic and stable across process restarts.
//!
//! Backends:
//! - `MemoryCache` — moka LRU, TTL, zero disk. Fast.
//! - `SqliteCache` — `rusqlite` behind `tokio::task::spawn_blocking`. Durable.
//!
//! Wrap any `ChatModel` with `CachedModel` to get transparent caching.

pub mod backend;
pub mod cached_model;
pub mod key;
pub mod semantic;
pub mod embedding_cache;

pub use backend::{Cache, MemoryCache, SqliteCache};
pub use cached_model::CachedModel;
pub use key::cache_key;
pub use semantic::{SemanticCache, SemanticCachedModel};
pub use embedding_cache::{
    embedding_key, CachedEmbeddings, EmbeddingCache, MemoryEmbeddingCache, SqliteEmbeddingCache,
};

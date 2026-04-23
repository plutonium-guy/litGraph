//! Cache layer for `Embeddings`. Different trait + storage from the existing
//! `Cache<ChatResponse>` because the value shape (`Vec<f32>`) and the keying
//! semantics (per-text, not per-conversation) are different enough that
//! sharing one trait would force generics everywhere or pointless boxing.
//!
//! The expensive thing on bulk indexing is "embed 100k documents"; if your
//! pipeline reruns on slightly-modified corpora, ~95% of those embeddings
//! are byte-identical to last run. Caching saves the API bill.
//!
//! Keying: `blake3((model, text))` → 32-byte hash → hex string. Stable across
//! restarts. Different model names produce different keys, so swapping
//! providers cleanly invalidates without manual purge.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use blake3;
use litgraph_core::{Embeddings, Error, Result};
use moka::future::Cache as MokaCache;
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};

#[async_trait]
pub trait EmbeddingCache: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<Vec<f32>>>;
    async fn put(&self, key: &str, value: Vec<f32>) -> Result<()>;
    async fn invalidate(&self, key: &str) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}

/// Moka-backed in-memory LRU. Pair with TTL for ephemeral pipelines; omit
/// TTL when the corpus is stable and you want indefinite caching.
pub struct MemoryEmbeddingCache {
    inner: MokaCache<String, Arc<Vec<f32>>>,
}

impl MemoryEmbeddingCache {
    pub fn new(max_capacity: u64) -> Self {
        Self { inner: MokaCache::builder().max_capacity(max_capacity).build() }
    }
    pub fn with_ttl(max_capacity: u64, ttl: Duration) -> Self {
        Self {
            inner: MokaCache::builder()
                .max_capacity(max_capacity)
                .time_to_live(ttl)
                .build(),
        }
    }
    pub fn entry_count(&self) -> u64 { self.inner.entry_count() }
}

impl Default for MemoryEmbeddingCache {
    fn default() -> Self { Self::new(100_000) }
}

#[async_trait]
impl EmbeddingCache for MemoryEmbeddingCache {
    async fn get(&self, key: &str) -> Result<Option<Vec<f32>>> {
        Ok(self.inner.get(key).await.map(|a| (*a).clone()))
    }
    async fn put(&self, key: &str, value: Vec<f32>) -> Result<()> {
        self.inner.insert(key.to_string(), Arc::new(value)).await;
        Ok(())
    }
    async fn invalidate(&self, key: &str) -> Result<()> {
        self.inner.invalidate(key).await;
        Ok(())
    }
    async fn clear(&self) -> Result<()> {
        self.inner.invalidate_all();
        // Wait briefly for the eviction tasks to drain so an immediate
        // entry_count() check sees the cleared state.
        self.inner.run_pending_tasks().await;
        Ok(())
    }
}

/// Build the deterministic cache key for `(model, text)`.
pub fn embedding_key(model: &str, text: &str) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(model.as_bytes());
    hasher.update(b"\0");
    hasher.update(text.as_bytes());
    hasher.finalize().to_hex().to_string()
}

/// Wraps any `Embeddings` provider with read-through caching. Batch
/// `embed_documents` is the hot path: cache hits are returned directly,
/// misses are batched into a single inner call so we don't make N API
/// requests for N partial misses. Result order is preserved.
///
/// We catch the case of `Error::other("...")` by NOT inserting on inner
/// failures — only successful embeddings get cached. (Error responses are
/// transient; we don't want a 429 to poison the next 5 minutes of runs.)
pub struct CachedEmbeddings {
    inner: Arc<dyn Embeddings>,
    cache: Arc<dyn EmbeddingCache>,
    /// Captured at construction so cache keys see a stable model name even
    /// if the inner impl's `name()` becomes dynamic later.
    model_name: String,
}

impl CachedEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, cache: Arc<dyn EmbeddingCache>) -> Self {
        let model_name = inner.name().to_string();
        Self { inner, cache, model_name }
    }
}

#[async_trait]
impl Embeddings for CachedEmbeddings {
    fn name(&self) -> &str { &self.model_name }
    fn dimensions(&self) -> usize { self.inner.dimensions() }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let key = embedding_key(&self.model_name, text);
        if let Some(v) = self.cache.get(&key).await? {
            return Ok(v);
        }
        let v = self.inner.embed_query(text).await?;
        self.cache.put(&key, v.clone()).await?;
        Ok(v)
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() { return Ok(vec![]); }

        // Pass 1: probe the cache for every text in parallel. (Sequential is
        // fine too — Moka is fast — but parallel matches the spirit of the
        // batch API and won't pessimize a SQLite-backed impl.)
        let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
        let mut miss_idx: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let key = embedding_key(&self.model_name, text);
            match self.cache.get(&key).await? {
                Some(v) => out.push(Some(v)),
                None => {
                    out.push(None);
                    miss_idx.push(i);
                    miss_texts.push(text.clone());
                }
            }
        }

        // Pass 2: ONE inner call for all misses (don't fan out N requests).
        if !miss_texts.is_empty() {
            let fresh = self.inner.embed_documents(&miss_texts).await?;
            if fresh.len() != miss_texts.len() {
                return Err(Error::other(format!(
                    "CachedEmbeddings: inner returned {} embeddings for {} texts",
                    fresh.len(), miss_texts.len()
                )));
            }
            for (slot, (text, vec)) in miss_idx.iter().zip(miss_texts.iter().zip(fresh.into_iter())) {
                let key = embedding_key(&self.model_name, text);
                self.cache.put(&key, vec.clone()).await?;
                out[*slot] = Some(vec);
            }
        }

        // out is now Some(_) at every slot.
        Ok(out.into_iter().map(|v| v.expect("filled above")).collect())
    }
}

/// SQLite-backed embedding cache. Durable across process restarts — if your
/// indexing job crashes or reruns tomorrow, cached embeddings survive.
/// Storage: one row per (key, blob, ts_ms); vector is packed little-endian
/// f32s (4 bytes/dim — a 3072-dim embedding is 12 KiB, vs ~20 KiB of
/// bincode-encoded JSON ~Vec). SQLite work runs in `tokio::task::spawn_blocking`
/// so the runtime isn't blocked.
pub struct SqliteEmbeddingCache {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteEmbeddingCache {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| Error::other(e.to_string()))?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "synchronous", "NORMAL").ok();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                ts_ms INTEGER NOT NULL
            );
            "#,
        )
        .map_err(|e| Error::other(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| Error::other(e.to_string()))?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS embedding_cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                ts_ms INTEGER NOT NULL
            );
            "#,
        )
        .map_err(|e| Error::other(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }

    pub fn len(&self) -> Result<u64> {
        let g = self.conn.lock();
        let n: i64 = g.query_row("SELECT COUNT(*) FROM embedding_cache", [], |r| r.get(0))
            .map_err(|e| Error::other(e.to_string()))?;
        Ok(n.max(0) as u64)
    }
}

fn pack_f32(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

fn unpack_f32(bytes: &[u8]) -> Result<Vec<f32>> {
    if bytes.len() % 4 != 0 {
        return Err(Error::other(format!(
            "embedding blob not a multiple of 4 bytes: got {}", bytes.len()
        )));
    }
    let mut out = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok(out)
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[async_trait]
impl EmbeddingCache for SqliteEmbeddingCache {
    async fn get(&self, key: &str) -> Result<Option<Vec<f32>>> {
        let conn = self.conn.clone();
        let k = key.to_string();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            let mut stmt = g
                .prepare("SELECT value FROM embedding_cache WHERE key = ?1")
                .map_err(|e| Error::other(e.to_string()))?;
            let blob: Option<Vec<u8>> = stmt
                .query_row(params![k], |r| r.get::<_, Vec<u8>>(0))
                .optional()
                .map_err(|e| Error::other(e.to_string()))?;
            match blob {
                Some(b) => Ok(Some(unpack_f32(&b)?)),
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))?
    }

    async fn put(&self, key: &str, value: Vec<f32>) -> Result<()> {
        let conn = self.conn.clone();
        let k = key.to_string();
        let bytes = pack_f32(&value);
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            g.execute(
                "INSERT OR REPLACE INTO embedding_cache (key, value, ts_ms) VALUES (?1, ?2, ?3)",
                params![k, bytes, now_ms() as i64],
            )
            .map_err(|e| Error::other(e.to_string()))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    async fn invalidate(&self, key: &str) -> Result<()> {
        let conn = self.conn.clone();
        let k = key.to_string();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            g.execute("DELETE FROM embedding_cache WHERE key = ?1", params![k])
                .map_err(|e| Error::other(e.to_string()))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            g.execute("DELETE FROM embedding_cache", [])
                .map_err(|e| Error::other(e.to_string()))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Inner that counts inner-API calls so we can verify cache hit rate.
    struct CountingInner {
        calls_query: AtomicU32,
        calls_batch_total_texts: AtomicU32,
    }
    #[async_trait]
    impl Embeddings for CountingInner {
        fn name(&self) -> &str { "counting" }
        fn dimensions(&self) -> usize { 4 }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            self.calls_query.fetch_add(1, Ordering::SeqCst);
            Ok(text_to_vec(text))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls_batch_total_texts.fetch_add(texts.len() as u32, Ordering::SeqCst);
            Ok(texts.iter().map(|t| text_to_vec(t)).collect())
        }
    }
    fn text_to_vec(t: &str) -> Vec<f32> {
        vec![t.len() as f32, t.chars().count() as f32, t.starts_with('a') as u8 as f32, 1.0]
    }

    #[test]
    fn embedding_key_is_stable_and_model_aware() {
        let k1 = embedding_key("gpt-emb", "hello");
        let k2 = embedding_key("gpt-emb", "hello");
        assert_eq!(k1, k2);
        // Different model → different key (so swapping providers invalidates cleanly).
        let k3 = embedding_key("voyage-3", "hello");
        assert_ne!(k1, k3);
        // Different text → different key.
        let k4 = embedding_key("gpt-emb", "world");
        assert_ne!(k1, k4);
    }

    #[tokio::test]
    async fn embed_query_caches_on_second_call() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(MemoryEmbeddingCache::default());
        let wrapped = CachedEmbeddings::new(inner.clone(), cache);

        let v1 = wrapped.embed_query("hello").await.unwrap();
        let v2 = wrapped.embed_query("hello").await.unwrap();
        assert_eq!(v1, v2);
        // Inner should have been called exactly once across the two queries.
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn embed_documents_partitions_hits_and_misses() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(MemoryEmbeddingCache::default());
        let wrapped = CachedEmbeddings::new(inner.clone(), cache);

        // Warm: cache "alpha" + "beta".
        wrapped.embed_documents(&["alpha".into(), "beta".into()]).await.unwrap();
        assert_eq!(inner.calls_batch_total_texts.load(Ordering::SeqCst), 2);

        // Mixed batch: "alpha" (cached), "gamma" (miss), "beta" (cached), "delta" (miss).
        let out = wrapped.embed_documents(&[
            "alpha".into(), "gamma".into(), "beta".into(), "delta".into(),
        ]).await.unwrap();
        assert_eq!(out.len(), 4);
        // Inner saw ONLY the 2 misses, not all 4.
        let total_after = inner.calls_batch_total_texts.load(Ordering::SeqCst);
        assert_eq!(total_after, 2 + 2, "expected +2 (misses only), got +{}", total_after - 2);

        // Order preserved: out[0] should match "alpha"'s vector.
        assert_eq!(out[0], text_to_vec("alpha"));
        assert_eq!(out[1], text_to_vec("gamma"));
        assert_eq!(out[2], text_to_vec("beta"));
        assert_eq!(out[3], text_to_vec("delta"));
    }

    #[tokio::test]
    async fn empty_texts_short_circuits_no_inner_call() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(MemoryEmbeddingCache::default());
        let wrapped = CachedEmbeddings::new(inner.clone(), cache);
        let out = wrapped.embed_documents(&[]).await.unwrap();
        assert!(out.is_empty());
        assert_eq!(inner.calls_batch_total_texts.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn cache_clear_forces_fresh_calls() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(MemoryEmbeddingCache::default());
        let wrapped = CachedEmbeddings::new(inner.clone(), cache.clone());

        wrapped.embed_query("hi").await.unwrap();
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 1);

        cache.clear().await.unwrap();
        wrapped.embed_query("hi").await.unwrap();
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn ttl_eviction_path() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(
            MemoryEmbeddingCache::with_ttl(100, Duration::from_millis(10))
        );
        let wrapped = CachedEmbeddings::new(inner.clone(), cache);

        wrapped.embed_query("hi").await.unwrap();
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 1);
        // Wait for TTL to expire then re-query.
        tokio::time::sleep(Duration::from_millis(50)).await;
        wrapped.embed_query("hi").await.unwrap();
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 2);
    }

    // ── SqliteEmbeddingCache ──────────────────────────────────────────────

    #[test]
    fn pack_unpack_roundtrip_preserves_f32_bits() {
        let original = vec![0.0f32, 1.5, -3.14, f32::INFINITY, f32::NEG_INFINITY, 1e-30];
        let packed = pack_f32(&original);
        assert_eq!(packed.len(), original.len() * 4);
        let restored = unpack_f32(&packed).unwrap();
        // Bit-exact — we didn't pass through any floating-point math.
        assert_eq!(restored, original);
    }

    #[test]
    fn unpack_rejects_non_multiple_of_4_bytes() {
        let err = unpack_f32(&[0u8, 0, 0]).unwrap_err();
        assert!(format!("{err}").contains("multiple of 4"));
    }

    #[tokio::test]
    async fn sqlite_get_put_roundtrip() {
        let cache = SqliteEmbeddingCache::in_memory().unwrap();
        assert!(cache.get("absent").await.unwrap().is_none());
        cache.put("k1", vec![0.1, 0.2, 0.3, 0.4]).await.unwrap();
        let got = cache.get("k1").await.unwrap().unwrap();
        assert_eq!(got, vec![0.1f32, 0.2, 0.3, 0.4]);
    }

    #[tokio::test]
    async fn sqlite_put_overwrites_existing_key() {
        let cache = SqliteEmbeddingCache::in_memory().unwrap();
        cache.put("k1", vec![1.0]).await.unwrap();
        cache.put("k1", vec![2.0, 2.0]).await.unwrap();
        assert_eq!(cache.get("k1").await.unwrap().unwrap(), vec![2.0f32, 2.0]);
        assert_eq!(cache.len().unwrap(), 1);
    }

    #[tokio::test]
    async fn sqlite_invalidate_clears_single_key() {
        let cache = SqliteEmbeddingCache::in_memory().unwrap();
        cache.put("a", vec![1.0]).await.unwrap();
        cache.put("b", vec![2.0]).await.unwrap();
        cache.invalidate("a").await.unwrap();
        assert!(cache.get("a").await.unwrap().is_none());
        assert!(cache.get("b").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn sqlite_clear_empties_table() {
        let cache = SqliteEmbeddingCache::in_memory().unwrap();
        for i in 0..5 {
            cache.put(&format!("k{i}"), vec![i as f32]).await.unwrap();
        }
        assert_eq!(cache.len().unwrap(), 5);
        cache.clear().await.unwrap();
        assert_eq!(cache.len().unwrap(), 0);
    }

    #[tokio::test]
    async fn sqlite_durability_across_connections_via_file() {
        // Open → write → drop connection → reopen same file → read-back.
        // This is the cross-process durability guarantee, simulated locally.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("emb.db");
        {
            let c1 = SqliteEmbeddingCache::open(&path).unwrap();
            c1.put("persisted", vec![0.25, 0.5, 0.75]).await.unwrap();
        }
        let c2 = SqliteEmbeddingCache::open(&path).unwrap();
        let got = c2.get("persisted").await.unwrap().unwrap();
        assert_eq!(got, vec![0.25f32, 0.5, 0.75]);
    }

    #[tokio::test]
    async fn cached_embeddings_works_over_sqlite_backend() {
        let inner = Arc::new(CountingInner {
            calls_query: AtomicU32::new(0),
            calls_batch_total_texts: AtomicU32::new(0),
        });
        let cache: Arc<dyn EmbeddingCache> = Arc::new(SqliteEmbeddingCache::in_memory().unwrap());
        let wrapped = CachedEmbeddings::new(inner.clone(), cache);
        wrapped.embed_query("foo").await.unwrap();
        wrapped.embed_query("foo").await.unwrap();
        assert_eq!(inner.calls_query.load(Ordering::SeqCst), 1,
            "sqlite cache should prevent 2nd inner call just like memory");
    }
}

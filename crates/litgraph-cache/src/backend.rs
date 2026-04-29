use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{ChatResponse, Error, Result};
use moka::future::Cache as MokaCache;
use parking_lot::Mutex;
use rusqlite::{Connection, OptionalExtension, params};

#[async_trait]
pub trait Cache: Send + Sync {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>>;
    async fn put(&self, key: &str, value: ChatResponse) -> Result<()>;
    async fn invalidate(&self, key: &str) -> Result<()>;
    async fn clear(&self) -> Result<()>;
}

/// Moka-backed LRU in-memory cache with optional TTL.
pub struct MemoryCache {
    inner: MokaCache<String, Arc<ChatResponse>>,
}

impl MemoryCache {
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
}

impl Default for MemoryCache {
    fn default() -> Self { Self::new(10_000) }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>> {
        Ok(self.inner.get(key).await.map(|a| (*a).clone()))
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<()> {
        self.inner.insert(key.to_string(), Arc::new(value)).await;
        Ok(())
    }

    async fn invalidate(&self, key: &str) -> Result<()> {
        self.inner.invalidate(key).await;
        Ok(())
    }

    async fn clear(&self) -> Result<()> {
        self.inner.invalidate_all();
        self.inner.run_pending_tasks().await;
        Ok(())
    }
}

/// SQLite-backed cache — durable across restarts. Binary-serialized `ChatResponse`
/// via bincode for compactness.
pub struct SqliteCache {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteCache {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| Error::other(e.to_string()))?;
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "synchronous", "NORMAL").ok();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS llm_cache (
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
            CREATE TABLE IF NOT EXISTS llm_cache (
                key TEXT PRIMARY KEY,
                value BLOB NOT NULL,
                ts_ms INTEGER NOT NULL
            );
            "#,
        )
        .map_err(|e| Error::other(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)) })
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[async_trait]
impl Cache for SqliteCache {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>> {
        let conn = self.conn.clone();
        let k = key.to_string();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            let mut stmt = g
                .prepare("SELECT value FROM llm_cache WHERE key = ?1")
                .map_err(|e| Error::other(e.to_string()))?;
            let blob: Option<Vec<u8>> = stmt
                .query_row(params![k], |r| r.get::<_, Vec<u8>>(0))
                .optional()
                .map_err(|e| Error::other(e.to_string()))?;
            match blob {
                Some(b) => Ok(Some(bincode::deserialize::<ChatResponse>(&b)
                    .map_err(|e| Error::other(e.to_string()))?)),
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))?
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<()> {
        let conn = self.conn.clone();
        let k = key.to_string();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock();
            let bytes = bincode::serialize(&value).map_err(|e| Error::other(e.to_string()))?;
            g.execute(
                "INSERT OR REPLACE INTO llm_cache (key, value, ts_ms) VALUES (?1, ?2, ?3)",
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
            g.execute("DELETE FROM llm_cache WHERE key = ?1", params![k])
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
            g.execute("DELETE FROM llm_cache", [])
                .map_err(|e| Error::other(e.to_string()))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }
}

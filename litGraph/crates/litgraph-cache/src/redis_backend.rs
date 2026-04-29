//! Redis-backed cache. Cross-process / multi-instance LLM-response cache
//! for distributed agent fleets. Pairs with the existing in-memory and
//! SQLite backends — pick by deployment shape:
//!
//! - **`MemoryCache`**: single-process, transient. Fast.
//! - **`SqliteCache`**: single-host, durable across restarts.
//! - **`RedisCache`** (this): multi-host, durable, shared across all
//!   workers. Canonical for serverless / autoscaled agent deployments.
//!
//! # Storage layout
//!
//! Each cache entry: `litgraph:cache:{key}` → bincode-serialized
//! `ChatResponse`. Optional TTL via Redis EXPIRE. Namespace prefix
//! protects against collision with other litGraph subsystems
//! (checkpointer uses `litgraph:cp:{thread_id}`).
//!
//! # Connection
//!
//! Uses `ConnectionManager` (auto-reconnect on network blip). Connect via
//! `RedisCache::connect("redis://127.0.0.1:6379/0")`. Pool/cluster setups
//! work — pass the URL the user's stack uses.

use async_trait::async_trait;
use litgraph_core::{ChatResponse, Error, Result};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use std::time::Duration;

use crate::backend::Cache;

const KEY_PREFIX: &str = "litgraph:cache:";

fn redis_key(key: &str) -> String { format!("{KEY_PREFIX}{key}") }

fn err<E: std::fmt::Display>(e: E) -> Error {
    Error::other(format!("redis cache: {e}"))
}

pub struct RedisCache {
    conn: ConnectionManager,
    /// Optional TTL applied on every `put`. `None` = no expiry (entries
    /// live until invalidated or the Redis instance is flushed).
    ttl: Option<Duration>,
}

impl RedisCache {
    /// Connect to Redis via URL (e.g. `redis://127.0.0.1:6379/0`).
    pub async fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(err)?;
        let conn = ConnectionManager::new(client).await.map_err(err)?;
        Ok(Self { conn, ttl: None })
    }

    /// Wrap an existing connection manager (sharing connections across
    /// litGraph subsystems — checkpointer + cache + ...).
    pub fn from_manager(conn: ConnectionManager) -> Self {
        Self { conn, ttl: None }
    }

    /// Set a default TTL applied to every `put`. Without this, entries
    /// never expire (useful for content-addressable caches where the key
    /// already encodes immutability).
    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }
}

#[async_trait]
impl Cache for RedisCache {
    async fn get(&self, key: &str) -> Result<Option<ChatResponse>> {
        let mut conn = self.conn.clone();
        let bytes: Option<Vec<u8>> = conn.get(redis_key(key)).await.map_err(err)?;
        match bytes {
            Some(b) => {
                let resp = bincode::deserialize::<ChatResponse>(&b).map_err(err)?;
                Ok(Some(resp))
            }
            None => Ok(None),
        }
    }

    async fn put(&self, key: &str, value: ChatResponse) -> Result<()> {
        let mut conn = self.conn.clone();
        let bytes = bincode::serialize(&value).map_err(err)?;
        let rk = redis_key(key);
        match self.ttl {
            Some(ttl) => {
                let secs = ttl.as_secs().max(1);
                let _: () = conn.set_ex(rk, bytes, secs).await.map_err(err)?;
            }
            None => {
                let _: () = conn.set(rk, bytes).await.map_err(err)?;
            }
        }
        Ok(())
    }

    async fn invalidate(&self, key: &str) -> Result<()> {
        let mut conn = self.conn.clone();
        let _: () = conn.del(redis_key(key)).await.map_err(err)?;
        Ok(())
    }

    /// Drop ALL litGraph cache entries (matches our prefix). Does NOT
    /// FLUSHDB — preserves checkpointer data + other tenants' keys.
    /// Uses SCAN + DEL in batches; safe on large caches.
    async fn clear(&self) -> Result<()> {
        let mut conn = self.conn.clone();
        let pattern = format!("{KEY_PREFIX}*");
        let mut cursor: u64 = 0;
        loop {
            let (next_cursor, keys): (u64, Vec<String>) = redis::cmd("SCAN")
                .arg(cursor)
                .arg("MATCH")
                .arg(&pattern)
                .arg("COUNT")
                .arg(500)
                .query_async(&mut conn)
                .await
                .map_err(err)?;
            if !keys.is_empty() {
                let _: () = conn.del(keys).await.map_err(err)?;
            }
            cursor = next_cursor;
            if cursor == 0 {
                break;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{FinishReason, TokenUsage};
    use litgraph_core::Message;

    fn fake_response(text: &str) -> ChatResponse {
        ChatResponse {
            message: Message::assistant(text),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage::default(),
            model: "test-model".into(),
        }
    }

    /// Skip if no Redis available — these are integration tests.
    /// Set REDIS_URL=redis://localhost:6379/15 to run them.
    fn try_connect() -> Option<String> {
        std::env::var("REDIS_URL").ok()
    }

    #[tokio::test]
    async fn put_get_roundtrips_chat_response() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap();
        cache.clear().await.unwrap();
        let resp = fake_response("hello");
        cache.put("k1", resp.clone()).await.unwrap();
        let got = cache.get("k1").await.unwrap().expect("cached");
        assert_eq!(got.message.text_content(), "hello");
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap();
        cache.clear().await.unwrap();
        let got = cache.get("nonexistent-key").await.unwrap();
        assert!(got.is_none());
    }

    #[tokio::test]
    async fn invalidate_removes_specific_key() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap();
        cache.clear().await.unwrap();
        cache.put("k1", fake_response("v1")).await.unwrap();
        cache.put("k2", fake_response("v2")).await.unwrap();
        cache.invalidate("k1").await.unwrap();
        assert!(cache.get("k1").await.unwrap().is_none());
        assert!(cache.get("k2").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn clear_removes_all_litgraph_keys_only() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap();
        let mut conn = cache.conn.clone();
        // Plant a non-litgraph key.
        let _: () = conn.set("other:foreign", "should-survive").await.unwrap();

        cache.clear().await.unwrap();
        cache.put("k1", fake_response("v1")).await.unwrap();
        cache.clear().await.unwrap();

        // Cache cleared.
        assert!(cache.get("k1").await.unwrap().is_none());
        // Foreign key untouched.
        let foreign: Option<String> = conn.get("other:foreign").await.unwrap();
        assert_eq!(foreign.as_deref(), Some("should-survive"));

        // Cleanup.
        let _: () = conn.del("other:foreign").await.unwrap();
    }

    #[tokio::test]
    async fn ttl_expires_entries() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap()
            .with_ttl(Duration::from_secs(1));
        cache.clear().await.unwrap();
        cache.put("ttl-key", fake_response("v")).await.unwrap();
        // Right after put: present.
        assert!(cache.get("ttl-key").await.unwrap().is_some());
        // After TTL: expired.
        tokio::time::sleep(Duration::from_secs(2)).await;
        assert!(cache.get("ttl-key").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn keys_namespaced_under_litgraph_cache_prefix() {
        let url = match try_connect() { Some(u) => u, None => return };
        let cache = RedisCache::connect(&url).await.unwrap();
        cache.clear().await.unwrap();
        cache.put("my-key", fake_response("v")).await.unwrap();
        let mut conn = cache.conn.clone();
        // Direct lookup with the namespaced key works.
        let bytes: Option<Vec<u8>> = conn.get("litgraph:cache:my-key").await.unwrap();
        assert!(bytes.is_some());
        // Bare key (without prefix) should NOT exist.
        let bare: Option<Vec<u8>> = conn.get("my-key").await.unwrap();
        assert!(bare.is_none());
    }

    #[test]
    fn redis_key_format_includes_prefix() {
        assert_eq!(redis_key("foo"), "litgraph:cache:foo");
        assert_eq!(redis_key(""), "litgraph:cache:");
    }
}

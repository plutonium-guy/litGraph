//! Redis-backed conversation history.
//!
//! Hot / ephemeral counterpart to `PostgresChatHistory` and
//! `SqliteChatHistory`. Same surface (`append`, `append_all`, `messages`,
//! `clear`, `delete_session`, `set_system`, `message_count`,
//! `list_sessions`) so callers can swap backends with one line.
//!
//! # Why Redis-backed history is its own crate
//!
//! Redis is the right call when:
//! - chat history is short-lived (per-session TTL), discarded after the
//!   user disconnects, and write throughput dominates;
//! - you want per-key TTL ([`RedisChatHistory::set_ttl`]) so abandoned
//!   sessions self-evict;
//! - you already run Redis for caching / pub-sub / queues and don't want
//!   to add Postgres for chat alone.
//!
//! Postgres is the right call when you need durable history months later
//! (analytics, audit, fine-tuning datasets) — Redis is RAM-bound. For a
//! mixed workload, push hot sessions to Redis with a TTL and snapshot
//! to Postgres on session close.
//!
//! # Storage layout
//!
//! Per session_id `s`:
//!
//! | key                      | type   | role                                    |
//! |--------------------------|--------|-----------------------------------------|
//! | `litgraph:msgs:{s}`      | LIST   | Messages, RPUSHed in chronological order|
//! | `litgraph:pin:{s}`       | STRING | System pin (one-per-session, REPLACE)   |
//! | `litgraph:sessions`      | SET    | All session ids in this store           |
//!
//! Sequencing is implicit in LIST insertion order — no MAX(seq) lookup
//! like the SQL backends; that's ~one round-trip removed per append.
//!
//! # Connection
//!
//! Uses `redis::aio::ConnectionManager` (workspace-pinned) so the
//! transport auto-reconnects after a network blip. Construct via
//! [`RedisChatHistory::connect`] (URL → manager) or [`from_manager`] to
//! share an existing manager with the checkpointer / cache crates.
//!
//! ```no_run
//! # async fn ex() -> litgraph_core::Result<()> {
//! use litgraph_memory_redis::RedisChatHistory;
//! use litgraph_core::Message;
//!
//! let h = RedisChatHistory::connect("redis://127.0.0.1:6379/0", "user-42").await?;
//! h.set_system(Some(Message::system("you are helpful"))).await?;
//! h.append(Message::user("hi")).await?;
//! let msgs = h.messages().await?;
//! # Ok(()) }
//! ```

use std::time::{SystemTime, UNIX_EPOCH};

use litgraph_core::{Error, Message, Result};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use tracing::debug;

const SESSIONS_KEY: &str = "litgraph:sessions";

fn msgs_key(session_id: &str) -> String {
    format!("litgraph:msgs:{session_id}")
}
fn pin_key(session_id: &str) -> String {
    format!("litgraph:pin:{session_id}")
}

fn err<E: std::fmt::Display>(stage: &'static str, e: E) -> Error {
    Error::other(format!("redis_chat {stage}: {e}"))
}

/// Durable-ish conversation history backed by Redis. Cheap to clone —
/// `ConnectionManager` is internally `Arc`'d.
#[derive(Clone)]
pub struct RedisChatHistory {
    conn: ConnectionManager,
    session_id: String,
    /// Per-session TTL in seconds; `None` = no expiry. Applied on every
    /// write so an active conversation keeps refreshing the timer.
    ttl_secs: Option<u64>,
}

impl RedisChatHistory {
    /// Connect via Redis URL (`redis://host:port/db`) for the given
    /// session. Auto-reconnects on transport failure.
    pub async fn connect(url: &str, session_id: impl Into<String>) -> Result<Self> {
        let client = redis::Client::open(url).map_err(|e| err("open", e))?;
        let conn = ConnectionManager::new(client)
            .await
            .map_err(|e| err("connect", e))?;
        debug!("redis_chat connected");
        Ok(Self {
            conn,
            session_id: session_id.into(),
            ttl_secs: None,
        })
    }

    /// Reuse an existing connection manager — preferred when the app
    /// already pools Redis for cache/checkpoints.
    pub fn from_manager(conn: ConnectionManager, session_id: impl Into<String>) -> Self {
        Self {
            conn,
            session_id: session_id.into(),
            ttl_secs: None,
        }
    }

    /// Set a per-session TTL (seconds). Refreshed on every write — so an
    /// abandoned session evicts after `ttl` of inactivity.
    pub fn with_ttl(mut self, ttl_secs: u64) -> Self {
        self.ttl_secs = Some(ttl_secs);
        self
    }

    /// Replace TTL after construction. Pass `None` to remove (and call
    /// `clear_ttl` to actually drop the EXPIRE on existing keys).
    pub fn set_ttl(&mut self, ttl_secs: Option<u64>) {
        self.ttl_secs = ttl_secs;
    }

    /// Same backing manager, addressed at a different session id. Cheap.
    /// TTL setting is inherited.
    pub fn session(&self, session_id: impl Into<String>) -> Self {
        Self {
            conn: self.conn.clone(),
            session_id: session_id.into(),
            ttl_secs: self.ttl_secs,
        }
    }

    pub fn session_id(&self) -> &str {
        &self.session_id
    }

    /// Append one message. O(1) — no MAX-seq round-trip needed; LIST
    /// preserves order natively.
    pub async fn append(&self, m: Message) -> Result<()> {
        let json = serde_json::to_string(&m).map_err(|e| err("serialize", e))?;
        let mut conn = self.conn.clone();
        let mkey = msgs_key(&self.session_id);
        let _: () = conn
            .rpush(&mkey, json)
            .await
            .map_err(|e| err("rpush", e))?;
        let _: () = conn
            .sadd(SESSIONS_KEY, &self.session_id)
            .await
            .map_err(|e| err("sadd", e))?;
        self.refresh_ttl(&mut conn, &mkey).await?;
        Ok(())
    }

    /// Bulk append — RPUSH variadic. One round-trip for the whole batch
    /// (vs. one-per-message for `append`).
    pub async fn append_all(&self, msgs: Vec<Message>) -> Result<()> {
        if msgs.is_empty() {
            return Ok(());
        }
        let mut serialized = Vec::with_capacity(msgs.len());
        for m in &msgs {
            serialized.push(serde_json::to_string(m).map_err(|e| err("serialize", e))?);
        }
        let mut conn = self.conn.clone();
        let mkey = msgs_key(&self.session_id);
        let _: () = conn
            .rpush(&mkey, serialized)
            .await
            .map_err(|e| err("rpush_all", e))?;
        let _: () = conn
            .sadd(SESSIONS_KEY, &self.session_id)
            .await
            .map_err(|e| err("sadd", e))?;
        self.refresh_ttl(&mut conn, &mkey).await?;
        Ok(())
    }

    /// All messages for this session in chronological order. System pin
    /// (if set) prepended so the result is ready for `ChatModel::invoke`.
    pub async fn messages(&self) -> Result<Vec<Message>> {
        let mut conn = self.conn.clone();
        let mut out = Vec::new();

        // System pin first.
        let pin: Option<String> = conn
            .get(pin_key(&self.session_id))
            .await
            .map_err(|e| err("get_pin", e))?;
        if let Some(s) = pin {
            let m: Message = serde_json::from_str(&s).map_err(|e| err("pin_parse", e))?;
            out.push(m);
        }

        // Conversation in LIST order.
        let raw: Vec<String> = conn
            .lrange(msgs_key(&self.session_id), 0, -1)
            .await
            .map_err(|e| err("lrange", e))?;
        for s in raw {
            let m: Message = serde_json::from_str(&s).map_err(|e| err("msg_parse", e))?;
            out.push(m);
        }
        Ok(out)
    }

    /// Drop conversation messages. System pin preserved.
    pub async fn clear(&self) -> Result<()> {
        let mut conn = self.conn.clone();
        let _: () = conn
            .del(msgs_key(&self.session_id))
            .await
            .map_err(|e| err("del_msgs", e))?;
        Ok(())
    }

    /// Drop ALL data for this session — messages + pin + sessions-set entry.
    pub async fn delete_session(&self) -> Result<()> {
        let mut conn = self.conn.clone();
        let keys = [msgs_key(&self.session_id), pin_key(&self.session_id)];
        let _: () = conn.del(&keys[..]).await.map_err(|e| err("del", e))?;
        let _: () = conn
            .srem(SESSIONS_KEY, &self.session_id)
            .await
            .map_err(|e| err("srem", e))?;
        Ok(())
    }

    /// Set or replace the system pin. `None` = remove.
    pub async fn set_system(&self, m: Option<Message>) -> Result<()> {
        let mut conn = self.conn.clone();
        let pkey = pin_key(&self.session_id);
        match m {
            Some(msg) => {
                let json = serde_json::to_string(&msg).map_err(|e| err("serialize", e))?;
                let _: () = conn.set(&pkey, json).await.map_err(|e| err("set_pin", e))?;
                let _: () = conn
                    .sadd(SESSIONS_KEY, &self.session_id)
                    .await
                    .map_err(|e| err("sadd", e))?;
                if let Some(ttl) = self.ttl_secs {
                    let _: () = conn
                        .expire(&pkey, ttl as i64)
                        .await
                        .map_err(|e| err("expire_pin", e))?;
                }
            }
            None => {
                let _: () = conn.del(&pkey).await.map_err(|e| err("del_pin", e))?;
            }
        }
        Ok(())
    }

    /// Total messages for this session (excludes the system pin).
    pub async fn message_count(&self) -> Result<usize> {
        let mut conn = self.conn.clone();
        let n: usize = conn
            .llen(msgs_key(&self.session_id))
            .await
            .map_err(|e| err("llen", e))?;
        Ok(n)
    }

    /// All distinct session ids in this store.
    ///
    /// Implementation note: we maintain a `litgraph:sessions` SET on
    /// every write. SCAN-based discovery would also work but blends with
    /// unrelated keys in shared Redis instances; a dedicated SET is
    /// cleaner and O(1) to query.
    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let mut conn = self.conn.clone();
        let v: Vec<String> = conn
            .smembers(SESSIONS_KEY)
            .await
            .map_err(|e| err("smembers", e))?;
        Ok(v)
    }

    /// Drop the EXPIRE on this session's keys. Use after `set_ttl(None)`
    /// if you want existing keys to become permanent (otherwise the old
    /// TTL keeps ticking until the next write refreshes — or doesn't).
    pub async fn clear_ttl(&self) -> Result<()> {
        let mut conn = self.conn.clone();
        // PERSIST returns 1 if a TTL was removed, 0 otherwise — both fine.
        let _: i64 = conn
            .persist(msgs_key(&self.session_id))
            .await
            .map_err(|e| err("persist_msgs", e))?;
        let _: i64 = conn
            .persist(pin_key(&self.session_id))
            .await
            .map_err(|e| err("persist_pin", e))?;
        Ok(())
    }

    async fn refresh_ttl(&self, conn: &mut ConnectionManager, mkey: &str) -> Result<()> {
        if let Some(ttl) = self.ttl_secs {
            let _: () = conn
                .expire(mkey, ttl as i64)
                .await
                .map_err(|e| err("expire_msgs", e))?;
            // Also refresh the pin if one exists; EXPIRE on a missing key
            // is a no-op (returns 0), so we don't need to check first.
            let _: () = conn
                .expire(pin_key(&self.session_id), ttl as i64)
                .await
                .map_err(|e| err("expire_pin", e))?;
        }
        Ok(())
    }
}

#[allow(dead_code)]
fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    //! Integration tests — gated on a live Redis. Run with:
    //! ```sh
    //! REDIS_URL=redis://127.0.0.1:6379/15 cargo test -p litgraph-memory-redis
    //! ```
    //! DB 15 is used to keep them away from production data. Each test
    //! uses a unique session id so tests can run in parallel without
    //! cross-contamination.

    use super::*;
    use litgraph_core::Role;

    fn try_url() -> Option<String> {
        std::env::var("REDIS_URL").ok()
    }
    /// Collision-proof session id. nanos alone collide when multiple
    /// tokio tests start in the same nanosecond on a fast machine; a
    /// per-process atomic counter resolves the tie.
    fn unique_session() -> String {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        format!("test-{nanos:x}-{n:x}")
    }

    #[tokio::test]
    async fn append_and_messages_roundtrips() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.append(Message::user("hello")).await.unwrap();
        h.append(Message::assistant("hi back")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::User);
        assert_eq!(msgs[1].role, Role::Assistant);
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn system_pin_prepends_in_messages() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("be brief"))).await.unwrap();
        h.append(Message::user("q")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn append_all_preserves_order() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.append_all(vec![
            Message::user("a"),
            Message::user("b"),
            Message::user("c"),
        ])
        .await
        .unwrap();
        let msgs = h.messages().await.unwrap();
        let texts: Vec<String> = msgs.iter().map(|m| m.text_content()).collect();
        assert_eq!(texts, vec!["a".to_string(), "b".to_string(), "c".to_string()]);
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn clear_keeps_pin() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("S"))).await.unwrap();
        h.append(Message::user("u")).await.unwrap();
        h.clear().await.unwrap();
        let msgs = h.messages().await.unwrap();
        // Only pin remains.
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].role, Role::System);
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn delete_session_wipes_all() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("S"))).await.unwrap();
        h.append(Message::user("u")).await.unwrap();
        h.delete_session().await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert!(msgs.is_empty());
        assert_eq!(h.message_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn message_count_excludes_pin() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("S"))).await.unwrap();
        h.append(Message::user("u1")).await.unwrap();
        h.append(Message::user("u2")).await.unwrap();
        assert_eq!(h.message_count().await.unwrap(), 2);
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn list_sessions_includes_active_session() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let sid = unique_session();
        let h = RedisChatHistory::connect(&url, &sid).await.unwrap();
        h.append(Message::user("ping")).await.unwrap();
        let sessions = h.list_sessions().await.unwrap();
        assert!(sessions.iter().any(|s| s == &sid));
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn session_clones_share_pool_distinct_id() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h_a = RedisChatHistory::connect(&url, unique_session()).await.unwrap();
        let h_b = h_a.session(unique_session());
        h_a.append(Message::user("A")).await.unwrap();
        h_b.append(Message::user("B")).await.unwrap();
        let msgs_a = h_a.messages().await.unwrap();
        let msgs_b = h_b.messages().await.unwrap();
        assert_eq!(msgs_a.len(), 1);
        assert_eq!(msgs_b.len(), 1);
        assert_ne!(h_a.session_id(), h_b.session_id());
        h_a.delete_session().await.unwrap();
        h_b.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn ttl_applied_on_write() {
        let url = match try_url() {
            Some(u) => u,
            None => return,
        };
        let h = RedisChatHistory::connect(&url, unique_session())
            .await
            .unwrap()
            .with_ttl(60);
        h.append(Message::user("hi")).await.unwrap();
        // Sanity: read TTL via raw redis call.
        let mut conn = h.conn.clone();
        let ttl: i64 = conn
            .ttl::<_, i64>(msgs_key(h.session_id()))
            .await
            .unwrap();
        // -1 = no expire, -2 = key missing. Anything ≥ 0 means EXPIRE applied.
        assert!(ttl > 0, "expected positive TTL, got {ttl}");
        h.delete_session().await.unwrap();
    }
}

//! Postgres-backed durable conversation history.
//!
//! Distributed counterpart to `SqliteChatHistory` (single-host) — for
//! cloud / multi-instance deployments where chat sessions need to be
//! readable + writable from multiple worker processes.
//!
//! # Storage layout
//!
//! Mirrors the sqlite schema: per-message rows + a separate system-pin
//! table (one-per-session, replace-not-append).
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS litgraph_messages (
//!     session_id TEXT   NOT NULL,
//!     seq        BIGINT NOT NULL,
//!     message_json TEXT NOT NULL,
//!     ts_ms      BIGINT NOT NULL,
//!     PRIMARY KEY (session_id, seq)
//! );
//! CREATE INDEX IF NOT EXISTS idx_litgraph_messages_session
//!     ON litgraph_messages (session_id, seq);
//!
//! CREATE TABLE IF NOT EXISTS litgraph_system_pins (
//!     session_id TEXT PRIMARY KEY,
//!     message_json TEXT NOT NULL,
//!     ts_ms BIGINT NOT NULL
//! );
//! ```
//!
//! Atomic seq via `COALESCE(MAX(seq)+1, 0)` in the INSERT — same
//! pattern as the sqlite impl. For high-write sessions, a future
//! enhancement can use a `serial`/`identity` column instead.
//!
//! # Connection
//!
//! Uses `deadpool-postgres` for pooling — recommended default. Connect
//! via libpq DSN: `postgres://user:pass@host:5432/db`. Or pass an
//! existing `Pool` via `from_pool` to share with the checkpointer +
//! other litGraph subsystems on the same Postgres instance.

use std::time::{SystemTime, UNIX_EPOCH};

use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use litgraph_core::{Error, Message, Result};
use tokio_postgres::NoTls;
use tracing::debug;

const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS litgraph_messages (
    session_id   TEXT   NOT NULL,
    seq          BIGINT NOT NULL,
    message_json TEXT   NOT NULL,
    ts_ms        BIGINT NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_litgraph_messages_session
    ON litgraph_messages (session_id, seq);
CREATE TABLE IF NOT EXISTS litgraph_system_pins (
    session_id   TEXT PRIMARY KEY,
    message_json TEXT NOT NULL,
    ts_ms        BIGINT NOT NULL
);
"#;

/// Durable conversation history. One handle per `(pool, session_id)` pair;
/// clone with `.session(other_id)` to address a different session in the
/// same store.
#[derive(Clone)]
pub struct PostgresChatHistory {
    pool: Pool,
    session_id: String,
}

impl PostgresChatHistory {
    /// Connect via libpq DSN (e.g. `postgres://user:pass@host:5432/db`)
    /// for the given session. Schema is created idempotently.
    pub async fn connect(dsn: &str, session_id: impl Into<String>) -> Result<Self> {
        let mut cfg = Config::new();
        cfg.url = Some(dsn.to_string());
        cfg.manager = Some(ManagerConfig { recycling_method: RecyclingMethod::Fast });
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        let this = Self { pool, session_id: session_id.into() };
        this.init().await?;
        Ok(this)
    }

    /// Use an existing pool. Preferred for apps that already pool Postgres
    /// (one connection budget across checkpointer + chat-history + ...).
    pub async fn from_pool(pool: Pool, session_id: impl Into<String>) -> Result<Self> {
        let this = Self { pool, session_id: session_id.into() };
        this.init().await?;
        Ok(this)
    }

    async fn init(&self) -> Result<()> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        client.batch_execute(DDL).await
            .map_err(|e| Error::other(format!("pg_chat schema: {e}")))?;
        debug!("pg_chat schema ready");
        Ok(())
    }

    /// Same backing pool, addressed at a different session id. Cheap —
    /// no new pool / connection.
    pub fn session(&self, session_id: impl Into<String>) -> Self {
        Self { pool: self.pool.clone(), session_id: session_id.into() }
    }

    pub fn session_id(&self) -> &str { &self.session_id }

    /// Append one message. Computes the next sequence number atomically.
    pub async fn append(&self, m: Message) -> Result<()> {
        let json = serde_json::to_string(&m)
            .map_err(|e| Error::other(format!("message serialize: {e}")))?;
        let ts = now_ms();
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        client.execute(
            "INSERT INTO litgraph_messages (session_id, seq, message_json, ts_ms) \
             VALUES ($1, COALESCE((SELECT MAX(seq)+1 FROM litgraph_messages WHERE session_id=$1), 0), $2, $3)",
            &[&self.session_id, &json, &ts],
        ).await
            .map_err(|e| Error::other(format!("pg_chat append: {e}")))?;
        Ok(())
    }

    /// Bulk append in a single transaction. Reuses one MAX-seq lookup +
    /// increments locally per insert — much faster than per-call append
    /// for batch loads.
    pub async fn append_all(&self, msgs: Vec<Message>) -> Result<()> {
        if msgs.is_empty() {
            return Ok(());
        }
        let mut serialized = Vec::with_capacity(msgs.len());
        for m in &msgs {
            serialized.push(
                serde_json::to_string(m)
                    .map_err(|e| Error::other(format!("message serialize: {e}")))?
            );
        }
        let ts = now_ms();
        let mut client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        let tx = client.transaction().await
            .map_err(|e| Error::other(format!("pg_chat tx: {e}")))?;
        let row = tx.query_one(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM litgraph_messages WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat seq: {e}")))?;
        let mut next_seq: i64 = row.try_get(0)
            .map_err(|e| Error::other(format!("pg_chat seq decode: {e}")))?;
        for json in &serialized {
            tx.execute(
                "INSERT INTO litgraph_messages (session_id, seq, message_json, ts_ms) VALUES ($1, $2, $3, $4)",
                &[&self.session_id, &next_seq, json, &ts],
            ).await
                .map_err(|e| Error::other(format!("pg_chat append_all: {e}")))?;
            next_seq += 1;
        }
        tx.commit().await
            .map_err(|e| Error::other(format!("pg_chat commit: {e}")))?;
        Ok(())
    }

    /// All messages for this session in chronological order. System pin
    /// (if set) prepended so the result is ready for `ChatModel::invoke`.
    pub async fn messages(&self) -> Result<Vec<Message>> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        // System pin first.
        let mut out = Vec::new();
        let pin_row = client.query_opt(
            "SELECT message_json FROM litgraph_system_pins WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat pin: {e}")))?;
        if let Some(r) = pin_row {
            let s: String = r.try_get(0)
                .map_err(|e| Error::other(format!("pg_chat pin decode: {e}")))?;
            let m: Message = serde_json::from_str(&s)
                .map_err(|e| Error::other(format!("pg_chat pin parse: {e}")))?;
            out.push(m);
        }
        // Conversation in seq order.
        let rows = client.query(
            "SELECT message_json FROM litgraph_messages WHERE session_id = $1 ORDER BY seq",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat select: {e}")))?;
        for r in rows {
            let s: String = r.try_get(0)
                .map_err(|e| Error::other(format!("pg_chat msg decode: {e}")))?;
            let m: Message = serde_json::from_str(&s)
                .map_err(|e| Error::other(format!("pg_chat msg parse: {e}")))?;
            out.push(m);
        }
        Ok(out)
    }

    /// Drop all conversation messages for this session. The system pin
    /// is preserved (use `delete_session()` to wipe everything).
    pub async fn clear(&self) -> Result<()> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        client.execute(
            "DELETE FROM litgraph_messages WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat clear: {e}")))?;
        Ok(())
    }

    /// Drop ALL data for this session — messages + system pin.
    pub async fn delete_session(&self) -> Result<()> {
        let mut client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        let tx = client.transaction().await
            .map_err(|e| Error::other(format!("pg_chat tx: {e}")))?;
        tx.execute(
            "DELETE FROM litgraph_messages WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat delete msgs: {e}")))?;
        tx.execute(
            "DELETE FROM litgraph_system_pins WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat delete pin: {e}")))?;
        tx.commit().await
            .map_err(|e| Error::other(format!("pg_chat commit: {e}")))?;
        Ok(())
    }

    /// Set or replace the system pin. Pass `None` to clear.
    pub async fn set_system(&self, m: Option<Message>) -> Result<()> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        match m {
            Some(msg) => {
                let json = serde_json::to_string(&msg)
                    .map_err(|e| Error::other(format!("message serialize: {e}")))?;
                let ts = now_ms();
                client.execute(
                    "INSERT INTO litgraph_system_pins (session_id, message_json, ts_ms) \
                     VALUES ($1, $2, $3) \
                     ON CONFLICT (session_id) DO UPDATE \
                     SET message_json = EXCLUDED.message_json, ts_ms = EXCLUDED.ts_ms",
                    &[&self.session_id, &json, &ts],
                ).await
                    .map_err(|e| Error::other(format!("pg_chat set_system: {e}")))?;
            }
            None => {
                client.execute(
                    "DELETE FROM litgraph_system_pins WHERE session_id = $1",
                    &[&self.session_id],
                ).await
                    .map_err(|e| Error::other(format!("pg_chat clear pin: {e}")))?;
            }
        }
        Ok(())
    }

    /// Total messages for this session (excludes the system pin).
    pub async fn message_count(&self) -> Result<usize> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        let row = client.query_one(
            "SELECT COUNT(*) FROM litgraph_messages WHERE session_id = $1",
            &[&self.session_id],
        ).await
            .map_err(|e| Error::other(format!("pg_chat count: {e}")))?;
        let n: i64 = row.try_get(0)
            .map_err(|e| Error::other(format!("pg_chat count decode: {e}")))?;
        Ok(n.max(0) as usize)
    }

    /// All distinct session ids in this store. Cheap (DISTINCT on indexed col).
    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let client = self.pool.get().await
            .map_err(|e| Error::other(format!("pg_chat pool: {e}")))?;
        let rows = client.query(
            "SELECT DISTINCT session_id FROM litgraph_messages",
            &[],
        ).await
            .map_err(|e| Error::other(format!("pg_chat sessions: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in rows {
            out.push(r.try_get::<_, String>(0)
                .map_err(|e| Error::other(format!("pg_chat sessions decode: {e}")))?);
        }
        Ok(out)
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::Role;

    /// Skip if no Postgres available — these are integration tests.
    /// Set PG_DSN=postgres://localhost:5432/litgraph_test to run them.
    fn try_dsn() -> Option<String> {
        std::env::var("PG_DSN").ok()
    }

    fn unique_session() -> String {
        format!("test-{}", uuid_lite())
    }

    /// Cheap pseudo-uuid based on nanos — no uuid dep needed for tests.
    fn uuid_lite() -> String {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        format!("{nanos:x}")
    }

    #[tokio::test]
    async fn append_and_messages_roundtrips() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        h.append(Message::user("hi")).await.unwrap();
        h.append(Message::assistant("hello")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text_content(), "hi");
        assert_eq!(msgs[1].text_content(), "hello");
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn append_all_batches_in_one_tx() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        let batch = vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ];
        h.append_all(batch).await.unwrap();
        assert_eq!(h.message_count().await.unwrap(), 3);
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs[0].text_content(), "a");
        assert_eq!(msgs[2].text_content(), "c");
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn system_pin_prepended_to_messages() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("you are helpful"))).await.unwrap();
        h.append(Message::user("q")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].role, Role::System);
        assert_eq!(msgs[0].text_content(), "you are helpful");
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn set_system_replaces_existing_pin() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("v1"))).await.unwrap();
        h.set_system(Some(Message::system("v2"))).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs[0].text_content(), "v2");
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn clear_drops_messages_keeps_pin() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("pin"))).await.unwrap();
        h.append(Message::user("x")).await.unwrap();
        h.clear().await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "pin");
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn delete_session_drops_everything() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        h.set_system(Some(Message::system("pin"))).await.unwrap();
        h.append(Message::user("x")).await.unwrap();
        h.delete_session().await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn session_clone_addresses_different_session_same_pool() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let id_a = unique_session();
        let id_b = unique_session();
        let h_a = PostgresChatHistory::connect(&dsn, &id_a).await.unwrap();
        let h_b = h_a.session(&id_b);
        h_a.append(Message::user("from-a")).await.unwrap();
        h_b.append(Message::user("from-b")).await.unwrap();
        assert_eq!(h_a.messages().await.unwrap()[0].text_content(), "from-a");
        assert_eq!(h_b.messages().await.unwrap()[0].text_content(), "from-b");
        h_a.delete_session().await.unwrap();
        h_b.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn list_sessions_includes_recent_session() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let id = unique_session();
        let h = PostgresChatHistory::connect(&dsn, &id).await.unwrap();
        h.append(Message::user("x")).await.unwrap();
        let sessions = h.list_sessions().await.unwrap();
        assert!(sessions.iter().any(|s| s == &id));
        h.delete_session().await.unwrap();
    }

    #[tokio::test]
    async fn empty_session_returns_empty_messages() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert!(msgs.is_empty());
    }

    #[tokio::test]
    async fn message_count_is_zero_for_empty_session() {
        let dsn = match try_dsn() { Some(d) => d, None => return };
        let h = PostgresChatHistory::connect(&dsn, unique_session()).await.unwrap();
        assert_eq!(h.message_count().await.unwrap(), 0);
    }
}

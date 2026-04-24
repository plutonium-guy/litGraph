//! Durable conversation history backed by SQLite.
//!
//! Survives process restarts; per-message storage for chronological replay
//! + cheap appends; multi-session isolation by `session_id` key.
//!
//! # Why per-message and not per-session-blob?
//!
//! LangChain's `SQLChatMessageHistory` chose per-message and so did we. The
//! tradeoff:
//! - **Per-message** (this impl): cheap append (one INSERT), cheap message
//!   count (COUNT), cheap session-listing (DISTINCT). Slightly more
//!   expensive `messages()` (N rows + N JSON parses) but the load is bounded
//!   by typical chat lengths (50–200 messages, not millions).
//! - **Per-session-blob**: cheap full-history load (one row), but every
//!   append is read-modify-write of the entire history. Burns I/O on long
//!   sessions.
//!
//! Per-message wins for prod chat workloads where appends dominate reads.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE messages (
//!     session_id TEXT NOT NULL,
//!     seq INTEGER NOT NULL,
//!     message_json TEXT NOT NULL,
//!     ts_ms INTEGER NOT NULL,
//!     PRIMARY KEY (session_id, seq)
//! );
//! CREATE TABLE system_pins (
//!     session_id TEXT PRIMARY KEY,
//!     message_json TEXT NOT NULL,
//!     ts_ms INTEGER NOT NULL
//! );
//! ```
//!
//! Two tables because the system pin has different semantics from the
//! conversation history (one-per-session, replace-not-append).
//!
//! # Concurrency
//!
//! All work runs in `tokio::task::spawn_blocking` (sqlite is sync). One
//! `Mutex<Connection>` shared across handles — sqlite serializes writes
//! anyway via WAL, and concurrent reads from a single connection just
//! contend on the mutex (acceptable; parallelize with multiple
//! `SqliteChatHistory::open` clones if you need read scalability).

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use litgraph_core::{Error, Message, Result};
use rusqlite::{Connection, params};
use tracing::debug;

const INIT_SQL: &str = "
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT NOT NULL,
    seq INTEGER NOT NULL,
    message_json TEXT NOT NULL,
    ts_ms INTEGER NOT NULL,
    PRIMARY KEY (session_id, seq)
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE TABLE IF NOT EXISTS system_pins (
    session_id TEXT PRIMARY KEY,
    message_json TEXT NOT NULL,
    ts_ms INTEGER NOT NULL
);
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
";

/// Durable conversation history. One handle per `(path, session_id)` pair;
/// clone with `.session(other_id)` to address a different session in the
/// same store.
#[derive(Clone)]
pub struct SqliteChatHistory {
    conn: Arc<Mutex<Connection>>,
    session_id: String,
    /// `:memory:` for the in-memory variant; the real path otherwise.
    /// Stored for `Display` / debugging only.
    _path: PathBuf,
}

impl SqliteChatHistory {
    /// Open a sqlite file at `path` for the given `session_id`. Creates
    /// the schema idempotently. Multiple handles can be `open()`-ed against
    /// the same file safely (sqlite WAL mode + per-handle Mutex).
    pub fn open(path: impl AsRef<Path>, session_id: impl Into<String>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path)
            .map_err(|e| Error::other(format!("sqlite_chat open: {e}")))?;
        conn.execute_batch(INIT_SQL)
            .map_err(|e| Error::other(format!("sqlite_chat schema: {e}")))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            session_id: session_id.into(),
            _path: path,
        })
    }

    /// In-memory variant — no file. Useful for tests / ephemeral sessions.
    pub fn in_memory(session_id: impl Into<String>) -> Result<Self> {
        let conn = Connection::open_in_memory()
            .map_err(|e| Error::other(format!("sqlite_chat open: {e}")))?;
        conn.execute_batch(INIT_SQL)
            .map_err(|e| Error::other(format!("sqlite_chat schema: {e}")))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
            session_id: session_id.into(),
            _path: PathBuf::from(":memory:"),
        })
    }

    /// Same backing store, addressed at a different session id. Cheap —
    /// no new connection.
    pub fn session(&self, session_id: impl Into<String>) -> Self {
        Self {
            conn: self.conn.clone(),
            session_id: session_id.into(),
            _path: self._path.clone(),
        }
    }

    pub fn session_id(&self) -> &str { &self.session_id }

    /// Append one message. Computes the next sequence number atomically.
    pub async fn append(&self, m: Message) -> Result<()> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        let json = serde_json::to_string(&m)
            .map_err(|e| Error::other(format!("message serialize: {e}")))?;
        let ts = now_ms();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            // Atomic: SELECT MAX(seq) + INSERT in one statement using
            // COALESCE so first-insert correctly seq=0.
            g.execute(
                "INSERT INTO messages (session_id, seq, message_json, ts_ms) \
                 VALUES (?1, COALESCE((SELECT MAX(seq)+1 FROM messages WHERE session_id=?1), 0), ?2, ?3)",
                params![sid, json, ts],
            )
            .map_err(|e| Error::other(format!("sqlite_chat append: {e}")))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        debug!(session = %self.session_id, "sqlite_chat append");
        Ok(())
    }

    /// Bulk append. Reuses one transaction for atomicity + speed.
    pub async fn append_all(&self, msgs: Vec<Message>) -> Result<()> {
        if msgs.is_empty() {
            return Ok(());
        }
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        let mut serialized = Vec::with_capacity(msgs.len());
        for m in &msgs {
            serialized.push(
                serde_json::to_string(m)
                    .map_err(|e| Error::other(format!("message serialize: {e}")))?
            );
        }
        let ts = now_ms();
        tokio::task::spawn_blocking(move || {
            let mut g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            let tx = g.transaction()
                .map_err(|e| Error::other(format!("sqlite_chat tx: {e}")))?;
            // Get the current max seq once, then increment locally.
            let mut next_seq: i64 = tx
                .query_row(
                    "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?1",
                    params![sid],
                    |r| r.get(0),
                )
                .map_err(|e| Error::other(format!("sqlite_chat seq: {e}")))?;
            for json in &serialized {
                tx.execute(
                    "INSERT INTO messages (session_id, seq, message_json, ts_ms) VALUES (?1, ?2, ?3, ?4)",
                    params![sid, next_seq, json, ts],
                )
                .map_err(|e| Error::other(format!("sqlite_chat append_all: {e}")))?;
                next_seq += 1;
            }
            tx.commit().map_err(|e| Error::other(format!("sqlite_chat commit: {e}")))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    /// Return all messages for this session in chronological order. The
    /// system pin (if set) is prepended so callers can hand the result
    /// straight to a `ChatModel::invoke()`.
    pub async fn messages(&self) -> Result<Vec<Message>> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<Message>> {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            // System pin first (if any).
            let mut out = Vec::new();
            let pin: Option<String> = g
                .query_row(
                    "SELECT message_json FROM system_pins WHERE session_id = ?1",
                    params![sid],
                    |r| r.get(0),
                )
                .ok();
            if let Some(p) = pin {
                let m: Message = serde_json::from_str(&p)
                    .map_err(|e| Error::other(format!("system pin parse: {e}")))?;
                out.push(m);
            }
            let mut stmt = g
                .prepare(
                    "SELECT message_json FROM messages WHERE session_id = ?1 ORDER BY seq ASC",
                )
                .map_err(|e| Error::other(format!("sqlite_chat messages prep: {e}")))?;
            let rows = stmt
                .query_map(params![sid], |r| r.get::<_, String>(0))
                .map_err(|e| Error::other(format!("sqlite_chat messages query: {e}")))?;
            for row in rows {
                let json = row.map_err(|e| Error::other(format!("sqlite_chat row: {e}")))?;
                let m: Message = serde_json::from_str(&json)
                    .map_err(|e| Error::other(format!("message parse: {e}")))?;
                out.push(m);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))?
    }

    /// Drop all conversation messages for this session. The system pin
    /// is preserved (it represents the agent's persona, not its memory).
    /// To wipe everything, use `delete_session()`.
    pub async fn clear(&self) -> Result<()> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            g.execute("DELETE FROM messages WHERE session_id = ?1", params![sid])
                .map_err(|e| Error::other(format!("sqlite_chat clear: {e}")))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    /// Drop the entire session — messages AND system pin.
    pub async fn delete_session(&self) -> Result<()> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            g.execute("DELETE FROM messages WHERE session_id = ?1", params![sid])
                .map_err(|e| Error::other(format!("sqlite_chat delete msgs: {e}")))?;
            g.execute("DELETE FROM system_pins WHERE session_id = ?1", params![sid])
                .map_err(|e| Error::other(format!("sqlite_chat delete pin: {e}")))?;
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    /// Set or clear the system pin. Pass `None` to remove.
    pub async fn set_system(&self, m: Option<Message>) -> Result<()> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        let payload = match m {
            Some(msg) => Some(
                serde_json::to_string(&msg)
                    .map_err(|e| Error::other(format!("system pin serialize: {e}")))?
            ),
            None => None,
        };
        let ts = now_ms();
        tokio::task::spawn_blocking(move || {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            match payload {
                Some(json) => {
                    g.execute(
                        "INSERT INTO system_pins (session_id, message_json, ts_ms) VALUES (?1, ?2, ?3) \
                         ON CONFLICT(session_id) DO UPDATE SET message_json=excluded.message_json, ts_ms=excluded.ts_ms",
                        params![sid, json, ts],
                    )
                    .map_err(|e| Error::other(format!("sqlite_chat set_system: {e}")))?;
                }
                None => {
                    g.execute("DELETE FROM system_pins WHERE session_id = ?1", params![sid])
                        .map_err(|e| Error::other(format!("sqlite_chat clear system: {e}")))?;
                }
            }
            Ok::<(), Error>(())
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(())
    }

    /// Count messages in this session (excluding system pin). Cheap O(1)
    /// via the (session_id, seq) primary index.
    pub async fn message_count(&self) -> Result<usize> {
        let conn = self.conn.clone();
        let sid = self.session_id.clone();
        let n: i64 = tokio::task::spawn_blocking(move || -> Result<i64> {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            g.query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?1",
                params![sid],
                |r| r.get(0),
            )
            .map_err(|e| Error::other(format!("sqlite_chat count: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(n as usize)
    }

    /// Enumerate every session id present in either table. Sorted for
    /// stable ordering across calls.
    pub async fn list_sessions(&self) -> Result<Vec<String>> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || -> Result<Vec<String>> {
            let g = conn.lock().map_err(|_| Error::other("mutex poisoned"))?;
            let mut stmt = g
                .prepare(
                    "SELECT session_id FROM messages \
                     UNION SELECT session_id FROM system_pins ORDER BY session_id ASC",
                )
                .map_err(|e| Error::other(format!("sqlite_chat list prep: {e}")))?;
            let rows = stmt
                .query_map([], |r| r.get::<_, String>(0))
                .map_err(|e| Error::other(format!("sqlite_chat list: {e}")))?;
            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(|e| Error::other(format!("row: {e}")))?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))?
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
    use tempfile::NamedTempFile;

    #[tokio::test]
    async fn append_then_messages_round_trip_in_order() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.append(Message::user("hello")).await.unwrap();
        h.append(Message::assistant("hi there")).await.unwrap();
        h.append(Message::user("how are you")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].text_content(), "hello");
        assert_eq!(msgs[1].text_content(), "hi there");
        assert_eq!(msgs[2].text_content(), "how are you");
    }

    #[tokio::test]
    async fn append_all_uses_one_transaction_and_keeps_order() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.append_all(vec![
            Message::user("a"),
            Message::assistant("b"),
            Message::user("c"),
        ])
        .await
        .unwrap();
        let msgs = h.messages().await.unwrap();
        let texts: Vec<String> = msgs.iter().map(|m| m.text_content()).collect();
        assert_eq!(texts, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn append_all_empty_is_no_op() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.append_all(vec![]).await.unwrap();
        assert_eq!(h.message_count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn system_pin_prepended_to_messages() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.set_system(Some(Message::system("you are helpful"))).await.unwrap();
        h.append(Message::user("hi")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].text_content(), "you are helpful");
        assert_eq!(msgs[1].text_content(), "hi");
    }

    #[tokio::test]
    async fn set_system_none_clears_the_pin() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.set_system(Some(Message::system("first"))).await.unwrap();
        h.set_system(None).await.unwrap();
        h.append(Message::user("hi")).await.unwrap();
        let msgs = h.messages().await.unwrap();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "hi");
    }

    #[tokio::test]
    async fn clear_drops_messages_but_keeps_system_pin() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.set_system(Some(Message::system("persona"))).await.unwrap();
        h.append(Message::user("a")).await.unwrap();
        h.append(Message::user("b")).await.unwrap();
        h.clear().await.unwrap();
        let msgs = h.messages().await.unwrap();
        // Only the system pin survives.
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].text_content(), "persona");
    }

    #[tokio::test]
    async fn delete_session_drops_messages_and_system_pin() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.set_system(Some(Message::system("persona"))).await.unwrap();
        h.append(Message::user("a")).await.unwrap();
        h.delete_session().await.unwrap();
        assert!(h.messages().await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn sessions_isolated_from_each_other() {
        let h1 = SqliteChatHistory::in_memory("s1").unwrap();
        let h2 = h1.session("s2");
        h1.append(Message::user("alpha")).await.unwrap();
        h2.append(Message::user("beta")).await.unwrap();
        h2.append(Message::user("gamma")).await.unwrap();
        assert_eq!(h1.message_count().await.unwrap(), 1);
        assert_eq!(h2.message_count().await.unwrap(), 2);
        let m1 = h1.messages().await.unwrap();
        assert_eq!(m1[0].text_content(), "alpha");
    }

    #[tokio::test]
    async fn list_sessions_includes_message_only_and_pin_only_sessions() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.append(Message::user("only-msg")).await.unwrap();
        h.session("s2").set_system(Some(Message::system("only-pin"))).await.unwrap();
        h.session("s3").append(Message::user("both")).await.unwrap();
        h.session("s3").set_system(Some(Message::system("both"))).await.unwrap();
        let sessions = h.list_sessions().await.unwrap();
        // UNION + ORDER BY session_id → ["s1", "s2", "s3"], dedup'd.
        assert_eq!(sessions, vec!["s1", "s2", "s3"]);
    }

    #[tokio::test]
    async fn message_count_excludes_system_pin() {
        let h = SqliteChatHistory::in_memory("s1").unwrap();
        h.set_system(Some(Message::system("persona"))).await.unwrap();
        h.append(Message::user("a")).await.unwrap();
        h.append(Message::user("b")).await.unwrap();
        assert_eq!(h.message_count().await.unwrap(), 2);
    }

    #[tokio::test]
    async fn durability_across_handle_drop_and_reopen() {
        // The promise: chat survives a "process restart". Open a real file,
        // write messages, drop the handle, reopen the same file, observe
        // messages are still there.
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let h = SqliteChatHistory::open(&path, "session-A").unwrap();
            h.set_system(Some(Message::system("persona"))).await.unwrap();
            h.append(Message::user("first message")).await.unwrap();
            h.append(Message::assistant("first reply")).await.unwrap();
        }
        // Drop scope → connection closed → file fully flushed.
        let h2 = SqliteChatHistory::open(&path, "session-A").unwrap();
        let msgs = h2.messages().await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[0].text_content(), "persona");
        assert_eq!(msgs[1].text_content(), "first message");
        assert_eq!(msgs[2].text_content(), "first reply");
    }

    #[tokio::test]
    async fn seq_numbers_continue_from_max_on_resumed_session() {
        // After reopening, append should pick up at max(seq)+1 — NOT
        // restart at 0 (which would PRIMARY KEY collide).
        let tmp = NamedTempFile::new().unwrap();
        let path = tmp.path().to_path_buf();
        {
            let h = SqliteChatHistory::open(&path, "s1").unwrap();
            h.append(Message::user("a")).await.unwrap();
            h.append(Message::user("b")).await.unwrap();
        }
        let h2 = SqliteChatHistory::open(&path, "s1").unwrap();
        h2.append(Message::user("c")).await.unwrap();
        let msgs = h2.messages().await.unwrap();
        assert_eq!(msgs.len(), 3);
        assert_eq!(msgs[2].text_content(), "c");
    }
}

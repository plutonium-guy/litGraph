//! SQLite-backed checkpointer. Blocking `rusqlite` calls are dispatched to
//! `tokio::task::spawn_blocking` so the async runtime stays responsive.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use litgraph_graph::{Checkpoint, Checkpointer, GraphError, Result};
use rusqlite::{Connection, OptionalExtension, params};

pub struct SqliteCheckpointer {
    conn: Arc<Mutex<Connection>>,
    #[allow(dead_code)]
    path: PathBuf,
}

impl SqliteCheckpointer {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let conn = Connection::open(&path).map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        // WAL = concurrent readers; good for a server holding many threads.
        conn.pragma_update(None, "journal_mode", "WAL").ok();
        conn.pragma_update(None, "synchronous", "NORMAL").ok();
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                state BLOB NOT NULL,
                next_nodes TEXT NOT NULL,
                pending_interrupt TEXT,
                ts_ms INTEGER NOT NULL,
                PRIMARY KEY (thread_id, step)
            );
            CREATE INDEX IF NOT EXISTS idx_checkpoints_thread ON checkpoints(thread_id);
            "#,
        )
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)), path })
    }

    pub fn in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory().map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        conn.execute_batch(
            r#"
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                state BLOB NOT NULL,
                next_nodes TEXT NOT NULL,
                pending_interrupt TEXT,
                ts_ms INTEGER NOT NULL,
                PRIMARY KEY (thread_id, step)
            );
            "#,
        )
        .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
        Ok(Self { conn: Arc::new(Mutex::new(conn)), path: PathBuf::from(":memory:") })
    }
}

fn row_to_checkpoint(row: &rusqlite::Row) -> rusqlite::Result<Checkpoint> {
    let thread_id: String = row.get(0)?;
    let step: i64 = row.get(1)?;
    let state: Vec<u8> = row.get(2)?;
    let next_nodes_json: String = row.get(3)?;
    let pending_json: Option<String> = row.get(4)?;
    let ts_ms: i64 = row.get(5)?;
    let next_nodes: Vec<String> = serde_json::from_str(&next_nodes_json).unwrap_or_default();
    let pending_interrupt = pending_json.and_then(|s| serde_json::from_str(&s).ok());
    // Sqlite backend doesn't persist next_sends yet (interrupt-during-fan-out
    // is rare); rebuild with empty sends. Add a column when fan-out + resume
    // becomes a load-bearing pattern for SQLite users.
    Ok(Checkpoint {
        thread_id, step: step as u64, state, next_nodes,
        next_sends: Vec::new(),
        pending_interrupt, ts_ms: ts_ms as u64,
    })
}

#[async_trait]
impl Checkpointer for SqliteCheckpointer {
    async fn put(&self, cp: Checkpoint) -> Result<()> {
        let conn = self.conn.clone();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            let next_nodes = serde_json::to_string(&cp.next_nodes)?;
            let pending = cp
                .pending_interrupt
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            guard
                .execute(
                    "INSERT OR REPLACE INTO checkpoints \
                     (thread_id, step, state, next_nodes, pending_interrupt, ts_ms) \
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![cp.thread_id, cp.step as i64, cp.state, next_nodes, pending, cp.ts_ms as i64],
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            Ok::<(), GraphError>(())
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))??;
        Ok(())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let conn = self.conn.clone();
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            let mut stmt = guard
                .prepare(
                    "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                     FROM checkpoints WHERE thread_id = ?1 ORDER BY step DESC LIMIT 1",
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            let out = stmt
                .query_row(params![tid], row_to_checkpoint)
                .optional()
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            Ok::<Option<Checkpoint>, GraphError>(out)
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))?
    }

    async fn get(&self, thread_id: &str, step: u64) -> Result<Option<Checkpoint>> {
        let conn = self.conn.clone();
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            let mut stmt = guard
                .prepare(
                    "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                     FROM checkpoints WHERE thread_id = ?1 AND step = ?2",
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            let out = stmt
                .query_row(params![tid, step as i64], row_to_checkpoint)
                .optional()
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            Ok::<Option<Checkpoint>, GraphError>(out)
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))?
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let conn = self.conn.clone();
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = conn.lock().map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            let mut stmt = guard
                .prepare(
                    "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                     FROM checkpoints WHERE thread_id = ?1 ORDER BY step ASC",
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            let rows = stmt
                .query_map(params![tid], row_to_checkpoint)
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(|e| GraphError::Checkpoint(e.to_string()))?);
            }
            Ok::<Vec<Checkpoint>, GraphError>(out)
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))?
    }

    /// Native DELETE-WHERE-step>target. Faster than the default
    /// list+clear+re-put because we never round-trip the surviving rows.
    async fn rewind_to(&self, thread_id: &str, target_step: u64) -> Result<usize> {
        let conn = self.conn.clone();
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = conn
                .lock()
                .map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            // Guard: target must exist.
            let exists: Option<i64> = guard
                .query_row(
                    "SELECT 1 FROM checkpoints WHERE thread_id = ?1 AND step = ?2",
                    params![tid, target_step as i64],
                    |r| r.get(0),
                )
                .optional()
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            if exists.is_none() {
                return Err(GraphError::Checkpoint(format!(
                    "rewind_to: thread `{tid}` has no checkpoint at step {target_step}"
                )));
            }
            let dropped = guard
                .execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?1 AND step > ?2",
                    params![tid, target_step as i64],
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            Ok::<usize, GraphError>(dropped)
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))?
    }

    async fn clear_thread(&self, thread_id: &str) -> Result<()> {
        let conn = self.conn.clone();
        let tid = thread_id.to_string();
        tokio::task::spawn_blocking(move || {
            let guard = conn
                .lock()
                .map_err(|_| GraphError::Checkpoint("mutex poisoned".into()))?;
            guard
                .execute(
                    "DELETE FROM checkpoints WHERE thread_id = ?1",
                    params![tid],
                )
                .map_err(|e| GraphError::Checkpoint(e.to_string()))?;
            Ok::<(), GraphError>(())
        })
        .await
        .map_err(|e| GraphError::Checkpoint(format!("join: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn put_latest_roundtrip() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        let c = Checkpoint {
            thread_id: "t1".into(),
            step: 1,
            state: vec![1, 2, 3],
            next_nodes: vec!["a".into()],
            next_sends: Vec::new(),
            pending_interrupt: None,
            ts_ms: 1000,
        };
        cp.put(c.clone()).await.unwrap();
        let got = cp.latest("t1").await.unwrap().unwrap();
        assert_eq!(got.step, 1);
        assert_eq!(got.state, vec![1, 2, 3]);
        assert_eq!(got.next_nodes, vec!["a".to_string()]);
    }

    #[tokio::test]
    async fn lists_in_order() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        for s in 1..=3 {
            cp.put(Checkpoint {
                thread_id: "t2".into(),
                step: s,
                state: vec![s as u8],
                next_nodes: vec![],
                next_sends: Vec::new(),
                pending_interrupt: None,
                ts_ms: s,
            })
            .await
            .unwrap();
        }
        let list = cp.list("t2").await.unwrap();
        assert_eq!(list.len(), 3);
        assert_eq!(list[0].step, 1);
        assert_eq!(list[2].step, 3);
    }

    fn mkcp(thread: &str, step: u64) -> Checkpoint {
        Checkpoint {
            thread_id: thread.into(),
            step,
            state: vec![step as u8],
            next_nodes: vec!["node".into()],
            next_sends: vec![],
            pending_interrupt: None,
            ts_ms: step * 100,
        }
    }

    #[tokio::test]
    async fn sqlite_rewind_drops_later_atomically() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        for s in 0..5 {
            cp.put(mkcp("t1", s)).await.unwrap();
        }
        let dropped = cp.rewind_to("t1", 2).await.unwrap();
        assert_eq!(dropped, 2);
        assert_eq!(cp.list("t1").await.unwrap().len(), 3);
        assert_eq!(cp.latest("t1").await.unwrap().unwrap().step, 2);
    }

    #[tokio::test]
    async fn sqlite_rewind_nonexistent_step_errors() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        cp.put(mkcp("t1", 0)).await.unwrap();
        let err = cp.rewind_to("t1", 99).await.unwrap_err();
        assert!(format!("{err}").contains("no checkpoint at step 99"));
        // Nothing was dropped.
        assert_eq!(cp.list("t1").await.unwrap().len(), 1);
    }

    #[tokio::test]
    async fn sqlite_fork_at_copies_history_into_new_thread() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        for s in 0..4 {
            cp.put(mkcp("main", s)).await.unwrap();
        }
        let copied = cp.fork_at("main", 1, "fork-a").await.unwrap();
        assert_eq!(copied, 2);
        let fork_hist = cp.list("fork-a").await.unwrap();
        assert_eq!(fork_hist.len(), 2);
        assert!(fork_hist.iter().all(|c| c.thread_id == "fork-a"));
        // Main unchanged.
        assert_eq!(cp.list("main").await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn sqlite_clear_thread_drops_all() {
        let cp = SqliteCheckpointer::in_memory().unwrap();
        cp.put(mkcp("t1", 0)).await.unwrap();
        cp.put(mkcp("t1", 1)).await.unwrap();
        cp.clear_thread("t1").await.unwrap();
        assert!(cp.list("t1").await.unwrap().is_empty());
    }
}

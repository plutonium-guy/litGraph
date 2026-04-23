//! Postgres-backed Checkpointer.
//!
//! Uses `deadpool-postgres` for connection pooling — recommended default for
//! long-running servers. Snapshots stored as `BYTEA` (bincode); JSON columns for
//! `next_nodes` and `pending_interrupt` so they're queryable with standard
//! Postgres JSON operators.
//!
//! # Schema
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS litgraph_checkpoints (
//!     thread_id         TEXT         NOT NULL,
//!     step              BIGINT       NOT NULL,
//!     state             BYTEA        NOT NULL,
//!     next_nodes        JSONB        NOT NULL,
//!     pending_interrupt JSONB,
//!     ts_ms             BIGINT       NOT NULL,
//!     PRIMARY KEY (thread_id, step)
//! );
//! CREATE INDEX IF NOT EXISTS idx_litgraph_checkpoints_thread_step
//!     ON litgraph_checkpoints (thread_id, step DESC);
//! ```
//!
//! `put()` uses `ON CONFLICT (thread_id, step) DO UPDATE` — safe to retry.

use async_trait::async_trait;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use litgraph_graph::{Checkpoint, Checkpointer, GraphError, Result};
use tokio_postgres::NoTls;
use tracing::debug;

const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS litgraph_checkpoints (
    thread_id         TEXT   NOT NULL,
    step              BIGINT NOT NULL,
    state             BYTEA  NOT NULL,
    next_nodes        TEXT   NOT NULL,
    pending_interrupt TEXT,
    ts_ms             BIGINT NOT NULL,
    PRIMARY KEY (thread_id, step)
);
CREATE INDEX IF NOT EXISTS idx_litgraph_checkpoints_thread_step
    ON litgraph_checkpoints (thread_id, step DESC);
"#;

pub struct PgCheckpointer {
    pool: Pool,
}

impl PgCheckpointer {
    /// Connect using a libpq-style URL, e.g. `postgres://user:pass@host:5432/db`.
    pub async fn connect(dsn: &str) -> Result<Self> {
        let mut cfg = Config::new();
        cfg.url = Some(dsn.to_string());
        cfg.manager = Some(ManagerConfig { recycling_method: RecyclingMethod::Fast });
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| GraphError::Checkpoint(format!("pg pool: {e}")))?;
        let this = Self { pool };
        this.init().await?;
        Ok(this)
    }

    /// Use an existing pool (preferred for apps that already pool Postgres).
    pub async fn from_pool(pool: Pool) -> Result<Self> {
        let this = Self { pool };
        this.init().await?;
        Ok(this)
    }

    async fn init(&self) -> Result<()> {
        let client = self.pool.get().await.map_err(pool_err)?;
        client.batch_execute(DDL).await.map_err(query_err)?;
        debug!("pg checkpointer schema ready");
        Ok(())
    }
}

fn pool_err<E: std::fmt::Display>(e: E) -> GraphError {
    GraphError::Checkpoint(format!("pg pool: {e}"))
}
fn query_err<E: std::fmt::Display>(e: E) -> GraphError {
    GraphError::Checkpoint(format!("pg query: {e}"))
}

fn row_to_checkpoint(row: &tokio_postgres::Row) -> std::result::Result<Checkpoint, GraphError> {
    let thread_id: String = row.try_get("thread_id").map_err(query_err)?;
    let step: i64 = row.try_get("step").map_err(query_err)?;
    let state: Vec<u8> = row.try_get("state").map_err(query_err)?;
    let next_nodes_s: String = row.try_get("next_nodes").map_err(query_err)?;
    let pending_s: Option<String> = row.try_get("pending_interrupt").map_err(query_err)?;
    let ts_ms: i64 = row.try_get("ts_ms").map_err(query_err)?;
    let next_nodes: Vec<String> =
        serde_json::from_str(&next_nodes_s).map_err(|e| query_err(e.to_string()))?;
    let pending_interrupt = pending_s
        .map(|s| serde_json::from_str::<litgraph_graph::Interrupt>(&s))
        .transpose()
        .map_err(|e| query_err(e.to_string()))?;
    Ok(Checkpoint {
        thread_id,
        step: step as u64,
        state,
        next_nodes,
        // Postgres backend doesn't persist next_sends yet — see SQLite note.
        next_sends: Vec::new(),
        pending_interrupt,
        ts_ms: ts_ms as u64,
    })
}

#[async_trait]
impl Checkpointer for PgCheckpointer {
    async fn put(&self, cp: Checkpoint) -> Result<()> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let stmt = client
            .prepare_cached(
                "INSERT INTO litgraph_checkpoints \
                 (thread_id, step, state, next_nodes, pending_interrupt, ts_ms) \
                 VALUES ($1, $2, $3, $4, $5, $6) \
                 ON CONFLICT (thread_id, step) DO UPDATE SET \
                   state = EXCLUDED.state, \
                   next_nodes = EXCLUDED.next_nodes, \
                   pending_interrupt = EXCLUDED.pending_interrupt, \
                   ts_ms = EXCLUDED.ts_ms",
            )
            .await
            .map_err(query_err)?;
        let next_nodes_s = serde_json::to_string(&cp.next_nodes)
            .map_err(|e| query_err(e.to_string()))?;
        let pending_s = cp
            .pending_interrupt
            .as_ref()
            .map(serde_json::to_string)
            .transpose()
            .map_err(|e| query_err(e.to_string()))?;
        client
            .execute(
                &stmt,
                &[
                    &cp.thread_id,
                    &(cp.step as i64),
                    &cp.state,
                    &next_nodes_s,
                    &pending_s,
                    &(cp.ts_ms as i64),
                ],
            )
            .await
            .map_err(query_err)?;
        Ok(())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let row = client
            .query_opt(
                "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                 FROM litgraph_checkpoints \
                 WHERE thread_id = $1 \
                 ORDER BY step DESC LIMIT 1",
                &[&thread_id],
            )
            .await
            .map_err(query_err)?;
        row.as_ref().map(row_to_checkpoint).transpose()
    }

    async fn get(&self, thread_id: &str, step: u64) -> Result<Option<Checkpoint>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let row = client
            .query_opt(
                "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                 FROM litgraph_checkpoints \
                 WHERE thread_id = $1 AND step = $2",
                &[&thread_id, &(step as i64)],
            )
            .await
            .map_err(query_err)?;
        row.as_ref().map(row_to_checkpoint).transpose()
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let client = self.pool.get().await.map_err(pool_err)?;
        let rows = client
            .query(
                "SELECT thread_id, step, state, next_nodes, pending_interrupt, ts_ms \
                 FROM litgraph_checkpoints \
                 WHERE thread_id = $1 \
                 ORDER BY step ASC",
                &[&thread_id],
            )
            .await
            .map_err(query_err)?;
        rows.iter().map(row_to_checkpoint).collect()
    }
}

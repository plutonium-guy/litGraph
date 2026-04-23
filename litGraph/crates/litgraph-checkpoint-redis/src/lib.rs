//! Redis-backed Checkpointer.
//!
//! # Storage layout
//!
//! For every `thread_id` we keep:
//! - `litgraph:cp:{thread_id}` — ZSET of `step -> bincode-serialized Checkpoint`
//!   payload (embedded step index → O(log n) latest lookup via ZREVRANGEBYSCORE).
//!
//! Each payload = bincode({state, next_nodes_json, pending_interrupt_json, ts_ms}).
//! We keep the `thread_id` + `step` in the struct for completeness even though
//! they are derivable.
//!
//! Use `ConnectionManager` so the pool auto-reconnects and the caller never has
//! to re-handshake after a network hiccup.

use async_trait::async_trait;
use litgraph_graph::{Checkpoint, Checkpointer, GraphError, Interrupt, Result};
use redis::AsyncCommands;
use redis::aio::ConnectionManager;
use serde::{Deserialize, Serialize};

fn key(thread_id: &str) -> String { format!("litgraph:cp:{thread_id}") }

fn err<E: std::fmt::Display>(e: E) -> GraphError {
    GraphError::Checkpoint(format!("redis: {e}"))
}

#[derive(Serialize, Deserialize)]
struct Payload {
    state: Vec<u8>,
    next_nodes: Vec<String>,
    pending_interrupt: Option<Interrupt>,
    ts_ms: u64,
    /// Pending Send fan-out commands; `#[serde(default)]` so legacy bincode
    /// payloads (pre-iter-77) still decode — they'll just have no sends.
    #[serde(default)]
    next_sends: Vec<litgraph_graph::Command>,
}

pub struct RedisCheckpointer {
    conn: ConnectionManager,
}

impl RedisCheckpointer {
    /// Connect via Redis URL (e.g. `redis://127.0.0.1:6379/0`).
    pub async fn connect(url: &str) -> Result<Self> {
        let client = redis::Client::open(url).map_err(err)?;
        let conn = ConnectionManager::new(client).await.map_err(err)?;
        Ok(Self { conn })
    }

    /// Wrap an existing connection manager.
    pub fn from_manager(conn: ConnectionManager) -> Self { Self { conn } }
}

fn decode(thread_id: String, step: u64, bytes: Vec<u8>) -> std::result::Result<Checkpoint, GraphError> {
    let p: Payload = bincode::deserialize(&bytes).map_err(err)?;
    Ok(Checkpoint {
        thread_id,
        step,
        state: p.state,
        next_nodes: p.next_nodes,
        next_sends: p.next_sends,
        pending_interrupt: p.pending_interrupt,
        ts_ms: p.ts_ms,
    })
}

#[async_trait]
impl Checkpointer for RedisCheckpointer {
    async fn put(&self, cp: Checkpoint) -> Result<()> {
        let payload = Payload {
            state: cp.state,
            next_nodes: cp.next_nodes,
            pending_interrupt: cp.pending_interrupt,
            ts_ms: cp.ts_ms,
            next_sends: cp.next_sends,
        };
        let bytes = bincode::serialize(&payload).map_err(err)?;
        let mut conn = self.conn.clone();
        let k = key(&cp.thread_id);
        // ZADD replaces if the same score already exists.
        let _: i64 = conn.zadd(k, bytes, cp.step as f64).await.map_err(err)?;
        Ok(())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let mut conn = self.conn.clone();
        let k = key(thread_id);
        // Highest score, one element; both the member bytes and its score.
        let items: Vec<(Vec<u8>, f64)> = conn
            .zrevrange_withscores(&k, 0, 0)
            .await
            .map_err(err)?;
        match items.into_iter().next() {
            Some((bytes, score)) => Ok(Some(decode(thread_id.to_string(), score as u64, bytes)?)),
            None => Ok(None),
        }
    }

    async fn get(&self, thread_id: &str, step: u64) -> Result<Option<Checkpoint>> {
        let mut conn = self.conn.clone();
        let k = key(thread_id);
        let items: Vec<(Vec<u8>, f64)> = conn
            .zrangebyscore_limit_withscores(&k, step as f64, step as f64, 0, 1)
            .await
            .map_err(err)?;
        match items.into_iter().next() {
            Some((bytes, _)) => Ok(Some(decode(thread_id.to_string(), step, bytes)?)),
            None => Ok(None),
        }
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let mut conn = self.conn.clone();
        let k = key(thread_id);
        let items: Vec<(Vec<u8>, f64)> = conn
            .zrange_withscores(&k, 0, -1)
            .await
            .map_err(err)?;
        let mut out = Vec::with_capacity(items.len());
        for (bytes, score) in items {
            out.push(decode(thread_id.to_string(), score as u64, bytes)?);
        }
        Ok(out)
    }
}

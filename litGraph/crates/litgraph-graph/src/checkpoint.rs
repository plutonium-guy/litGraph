use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::{GraphError, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Checkpoint {
    pub thread_id: String,
    pub step: u64,
    /// Bincode-serialized state snapshot (fast, compact).
    pub state: Vec<u8>,
    pub next_nodes: Vec<String>,
    /// Pending Send-style fan-out commands queued for the next superstep.
    /// Each carries a target node + a state override to fork into for that
    /// specific sub-invocation (LangGraph `Send` semantics). `#[serde(default)]`
    /// keeps old checkpoints (pre-iter-77) readable — they just had no sends.
    #[serde(default)]
    pub next_sends: Vec<crate::interrupt::Command>,
    pub pending_interrupt: Option<crate::interrupt::Interrupt>,
    pub ts_ms: u64,
}

#[async_trait]
pub trait Checkpointer: Send + Sync {
    async fn put(&self, cp: Checkpoint) -> Result<()>;
    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>>;
    async fn get(&self, thread_id: &str, step: u64) -> Result<Option<Checkpoint>>;
    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>>;
}

/// In-memory checkpointer — default; swap for SQLite/Postgres in prod.
#[derive(Default)]
pub struct MemoryCheckpointer {
    inner: Mutex<HashMap<String, Vec<Checkpoint>>>,
}

impl MemoryCheckpointer {
    pub fn new() -> Self { Self::default() }
}

#[async_trait]
impl Checkpointer for MemoryCheckpointer {
    async fn put(&self, cp: Checkpoint) -> Result<()> {
        let mut g = self.inner.lock().map_err(|_| GraphError::Checkpoint("poisoned".into()))?;
        g.entry(cp.thread_id.clone()).or_default().push(cp);
        Ok(())
    }

    async fn latest(&self, thread_id: &str) -> Result<Option<Checkpoint>> {
        let g = self.inner.lock().map_err(|_| GraphError::Checkpoint("poisoned".into()))?;
        Ok(g.get(thread_id).and_then(|v| v.last()).cloned())
    }

    async fn get(&self, thread_id: &str, step: u64) -> Result<Option<Checkpoint>> {
        let g = self.inner.lock().map_err(|_| GraphError::Checkpoint("poisoned".into()))?;
        Ok(g.get(thread_id).and_then(|v| v.iter().find(|c| c.step == step)).cloned())
    }

    async fn list(&self, thread_id: &str) -> Result<Vec<Checkpoint>> {
        let g = self.inner.lock().map_err(|_| GraphError::Checkpoint("poisoned".into()))?;
        Ok(g.get(thread_id).cloned().unwrap_or_default())
    }
}

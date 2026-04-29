use std::collections::HashMap;
use std::sync::Mutex;

use async_trait::async_trait;
use serde::de::DeserializeOwned;
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

    /// Drop checkpoints with `step > target_step` for `thread_id`. After
    /// this, the next resume will pick up from `target_step` — effectively
    /// rewinding execution. The retained checkpoint at `target_step`
    /// becomes the new latest.
    ///
    /// Default impl: list, identify the survivors, clear the thread,
    /// re-put the survivors. Storage backends with native delete-by-range
    /// (Postgres, Redis) should override for performance.
    ///
    /// Returns the number of checkpoints dropped. Errors if the thread
    /// has no checkpoint at `target_step` (would silently fork to an
    /// empty state otherwise).
    async fn rewind_to(&self, thread_id: &str, target_step: u64) -> Result<usize> {
        let history = self.list(thread_id).await?;
        if !history.iter().any(|c| c.step == target_step) {
            return Err(GraphError::Checkpoint(format!(
                "rewind_to: thread `{thread_id}` has no checkpoint at step {target_step}"
            )));
        }
        let survivors: Vec<Checkpoint> = history
            .iter()
            .filter(|c| c.step <= target_step)
            .cloned()
            .collect();
        let dropped = history.len() - survivors.len();
        // Default rewind = nuke + restore via a fresh thread roundtrip.
        // Backends override to do this in one transaction.
        self.clear_thread(thread_id).await?;
        for cp in survivors {
            self.put(cp).await?;
        }
        Ok(dropped)
    }

    /// Drop ALL checkpoints for `thread_id`. Used by `rewind_to`'s
    /// default impl + by callers that want to scrub a session.
    /// Default impl: re-put an empty thread (NOT enough for sqlite/etc —
    /// so backends override). The default panics-by-erroring to force
    /// override; in-memory has its own implementation.
    async fn clear_thread(&self, _thread_id: &str) -> Result<()> {
        Err(GraphError::Checkpoint(
            "clear_thread: backend must override (default impl is unsafe)".to_string(),
        ))
    }

    /// Copy `thread_id`'s checkpoints with `step <= source_step` into
    /// `new_thread_id`. The new thread becomes an independent timeline
    /// branching off the original at `source_step` — resuming the new
    /// thread runs forward from there without affecting the source.
    ///
    /// Use for: A/B-testing alternative paths from the same state,
    /// human-in-the-loop "what if I edited this state and continued",
    /// LangGraph-style time travel + branch.
    ///
    /// Default impl: list + filter + put each into the new thread.
    /// Errors if `new_thread_id` already has any checkpoints (would
    /// merge two timelines silently otherwise).
    async fn fork_at(
        &self,
        thread_id: &str,
        source_step: u64,
        new_thread_id: &str,
    ) -> Result<usize> {
        if !self.list(new_thread_id).await?.is_empty() {
            return Err(GraphError::Checkpoint(format!(
                "fork_at: target thread `{new_thread_id}` already has checkpoints"
            )));
        }
        let history = self.list(thread_id).await?;
        if !history.iter().any(|c| c.step == source_step) {
            return Err(GraphError::Checkpoint(format!(
                "fork_at: thread `{thread_id}` has no checkpoint at step {source_step}"
            )));
        }
        let mut copied = 0usize;
        for cp in history.into_iter().filter(|c| c.step <= source_step) {
            let mut forked = cp;
            forked.thread_id = new_thread_id.to_string();
            self.put(forked).await?;
            copied += 1;
        }
        Ok(copied)
    }
}

/// One entry in a typed state history. Decoded from `Checkpoint.state`.
#[derive(Debug, Clone)]
pub struct StateHistoryEntry<S> {
    pub thread_id: String,
    pub step: u64,
    pub state: S,
    pub next_nodes: Vec<String>,
    pub pending_interrupt: Option<crate::interrupt::Interrupt>,
    pub ts_ms: u64,
}

/// Decode the bincode state blobs back into typed states. The 80% caller
/// pattern: "show me the trajectory" — list of `(step, state)` pairs.
///
/// Returns entries in `step` ascending order (the storage layer doesn't
/// guarantee ordering across backends; we sort to make the API stable).
///
/// `S: DeserializeOwned` — caller picks the state type (must match
/// what was checkpointed during execution).
pub async fn state_history<S>(
    checkpointer: &dyn Checkpointer,
    thread_id: &str,
) -> Result<Vec<StateHistoryEntry<S>>>
where
    S: DeserializeOwned,
{
    let mut raw = checkpointer.list(thread_id).await?;
    raw.sort_by_key(|c| c.step);
    let mut out = Vec::with_capacity(raw.len());
    for cp in raw {
        let state: S = rmp_serde::from_slice(&cp.state)
            .map_err(|e| GraphError::Checkpoint(format!("decode state: {e}")))?;
        out.push(StateHistoryEntry {
            thread_id: cp.thread_id,
            step: cp.step,
            state,
            next_nodes: cp.next_nodes,
            pending_interrupt: cp.pending_interrupt,
            ts_ms: cp.ts_ms,
        });
    }
    Ok(out)
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

    async fn clear_thread(&self, thread_id: &str) -> Result<()> {
        let mut g = self.inner.lock().map_err(|_| GraphError::Checkpoint("poisoned".into()))?;
        g.remove(thread_id);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
    struct Counter {
        n: i32,
    }

    fn cp(thread: &str, step: u64, n: i32) -> Checkpoint {
        let state = rmp_serde::to_vec(&Counter { n }).unwrap();
        Checkpoint {
            thread_id: thread.into(),
            step,
            state,
            next_nodes: vec!["node".into()],
            next_sends: vec![],
            pending_interrupt: None,
            ts_ms: step * 100,
        }
    }

    #[tokio::test]
    async fn state_history_decodes_typed_states_sorted_by_step() {
        let m = MemoryCheckpointer::new();
        m.put(cp("t1", 2, 20)).await.unwrap();
        m.put(cp("t1", 0, 0)).await.unwrap();
        m.put(cp("t1", 1, 10)).await.unwrap();
        let hist: Vec<StateHistoryEntry<Counter>> =
            state_history(&m, "t1").await.unwrap();
        assert_eq!(hist.len(), 3);
        let steps: Vec<u64> = hist.iter().map(|e| e.step).collect();
        assert_eq!(steps, vec![0, 1, 2]);
        assert_eq!(hist[0].state, Counter { n: 0 });
        assert_eq!(hist[2].state, Counter { n: 20 });
    }

    #[tokio::test]
    async fn state_history_empty_thread_returns_empty_vec() {
        let m = MemoryCheckpointer::new();
        let hist: Vec<StateHistoryEntry<Counter>> =
            state_history(&m, "no-such-thread").await.unwrap();
        assert!(hist.is_empty());
    }

    #[tokio::test]
    async fn rewind_to_drops_later_checkpoints_keeps_target_as_latest() {
        let m = MemoryCheckpointer::new();
        for s in 0..5 {
            m.put(cp("t1", s, s as i32 * 10)).await.unwrap();
        }
        let dropped = m.rewind_to("t1", 2).await.unwrap();
        assert_eq!(dropped, 2); // dropped steps 3, 4
        let latest = m.latest("t1").await.unwrap().unwrap();
        assert_eq!(latest.step, 2);
        let hist: Vec<Checkpoint> = m.list("t1").await.unwrap();
        assert_eq!(hist.len(), 3);
    }

    #[tokio::test]
    async fn rewind_to_nonexistent_step_errors() {
        let m = MemoryCheckpointer::new();
        m.put(cp("t1", 0, 0)).await.unwrap();
        m.put(cp("t1", 1, 10)).await.unwrap();
        let err = m.rewind_to("t1", 42).await.unwrap_err();
        assert!(format!("{err}").contains("no checkpoint at step 42"));
        // State unchanged — rewind to a non-existent step is a no-op.
        assert_eq!(m.list("t1").await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn fork_at_copies_history_up_to_source_step_into_new_thread() {
        let m = MemoryCheckpointer::new();
        for s in 0..5 {
            m.put(cp("main", s, s as i32 * 10)).await.unwrap();
        }
        let copied = m.fork_at("main", 2, "fork-a").await.unwrap();
        assert_eq!(copied, 3); // steps 0, 1, 2

        // Fork has 3 checkpoints, all with thread_id="fork-a".
        let fork_hist = m.list("fork-a").await.unwrap();
        assert_eq!(fork_hist.len(), 3);
        assert!(fork_hist.iter().all(|c| c.thread_id == "fork-a"));
        assert_eq!(fork_hist.last().unwrap().step, 2);

        // Main is unchanged.
        assert_eq!(m.list("main").await.unwrap().len(), 5);
    }

    #[tokio::test]
    async fn fork_at_refuses_to_merge_into_existing_thread() {
        let m = MemoryCheckpointer::new();
        m.put(cp("main", 0, 0)).await.unwrap();
        m.put(cp("main", 1, 10)).await.unwrap();
        // Target already has a checkpoint → fork_at must refuse.
        m.put(cp("existing", 0, 999)).await.unwrap();

        let err = m.fork_at("main", 0, "existing").await.unwrap_err();
        assert!(format!("{err}").contains("already has checkpoints"));
    }

    #[tokio::test]
    async fn fork_at_nonexistent_step_errors() {
        let m = MemoryCheckpointer::new();
        m.put(cp("main", 0, 0)).await.unwrap();
        let err = m.fork_at("main", 99, "fork").await.unwrap_err();
        assert!(format!("{err}").contains("no checkpoint at step 99"));
    }

    #[tokio::test]
    async fn fork_then_rewind_creates_independent_branches() {
        // Typical workflow: main ran 5 steps, decided step 3 was wrong,
        // fork at step 2 to a new thread, resume there.
        let m = MemoryCheckpointer::new();
        for s in 0..5 {
            m.put(cp("main", s, s as i32 * 10)).await.unwrap();
        }
        m.fork_at("main", 2, "fork").await.unwrap();
        // Add a divergent step to the fork.
        m.put(cp("fork", 3, 999)).await.unwrap();

        // Main unaffected.
        let main_latest = m.latest("main").await.unwrap().unwrap();
        assert_eq!(main_latest.step, 4);
        // Fork has its own step 3 with divergent state.
        let fork_hist: Vec<StateHistoryEntry<Counter>> =
            state_history(&m, "fork").await.unwrap();
        assert_eq!(fork_hist.last().unwrap().state.n, 999);
        assert_eq!(fork_hist.len(), 4);
    }

    #[tokio::test]
    async fn state_history_entry_carries_metadata() {
        let m = MemoryCheckpointer::new();
        let mut c = cp("t1", 5, 100);
        c.next_nodes = vec!["node_a".into(), "node_b".into()];
        c.ts_ms = 42_000;
        m.put(c).await.unwrap();
        let hist: Vec<StateHistoryEntry<Counter>> =
            state_history(&m, "t1").await.unwrap();
        let e = &hist[0];
        assert_eq!(e.step, 5);
        assert_eq!(e.next_nodes, vec!["node_a", "node_b"]);
        assert_eq!(e.ts_ms, 42_000);
    }
}

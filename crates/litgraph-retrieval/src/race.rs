//! `RaceRetriever` — invoke N retrievers concurrently, return the
//! first successful response, abort the rest.
//!
//! Completes the **race** pattern across the three primary read-side
//! async traits:
//!
//! | Iter | Type                                     | Domain        |
//! |------|------------------------------------------|---------------|
//! | 184  | [`RaceChatModel`](litgraph_resilience::RaceChatModel) | `ChatModel::invoke` |
//! | 192  | [`RaceEmbeddings`](litgraph_resilience::RaceEmbeddings) | `Embeddings::embed_*` |
//! | 193  | `RaceRetriever` (this)                   | `Retriever::retrieve` |
//!
//! # Why use this over `EnsembleRetriever`
//!
//! - [`EnsembleRetriever`](crate::EnsembleRetriever) waits for **every**
//!   child and **fuses** their results via weighted RRF. Output is the
//!   union of contributions; latency = `max(t_i)`; cost = `Σ children`.
//! - `RaceRetriever` waits for the **first** child to succeed and
//!   returns its raw result. Latency = `min(t_i)`; cost = `~all
//!   children` (the slow ones get aborted as soon as a winner emerges,
//!   but their HTTP requests / index scans were already in flight).
//!
//! Use `EnsembleRetriever` for **quality** (multiple sources of truth
//! fused). Use `RaceRetriever` for **latency** (hedge a fast cache
//! retriever against a slow but authoritative one; take the first hit).
//!
//! # Typical patterns
//!
//! - **Cache vs primary**: in-memory cached retriever races against
//!   pgvector / qdrant; cache wins on hot keys, primary fills cold.
//! - **Multi-region**: us-east + eu-west; pod-local network distance
//!   wins.
//! - **Sparse vs dense**: BM25 (microseconds) races against a remote
//!   dense embedder + vector store (tens of ms); BM25 wins on
//!   keyword-heavy queries.
//!
//! # Failure
//!
//! Returns `Ok` as soon as **any** child returns `Ok`. Returns `Err`
//! only when **every** child fails — the error message aggregates
//! all failures (newline-separated) so the caller can debug.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};

use crate::retriever::Retriever;

pub struct RaceRetriever {
    pub inners: Vec<Arc<dyn Retriever>>,
}

impl RaceRetriever {
    /// Build a race set. Panics if `inners` is empty (a race with
    /// no runners can't yield a winner).
    pub fn new(inners: Vec<Arc<dyn Retriever>>) -> Self {
        assert!(
            !inners.is_empty(),
            "RaceRetriever: need at least one inner retriever",
        );
        Self { inners }
    }
}

#[async_trait]
impl Retriever for RaceRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        use tokio::task::JoinSet;

        // Single-inner shortcut keeps the spawn-overhead off the hot
        // path for users who probe with a one-element race.
        if self.inners.len() == 1 {
            return self.inners[0].retrieve(query, k).await;
        }

        let mut set: JoinSet<Result<Vec<Document>>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let q = query.to_string();
            set.spawn(async move { inner.retrieve(&q, k).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(docs)) => {
                    set.abort_all();
                    return Ok(docs);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceRetriever: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Sleeps `delay_ms` then returns docs labelled with `marker`, or
    /// errors if `succeed=false`.
    struct DelayedRetriever {
        delay_ms: u64,
        marker: &'static str,
        succeed: bool,
        invocations: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Retriever for DelayedRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if !self.succeed {
                return Err(Error::other(format!("{} failed", self.marker)));
            }
            Ok((0..k.min(3))
                .map(|i| {
                    Document::new(format!("{}-{i}", self.marker)).with_id(format!("{}-{i}", self.marker))
                })
                .collect())
        }
    }

    fn arc_dr(
        marker: &'static str,
        delay_ms: u64,
        succeed: bool,
    ) -> (Arc<dyn Retriever>, Arc<AtomicUsize>) {
        let count = Arc::new(AtomicUsize::new(0));
        let r = DelayedRetriever {
            delay_ms,
            marker,
            succeed,
            invocations: count.clone(),
        };
        (Arc::new(r), count)
    }

    #[tokio::test]
    #[should_panic(expected = "at least one inner retriever")]
    async fn panics_on_empty() {
        let _ = RaceRetriever::new(vec![]);
    }

    #[tokio::test]
    async fn returns_first_winner() {
        // Fast finishes in 5ms, slow in 50ms — fast must win.
        let (fast, _) = arc_dr("fast", 5, true);
        let (slow, _) = arc_dr("slow", 50, true);
        let race = RaceRetriever::new(vec![fast, slow]);
        let docs = race.retrieve("q", 2).await.unwrap();
        assert!(docs.iter().all(|d| d.id.as_deref().unwrap().starts_with("fast")));
    }

    #[tokio::test]
    async fn falls_through_failures() {
        // First fails, second succeeds slowly — second wins.
        let (a, _) = arc_dr("a", 1, false);
        let (b, _) = arc_dr("b", 10, true);
        let race = RaceRetriever::new(vec![a, b]);
        let docs = race.retrieve("q", 1).await.unwrap();
        assert!(docs[0].id.as_deref().unwrap().starts_with("b"));
    }

    #[tokio::test]
    async fn aggregates_when_all_fail() {
        let (a, _) = arc_dr("a", 1, false);
        let (b, _) = arc_dr("b", 2, false);
        let (c, _) = arc_dr("c", 3, false);
        let race = RaceRetriever::new(vec![a, b, c]);
        let err = race.retrieve("q", 1).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("all 3 inners failed"), "got: {s}");
        assert!(s.contains("a failed"));
        assert!(s.contains("b failed"));
        assert!(s.contains("c failed"));
    }

    #[tokio::test]
    async fn single_inner_passes_through() {
        let (only, count) = arc_dr("only", 0, true);
        let race = RaceRetriever::new(vec![only]);
        let _ = race.retrieve("q", 1).await.unwrap();
        assert_eq!(count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn aborts_losers_after_winner() {
        // Fast wins in 5ms; slow sleeps 500ms. Wall-clock should be
        // closer to 5ms than 500ms.
        let (fast, _) = arc_dr("fast", 5, true);
        let (slow, _) = arc_dr("slow", 500, true);
        let race = RaceRetriever::new(vec![fast, slow]);
        let started = std::time::Instant::now();
        let _ = race.retrieve("q", 1).await.unwrap();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 200,
            "elapsed {elapsed_ms}ms — losers were not aborted",
        );
    }

    #[tokio::test]
    async fn preserves_winner_doc_count() {
        let (fast, _) = arc_dr("fast", 0, true);
        let race = RaceRetriever::new(vec![fast]);
        let docs = race.retrieve("q", 5).await.unwrap();
        assert_eq!(docs.len(), 3); // DelayedRetriever returns min(k, 3)
    }
}

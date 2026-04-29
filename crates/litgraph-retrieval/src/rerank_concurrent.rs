//! `rerank_concurrent` — fan out one reranker over N
//! `(query, candidates)` pairs via Tokio + Semaphore.
//!
//! Adds a fifth axis to the parallel-batch family across primary
//! async traits:
//!
//! | Iter | Helper                          | Trait               |
//! |------|---------------------------------|---------------------|
//! | 182  | `batch_concurrent`              | `ChatModel::invoke` |
//! | 183  | `embed_documents_concurrent`    | `Embeddings::*`     |
//! | 190  | `retrieve_concurrent`           | `Retriever::retrieve` |
//! | 191  | `tool_dispatch_concurrent`      | `Tool::run`         |
//! | 197  | `rerank_concurrent` (this)      | `Reranker::rerank`  |
//!
//! # Distinct from `EnsembleReranker` (iter 186)
//!
//! - `EnsembleReranker`: **N rerankers**, ONE `(query, candidates)`,
//!   fused via weighted RRF → quality-min single result.
//! - `rerank_concurrent`: ONE reranker, **N independent**
//!   `(query, candidates)` pairs → batch of N results.
//!
//! Real use cases:
//! - **Eval harness** scoring a single reranker over hundreds of
//!   `(query, gold-set)` pairs.
//! - **Plan-and-Execute**: an agent emits multiple sub-questions,
//!   each with its own retrieved candidates needing rerank.
//! - **Multi-tenant** dashboard issuing batch rerank requests.
//!
//! # Guarantees
//!
//! 1. Output index `i` matches input `i`, regardless of completion order.
//! 2. Per-call `Result` so one failure doesn't tank the rest.
//!    Use [`rerank_concurrent_fail_fast`] for all-or-nothing.
//! 3. `max_concurrency` enforced via [`tokio::sync::Semaphore`].

use std::sync::Arc;

use litgraph_core::{Document, Error, Progress, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::rerank::Reranker;

/// Fan out N `(query, candidates)` pairs against a single reranker
/// concurrently, capped at `max_concurrency` in flight.
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub async fn rerank_concurrent(
    reranker: Arc<dyn Reranker>,
    pairs: Vec<(String, Vec<Document>)>,
    top_k: usize,
    max_concurrency: usize,
) -> Vec<Result<Vec<Document>>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Document>>)> = JoinSet::new();

    for (idx, (query, candidates)) in pairs.into_iter().enumerate() {
        let sem = sem.clone();
        let reranker = reranker.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("rerank semaphore closed"))),
            };
            let r = reranker.rerank(&query, candidates, top_k).await;
            (idx, r)
        });
    }

    let n = set.len();
    let mut results: Vec<Option<Result<Vec<Document>>>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => results[idx] = Some(r),
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("rerank task join: {e}"))));
                }
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("rerank slot lost"))))
        .collect()
}

/// Counters maintained by [`rerank_concurrent_with_progress`].
/// Snapshot from any `Progress<RerankProgress>` observer to drive a
/// progress bar over a multi-pair eval run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RerankProgress {
    /// Total `(query, candidates)` pairs submitted (set on entry).
    pub total: u64,
    /// Pairs whose `rerank` has finished, success or failure.
    pub completed: u64,
    /// Total candidates submitted across all pairs (set on entry).
    /// Useful for eval reports that compute "X% of all candidates
    /// reranked" rather than "X% of pairs finished".
    pub total_candidates: u64,
    /// Total docs returned across all successful pairs.
    pub docs_returned: u64,
    /// Subset of `completed` that returned `Err`.
    pub errors: u64,
}

/// Same as [`rerank_concurrent`] but updates `progress` as each pair
/// completes. Sixth (and final) progress-aware sibling, completing
/// the family across all parallel-batch axes:
///
///   200  ingest_to_stream_with_progress             pipeline
///   205  batch_concurrent_with_progress             ChatModel
///   206  embed_documents_concurrent_with_progress   Embeddings
///   207  retrieve_concurrent_with_progress          Retriever
///   208  tool_dispatch_concurrent_with_progress     Tool
///   209  rerank_concurrent_with_progress (this)     Reranker
///
/// Real prod use: an eval harness scoring a reranker over hundreds
/// of `(query, gold-set)` pairs with a live counter. `total` and
/// `total_candidates` set up front so observers see run shape
/// before the first completion arrives.
pub async fn rerank_concurrent_with_progress(
    reranker: Arc<dyn Reranker>,
    pairs: Vec<(String, Vec<Document>)>,
    top_k: usize,
    max_concurrency: usize,
    progress: Progress<RerankProgress>,
) -> Vec<Result<Vec<Document>>> {
    if pairs.is_empty() {
        return Vec::new();
    }
    let total = pairs.len() as u64;
    let total_candidates: u64 = pairs.iter().map(|(_, c)| c.len() as u64).sum();
    let _ = progress.update(|p| RerankProgress {
        total,
        total_candidates,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Document>>)> = JoinSet::new();

    for (idx, (query, candidates)) in pairs.into_iter().enumerate() {
        let sem = sem.clone();
        let reranker = reranker.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("rerank semaphore closed"))),
            };
            let r = reranker.rerank(&query, candidates, top_k).await;
            (idx, r)
        });
    }

    let n = set.len();
    let mut results: Vec<Option<Result<Vec<Document>>>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => {
                let (n_docs, is_err) = match &r {
                    Ok(docs) => (docs.len() as u64, false),
                    Err(_) => (0, true),
                };
                results[idx] = Some(r);
                let _ = progress.update(|p| RerankProgress {
                    completed: p.completed + 1,
                    docs_returned: p.docs_returned + n_docs,
                    errors: p.errors + if is_err { 1 } else { 0 },
                    ..p.clone()
                });
            }
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("rerank task join: {e}"))));
                }
                let _ = progress.update(|p| RerankProgress {
                    completed: p.completed + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("rerank slot lost"))))
        .collect()
}

/// Fail-fast variant: returns `Err` on the first failed pair, drops
/// the rest. Output aligned to input only on success.
pub async fn rerank_concurrent_fail_fast(
    reranker: Arc<dyn Reranker>,
    pairs: Vec<(String, Vec<Document>)>,
    top_k: usize,
    max_concurrency: usize,
) -> Result<Vec<Vec<Document>>> {
    let n = pairs.len();
    let results = rerank_concurrent(reranker, pairs, top_k, max_concurrency).await;
    let mut out = Vec::with_capacity(n);
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Reverses the candidate order. Trivial deterministic reranker.
    /// Tracks peak concurrent invocations for the cap test.
    struct ReverseRerank {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        delay_ms: u64,
    }

    #[async_trait]
    impl Reranker for ReverseRerank {
        async fn rerank(
            &self,
            _query: &str,
            mut docs: Vec<Document>,
            top_k: usize,
        ) -> Result<Vec<Document>> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self
                    .peak
                    .compare_exchange(p, now, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            docs.reverse();
            docs.truncate(top_k);
            Ok(docs)
        }
    }

    fn probe(delay_ms: u64) -> (Arc<dyn Reranker>, Arc<AtomicUsize>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let r = ReverseRerank {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: peak.clone(),
            delay_ms,
        };
        (Arc::new(r), peak)
    }

    fn doc(id: &str) -> Document {
        Document {
            id: Some(id.into()),
            content: format!("doc {id}"),
            metadata: std::collections::HashMap::new(),
            score: None,
        }
    }

    #[tokio::test]
    async fn empty_input_returns_empty() {
        let (r, _peak) = probe(0);
        let out = rerank_concurrent(r, vec![], 5, 4).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_aligned_to_input_order() {
        let (r, _peak) = probe(2);
        let pairs = vec![
            ("q0".into(), vec![doc("a0"), doc("b0")]),
            ("q1".into(), vec![doc("a1"), doc("b1")]),
            ("q2".into(), vec![doc("a2"), doc("b2"), doc("c2")]),
        ];
        let out = rerank_concurrent(r, pairs, 10, 4).await;
        assert_eq!(out.len(), 3);
        // The reverser flips order; each slot's first id is the
        // original last candidate.
        assert_eq!(out[0].as_ref().unwrap()[0].id.as_deref(), Some("b0"));
        assert_eq!(out[1].as_ref().unwrap()[0].id.as_deref(), Some("b1"));
        assert_eq!(out[2].as_ref().unwrap()[0].id.as_deref(), Some("c2"));
    }

    #[tokio::test]
    async fn concurrency_cap_honoured() {
        let (r, peak) = probe(25);
        let pairs: Vec<_> = (0..15)
            .map(|i| (format!("q{i}"), vec![doc(&format!("d{i}"))]))
            .collect();
        let _ = rerank_concurrent(r, pairs, 5, 3).await;
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 3, "peak {observed} > cap 3");
        assert!(observed >= 2, "peak {observed} — concurrency never engaged");
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let (r, peak) = probe(2);
        let pairs: Vec<_> = (0..4)
            .map(|i| (format!("q{i}"), vec![doc("x")]))
            .collect();
        let _ = rerank_concurrent(r, pairs, 5, 0).await;
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn per_pair_failure_isolated() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait]
        impl Reranker for FailOnQ {
            async fn rerank(
                &self,
                query: &str,
                docs: Vec<Document>,
                top_k: usize,
            ) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(docs.into_iter().take(top_k).collect())
                }
            }
        }
        let r: Arc<dyn Reranker> = Arc::new(FailOnQ { bad: "q2".into() });
        let pairs: Vec<_> = (0..5)
            .map(|i| (format!("q{i}"), vec![doc("x")]))
            .collect();
        let out = rerank_concurrent(r, pairs, 1, 4).await;
        assert!(out[0].is_ok());
        assert!(out[1].is_ok());
        assert!(out[2].is_err());
        assert!(out[3].is_ok());
        assert!(out[4].is_ok());
    }

    #[tokio::test]
    async fn fail_fast_returns_first_error() {
        struct AlwaysFail;
        #[async_trait]
        impl Reranker for AlwaysFail {
            async fn rerank(
                &self,
                _q: &str,
                _d: Vec<Document>,
                _k: usize,
            ) -> Result<Vec<Document>> {
                Err(Error::other("nope"))
            }
        }
        let r: Arc<dyn Reranker> = Arc::new(AlwaysFail);
        let pairs: Vec<_> = (0..3)
            .map(|i| (format!("q{i}"), vec![doc("x")]))
            .collect();
        let res = rerank_concurrent_fail_fast(r, pairs, 1, 4).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn fail_fast_succeeds_when_all_ok() {
        let (r, _peak) = probe(0);
        let pairs: Vec<_> = (0..3)
            .map(|i| (format!("q{i}"), vec![doc("a"), doc("b")]))
            .collect();
        let out = rerank_concurrent_fail_fast(r, pairs, 1, 4).await.unwrap();
        assert_eq!(out.len(), 3);
        // Reverser flips → first elem is "b" for every pair.
        for ranked in &out {
            assert_eq!(ranked[0].id.as_deref(), Some("b"));
        }
    }

    #[tokio::test]
    async fn top_k_caps_each_slot() {
        let (r, _peak) = probe(0);
        let pairs: Vec<_> = (0..2)
            .map(|i| {
                (
                    format!("q{i}"),
                    vec![doc("a"), doc("b"), doc("c"), doc("d")],
                )
            })
            .collect();
        let out = rerank_concurrent(r, pairs, 2, 4).await;
        for r in &out {
            assert_eq!(r.as_ref().unwrap().len(), 2);
        }
    }

    // ---- rerank_concurrent_with_progress tests ------------------------

    #[tokio::test]
    async fn progress_total_and_candidates_set_at_start() {
        let (r, _peak) = probe(0);
        let pairs: Vec<_> = vec![
            ("q1".into(), vec![doc("a"), doc("b")]),
            ("q2".into(), vec![doc("c"), doc("d"), doc("e")]),
            ("q3".into(), vec![doc("f")]),
        ];
        let progress = Progress::new(RerankProgress::default());
        let obs = progress.observer();
        let _ = rerank_concurrent_with_progress(r, pairs, 5, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 3);
        assert_eq!(snap.total_candidates, 6); // 2 + 3 + 1
        assert_eq!(snap.completed, 3);
        assert_eq!(snap.errors, 0);
        // Reverser returns up to top_k from each set: min(top_k, candidates_in_pair).
        assert_eq!(snap.docs_returned, 6);
    }

    #[tokio::test]
    async fn progress_records_per_pair_errors() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait]
        impl Reranker for FailOnQ {
            async fn rerank(
                &self,
                query: &str,
                docs: Vec<Document>,
                top_k: usize,
            ) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(docs.into_iter().take(top_k).collect())
                }
            }
        }
        let r: Arc<dyn Reranker> = Arc::new(FailOnQ {
            bad: "q2".into(),
        });
        let pairs: Vec<_> = (0..5)
            .map(|i| (format!("q{i}"), vec![doc("x"), doc("y")]))
            .collect();
        let progress = Progress::new(RerankProgress::default());
        let obs = progress.observer();
        let _ = rerank_concurrent_with_progress(r, pairs, 2, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 5);
        assert_eq!(snap.total_candidates, 10);
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.errors, 1);
        // 4 successful pairs × 2 docs = 8 returned.
        assert_eq!(snap.docs_returned, 8);
    }

    #[tokio::test]
    async fn progress_observer_polls_mid_run() {
        let (r, _peak) = probe(15);
        let pairs: Vec<_> = (0..6)
            .map(|i| (format!("q{i}"), vec![doc("a"), doc("b")]))
            .collect();
        let progress = Progress::new(RerankProgress::default());
        let mut obs = progress.observer();
        let progress_clone = progress.clone();
        let h = tokio::spawn(async move {
            rerank_concurrent_with_progress(r, pairs, 1, 2, progress_clone).await
        });
        let _ = obs.changed().await;
        let mid = obs.snapshot();
        assert_eq!(mid.total, 6);
        assert_eq!(mid.total_candidates, 12);
        let _ = h.await.unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 6);
    }

    #[tokio::test]
    async fn progress_empty_pairs_no_updates() {
        let (r, _peak) = probe(0);
        let progress = Progress::new(RerankProgress::default());
        let obs = progress.observer();
        let out = rerank_concurrent_with_progress(r, vec![], 5, 4, progress).await;
        assert!(out.is_empty());
        assert_eq!(obs.snapshot(), RerankProgress::default());
    }
}

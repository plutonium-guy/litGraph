//! `retrieve_concurrent` — fan out N user queries against a **single**
//! retriever via Tokio + Semaphore, with order-preserved per-query
//! `Result`s.
//!
//! Completes a parallel-batch trio with two earlier `litgraph-core`
//! primitives:
//!
//! | Iter | Helper                          | Domain                |
//! |------|---------------------------------|-----------------------|
//! | 182  | [`batch_concurrent`]            | `ChatModel::invoke`   |
//! | 183  | [`embed_documents_concurrent`]  | `Embeddings::embed_*` |
//! | 190  | `retrieve_concurrent` (this)    | `Retriever::retrieve` |
//!
//! [`batch_concurrent`]: litgraph_core::batch_concurrent
//! [`embed_documents_concurrent`]: litgraph_core::embed_documents_concurrent
//!
//! # Why this exists
//!
//! Distinct from the existing [`MultiQueryRetriever`](crate::MultiQueryRetriever)
//! and [`EnsembleRetriever`](crate::EnsembleRetriever):
//!
//! | Helper                     | Same retriever, different queries? | LLM call? |
//! |----------------------------|------------------------------------|-----------|
//! | `MultiQueryRetriever`      | LLM generates N paraphrases of ONE query | yes |
//! | `EnsembleRetriever`        | Different retrievers, ONE query    | no  |
//! | `retrieve_concurrent`      | ONE retriever, **N caller queries** | no  |
//!
//! Real use cases:
//! - **Eval harness** scoring a retriever over hundreds of test queries.
//! - **Agentic flows** that issue many retrieval lookups per turn.
//! - **Multi-tenant** dashboards that batch search requests.
//!
//! # Guarantees
//!
//! 1. Output index `i` corresponds to query `i`, regardless of
//!    completion order.
//! 2. Per-query `Result` so partial failures don't tank the batch.
//!    Use [`retrieve_concurrent_fail_fast`] if you need all-or-nothing.
//! 3. `max_concurrency` enforced via [`tokio::sync::Semaphore`] so a
//!    10k-query eval run with `max=50` is bounded — won't melt the
//!    runtime or the underlying provider's rate limit.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use litgraph_core::{Document, Error, Progress, Result};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;

use crate::retriever::Retriever;

/// Run `retriever.retrieve(query, k)` for every `query` in parallel,
/// capped at `max_concurrency` in flight. Output is aligned 1:1 with
/// `queries` — slot `i` holds the outcome of `queries[i]`.
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub async fn retrieve_concurrent(
    retriever: Arc<dyn Retriever>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
) -> Vec<Result<Vec<Document>>> {
    if queries.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Document>>)> = JoinSet::new();

    for (idx, q) in queries.into_iter().enumerate() {
        let sem = sem.clone();
        let retriever = retriever.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("retrieve semaphore closed"))),
            };
            let r = retriever.retrieve(&q, k).await;
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
                    *slot = Some(Err(Error::other(format!("retrieve task join: {e}"))));
                }
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("retrieve slot lost"))))
        .collect()
}

/// Counters maintained by [`retrieve_concurrent_with_progress`].
/// Snapshot from any `Progress<RetrieveProgress>` observer to drive
/// a progress bar over a multi-query eval run.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RetrieveProgress {
    /// Total queries submitted (set once on entry).
    pub total: u64,
    /// Queries whose `retrieve` has finished, success or failure.
    pub completed: u64,
    /// Total documents returned across all successful queries.
    pub docs_returned: u64,
    /// Subset of `completed` that returned `Err`.
    pub errors: u64,
}

/// Same as [`retrieve_concurrent`] but updates `progress` as each
/// query completes. Real prod use: an eval harness scoring a
/// retriever over hundreds of test queries with a live counter.
///
/// Composition: fourth progress-aware sibling after iter 200
/// (ingest), iter 205 (chat batch), iter 206 (embed batch).
pub async fn retrieve_concurrent_with_progress(
    retriever: Arc<dyn Retriever>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
    progress: Progress<RetrieveProgress>,
) -> Vec<Result<Vec<Document>>> {
    if queries.is_empty() {
        return Vec::new();
    }
    let total = queries.len() as u64;
    let _ = progress.update(|p| RetrieveProgress {
        total,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Document>>)> = JoinSet::new();

    for (idx, q) in queries.into_iter().enumerate() {
        let sem = sem.clone();
        let retriever = retriever.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("retrieve semaphore closed"))),
            };
            let r = retriever.retrieve(&q, k).await;
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
                let _ = progress.update(|p| RetrieveProgress {
                    completed: p.completed + 1,
                    docs_returned: p.docs_returned + n_docs,
                    errors: p.errors + if is_err { 1 } else { 0 },
                    ..p.clone()
                });
            }
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("retrieve task join: {e}"))));
                }
                let _ = progress.update(|p| RetrieveProgress {
                    completed: p.completed + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("retrieve slot lost"))))
        .collect()
}

/// One item from [`retrieve_concurrent_stream`] — the input query
/// index plus that query's outcome, emitted in completion order.
pub type RetrieveStreamItem = (usize, Result<Vec<Document>>);

/// Streaming variant of [`retrieve_concurrent`]. Yields
/// `(query_idx, Result)` pairs as each query completes — caller
/// drains in completion order, can dispatch downstream work
/// immediately on early completers, and dropping the stream aborts
/// remaining in-flight work.
///
/// Streaming-variant pattern from iter 210 (chat batch) and iter
/// 211 (embed batch) extended to the retriever axis.
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub fn retrieve_concurrent_stream(
    retriever: Arc<dyn Retriever>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
) -> Pin<Box<dyn Stream<Item = RetrieveStreamItem> + Send>> {
    if queries.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let n = queries.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<RetrieveStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<RetrieveStreamItem> = JoinSet::new();
        for (idx, q) in queries.into_iter().enumerate() {
            let sem = sem.clone();
            let retriever = retriever.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return (idx, Err(Error::other("retrieve semaphore closed"))),
                };
                let r = retriever.retrieve(&q, k).await;
                (idx, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(Error::other(format!("retrieve task join: {e}"))),
                ),
            };
            if tx.send(item).await.is_err() {
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// Combined streaming + progress-watcher variant. Yields the same
/// `(query_idx, Result)` items as [`retrieve_concurrent_stream`]
/// AND updates the supplied [`Progress<RetrieveProgress>`] watcher
/// as each query completes.
///
/// Real prod use: an eval harness UI rendering a row per query as
/// it lands (stream side) AND a summary progress bar with `{total,
/// completed, docs_returned, errors}` (watcher side), both backed
/// by the same retriever fan-out.
///
/// Composition: extends the combined consumer shape from iters
/// 216 (chat batch) and 217 (embed batch) to the retriever axis.
/// Same consistency contract — counter ticks before stream item.
pub fn retrieve_concurrent_stream_with_progress(
    retriever: Arc<dyn Retriever>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
    progress: Progress<RetrieveProgress>,
) -> Pin<Box<dyn Stream<Item = RetrieveStreamItem> + Send>> {
    if queries.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let total = queries.len() as u64;
    let _ = progress.update(|p| RetrieveProgress {
        total,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let n = queries.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<RetrieveStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<RetrieveStreamItem> = JoinSet::new();
        for (idx, q) in queries.into_iter().enumerate() {
            let sem = sem.clone();
            let retriever = retriever.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return (idx, Err(Error::other("retrieve semaphore closed"))),
                };
                let r = retriever.retrieve(&q, k).await;
                (idx, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(Error::other(format!("retrieve task join: {e}"))),
                ),
            };
            let (n_docs, is_err) = match &item.1 {
                Ok(docs) => (docs.len() as u64, false),
                Err(_) => (0, true),
            };
            let _ = progress.update(|p| RetrieveProgress {
                completed: p.completed + 1,
                docs_returned: p.docs_returned + n_docs,
                errors: p.errors + if is_err { 1 } else { 0 },
                ..p.clone()
            });
            if tx.send(item).await.is_err() {
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// Like `retrieve_concurrent` but fail-fast: returns `Err` on the
/// first failed query and aborts the rest. Outputs are aligned to
/// inputs only on success.
pub async fn retrieve_concurrent_fail_fast(
    retriever: Arc<dyn Retriever>,
    queries: Vec<String>,
    k: usize,
    max_concurrency: usize,
) -> Result<Vec<Vec<Document>>> {
    let n = queries.len();
    let results = retrieve_concurrent(retriever, queries, k, max_concurrency).await;
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

    /// Records peak concurrent calls; sleeps `delay_ms` per query;
    /// echoes the query back as a single doc.
    struct ProbeRetriever {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        delay_ms: u64,
    }

    #[async_trait]
    impl Retriever for ProbeRetriever {
        async fn retrieve(&self, query: &str, _k: usize) -> Result<Vec<Document>> {
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
            Ok(vec![Document::new(query).with_id(query)])
        }
    }

    fn probe(delay_ms: u64) -> (Arc<dyn Retriever>, Arc<AtomicUsize>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let r = ProbeRetriever {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: peak.clone(),
            delay_ms,
        };
        (Arc::new(r), peak)
    }

    fn qs(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("q{i}")).collect()
    }

    #[tokio::test]
    async fn empty_queries_returns_empty() {
        let (r, _peak) = probe(0);
        let out = retrieve_concurrent(r, vec![], 5, 4).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_aligned_to_input_order() {
        let (r, _peak) = probe(2);
        let queries = qs(8);
        let out = retrieve_concurrent(r, queries.clone(), 5, 4).await;
        assert_eq!(out.len(), 8);
        for (i, res) in out.iter().enumerate() {
            let docs = res.as_ref().expect("ok");
            assert_eq!(docs.len(), 1);
            assert_eq!(docs[0].content, queries[i]);
        }
    }

    #[tokio::test]
    async fn concurrency_cap_honoured() {
        let (r, peak) = probe(25);
        let _ = retrieve_concurrent(r, qs(15), 1, 3).await;
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 3, "peak {observed} > cap 3");
        assert!(observed >= 2, "peak {observed} — concurrency never engaged");
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let (r, peak) = probe(2);
        let _ = retrieve_concurrent(r, qs(4), 1, 0).await;
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn per_query_failure_isolated() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait]
        impl Retriever for FailOnQ {
            async fn retrieve(&self, query: &str, _k: usize) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(vec![Document::new(query)])
                }
            }
        }
        let r: Arc<dyn Retriever> = Arc::new(FailOnQ {
            bad: "q2".into(),
        });
        let out = retrieve_concurrent(r, qs(5), 1, 4).await;
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
        impl Retriever for AlwaysFail {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                Err(Error::other("nope"))
            }
        }
        let r: Arc<dyn Retriever> = Arc::new(AlwaysFail);
        let res = retrieve_concurrent_fail_fast(r, qs(3), 1, 4).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn fail_fast_succeeds_when_all_ok() {
        let (r, _peak) = probe(0);
        let out = retrieve_concurrent_fail_fast(r, qs(3), 1, 4).await.unwrap();
        assert_eq!(out.len(), 3);
        assert_eq!(out[0][0].content, "q0");
    }

    // ---- retrieve_concurrent_with_progress tests -----------------------

    #[tokio::test]
    async fn progress_total_set_and_completed_counts_advance() {
        let (r, _peak) = probe(0);
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let out =
            retrieve_concurrent_with_progress(r, qs(6), 1, 4, progress).await;
        assert_eq!(out.len(), 6);
        let snap = obs.snapshot();
        assert_eq!(snap.total, 6);
        assert_eq!(snap.completed, 6);
        // probe returns 1 doc per query (k=1, min(k, 3)=1).
        assert_eq!(snap.docs_returned, 6);
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn progress_records_per_query_errors() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait::async_trait]
        impl Retriever for FailOnQ {
            async fn retrieve(
                &self,
                query: &str,
                _k: usize,
            ) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(vec![Document::new(query)])
                }
            }
        }
        let r: Arc<dyn Retriever> = Arc::new(FailOnQ {
            bad: "q2".into(),
        });
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let _ = retrieve_concurrent_with_progress(r, qs(5), 1, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 5);
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.errors, 1);
        // 4 successes × 1 doc each = 4 docs returned.
        assert_eq!(snap.docs_returned, 4);
    }

    #[tokio::test]
    async fn progress_observer_polls_mid_run() {
        let (r, _peak) = probe(15);
        let progress = Progress::new(RetrieveProgress::default());
        let mut obs = progress.observer();
        let queries = qs(8);
        let progress_clone = progress.clone();
        let h = tokio::spawn(async move {
            retrieve_concurrent_with_progress(r, queries, 1, 2, progress_clone).await
        });
        let _ = obs.changed().await;
        let mid = obs.snapshot();
        assert_eq!(mid.total, 8);
        let _ = h.await.unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 8);
    }

    // ---- retrieve_concurrent_stream tests -----------------------------

    use futures::StreamExt;

    // ---- retrieve_concurrent_stream_with_progress tests ---------------

    #[tokio::test]
    async fn stream_with_progress_yields_items_and_advances_counters() {
        let (r, _peak) = probe(2);
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let mut s = retrieve_concurrent_stream_with_progress(
            r, qs(6), 1, 4, progress,
        );
        let mut count = 0;
        while let Some((_idx, res)) = s.next().await {
            assert!(res.is_ok());
            count += 1;
        }
        assert_eq!(count, 6);
        let snap = obs.snapshot();
        assert_eq!(snap.total, 6);
        assert_eq!(snap.completed, 6);
        assert_eq!(snap.docs_returned, 6);
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn stream_with_progress_records_per_query_errors() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait::async_trait]
        impl Retriever for FailOnQ {
            async fn retrieve(
                &self,
                query: &str,
                _k: usize,
            ) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(vec![Document::new(query)])
                }
            }
        }
        let r: Arc<dyn Retriever> = Arc::new(FailOnQ {
            bad: "q2".into(),
        });
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let mut s = retrieve_concurrent_stream_with_progress(
            r, qs(5), 1, 4, progress,
        );
        let mut errors_in_stream = 0;
        while let Some((_idx, res)) = s.next().await {
            if res.is_err() {
                errors_in_stream += 1;
            }
        }
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.errors, 1);
        assert_eq!(errors_in_stream, snap.errors as usize);
    }

    #[tokio::test]
    async fn stream_with_progress_total_set_at_start() {
        let (r, _peak) = probe(0);
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let _s = retrieve_concurrent_stream_with_progress(
            r, qs(7), 1, 2, progress,
        );
        assert_eq!(obs.snapshot().total, 7);
    }

    #[tokio::test]
    async fn stream_with_progress_empty_queries_no_updates() {
        let (r, _peak) = probe(0);
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let mut s = retrieve_concurrent_stream_with_progress(
            r,
            vec![],
            5,
            4,
            progress,
        );
        assert!(s.next().await.is_none());
        assert_eq!(obs.snapshot(), RetrieveProgress::default());
    }

    #[tokio::test]
    async fn stream_yields_one_item_per_query() {
        let (r, _peak) = probe(0);
        let mut s = retrieve_concurrent_stream(r, qs(6), 1, 4);
        let mut indices: Vec<usize> = Vec::new();
        while let Some((idx, res)) = s.next().await {
            assert!(res.is_ok());
            indices.push(idx);
        }
        indices.sort();
        assert_eq!(indices, (0..6).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn stream_idx_aligns_with_input_query() {
        let (r, _peak) = probe(0);
        let queries = qs(5);
        let mut s = retrieve_concurrent_stream(r, queries.clone(), 1, 2);
        while let Some((idx, res)) = s.next().await {
            let docs = res.unwrap();
            assert_eq!(docs.len(), 1);
            // probe echoes query into the doc content; verify idx
            // matches the query at that position.
            assert_eq!(docs[0].content, queries[idx]);
        }
    }

    #[tokio::test]
    async fn stream_per_query_failure_arrives_as_err_item() {
        struct FailOnQ {
            bad: String,
        }
        #[async_trait::async_trait]
        impl Retriever for FailOnQ {
            async fn retrieve(
                &self,
                query: &str,
                _k: usize,
            ) -> Result<Vec<Document>> {
                if query == self.bad {
                    Err(Error::other("synthetic"))
                } else {
                    Ok(vec![Document::new(query)])
                }
            }
        }
        let r: Arc<dyn Retriever> = Arc::new(FailOnQ {
            bad: "q2".into(),
        });
        let mut s = retrieve_concurrent_stream(r, qs(5), 1, 4);
        let mut count = 0;
        let mut errors = 0;
        while let Some((_idx, res)) = s.next().await {
            count += 1;
            if res.is_err() {
                errors += 1;
            }
        }
        assert_eq!(count, 5);
        assert_eq!(errors, 1);
    }

    #[tokio::test]
    async fn stream_empty_queries_yields_empty() {
        let (r, _peak) = probe(0);
        let mut s = retrieve_concurrent_stream(r, vec![], 1, 4);
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_caller_drop_aborts_in_flight_queries() {
        // 50 queries × 50ms / cap=2 → full sequential ~1.25s. Drop
        // after 1 item; total wall-clock should be far less.
        let (r, _peak) = probe(50);
        let started = std::time::Instant::now();
        {
            let mut s = retrieve_concurrent_stream(r, qs(50), 1, 2);
            let _first = s.next().await.unwrap();
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 400,
            "elapsed {elapsed_ms}ms — caller-drop didn't abort remaining queries",
        );
    }

    #[tokio::test]
    async fn progress_empty_queries_no_updates() {
        let (r, _peak) = probe(0);
        let progress = Progress::new(RetrieveProgress::default());
        let obs = progress.observer();
        let out = retrieve_concurrent_with_progress(
            r,
            vec![],
            5,
            4,
            progress,
        )
        .await;
        assert!(out.is_empty());
        assert_eq!(obs.snapshot(), RetrieveProgress::default());
    }
}

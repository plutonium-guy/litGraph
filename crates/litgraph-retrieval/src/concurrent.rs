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

use std::sync::Arc;

use litgraph_core::{Document, Error, Result};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

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
}

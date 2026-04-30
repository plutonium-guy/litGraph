//! `TimeoutRetriever` — per-call deadline wrapper for any
//! [`Retriever`].
//!
//! Mirrors `TimeoutChatModel` / `TimeoutEmbeddings` (iter 194)
//! pattern: wrap any inner `Retriever` so each `retrieve` call
//! must complete within `timeout`. On timeout, returns
//! `Error::Timeout`; the inner future is cancelled
//! (`tokio::time::timeout` drops it, releasing whatever
//! HTTP/DB connection it held).
//!
//! # Why this exists
//!
//! Some vector stores have unbounded p99 latency under load
//! (HNSW with cold pages, pgvector under heavy write
//! contention). An interactive UI that issues a retrieval and
//! waits 30s degrades the user experience badly. Wrap the
//! retriever with a 2s deadline; let the agent fall through to
//! a backup retriever (or a degraded "no context" answer) on
//! timeout instead of hanging the request.
//!
//! # Composition
//!
//! Stacks naturally with the rest of the retriever toolkit:
//! `TimeoutRetriever(EnsembleRetriever([primary, backup]))` to
//! enforce a deadline on the ensemble; or
//! `EnsembleRetriever([TimeoutRetriever(slow), fast])` to
//! short-circuit only the slow branch.
//!
//! # Why not a generic combinator
//!
//! `tokio::time::timeout(d, retriever.retrieve(...))` already
//! works. The wrapper exists so `TimeoutRetriever` can be
//! passed wherever an `Arc<dyn Retriever>` is expected — i.e.,
//! it composes through `EnsembleRetriever`, `RagFusion`,
//! `RerankingRetriever`, etc., without each composer needing to
//! add per-branch timeout knobs.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};

use crate::retriever::Retriever;

pub struct TimeoutRetriever {
    pub inner: Arc<dyn Retriever>,
    pub timeout: Duration,
}

impl TimeoutRetriever {
    pub fn new(inner: Arc<dyn Retriever>, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

#[async_trait]
impl Retriever for TimeoutRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        match tokio::time::timeout(self.timeout, self.inner.retrieve(query, k)).await {
            Ok(r) => r,
            Err(_) => Err(Error::Timeout),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    /// Sleeps `delay_ms` then returns a fixed Vec.
    struct DelayRetriever {
        delay_ms: u64,
        docs: Vec<Document>,
    }

    #[async_trait]
    impl Retriever for DelayRetriever {
        async fn retrieve(&self, _query: &str, _k: usize) -> Result<Vec<Document>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(self.docs.clone())
        }
    }

    #[tokio::test]
    async fn under_deadline_returns_inner_result() {
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 10,
            docs: vec![doc("a"), doc("b")],
        });
        let r = TimeoutRetriever::new(inner, Duration::from_millis(100));
        let out = r.retrieve("q", 5).await.unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].id.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn over_deadline_returns_timeout_error() {
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 100,
            docs: vec![doc("a")],
        });
        let r = TimeoutRetriever::new(inner, Duration::from_millis(20));
        let started = std::time::Instant::now();
        let err = r.retrieve("q", 5).await.unwrap_err();
        let elapsed = started.elapsed();
        assert!(matches!(err, Error::Timeout));
        // Returned within ~deadline, NOT the inner's full delay.
        assert!(
            elapsed < Duration::from_millis(80),
            "timeout didn't drop inner future: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn zero_timeout_fails_immediately() {
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 50,
            docs: vec![doc("a")],
        });
        let r = TimeoutRetriever::new(inner, Duration::from_millis(0));
        let err = r.retrieve("q", 5).await.unwrap_err();
        assert!(matches!(err, Error::Timeout));
    }

    #[tokio::test]
    async fn inner_error_is_propagated_unchanged() {
        struct Failing;
        #[async_trait]
        impl Retriever for Failing {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                Err(Error::other("synthetic"))
            }
        }
        let inner: Arc<dyn Retriever> = Arc::new(Failing);
        let r = TimeoutRetriever::new(inner, Duration::from_millis(100));
        let err = r.retrieve("q", 5).await.unwrap_err();
        // Not Timeout — passthrough of the inner error.
        assert!(!matches!(err, Error::Timeout));
        assert!(err.to_string().contains("synthetic"));
    }
}

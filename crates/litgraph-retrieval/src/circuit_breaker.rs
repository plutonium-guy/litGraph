//! `CircuitBreakerRetriever` — fail-fast wrapper around any
//! [`Retriever`] using the iter-243 [`CircuitBreaker`] primitive.
//!
//! Mirrors `CircuitBreakerChatModel` (iter 244) /
//! `CircuitBreakerEmbeddings` (iter 245) on the retriever axis.
//! With this iter the breaker covers chat / embed / retriever —
//! the three primary axes where it makes sense (tool axis has
//! different resilience semantics via iter-159 `TimeoutTool` /
//! `RetryTool`).
//!
//! # Why
//!
//! Vector stores can go down. When pgvector / qdrant / weaviate
//! is failing, retrying every retrieval just amplifies load
//! against the sick service and delays recovery. After N
//! consecutive failures the breaker opens; subsequent retrievals
//! return `Error::Provider("circuit breaker open")` for the
//! cooldown window so agents can fall through to a backup
//! retriever (e.g., `EnsembleRetriever([primary_with_breaker,
//! backup_bm25])`) immediately rather than spending 30s
//! retrying.
//!
//! After cooldown, a single probe call is allowed. If it
//! succeeds the breaker closes; if it fails the cooldown resets.
//!
//! # Composition
//!
//! Stack with the retry/timeout wrappers for a full prod chain:
//!
//! ```ignore
//! let inner = Arc::new(MyVectorRetriever::new(...));
//! let timed = Arc::new(TimeoutRetriever::new(inner, Duration::from_secs(2)));
//! let breaker = Arc::new(CircuitBreaker::new(5, Duration::from_secs(30)));
//! let cb = Arc::new(CircuitBreakerRetriever::new(timed, breaker));
//! let retried = Arc::new(RetryingRetriever::new(cb, RetryConfig::default()));
//! ```
//!
//! Outer-to-inner: retry → breaker → timeout → store. Retry sees
//! `Error::Provider("circuit breaker open")` (which doesn't
//! match the 5xx pattern in `is_transient`) and exits without
//! retry — exactly the right behavior, since hammering an open
//! breaker just delays recovery.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{
    CircuitBreaker, CircuitCallError, Document, Error, Result,
};

use crate::retriever::Retriever;

pub struct CircuitBreakerRetriever {
    pub inner: Arc<dyn Retriever>,
    pub breaker: Arc<CircuitBreaker>,
}

impl CircuitBreakerRetriever {
    pub fn new(inner: Arc<dyn Retriever>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }
}

#[async_trait]
impl Retriever for CircuitBreakerRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let r = self
            .breaker
            .call(move || async move { inner.retrieve(&q, k).await })
            .await;
        match r {
            Ok(docs) => Ok(docs),
            Err(CircuitCallError::CircuitOpen) => {
                Err(Error::Provider("circuit breaker open".into()))
            }
            Err(CircuitCallError::Inner(e)) => Err(e),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    /// Always-fails retriever. Counts invocations so we can verify
    /// the breaker short-circuits without invoking inner.
    struct AlwaysFailRetriever {
        seen: AtomicU32,
    }
    #[async_trait]
    impl Retriever for AlwaysFailRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Err(Error::provider("502 sick upstream"))
        }
    }

    struct AlwaysOkRetriever;
    #[async_trait]
    impl Retriever for AlwaysOkRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            Ok(vec![doc("a")])
        }
    }

    #[tokio::test]
    async fn passes_through_inner_errors_until_threshold() {
        let inner = Arc::new(AlwaysFailRetriever {
            seen: AtomicU32::new(0),
        });
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(60)));
        let cb = CircuitBreakerRetriever::new(
            inner.clone() as Arc<dyn Retriever>,
            breaker.clone(),
        );
        for _ in 0..2 {
            let err = cb.retrieve("q", 5).await.unwrap_err();
            assert!(
                matches!(err, Error::Provider(ref m) if m.contains("502")),
                "expected pass-through, got {err:?}",
            );
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn short_circuits_after_threshold() {
        let inner = Arc::new(AlwaysFailRetriever {
            seen: AtomicU32::new(0),
        });
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerRetriever::new(
            inner.clone() as Arc<dyn Retriever>,
            breaker.clone(),
        );
        for _ in 0..2 {
            let _ = cb.retrieve("q", 5).await;
        }
        // Subsequent calls fail-fast WITHOUT invoking inner.
        for _ in 0..3 {
            let err = cb.retrieve("q", 5).await.unwrap_err();
            assert!(
                matches!(err, Error::Provider(ref m) if m.contains("circuit breaker open")),
            );
        }
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            2,
            "inner was invoked while breaker was open",
        );
    }

    #[tokio::test]
    async fn closed_passes_successes_through() {
        let inner: Arc<dyn Retriever> = Arc::new(AlwaysOkRetriever);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerRetriever::new(inner, breaker);
        for _ in 0..5 {
            let docs = cb.retrieve("q", 5).await.unwrap();
            assert_eq!(docs.len(), 1);
        }
    }

    #[tokio::test]
    async fn half_open_probe_success_closes_breaker() {
        // Use a mutable retriever that fails N times then succeeds.
        struct FailNThenOk {
            seen: AtomicU32,
            fail_first: u32,
        }
        #[async_trait]
        impl Retriever for FailNThenOk {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst);
                if n < self.fail_first {
                    Err(Error::provider("502 sick"))
                } else {
                    Ok(vec![doc("recovered")])
                }
            }
        }
        let inner: Arc<dyn Retriever> = Arc::new(FailNThenOk {
            seen: AtomicU32::new(0),
            fail_first: 1,
        });
        let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_millis(20)));
        let cb = CircuitBreakerRetriever::new(inner, breaker);
        // First call fails, breaker opens.
        let _ = cb.retrieve("q", 5).await;
        // Wait past cooldown.
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Probe succeeds → breaker closes; result returned.
        let docs = cb.retrieve("q", 5).await.unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("recovered"));
        // Subsequent calls go through normally.
        let docs2 = cb.retrieve("q", 5).await.unwrap();
        assert_eq!(docs2[0].id.as_deref(), Some("recovered"));
    }
}

//! `RetryingRetriever` — retry transient retrieval errors with
//! jittered exponential backoff.
//!
//! Mirrors `RetryingChatModel` / `RetryingEmbeddings` (early
//! resilience iters) for the retriever axis. With this iter the
//! retry-wrapper trio is complete: chat / embed / retriever.
//!
//! # What's retried
//!
//! - `Error::RateLimited` — common from managed vector stores
//!   (Qdrant Cloud, Weaviate Cloud) that throttle reads under
//!   load.
//! - `Error::Timeout` — pairs with [`crate::TimeoutRetriever`]
//!   (iter 271) so a `Retrying(Timeout(inner))` chain
//!   automatically retries on per-call deadline expiry.
//! - `Error::Provider(s)` where `s` matches a 5xx / connection-
//!   reset pattern — covers transient network errors.
//!
//! Everything else (`InvalidInput`, parse errors, etc) is
//! terminal — replays just waste cycles.
//!
//! # Composition
//!
//! Stack with `TimeoutRetriever` for "retry on per-call deadline":
//!
//! ```ignore
//! let inner = Arc::new(MyVectorRetriever::new(...));
//! let timed = Arc::new(TimeoutRetriever::new(inner, Duration::from_secs(2)));
//! let retried = Arc::new(RetryingRetriever::new(timed, RetryConfig::default()));
//! ```
//!
//! Or stack with `EnsembleRetriever` so each branch has its own
//! retry policy.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use litgraph_core::{Document, Error, Result};

use crate::retriever::Retriever;

/// Backoff schedule. Same shape as the resilience crate's
/// `RetryConfig` — kept independent so retrieval doesn't
/// circularly depend on resilience.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub min_delay: Duration,
    pub max_delay: Duration,
    pub factor: f32,
    pub max_times: usize,
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            min_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(10),
            factor: 2.0,
            max_times: 5,
            jitter: true,
        }
    }
}

impl RetryConfig {
    fn to_builder(&self) -> ExponentialBuilder {
        let mut b = ExponentialBuilder::default()
            .with_min_delay(self.min_delay)
            .with_max_delay(self.max_delay)
            .with_factor(self.factor)
            .with_max_times(self.max_times);
        if self.jitter {
            b = b.with_jitter();
        }
        b
    }
}

/// Classify whether an `Error` is worth retrying.
fn is_transient(e: &Error) -> bool {
    match e {
        Error::RateLimited { .. } => true,
        Error::Timeout => true,
        Error::Provider(msg) => {
            let m = msg.to_ascii_lowercase();
            m.contains("500 ")
                || m.contains("502 ")
                || m.contains("503 ")
                || m.contains("504 ")
                || m.contains("connection reset")
                || m.contains("connection closed")
                || m.contains("connect error")
        }
        _ => false,
    }
}

pub struct RetryingRetriever {
    pub inner: Arc<dyn Retriever>,
    pub cfg: RetryConfig,
}

impl RetryingRetriever {
    pub fn new(inner: Arc<dyn Retriever>, cfg: RetryConfig) -> Self {
        Self { inner, cfg }
    }
}

#[async_trait]
impl Retriever for RetryingRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let inner = self.inner.clone();
        let q = query.to_string();
        let op = move || {
            let inner = inner.clone();
            let q = q.clone();
            async move { inner.retrieve(&q, k).await }
        };
        op.retry(self.cfg.to_builder())
            .when(is_transient)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    /// Errors first N calls, then succeeds.
    struct FlakyRetriever {
        fails_remaining: AtomicU32,
        kind: ErrKind,
        docs: Vec<Document>,
    }

    enum ErrKind {
        RateLimited,
        Provider5xx,
        InvalidInput,
    }

    #[async_trait]
    impl Retriever for FlakyRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    ErrKind::RateLimited => {
                        Error::RateLimited { retry_after_ms: None }
                    }
                    ErrKind::Provider5xx => Error::provider("502 bad gateway"),
                    ErrKind::InvalidInput => Error::invalid("bad query"),
                });
            }
            Ok(self.docs.clone())
        }
    }

    fn quick_cfg() -> RetryConfig {
        RetryConfig {
            min_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(10),
            factor: 2.0,
            max_times: 5,
            jitter: false,
        }
    }

    #[tokio::test]
    async fn retries_rate_limited_then_succeeds() {
        let inner = Arc::new(FlakyRetriever {
            fails_remaining: AtomicU32::new(2),
            kind: ErrKind::RateLimited,
            docs: vec![doc("a"), doc("b")],
        });
        let r = RetryingRetriever::new(inner.clone() as Arc<dyn Retriever>, quick_cfg());
        let docs = r.retrieve("q", 5).await.unwrap();
        assert_eq!(docs.len(), 2);
        // 0 remaining → all 2 failures consumed.
        assert_eq!(inner.fails_remaining.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn retries_5xx_then_succeeds() {
        let inner = Arc::new(FlakyRetriever {
            fails_remaining: AtomicU32::new(3),
            kind: ErrKind::Provider5xx,
            docs: vec![doc("a")],
        });
        let r = RetryingRetriever::new(inner as Arc<dyn Retriever>, quick_cfg());
        let docs = r.retrieve("q", 5).await.unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[tokio::test]
    async fn does_not_retry_invalid_input() {
        let inner = Arc::new(FlakyRetriever {
            fails_remaining: AtomicU32::new(10),
            kind: ErrKind::InvalidInput,
            docs: vec![],
        });
        let r = RetryingRetriever::new(inner.clone() as Arc<dyn Retriever>, quick_cfg());
        let err = r.retrieve("q", 5).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        // Only 1 call consumed (no retry on terminal error).
        assert_eq!(inner.fails_remaining.load(Ordering::SeqCst), 9);
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let inner = Arc::new(FlakyRetriever {
            fails_remaining: AtomicU32::new(100),
            kind: ErrKind::RateLimited,
            docs: vec![],
        });
        let r = RetryingRetriever::new(inner as Arc<dyn Retriever>, quick_cfg());
        let err = r.retrieve("q", 5).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }));
    }

    #[tokio::test]
    async fn retries_timeout_then_succeeds() {
        struct TimeoutThenOk {
            fails_remaining: AtomicU32,
        }
        #[async_trait]
        impl Retriever for TimeoutThenOk {
            async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
                let n = self.fails_remaining.load(Ordering::SeqCst);
                if n > 0 {
                    self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                    return Err(Error::Timeout);
                }
                Ok(vec![doc("ok")])
            }
        }
        let inner: Arc<dyn Retriever> = Arc::new(TimeoutThenOk {
            fails_remaining: AtomicU32::new(2),
        });
        let r = RetryingRetriever::new(inner, quick_cfg());
        let docs = r.retrieve("q", 5).await.unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("ok"));
    }
}

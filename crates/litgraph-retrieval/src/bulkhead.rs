//! `BulkheadRetriever` — concurrent-call cap with rejection on
//! the retriever axis. Mirrors iter-249 `BulkheadChatModel` /
//! `BulkheadEmbeddings` for the retrieve axis.
//!
//! # Why
//!
//! Vector stores have finite resources: pgvector has a connection
//! pool, qdrant has CPU+RAM caps, in-process HNSW has memory
//! pressure. Unbounded retrieval fan-out from a multi-agent
//! deployment can exhaust those resources and degrade everyone.
//! A bulkhead caps concurrent retrievals: at-cap callers either
//! wait up to a deadline (`WaitUpTo` mode) or reject immediately
//! with `Error::RateLimited` (`Reject` mode).
//!
//! Choice of `Error::RateLimited` as the at-cap outcome means
//! the existing `is_transient` classifier in `RetryingRetriever`
//! treats this as retryable — so wrapping
//! `RetryingRetriever(BulkheadRetriever(inner))` will retry on
//! rejection, giving the slot a chance to free up. Same
//! composition story as iter-249 chat/embed siblings.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Bulkhead, Document, Error, Result};

use crate::retriever::Retriever;

/// How an over-cap caller is handled when the [`Bulkhead`] is
/// saturated. Mirrors `litgraph_resilience::BulkheadMode`.
#[derive(Debug, Clone)]
pub enum BulkheadMode {
    /// Reject immediately. Returns `Error::RateLimited`.
    Reject,
    /// Block up to `Duration` for a slot, then reject.
    WaitUpTo(Duration),
}

pub struct BulkheadRetriever {
    pub inner: Arc<dyn Retriever>,
    pub bulkhead: Arc<Bulkhead>,
    pub mode: BulkheadMode,
}

impl BulkheadRetriever {
    pub fn new(
        inner: Arc<dyn Retriever>,
        bulkhead: Arc<Bulkhead>,
        mode: BulkheadMode,
    ) -> Self {
        Self {
            inner,
            bulkhead,
            mode,
        }
    }

    async fn enter(&self) -> Option<litgraph_core::BulkheadGuard> {
        match self.mode {
            BulkheadMode::Reject => self.bulkhead.try_enter(),
            BulkheadMode::WaitUpTo(d) => self.bulkhead.enter_with_timeout(d).await,
        }
    }
}

#[async_trait]
impl Retriever for BulkheadRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let _guard = match self.enter().await {
            Some(g) => g,
            None => return Err(Error::RateLimited { retry_after_ms: None }),
        };
        self.inner.retrieve(query, k).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    /// Sleeps `delay_ms` per call, tracks peak concurrent invocations.
    struct DelayRetriever {
        delay_ms: u64,
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }
    #[async_trait]
    impl Retriever for DelayRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self.peak.compare_exchange(
                    p,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![doc("a")])
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
    async fn rejects_with_rate_limited_when_at_cap() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 30,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(2));
        let r = Arc::new(BulkheadRetriever::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::Reject,
        ));
        // 5 concurrent calls; cap=2.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let r = r.clone();
            handles.push(tokio::spawn(async move { r.retrieve("q", 5).await }));
        }
        let mut oks = 0;
        let mut rejects = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => oks += 1,
                Err(Error::RateLimited { .. }) => rejects += 1,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(oks >= 2, "fewer than cap calls succeeded: {oks}");
        assert!(rejects >= 1, "expected rejections: {rejects}");
        assert_eq!(rejects as u64, bulkhead.rejected_count());
        assert!(
            peak.load(Ordering::SeqCst) <= 2,
            "peak in_flight exceeded cap",
        );
    }

    #[tokio::test]
    async fn wait_mode_blocks_then_succeeds() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 20,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(1));
        let r = Arc::new(BulkheadRetriever::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::WaitUpTo(Duration::from_millis(100)),
        ));
        let mut handles = Vec::new();
        for _ in 0..3 {
            let r = r.clone();
            handles.push(tokio::spawn(async move { r.retrieve("q", 5).await }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1);
        assert_eq!(bulkhead.rejected_count(), 0);
    }

    #[tokio::test]
    async fn wait_mode_rejects_after_deadline() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn Retriever> = Arc::new(DelayRetriever {
            delay_ms: 100,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(1));
        let r = Arc::new(BulkheadRetriever::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::WaitUpTo(Duration::from_millis(15)),
        ));
        // First call holds the slot 100ms.
        let r1 = r.clone();
        let h1 = tokio::spawn(async move { r1.retrieve("q", 5).await });
        tokio::time::sleep(Duration::from_millis(5)).await;
        // Second call: waits 15ms, then rejects.
        let r2 = r.retrieve("q", 5).await;
        assert!(matches!(r2, Err(Error::RateLimited { .. })));
        h1.await.unwrap().unwrap();
        assert_eq!(bulkhead.rejected_count(), 1);
    }

    #[tokio::test]
    async fn under_cap_passes_through() {
        let bulkhead = Arc::new(Bulkhead::new(5));
        let inner: Arc<dyn Retriever> = Arc::new(AlwaysOkRetriever);
        let r = BulkheadRetriever::new(inner, bulkhead.clone(), BulkheadMode::Reject);
        for _ in 0..5 {
            r.retrieve("q", 5).await.unwrap();
        }
        assert_eq!(bulkhead.rejected_count(), 0);
    }
}

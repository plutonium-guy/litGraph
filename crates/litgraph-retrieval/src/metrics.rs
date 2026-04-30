//! `MetricsRetriever` — auto-instrument any retriever against
//! a [`litgraph_core::MetricsRegistry`].
//!
//! Mirrors `MetricsChatModel` / `MetricsEmbeddings` (iter 261)
//! and `MetricsTool` (iter 262). With this iter the
//! metrics-instrumentation matrix covers all four primary axes:
//! chat, embed, tool, retrieve.
//!
//! Bumps the standard four metrics on every `retrieve` call:
//!
//! - `<prefix>_invocations_total` — counter.
//! - `<prefix>_errors_total` — counter.
//! - `<prefix>_in_flight` — gauge (RAII-guarded so cancellation
//!   / panic paths leave it correct).
//! - `<prefix>_latency_seconds` — histogram.
//!
//! Default prefix: `"retrieve"`. Override with `with_prefix` for
//! per-retriever labeling (e.g., `pgvector_invocations_total`
//! vs `qdrant_invocations_total` when wrapping multiple
//! retrievers).
//!
//! Lives in `litgraph-retrieval` (not `litgraph-resilience`) to
//! avoid a circular dep — same placement rationale as iter-271
//! `TimeoutRetriever` and iter-272 `RetryingRetriever`.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{
    Counter, Document, Gauge, Histogram, MetricsRegistry, Result,
};

use crate::retriever::Retriever;

/// Default histogram buckets in seconds. Geometric-ish spread
/// covering typical retrieval latency from 1ms to 10s. Tighter
/// at the low end than the chat/embed defaults because retrieval
/// is usually faster than LLM calls.
pub const DEFAULT_RETRIEVER_LATENCY_BUCKETS_SECS: &[f64] = &[
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 10.0,
];

struct Handles {
    invocations: Arc<Counter>,
    errors: Arc<Counter>,
    in_flight: Arc<Gauge>,
    latency: Arc<Histogram>,
}

struct InFlightGuard {
    gauge: Arc<Gauge>,
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.gauge.dec();
    }
}

fn resolve_handles(
    registry: &MetricsRegistry,
    prefix: &str,
    buckets: &[f64],
) -> Handles {
    Handles {
        invocations: registry.counter(&format!("{prefix}_invocations_total")),
        errors: registry.counter(&format!("{prefix}_errors_total")),
        in_flight: registry.gauge(&format!("{prefix}_in_flight")),
        latency: registry
            .histogram(&format!("{prefix}_latency_seconds"), buckets),
    }
}

pub struct MetricsRetriever {
    pub inner: Arc<dyn Retriever>,
    handles: Handles,
}

impl MetricsRetriever {
    pub fn new(inner: Arc<dyn Retriever>, registry: &MetricsRegistry) -> Self {
        Self {
            inner,
            handles: resolve_handles(
                registry,
                "retrieve",
                DEFAULT_RETRIEVER_LATENCY_BUCKETS_SECS,
            ),
        }
    }

    pub fn with_prefix(
        inner: Arc<dyn Retriever>,
        registry: &MetricsRegistry,
        prefix: &str,
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(
                registry,
                prefix,
                DEFAULT_RETRIEVER_LATENCY_BUCKETS_SECS,
            ),
        }
    }

    pub fn with_buckets(
        inner: Arc<dyn Retriever>,
        registry: &MetricsRegistry,
        prefix: &str,
        buckets: &[f64],
    ) -> Self {
        Self {
            inner,
            handles: resolve_handles(registry, prefix, buckets),
        }
    }
}

#[async_trait]
impl Retriever for MetricsRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        self.handles.invocations.inc();
        self.handles.in_flight.inc();
        let _guard = InFlightGuard {
            gauge: self.handles.in_flight.clone(),
        };
        let started = std::time::Instant::now();
        let r = self.inner.retrieve(query, k).await;
        self.handles
            .latency
            .observe(started.elapsed().as_secs_f64());
        if r.is_err() {
            self.handles.errors.inc();
        }
        r
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::Error;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    struct OkRetriever {
        docs: Vec<Document>,
    }
    #[async_trait]
    impl Retriever for OkRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            Ok(self.docs.clone())
        }
    }

    struct ErrRetriever;
    #[async_trait]
    impl Retriever for ErrRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            Err(Error::other("synthetic"))
        }
    }

    struct SlowRetriever {
        delay_ms: u64,
    }
    #[async_trait]
    impl Retriever for SlowRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(vec![doc("a")])
        }
    }

    #[tokio::test]
    async fn records_invocations_and_latency() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("a"), doc("b")],
        });
        let m = MetricsRetriever::new(inner, &registry);
        for _ in 0..3 {
            m.retrieve("q", 5).await.unwrap();
        }
        assert_eq!(registry.counter("retrieve_invocations_total").get(), 3);
        assert_eq!(registry.counter("retrieve_errors_total").get(), 0);
        assert_eq!(registry.gauge("retrieve_in_flight").get(), 0);
        assert_eq!(
            registry
                .histogram(
                    "retrieve_latency_seconds",
                    DEFAULT_RETRIEVER_LATENCY_BUCKETS_SECS,
                )
                .count(),
            3,
        );
    }

    #[tokio::test]
    async fn counts_errors_and_decs_gauge_on_err_path() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn Retriever> = Arc::new(ErrRetriever);
        let m = MetricsRetriever::new(inner, &registry);
        for _ in 0..4 {
            let _ = m.retrieve("q", 5).await;
        }
        assert_eq!(registry.counter("retrieve_invocations_total").get(), 4);
        assert_eq!(registry.counter("retrieve_errors_total").get(), 4);
        // RAII guard decremented on every error.
        assert_eq!(registry.gauge("retrieve_in_flight").get(), 0);
    }

    #[tokio::test]
    async fn in_flight_gauge_tracks_concurrent_calls() {
        let registry = Arc::new(MetricsRegistry::new());
        let inner: Arc<dyn Retriever> = Arc::new(SlowRetriever { delay_ms: 30 });
        let m = Arc::new(MetricsRetriever::new(inner, registry.as_ref()));
        let mut handles = Vec::new();
        for _ in 0..3 {
            let m = m.clone();
            handles.push(tokio::spawn(async move { m.retrieve("q", 5).await }));
        }
        // Sample mid-flight.
        tokio::time::sleep(Duration::from_millis(10)).await;
        let mid = registry.gauge("retrieve_in_flight").get();
        for h in handles {
            h.await.unwrap().unwrap();
        }
        assert!(mid >= 1, "in_flight gauge never observed concurrent work");
        assert_eq!(registry.gauge("retrieve_in_flight").get(), 0);
    }

    #[tokio::test]
    async fn with_prefix_uses_custom_name() {
        let registry = MetricsRegistry::new();
        let inner: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("a")],
        });
        let m = MetricsRetriever::with_prefix(inner, &registry, "pgvector");
        m.retrieve("q", 5).await.unwrap();
        assert_eq!(registry.counter("pgvector_invocations_total").get(), 1);
        let prom = registry.to_prometheus();
        assert!(prom.contains("pgvector_invocations_total 1"));
        assert!(!prom.contains("\nretrieve_invocations_total "));
    }

    #[tokio::test]
    async fn shared_registry_aggregates_across_retrievers() {
        // Two MetricsRetrievers wrapping different inner retrievers
        // but sharing one registry — invocations counter is per-prefix,
        // not per-instance.
        let registry = MetricsRegistry::new();
        let m1 = MetricsRetriever::with_prefix(
            Arc::new(OkRetriever { docs: vec![] }) as Arc<dyn Retriever>,
            &registry,
            "pg",
        );
        let m2 = MetricsRetriever::with_prefix(
            Arc::new(OkRetriever { docs: vec![] }) as Arc<dyn Retriever>,
            &registry,
            "qdrant",
        );
        m1.retrieve("q", 5).await.unwrap();
        m1.retrieve("q", 5).await.unwrap();
        m2.retrieve("q", 5).await.unwrap();
        assert_eq!(registry.counter("pg_invocations_total").get(), 2);
        assert_eq!(registry.counter("qdrant_invocations_total").get(), 1);
    }
}

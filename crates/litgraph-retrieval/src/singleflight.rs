//! `SingleflightRetriever` — request-coalescing wrapper for the
//! retriever axis. Mirrors iter-253 `SingleflightEmbeddings` and
//! iter-263 `SingleflightTool`.
//!
//! # The pattern
//!
//! When N agents concurrently ask the same `retrieve(query, k)`,
//! only ONE inner call runs. The other N-1 agents subscribe to
//! the leader's broadcast and receive the same `Vec<Document>`.
//! Saves vector-store load when popular queries are issued in a
//! tight burst (e.g., a multi-agent system where every worker
//! starts the same RAG sub-task).
//!
//! # Hash key
//!
//! blake3 over canonical JSON of `(query, k)` (same shape as the
//! iter-274 retriever-cassette hash key). Different `k` values
//! coalesce independently — `retrieve("X", 5)` and
//! `retrieve("X", 50)` are distinct requests.
//!
//! # Error handling
//!
//! Errors broadcast as `Arc<String>` (the inner `Error`'s
//! `to_string()`) — same lossy-by-design tradeoff as iter 253 /
//! iter 263. `litgraph_core::Error` isn't Clone, and storing
//! `Arc<Error>` would need `try_unwrap` at the receiver side
//! which fails with multiple followers. Stringification keeps
//! the broadcast value type Clone.
//!
//! # When to coalesce vs not
//!
//! Coalesce when retrieval is a pure function of `(query, k)`:
//! every caller wants the same docs, the underlying store is
//! idempotent, and we want to dedupe the load. Don't coalesce
//! when retrieval has side effects (telemetry that must record
//! per-call, audit logs, etc) or when there's a per-caller
//! authorization check that varies the result.

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result, Singleflight};

use crate::record_replay::retrieve_hash;
use crate::retriever::Retriever;

pub struct SingleflightRetriever {
    pub inner: Arc<dyn Retriever>,
    sf: Arc<Singleflight<String, Arc<std::result::Result<Vec<Document>, String>>>>,
}

impl SingleflightRetriever {
    pub fn new(inner: Arc<dyn Retriever>) -> Self {
        Self {
            inner,
            sf: Arc::new(Singleflight::new()),
        }
    }
}

#[async_trait]
impl Retriever for SingleflightRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let key = retrieve_hash(query, k);
        let inner = self.inner.clone();
        let q = query.to_string();
        let r = self
            .sf
            .get_or_compute(key, move || async move {
                let res = inner.retrieve(&q, k).await;
                Arc::new(match res {
                    Ok(v) => Ok(v),
                    Err(e) => Err(e.to_string()),
                })
            })
            .await;
        match &*r {
            Ok(v) => Ok(v.clone()),
            Err(s) => Err(Error::Provider(s.clone())),
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

    /// Slow-counting retriever: sleeps `delay_ms`, returns fixed
    /// docs, increments `seen`.
    struct SlowCountingRetriever {
        seen: AtomicU32,
        delay_ms: u64,
        docs: Vec<Document>,
    }
    #[async_trait]
    impl Retriever for SlowCountingRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(self.docs.clone())
        }
    }

    struct AlwaysFailRetriever;
    #[async_trait]
    impl Retriever for AlwaysFailRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            Err(Error::other("synthetic"))
        }
    }

    #[tokio::test]
    async fn concurrent_same_query_one_inner_call() {
        let inner = Arc::new(SlowCountingRetriever {
            seen: AtomicU32::new(0),
            delay_ms: 30,
            docs: vec![doc("a"), doc("b"), doc("c")],
        });
        let sf = Arc::new(SingleflightRetriever::new(
            inner.clone() as Arc<dyn Retriever>,
        ));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move {
                sf.retrieve("popular query", 5).await
            }));
        }
        for h in handles {
            let docs = h.await.unwrap().unwrap();
            assert_eq!(docs.len(), 3);
            assert_eq!(docs[0].id.as_deref(), Some("a"));
        }
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            1,
            "inner ran more than once",
        );
    }

    #[tokio::test]
    async fn different_query_runs_independently() {
        let inner = Arc::new(SlowCountingRetriever {
            seen: AtomicU32::new(0),
            delay_ms: 5,
            docs: vec![doc("a")],
        });
        let sf = Arc::new(SingleflightRetriever::new(
            inner.clone() as Arc<dyn Retriever>,
        ));
        let mut handles = Vec::new();
        for q in ["q1", "q2", "q3"] {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move { sf.retrieve(q, 5).await }));
        }
        for h in handles {
            let _ = h.await.unwrap().unwrap();
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn different_k_runs_independently() {
        let inner = Arc::new(SlowCountingRetriever {
            seen: AtomicU32::new(0),
            delay_ms: 5,
            docs: vec![doc("a")],
        });
        let sf = Arc::new(SingleflightRetriever::new(
            inner.clone() as Arc<dyn Retriever>,
        ));
        // Same query, different k → distinct hash keys, distinct
        // coalescing windows.
        let mut handles = Vec::new();
        for k in [5, 10, 20] {
            let sf = sf.clone();
            handles.push(tokio::spawn(async move { sf.retrieve("q", k).await }));
        }
        for h in handles {
            let _ = h.await.unwrap().unwrap();
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn propagates_errors_as_provider_error() {
        let inner: Arc<dyn Retriever> = Arc::new(AlwaysFailRetriever);
        let sf = SingleflightRetriever::new(inner);
        let r = sf.retrieve("q", 5).await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("synthetic"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn window_closes_after_completion() {
        let inner = Arc::new(SlowCountingRetriever {
            seen: AtomicU32::new(0),
            delay_ms: 0,
            docs: vec![doc("a")],
        });
        let sf = SingleflightRetriever::new(inner.clone() as Arc<dyn Retriever>);
        // Two sequential calls — coalescing only inside an in-flight
        // window. Both should run inner.
        sf.retrieve("q", 5).await.unwrap();
        sf.retrieve("q", 5).await.unwrap();
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }
}

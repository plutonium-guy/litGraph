//! `HedgedRetriever` — tail-latency mitigation for slow vector
//! stores. Mirrors iter-251 `HedgedChatModel` / `HedgedEmbeddings`
//! for the retriever axis.
//!
//! # Distinct from `RaceRetriever`
//!
//! `RaceRetriever` (iter 193) issues to ALL inner retrievers
//! simultaneously — every call doubles cost. `HedgedRetriever`
//! issues to `primary` alone for `hedge_delay`; only if primary
//! is slow does `backup` also fire. Fast-path requests pay zero
//! overhead — only the slow tail incurs the second-call cost.
//!
//! Standard pattern from Dean & Barroso 2013 "The Tail At Scale".
//! Right when median latency is fine and you only want to insure
//! against the p99.
//!
//! # Real prod use
//!
//! - **HNSW cold pages**: in-memory index with 1ms p50 / 50ms p99
//!   from rare cold-paging. Hedge after 5ms — most calls
//!   complete fast, slow tail covered by backup HNSW replica.
//! - **Multi-region failover**: primary in us-east-1 (close, fast
//!   normally); backup in us-west-2 (far, slower normally but
//!   alive when us-east-1 is having a bad day). Hedge after
//!   500ms.
//! - **Replica hedging**: primary against the read replica that
//!   was just promoted; backup against a known-stable replica.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{hedged_call, Document, Result};

use crate::retriever::Retriever;

pub struct HedgedRetriever {
    pub primary: Arc<dyn Retriever>,
    pub backup: Arc<dyn Retriever>,
    pub hedge_delay: Duration,
}

impl HedgedRetriever {
    pub fn new(
        primary: Arc<dyn Retriever>,
        backup: Arc<dyn Retriever>,
        hedge_delay: Duration,
    ) -> Self {
        Self {
            primary,
            backup,
            hedge_delay,
        }
    }
}

#[async_trait]
impl Retriever for HedgedRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let primary = self.primary.clone();
        let backup = self.backup.clone();
        let q_p = query.to_string();
        let q_b = query.to_string();
        hedged_call(
            move || async move { primary.retrieve(&q_p, k).await },
            move || async move { backup.retrieve(&q_b, k).await },
            self.hedge_delay,
        )
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn doc(tag: &str) -> Document {
        Document::new(tag.to_string()).with_id(tag.to_string())
    }

    /// Sleeps `delay_ms` then returns docs tagged with `label`.
    /// Counts invocations so we can verify backup-was-fired.
    struct LabeledDelayRetriever {
        delay_ms: u64,
        label: String,
        seen: AtomicU32,
    }
    #[async_trait]
    impl Retriever for LabeledDelayRetriever {
        async fn retrieve(&self, _q: &str, _k: usize) -> Result<Vec<Document>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(vec![doc(&self.label)])
        }
    }

    #[tokio::test]
    async fn primary_wins_when_fast_no_backup_invoked() {
        let primary = Arc::new(LabeledDelayRetriever {
            delay_ms: 10,
            label: "primary".into(),
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(LabeledDelayRetriever {
            delay_ms: 10,
            label: "backup".into(),
            seen: AtomicU32::new(0),
        });
        let h = HedgedRetriever::new(
            primary.clone() as Arc<dyn Retriever>,
            backup.clone() as Arc<dyn Retriever>,
            Duration::from_millis(50),
        );
        let docs = h.retrieve("q", 5).await.unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("primary"));
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        // Backup never even invoked — primary finished within hedge window.
        assert_eq!(backup.seen.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn backup_wins_when_primary_slow() {
        let primary = Arc::new(LabeledDelayRetriever {
            delay_ms: 100,
            label: "primary".into(),
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(LabeledDelayRetriever {
            delay_ms: 10,
            label: "backup".into(),
            seen: AtomicU32::new(0),
        });
        let h = HedgedRetriever::new(
            primary.clone() as Arc<dyn Retriever>,
            backup.clone() as Arc<dyn Retriever>,
            Duration::from_millis(20),
        );
        let docs = h.retrieve("q", 5).await.unwrap();
        // Primary 100ms, backup starts at 20ms and finishes at ~30ms.
        assert_eq!(docs[0].id.as_deref(), Some("backup"));
        assert_eq!(primary.seen.load(Ordering::SeqCst), 1);
        assert_eq!(backup.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn primary_can_still_win_after_hedge_fires() {
        let primary = Arc::new(LabeledDelayRetriever {
            delay_ms: 30,
            label: "primary".into(),
            seen: AtomicU32::new(0),
        });
        let backup = Arc::new(LabeledDelayRetriever {
            delay_ms: 100,
            label: "backup".into(),
            seen: AtomicU32::new(0),
        });
        let h = HedgedRetriever::new(
            primary as Arc<dyn Retriever>,
            backup as Arc<dyn Retriever>,
            Duration::from_millis(10),
        );
        let docs = h.retrieve("q", 5).await.unwrap();
        // Both started; primary finishes at 30ms (faster than backup's 100ms).
        assert_eq!(docs[0].id.as_deref(), Some("primary"));
    }
}

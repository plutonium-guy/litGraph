//! Record/replay for retriever calls. Fourth axis of the VCR-
//! style testing infra (after chat iter 254, embed iter 255,
//! tool iter 256).
//!
//! # Why
//!
//! Agent tests that exercise retrieval in CI either need a real
//! vector store (slow, requires fixture data, flakes on
//! infrastructure issues) or a hand-mocked retriever (tedious to
//! maintain, drifts from real behavior). Record/replay solves
//! both: record real retrievals once during a real-traffic
//! test run, save to a JSON cassette, replay deterministically
//! in CI without the live store.
//!
//! Hash key: blake3 over canonical JSON of `(query, k)`. Same
//! request → same hash → same cached response. The shared
//! cassette format mirrors the chat/embed/tool cassettes.

use std::path::Path;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use parking_lot::Mutex as PlMutex;

use crate::retriever::Retriever;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RetrieverExchange {
    pub request_hash: String,
    pub query: String,
    pub k: usize,
    pub response: Vec<Document>,
}

/// Persistable retriever-call record. Same shape as
/// `Cassette` / `EmbedCassette` / `ToolCassette`.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RetrieverCassette {
    #[serde(default = "default_version")]
    pub version: u32,
    pub exchanges: Vec<RetrieverExchange>,
}

fn default_version() -> u32 {
    1
}

impl RetrieverCassette {
    pub fn load_from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path).map_err(|e| {
            Error::other(format!("read retriever cassette {path:?}: {e}"))
        })?;
        serde_json::from_str(&s).map_err(|e| {
            Error::other(format!("parse retriever cassette {path:?}: {e}"))
        })
    }

    pub fn save_to_file(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::other(format!("mkdir {parent:?}: {e}")))?;
            }
        }
        let s = serde_json::to_string_pretty(self).map_err(|e| {
            Error::other(format!("serialize retriever cassette: {e}"))
        })?;
        std::fs::write(path, s).map_err(|e| {
            Error::other(format!("write retriever cassette {path:?}: {e}"))
        })?;
        Ok(())
    }
}

/// blake3 over canonical JSON of `(query, k)`.
pub fn retrieve_hash(query: &str, k: usize) -> String {
    let req = serde_json::json!({ "query": query, "k": k });
    let s = serde_json::to_string(&req).unwrap_or_default();
    blake3::hash(s.as_bytes()).to_hex().to_string()
}

/// Wrap any retriever to record every `retrieve` call into a
/// shared cassette.
pub struct RecordingRetriever {
    pub inner: Arc<dyn Retriever>,
    cassette: Arc<PlMutex<RetrieverCassette>>,
}

impl RecordingRetriever {
    pub fn new(
        inner: Arc<dyn Retriever>,
        cassette: Arc<PlMutex<RetrieverCassette>>,
    ) -> Self {
        Self { inner, cassette }
    }
}

#[async_trait]
impl Retriever for RecordingRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let response = self.inner.retrieve(query, k).await?;
        let exchange = RetrieverExchange {
            request_hash: retrieve_hash(query, k),
            query: query.to_string(),
            k,
            response: response.clone(),
        };
        self.cassette.lock().exchanges.push(exchange);
        Ok(response)
    }
}

/// Replay recorded retrievals from a cassette. Optional
/// `passthrough` for "record-then-fill-gaps" workflows where a
/// cassette miss falls through to the live retriever.
pub struct ReplayingRetriever {
    pub cassette: RetrieverCassette,
    pub passthrough: Option<Arc<dyn Retriever>>,
}

impl ReplayingRetriever {
    pub fn new(
        cassette: RetrieverCassette,
        passthrough: Option<Arc<dyn Retriever>>,
    ) -> Self {
        Self {
            cassette,
            passthrough,
        }
    }
}

#[async_trait]
impl Retriever for ReplayingRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let hash = retrieve_hash(query, k);
        if let Some(ex) = self
            .cassette
            .exchanges
            .iter()
            .find(|e| e.request_hash == hash)
        {
            return Ok(ex.response.clone());
        }
        if let Some(pt) = &self.passthrough {
            return pt.retrieve(query, k).await;
        }
        Err(Error::Provider(format!(
            "no recorded retrieve response for hash {hash}",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[tokio::test]
    async fn record_then_replay_round_trip() {
        let inner: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("a"), doc("b")],
        });
        let cass = Arc::new(PlMutex::new(RetrieverCassette::default()));
        let recorder = RecordingRetriever::new(inner, cass.clone());
        recorder.retrieve("question 1", 5).await.unwrap();
        recorder.retrieve("question 2", 3).await.unwrap();
        let snap = cass.lock().clone();
        assert_eq!(snap.exchanges.len(), 2);

        let player = ReplayingRetriever::new(snap, None);
        let docs = player.retrieve("question 1", 5).await.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id.as_deref(), Some("a"));
    }

    #[tokio::test]
    async fn replay_miss_returns_error_when_no_passthrough() {
        let cass = RetrieverCassette::default();
        let player = ReplayingRetriever::new(cass, None);
        let r = player.retrieve("never seen", 5).await;
        match r {
            Err(Error::Provider(msg)) => {
                assert!(msg.contains("no recorded retrieve response"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn replay_miss_falls_through_to_passthrough() {
        let cass = RetrieverCassette::default();
        let live: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("live")],
        });
        let player = ReplayingRetriever::new(cass, Some(live));
        let docs = player.retrieve("anything", 5).await.unwrap();
        assert_eq!(docs[0].id.as_deref(), Some("live"));
    }

    #[tokio::test]
    async fn cassette_save_and_load_round_trip_through_disk() {
        let inner: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("disk1"), doc("disk2")],
        });
        let cass = Arc::new(PlMutex::new(RetrieverCassette::default()));
        let recorder = RecordingRetriever::new(inner, cass.clone());
        recorder.retrieve("disk-q", 4).await.unwrap();
        let snap = cass.lock().clone();

        let tmp = std::env::temp_dir().join(format!(
            "litgraph_retriever_cassette_{}.json",
            uuid::Uuid::new_v4()
        ));
        snap.save_to_file(&tmp).unwrap();
        let restored = RetrieverCassette::load_from_file(&tmp).unwrap();
        let _ = std::fs::remove_file(&tmp);

        assert_eq!(restored.exchanges.len(), 1);
        let player = ReplayingRetriever::new(restored, None);
        let docs = player.retrieve("disk-q", 4).await.unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].id.as_deref(), Some("disk1"));
    }

    #[tokio::test]
    async fn hash_distinguishes_query_and_k() {
        let h1 = retrieve_hash("a", 5);
        let h2 = retrieve_hash("a", 5);
        let h3 = retrieve_hash("b", 5);
        let h4 = retrieve_hash("a", 10);
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
        assert_ne!(h1, h4);
    }

    #[tokio::test]
    async fn different_k_values_replay_separately() {
        let inner: Arc<dyn Retriever> = Arc::new(OkRetriever {
            docs: vec![doc("a"), doc("b"), doc("c")],
        });
        let cass = Arc::new(PlMutex::new(RetrieverCassette::default()));
        let recorder = RecordingRetriever::new(inner, cass.clone());
        recorder.retrieve("q", 5).await.unwrap();
        recorder.retrieve("q", 10).await.unwrap();
        let snap = cass.lock().clone();
        assert_eq!(snap.exchanges.len(), 2);

        let player = ReplayingRetriever::new(snap, None);
        // Different k → different hash → different cassette entry, both
        // hit successfully on replay.
        let r1 = player.retrieve("q", 5).await.unwrap();
        let r2 = player.retrieve("q", 10).await.unwrap();
        assert_eq!(r1.len(), r2.len()); // both record OkRetriever's full Vec
    }
}

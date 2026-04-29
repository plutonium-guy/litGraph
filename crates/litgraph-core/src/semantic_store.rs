//! `SemanticStore` — wraps any [`Store`] with an [`Embeddings`] backend so
//! callers can do brute-force cosine semantic search across a namespace.
//!
//! Closes Tier 1 #3 from `ROADMAP.md`: "Vector-indexed semantic search
//! on `Store`." LangGraph's `BaseStore` exposes a `search` parameter
//! that takes a string and matches by *meaning* — semantic search in
//! Python is opt-in via a separately-configured embedder. This module
//! is the same idea: opt-in, BYO embedder, drops in over any
//! `Arc<dyn Store>` (InMemory, Postgres, Redis).
//!
//! # Storage shape
//!
//! Each `put` writes a JSON value of the shape:
//!
//! ```json
//! {
//!   "_emb":   [0.12, -0.04, ...],
//!   "_text":  "the original text",
//!   "value":  { /* whatever you passed */ }
//! }
//! ```
//!
//! That keeps the embedding **inside the store** — durable across
//! crashes, sharable across processes for distributed backends. The
//! cost is one extra JSON-array-of-floats per row; for typical
//! 384-1536 dim embeddings, that's ~3-12 KB serialized as JSON.
//!
//! # Search
//!
//! `semantic_search` fetches all items in the namespace via
//! [`Store::search`] (with optional pre-filter), then runs cosine
//! similarity in **parallel via Rayon** before sorting top-k. This is
//! brute force — fine up to ~10k items per namespace. For larger
//! namespaces, use a real vector store (`pgvector` / `hnsw`) and
//! `VectorRetriever`.
//!
//! # Why brute force, not HNSW
//!
//! `SemanticStore` is a *long-term memory* primitive, not a RAG
//! corpus. Real-world LangGraph memory namespaces top out at hundreds
//! of items (per-user preferences, episodic memories, scratchpads).
//! Brute-force cosine over that scale finishes in microseconds and
//! avoids the index-rebuild + recall-loss tradeoffs of HNSW. Reach
//! for a vector store when you have ≥10k items.

use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::error::{Error, Result};
use crate::model::Embeddings;
use crate::store::{Namespace, SearchFilter, Store, StoreItem};

const EMBED_KEY: &str = "_emb";
const TEXT_KEY: &str = "_text";
const VALUE_KEY: &str = "value";

/// One semantic-search hit. The `score` is cosine similarity in
/// `[-1.0, 1.0]`; bigger = more relevant.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SemanticHit {
    pub namespace: Vec<String>,
    pub key: String,
    pub text: String,
    pub value: Value,
    pub score: f32,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

/// Wraps any `Store` to add semantic put/get/search backed by an
/// `Embeddings` provider.
#[derive(Clone)]
pub struct SemanticStore {
    pub inner: Arc<dyn Store>,
    pub embedder: Arc<dyn Embeddings>,
}

impl SemanticStore {
    pub fn new(inner: Arc<dyn Store>, embedder: Arc<dyn Embeddings>) -> Self {
        Self { inner, embedder }
    }

    /// Insert or replace one item, embedding `text` for later
    /// semantic-search recall. `value` is the user payload — round-trips
    /// untouched through `get` / `semantic_search`.
    pub async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        text: &str,
        value: Value,
        ttl_ms: Option<u64>,
    ) -> Result<()> {
        let emb = self.embedder.embed_query(text).await?;
        let wrapped = json!({
            EMBED_KEY: emb,
            TEXT_KEY: text,
            VALUE_KEY: value,
        });
        self.inner.put(namespace, key, &wrapped, ttl_ms).await
    }

    /// Fetch by exact key. Returns `(text, value)` unwrapped from the
    /// internal storage shape; the embedding is dropped on the way
    /// out so callers never see it.
    pub async fn get(
        &self,
        namespace: &Namespace,
        key: &str,
    ) -> Result<Option<(String, Value)>> {
        let Some(item) = self.inner.get(namespace, key).await? else {
            return Ok(None);
        };
        let (text, value) = unwrap(&item.value).ok_or_else(|| {
            Error::other(format!(
                "SemanticStore: item at {namespace:?}/{key} not in semantic shape"
            ))
        })?;
        Ok(Some((text, value)))
    }

    /// Delete by key. Same semantics as `Store::delete`.
    pub async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool> {
        self.inner.delete(namespace, key).await
    }

    /// Embed `query`, fetch all items in `namespace_prefix` (optionally
    /// pre-filtered by `filter`), score each in parallel via Rayon, and
    /// return the top-`k` by cosine similarity.
    pub async fn semantic_search(
        &self,
        namespace_prefix: &Namespace,
        query: &str,
        k: usize,
        filter: Option<SearchFilter>,
    ) -> Result<Vec<SemanticHit>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let q_emb = self.embedder.embed_query(query).await?;
        // Pull every item in the namespace prefix. Use a generous
        // limit since brute-force scan dominates the cost anyway.
        let mut f = filter.unwrap_or_default();
        if f.limit.is_none() {
            f.limit = Some(usize::MAX);
        }
        let items = self.inner.search(namespace_prefix, &f).await?;

        // Rayon parallel cosine — embedded in JSON, so we have to
        // pay one f64→f32 cast per dim per item, but rayon makes it
        // trivially parallel.
        let mut scored: Vec<(StoreItem, f32, String)> = items
            .into_par_iter()
            .filter_map(|item| {
                let emb = item.value.get(EMBED_KEY)?.as_array()?;
                let text = item
                    .value
                    .get(TEXT_KEY)
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let cand: Vec<f32> = emb
                    .iter()
                    .filter_map(|n| n.as_f64().map(|x| x as f32))
                    .collect();
                let score = cosine_sim(&q_emb, &cand);
                if score.is_finite() {
                    Some((item, score, text))
                } else {
                    None
                }
            })
            .collect();

        // Sort descending by score; truncate; convert to hits.
        scored.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        Ok(scored
            .into_iter()
            .map(|(item, score, text)| {
                let value = item
                    .value
                    .get(VALUE_KEY)
                    .cloned()
                    .unwrap_or(Value::Null);
                SemanticHit {
                    namespace: item.namespace,
                    key: item.key,
                    text,
                    value,
                    score,
                    created_at_ms: item.created_at_ms,
                    updated_at_ms: item.updated_at_ms,
                }
            })
            .collect())
    }
}

fn unwrap(v: &Value) -> Option<(String, Value)> {
    let text = v.get(TEXT_KEY)?.as_str()?.to_string();
    let value = v.get(VALUE_KEY).cloned().unwrap_or(Value::Null);
    Some((text, value))
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryStore;
    use async_trait::async_trait;

    /// 2-D toy embedder: ("rust", 0) → [1, 0]; ("memory", 0) → [0, 1];
    /// any other text → [0.5, 0.5]. Lets us hand-pick which items
    /// are "near" a query.
    struct ToyEmbedder;

    #[async_trait]
    impl Embeddings for ToyEmbedder {
        fn name(&self) -> &str {
            "toy"
        }
        fn dimensions(&self) -> usize {
            2
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            let lower = text.to_lowercase();
            let vec = if lower.contains("rust") {
                vec![1.0, 0.0]
            } else if lower.contains("memory") {
                vec![0.0, 1.0]
            } else {
                vec![0.5, 0.5]
            };
            Ok(vec)
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            let mut out = Vec::with_capacity(texts.len());
            for t in texts {
                out.push(self.embed_query(t).await?);
            }
            Ok(out)
        }
    }

    fn fixture() -> SemanticStore {
        let store = Arc::new(InMemoryStore::new()) as Arc<dyn Store>;
        let emb = Arc::new(ToyEmbedder) as Arc<dyn Embeddings>;
        SemanticStore::new(store, emb)
    }

    #[tokio::test]
    async fn put_get_round_trip_strips_embedding() {
        let s = fixture();
        let ns = vec!["users".to_string(), "alice".to_string()];
        s.put(&ns, "fact:1", "rust is fast", json!({"k": 1}), None)
            .await
            .unwrap();
        let (text, value) = s.get(&ns, "fact:1").await.unwrap().unwrap();
        assert_eq!(text, "rust is fast");
        assert_eq!(value, json!({"k": 1}));
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let s = fixture();
        let ns = vec!["x".to_string()];
        assert!(s.get(&ns, "nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn semantic_search_ranks_by_cosine() {
        let s = fixture();
        let ns = vec!["facts".to_string()];
        s.put(&ns, "a", "rust safety", json!({"id": "a"}), None)
            .await
            .unwrap();
        s.put(&ns, "b", "memory leaks", json!({"id": "b"}), None)
            .await
            .unwrap();
        s.put(&ns, "c", "javascript closures", json!({"id": "c"}), None)
            .await
            .unwrap();
        // Query embedding is [1, 0] (rust direction) → "rust safety"
        // wins, "javascript closures" gets the diagonal score.
        let hits = s.semantic_search(&ns, "rust performance", 3, None).await.unwrap();
        assert_eq!(hits.len(), 3);
        assert_eq!(hits[0].key, "a");
        assert!(hits[0].score >= hits[1].score);
        assert!(hits[1].score >= hits[2].score);
    }

    #[tokio::test]
    async fn semantic_search_respects_k() {
        let s = fixture();
        let ns = vec!["facts".to_string()];
        for i in 0..6 {
            s.put(&ns, &format!("k{i}"), "rust thing", json!(i), None)
                .await
                .unwrap();
        }
        let hits = s.semantic_search(&ns, "rust", 3, None).await.unwrap();
        assert_eq!(hits.len(), 3);
    }

    #[tokio::test]
    async fn semantic_search_k_zero_returns_empty() {
        let s = fixture();
        let ns = vec!["facts".to_string()];
        s.put(&ns, "a", "rust", json!(null), None).await.unwrap();
        let hits = s.semantic_search(&ns, "rust", 0, None).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn delete_removes_from_search() {
        let s = fixture();
        let ns = vec!["facts".to_string()];
        s.put(&ns, "a", "rust safety", json!(null), None).await.unwrap();
        s.put(&ns, "b", "rust threads", json!(null), None).await.unwrap();
        let removed = s.delete(&ns, "a").await.unwrap();
        assert!(removed);
        let hits = s.semantic_search(&ns, "rust", 5, None).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key, "b");
    }

    #[tokio::test]
    async fn semantic_search_empty_namespace_is_empty() {
        let s = fixture();
        let ns = vec!["empty".to_string()];
        let hits = s.semantic_search(&ns, "anything", 5, None).await.unwrap();
        assert!(hits.is_empty());
    }

    #[tokio::test]
    async fn semantic_search_namespace_isolation() {
        let s = fixture();
        let ns_a = vec!["users".to_string(), "alice".to_string()];
        let ns_b = vec!["users".to_string(), "bob".to_string()];
        s.put(&ns_a, "k", "rust", json!("alice's"), None).await.unwrap();
        s.put(&ns_b, "k", "rust", json!("bob's"), None).await.unwrap();
        let hits_b = s.semantic_search(&ns_b, "rust", 5, None).await.unwrap();
        assert_eq!(hits_b.len(), 1);
        assert_eq!(hits_b[0].value, json!("bob's"));
    }

    #[test]
    fn cosine_handles_empty_or_zero() {
        assert_eq!(cosine_sim(&[], &[]), 0.0);
        assert_eq!(cosine_sim(&[0.0, 0.0], &[1.0, 1.0]), 0.0);
        let v = (cosine_sim(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs();
        assert!(v < 1e-6);
    }
}

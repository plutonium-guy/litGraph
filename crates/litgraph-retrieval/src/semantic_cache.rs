//! `SemanticCachedRetriever` — cache retrieval results keyed by
//! semantic similarity, not exact-string match.
//!
//! # Distinct from existing wrappers
//!
//! - **iter-278 `SingleflightRetriever`**: coalesces *concurrent*
//!   identical calls (same query+k arriving at the same time
//!   share one inner call).
//! - **iter-271 `TimeoutRetriever`**, **iter-272
//!   `RetryingRetriever`**, etc: resilience.
//! - **`SemanticCachedRetriever` (this)**: caches *across* calls.
//!   When a new query embeds close to a previous query
//!   (cosine ≥ threshold), reuse the previous result. Real prod
//!   use: FAQ agents where users phrase "How do I reset my
//!   password?" / "I forgot my password, help" / "password
//!   recovery steps" — semantically the same query.
//!
//! # Knobs
//!
//! - `similarity_threshold` (default 0.95) — cosine similarity
//!   for a hit. Tune lower (e.g. 0.85) for chattier-FAQ matching;
//!   higher (e.g. 0.98) for strict near-exact-paraphrase.
//! - `max_entries` (default 1000) — LRU cap. Oldest entries
//!   evicted when full.
//! - `ttl` (default `None`) — optional expiry. Useful when the
//!   underlying corpus updates (cached retrieval becomes stale).
//!
//! # Caveats
//!
//! - Different `k` values keep separate cache lines (a hit for
//!   `(query, 5)` doesn't satisfy `(query, 50)` because the
//!   result list shape differs).
//! - The cache scans linearly. For thousands of entries that's
//!   thousands of cosine similarity computations per call —
//!   tune `max_entries` to your latency budget.
//! - Embedding cost: one extra `embed_query` per call (cache or
//!   miss). The win is when retrieval is more expensive than
//!   embedding, which is typical for hybrid / reranked / large-
//!   corpus retrievers.

use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use litgraph_core::{Document, Embeddings, Result};
use parking_lot::Mutex as PlMutex;

use crate::retriever::Retriever;

struct CacheEntry {
    embedding: Vec<f32>,
    response: Vec<Document>,
    k: usize,
    inserted_at: Instant,
}

pub struct SemanticCachedRetriever {
    pub inner: Arc<dyn Retriever>,
    pub embeddings: Arc<dyn Embeddings>,
    pub similarity_threshold: f32,
    pub max_entries: usize,
    pub ttl: Option<Duration>,
    cache: PlMutex<Vec<CacheEntry>>,
}

impl SemanticCachedRetriever {
    pub fn new(inner: Arc<dyn Retriever>, embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            inner,
            embeddings,
            similarity_threshold: 0.95,
            max_entries: 1000,
            ttl: None,
            cache: PlMutex::new(Vec::new()),
        }
    }

    pub fn with_threshold(mut self, t: f32) -> Self {
        self.similarity_threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn with_max_entries(mut self, n: usize) -> Self {
        self.max_entries = n.max(1);
        self
    }

    pub fn with_ttl(mut self, ttl: Duration) -> Self {
        self.ttl = Some(ttl);
        self
    }

    /// Approximate count of entries in the cache. Useful for
    /// telemetry / smoke tests.
    pub fn len(&self) -> usize {
        self.cache.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.lock().is_empty()
    }

    /// Drop all cached entries. Useful for tests or when the
    /// underlying corpus is known to have changed.
    pub fn clear(&self) {
        self.cache.lock().clear();
    }
}

#[async_trait]
impl Retriever for SemanticCachedRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let q_emb = self.embeddings.embed_query(query).await?;
        let now = Instant::now();

        // Cache lookup phase. Hold the lock briefly.
        let hit = {
            let mut cache = self.cache.lock();
            // Evict expired entries before searching.
            if let Some(ttl) = self.ttl {
                cache.retain(|e| now.duration_since(e.inserted_at) <= ttl);
            }
            let mut best: Option<(usize, f32)> = None;
            for (i, entry) in cache.iter().enumerate() {
                if entry.k != k {
                    continue;
                }
                let sim = cosine_similarity(&q_emb, &entry.embedding);
                if sim >= self.similarity_threshold {
                    if best.map(|(_, s)| sim > s).unwrap_or(true) {
                        best = Some((i, sim));
                    }
                }
            }
            best.map(|(i, _)| cache[i].response.clone())
        };
        if let Some(docs) = hit {
            return Ok(docs);
        }

        // Cache miss — invoke inner.
        let docs = self.inner.retrieve(query, k).await?;
        // Insert into cache; evict oldest if over cap.
        {
            let mut cache = self.cache.lock();
            cache.push(CacheEntry {
                embedding: q_emb,
                response: docs.clone(),
                k,
                inserted_at: now,
            });
            while cache.len() > self.max_entries {
                cache.remove(0);
            }
        }
        Ok(docs)
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn doc(id: &str) -> Document {
        Document::new("x".to_string()).with_id(id.to_string())
    }

    /// Scripted retriever: returns docs keyed by query string.
    /// Counts inner invocations for cache-hit/miss verification.
    struct ScriptedRetriever {
        responses: HashMap<String, Vec<Document>>,
        seen: AtomicU32,
    }
    #[async_trait]
    impl Retriever for ScriptedRetriever {
        async fn retrieve(&self, q: &str, _k: usize) -> Result<Vec<Document>> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Ok(self.responses.get(q).cloned().unwrap_or_default())
        }
    }

    /// Embeddings shim: returns a hand-coded embedding per query
    /// from a map. Lets us control similarity precisely.
    struct ScriptedEmbeddings {
        map: HashMap<String, Vec<f32>>,
    }
    #[async_trait]
    impl Embeddings for ScriptedEmbeddings {
        fn name(&self) -> &str {
            "scripted"
        }
        fn dimensions(&self) -> usize {
            3
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self
                .map
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0; 3]))
        }
        async fn embed_documents(&self, _t: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(Vec::new())
        }
    }

    fn make_setup(
        retriever_responses: &[(&str, Vec<Document>)],
        embedding_map: &[(&str, Vec<f32>)],
    ) -> (
        Arc<ScriptedRetriever>,
        Arc<dyn Embeddings>,
    ) {
        let r = Arc::new(ScriptedRetriever {
            responses: retriever_responses
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
            seen: AtomicU32::new(0),
        });
        let e: Arc<dyn Embeddings> = Arc::new(ScriptedEmbeddings {
            map: embedding_map
                .iter()
                .map(|(k, v)| (k.to_string(), v.clone()))
                .collect(),
        });
        (r, e)
    }

    #[tokio::test]
    async fn exact_same_query_hits_cache() {
        let (r, e) = make_setup(
            &[("hello", vec![doc("a")])],
            &[("hello", vec![1.0, 0.0, 0.0])],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e);
        cached.retrieve("hello", 5).await.unwrap();
        cached.retrieve("hello", 5).await.unwrap();
        // Inner called once; second call hit cache.
        assert_eq!(r.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn semantically_similar_query_hits_cache() {
        // Two near-identical embeddings (cosine 0.99). Hit at default 0.95.
        let (r, e) = make_setup(
            &[
                ("password reset", vec![doc("p1")]),
                ("forgot my password", vec![doc("p2")]),
            ],
            &[
                ("password reset", vec![1.0, 0.0, 0.0]),
                ("forgot my password", vec![0.99, 0.14, 0.0]),
            ],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e);
        let first = cached.retrieve("password reset", 5).await.unwrap();
        let second = cached.retrieve("forgot my password", 5).await.unwrap();
        // The second query was cache-served from the first because their
        // embeddings have cosine > 0.95.
        assert_eq!(first[0].id, second[0].id);
        assert_eq!(r.seen.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn dissimilar_query_misses_cache() {
        let (r, e) = make_setup(
            &[
                ("apple", vec![doc("a")]),
                ("banana", vec![doc("b")]),
            ],
            &[
                ("apple", vec![1.0, 0.0, 0.0]),
                ("banana", vec![0.0, 1.0, 0.0]),
            ],
        );
        // Orthogonal embeddings → cosine 0 → no hit.
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e);
        cached.retrieve("apple", 5).await.unwrap();
        cached.retrieve("banana", 5).await.unwrap();
        assert_eq!(r.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn different_k_values_keep_separate_cache_lines() {
        let (r, e) = make_setup(
            &[("q", vec![doc("d")])],
            &[("q", vec![1.0, 0.0, 0.0])],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e);
        cached.retrieve("q", 5).await.unwrap();
        cached.retrieve("q", 10).await.unwrap();
        // Same query but k differs → 2 inner calls.
        assert_eq!(r.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn ttl_expires_old_entries() {
        let (r, e) = make_setup(
            &[("q", vec![doc("d")])],
            &[("q", vec![1.0, 0.0, 0.0])],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e)
            .with_ttl(Duration::from_millis(20));
        cached.retrieve("q", 5).await.unwrap();
        // Wait past TTL.
        tokio::time::sleep(Duration::from_millis(40)).await;
        cached.retrieve("q", 5).await.unwrap();
        // Cache miss after expiry → 2 inner calls.
        assert_eq!(r.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn max_entries_evicts_oldest() {
        let (r, e) = make_setup(
            &[
                ("a", vec![doc("1")]),
                ("b", vec![doc("2")]),
                ("c", vec![doc("3")]),
            ],
            &[
                ("a", vec![1.0, 0.0, 0.0]),
                ("b", vec![0.0, 1.0, 0.0]),
                ("c", vec![0.0, 0.0, 1.0]),
            ],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e)
            .with_max_entries(2);
        cached.retrieve("a", 5).await.unwrap();
        cached.retrieve("b", 5).await.unwrap();
        cached.retrieve("c", 5).await.unwrap();
        // Cache cap=2; "a" should have been evicted.
        assert_eq!(cached.len(), 2);
        let baseline = r.seen.load(Ordering::SeqCst);
        // Re-query "a" — should miss cache (was evicted).
        cached.retrieve("a", 5).await.unwrap();
        assert_eq!(r.seen.load(Ordering::SeqCst), baseline + 1);
    }

    #[tokio::test]
    async fn clear_drops_all_entries() {
        let (r, e) = make_setup(
            &[("q", vec![doc("d")])],
            &[("q", vec![1.0, 0.0, 0.0])],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e);
        cached.retrieve("q", 5).await.unwrap();
        assert_eq!(cached.len(), 1);
        cached.clear();
        assert!(cached.is_empty());
        cached.retrieve("q", 5).await.unwrap();
        // Cache miss after clear → 2 inner calls.
        assert_eq!(r.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn higher_threshold_rejects_lower_similarity() {
        // Cosine 0.99 — would hit at default 0.95 but not at 0.999.
        let (r, e) = make_setup(
            &[
                ("a", vec![doc("d")]),
                ("b", vec![doc("d")]),
            ],
            &[
                ("a", vec![1.0, 0.0, 0.0]),
                ("b", vec![0.99, 0.14, 0.0]),
            ],
        );
        let cached = SemanticCachedRetriever::new(r.clone() as Arc<dyn Retriever>, e)
            .with_threshold(0.999);
        cached.retrieve("a", 5).await.unwrap();
        cached.retrieve("b", 5).await.unwrap();
        // Threshold too strict → cache miss → 2 inner calls.
        assert_eq!(r.seen.load(Ordering::SeqCst), 2);
    }
}

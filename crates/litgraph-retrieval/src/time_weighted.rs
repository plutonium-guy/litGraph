//! TimeWeightedVectorStoreRetriever — boost recently-accessed documents via
//! exponential decay. Direct LangChain parity.
//!
//! # Use case
//!
//! Agent memory / chat-history retrieval. The user's recent conversation
//! turns matter more than turns from a week ago, even if cosine similarity
//! says otherwise. Without time-weighting, naive vector search will keep
//! surfacing the same old "best match" forever.
//!
//! # Scoring
//!
//! ```text
//! combined_score = similarity_score + (1 - decay_rate)^hours_since_access
//! ```
//!
//! LangChain uses additive (not multiplicative) so a stale doc with high
//! similarity can still beat a fresh doc with poor similarity. Decay rate
//! default 0.01/hour (~25% influence after 30 days). Set 0.0 for no time
//! effect; set 1.0 for "only freshly-accessed survives."
//!
//! # Side-effect: last_accessed bumps on retrieval
//!
//! Retrieved docs have their `last_accessed` timestamp updated to `now`.
//! That's how the system "remembers" what got used recently. The internal
//! timestamp map lives in the retriever (NOT in the vector store), so this
//! works with any `VectorStore` impl — no metadata-update plumbing needed.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use litgraph_core::{Document, Embeddings, Result};
use parking_lot::Mutex;

use crate::retriever::Retriever;
use crate::store::VectorStore;

const DEFAULT_DECAY_RATE: f32 = 0.01;
const MS_PER_HOUR: f64 = 3600.0 * 1000.0;

pub type ClockFn = Arc<dyn Fn() -> i64 + Send + Sync>;

pub struct TimeWeightedRetriever {
    pub embeddings: Arc<dyn Embeddings>,
    pub store: Arc<dyn VectorStore>,
    /// Per-doc-id last-accessed timestamp (ms since epoch). Read on every
    /// retrieve; written for every returned doc to bump its recency.
    last_accessed: Arc<Mutex<HashMap<String, i64>>>,
    pub decay_rate: f32,
    /// Clock callable — `now_ms()` in prod, fake-clock in tests. Lets tests
    /// be deterministic without sleeping for hours.
    clock: ClockFn,
    /// Over-fetch factor for the underlying vector search. Default 4 — gives
    /// the time-weighted scorer headroom to promote a high-recency / low-
    /// similarity doc above a low-recency / high-similarity one.
    pub over_fetch_factor: usize,
}

impl TimeWeightedRetriever {
    pub fn new(embeddings: Arc<dyn Embeddings>, store: Arc<dyn VectorStore>) -> Self {
        Self {
            embeddings,
            store,
            last_accessed: Arc::new(Mutex::new(HashMap::new())),
            decay_rate: DEFAULT_DECAY_RATE,
            clock: Arc::new(now_ms),
            over_fetch_factor: 4,
        }
    }

    pub fn with_decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = rate.clamp(0.0, 1.0); self
    }

    /// Override the clock — primarily for tests so we can simulate hours/days
    /// of elapsed time without sleeping. Production callers should leave the
    /// default `now_ms()`.
    pub fn with_clock<F>(mut self, f: F) -> Self
    where F: Fn() -> i64 + Send + Sync + 'static,
    { self.clock = Arc::new(f); self }

    pub fn with_over_fetch_factor(mut self, n: usize) -> Self {
        self.over_fetch_factor = n.max(1); self
    }

    /// Add documents to the store AND record their initial last_accessed
    /// timestamp. New docs start "fresh" (last_accessed = now), so they
    /// rank highest by recency until they're pushed down by decay or
    /// out-ranked by other recently-accessed docs.
    pub async fn add_documents(
        &self,
        docs: Vec<Document>,
        embeddings: Vec<Vec<f32>>,
    ) -> Result<Vec<String>> {
        let ids = self.store.add(docs, embeddings).await?;
        let now = (self.clock)();
        let mut g = self.last_accessed.lock();
        for id in &ids {
            g.insert(id.clone(), now);
        }
        Ok(ids)
    }

    /// Read-only snapshot of (id, last_accessed_ms). Useful for tests +
    /// observability dashboards ("which docs has the system actually used?").
    pub fn access_log(&self) -> Vec<(String, i64)> {
        let g = self.last_accessed.lock();
        let mut v: Vec<_> = g.iter().map(|(k, v)| (k.clone(), *v)).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0));
        v
    }
}

#[async_trait]
impl Retriever for TimeWeightedRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let q_emb = self.embeddings.embed_query(query).await?;
        let raw_k = (k * self.over_fetch_factor).max(k);
        let mut candidates = self
            .store
            .similarity_search(&q_emb, raw_k, None)
            .await?;
        if candidates.is_empty() {
            return Ok(candidates);
        }

        let now = (self.clock)();
        // Compute combined score per candidate.
        let mut scored: Vec<(f32, Document)> = candidates
            .drain(..)
            .map(|d| {
                let id = d.id.clone().unwrap_or_default();
                let sim = d.score.unwrap_or(0.0);
                let last = self
                    .last_accessed
                    .lock()
                    .get(&id)
                    .copied()
                    .unwrap_or(now); // Unknown id = treat as just-seen (no decay penalty).
                let hours = ((now - last) as f64 / MS_PER_HOUR).max(0.0) as f32;
                let recency = (1.0_f32 - self.decay_rate).powf(hours);
                (sim + recency, d)
            })
            .collect();
        // Sort descending by combined score.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        // Side effect: bump last_accessed on the returned docs to `now`.
        // This is the "memory of usage" mechanism — frequently-retrieved
        // docs stay surfaced; never-retrieved docs decay out over time.
        {
            let mut g = self.last_accessed.lock();
            for (_, d) in &scored {
                if let Some(id) = &d.id {
                    g.insert(id.clone(), now);
                }
            }
        }

        // Update score on the returned docs so callers can see the time-
        // weighted score (not the raw similarity).
        Ok(scored
            .into_iter()
            .map(|(s, mut d)| {
                d.score = Some(s);
                d
            })
            .collect())
    }
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::{Result as LgResult};
    use std::sync::atomic::{AtomicI64, Ordering};

    /// Constant-vector embedder — every text → same vector. Forces the
    /// similarity score to be deterministic regardless of text content,
    /// so tests isolate the time-weighting effect.
    struct ConstEmb;
    #[async_trait]
    impl Embeddings for ConstEmb {
        fn name(&self) -> &str { "const" }
        fn dimensions(&self) -> usize { 4 }
        async fn embed_query(&self, _q: &str) -> LgResult<Vec<f32>> {
            Ok(vec![1.0, 0.0, 0.0, 0.0])
        }
        async fn embed_documents(&self, ts: &[String]) -> LgResult<Vec<Vec<f32>>> {
            Ok(ts.iter().map(|_| vec![1.0, 0.0, 0.0, 0.0]).collect())
        }
    }

    /// Store that returns docs in insertion order with fixed similarity score.
    struct FixedStore {
        rows: Mutex<Vec<Document>>,
        score: f32,
    }
    impl FixedStore {
        fn new(score: f32) -> Self {
            Self { rows: Mutex::new(Vec::new()), score }
        }
    }
    #[async_trait]
    impl VectorStore for FixedStore {
        async fn add(
            &self,
            docs: Vec<Document>,
            _embs: Vec<Vec<f32>>,
        ) -> LgResult<Vec<String>> {
            let mut g = self.rows.lock();
            let mut ids = Vec::new();
            for mut d in docs {
                let id = d.id.clone().unwrap_or_else(|| format!("auto-{}", g.len()));
                d.id = Some(id.clone());
                g.push(d);
                ids.push(id);
            }
            Ok(ids)
        }
        async fn similarity_search(
            &self,
            _q: &[f32],
            k: usize,
            _f: Option<&crate::store::Filter>,
        ) -> LgResult<Vec<Document>> {
            let g = self.rows.lock();
            let mut out = Vec::new();
            for d in g.iter().take(k) {
                let mut copy = d.clone();
                copy.score = Some(self.score);
                out.push(copy);
            }
            Ok(out)
        }
        async fn delete(&self, _ids: &[String]) -> LgResult<()> { Ok(()) }
        async fn len(&self) -> usize { self.rows.lock().len() }
    }

    fn doc_with_id(id: &str, content: &str) -> Document {
        let mut d = Document::new(content);
        d.id = Some(id.into());
        d
    }

    #[tokio::test]
    async fn fresh_doc_outranks_stale_doc_when_similarity_equal() {
        // Two docs, identical similarity. Doc "old" was added 100 hours ago;
        // doc "new" was added now. With default decay rate 0.01, new's
        // recency component is much higher → new ranks first.
        let clock = Arc::new(AtomicI64::new(0));
        let clock_for_retriever = clock.clone();
        let store = Arc::new(FixedStore::new(0.5));
        let r = TimeWeightedRetriever::new(Arc::new(ConstEmb), store.clone())
            .with_clock(move || clock_for_retriever.load(Ordering::SeqCst));

        // t = 0: add "old" doc.
        clock.store(0, Ordering::SeqCst);
        r.add_documents(vec![doc_with_id("old", "first")], vec![vec![0.0; 4]])
            .await.unwrap();

        // t = 100h later: add "new" doc.
        let later = 100 * MS_PER_HOUR as i64;
        clock.store(later, Ordering::SeqCst);
        r.add_documents(vec![doc_with_id("new", "second")], vec![vec![0.0; 4]])
            .await.unwrap();

        // Retrieve at t = 100h. Old: similarity 0.5 + (0.99)^100 ≈ 0.5 + 0.366 = 0.866.
        // New: similarity 0.5 + (0.99)^0 = 0.5 + 1.0 = 1.5.
        let hits = r.retrieve("query", 2).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id.as_deref(), Some("new"));
        assert_eq!(hits[1].id.as_deref(), Some("old"));
        // Combined scores reflect the additive recency.
        assert!(hits[0].score.unwrap() > hits[1].score.unwrap());
    }

    #[tokio::test]
    async fn decay_rate_zero_means_pure_similarity_ordering() {
        // With decay_rate=0, recency is always 1.0 (constant), so every
        // doc gets +1.0 — relative ordering is determined purely by
        // similarity (which we've fixed at 0.5 for all docs). Order
        // preserved from the underlying store.
        let store = Arc::new(FixedStore::new(0.5));
        let r = TimeWeightedRetriever::new(Arc::new(ConstEmb), store.clone())
            .with_decay_rate(0.0);
        r.add_documents(
            vec![
                doc_with_id("a", "first"),
                doc_with_id("b", "second"),
                doc_with_id("c", "third"),
            ],
            vec![vec![0.0; 4]; 3],
        ).await.unwrap();
        let hits = r.retrieve("q", 3).await.unwrap();
        assert_eq!(hits.len(), 3);
        // No decay → all equally fresh → store insertion order preserved.
        assert_eq!(hits[0].id.as_deref(), Some("a"));
        assert_eq!(hits[1].id.as_deref(), Some("b"));
        assert_eq!(hits[2].id.as_deref(), Some("c"));
    }

    #[tokio::test]
    async fn retrieve_bumps_last_accessed_on_returned_docs() {
        // The "memory of usage" invariant. After retrieving a doc, its
        // last_accessed should be `now` — so a follow-up query later
        // ranks it as fresh again, even if it was originally stale.
        let clock = Arc::new(AtomicI64::new(0));
        let clock_for_retriever = clock.clone();
        let store = Arc::new(FixedStore::new(0.5));
        let r = TimeWeightedRetriever::new(Arc::new(ConstEmb), store.clone())
            .with_clock(move || clock_for_retriever.load(Ordering::SeqCst));

        clock.store(0, Ordering::SeqCst);
        r.add_documents(vec![doc_with_id("d1", "x")], vec![vec![0.0; 4]])
            .await.unwrap();

        // Jump 10h ahead, retrieve.
        clock.store(10 * MS_PER_HOUR as i64, Ordering::SeqCst);
        let _ = r.retrieve("q", 1).await.unwrap();

        // last_accessed["d1"] should now be 10h not 0.
        let log = r.access_log();
        let entry = log.iter().find(|(id, _)| id == "d1").unwrap();
        assert_eq!(entry.1, 10 * MS_PER_HOUR as i64);
    }

    #[tokio::test]
    async fn add_documents_initializes_last_accessed_to_now() {
        let clock = Arc::new(AtomicI64::new(0));
        let clock_for_retriever = clock.clone();
        let store = Arc::new(FixedStore::new(0.5));
        let r = TimeWeightedRetriever::new(Arc::new(ConstEmb), store.clone())
            .with_clock(move || clock_for_retriever.load(Ordering::SeqCst));

        clock.store(42_000, Ordering::SeqCst);  // fake "now" = 42 seconds
        r.add_documents(
            vec![doc_with_id("a", "x"), doc_with_id("b", "y")],
            vec![vec![0.0; 4]; 2],
        ).await.unwrap();
        let log = r.access_log();
        for (_, ts) in &log {
            assert_eq!(*ts, 42_000);
        }
    }

    #[tokio::test]
    async fn over_fetch_factor_lets_fresh_doc_overtake_high_sim_stale_doc() {
        // Without over-fetch (raw_k = k), if k=1 we'd only get the top-1
        // by similarity — no chance to promote the fresh-but-lower-sim
        // doc. With over_fetch_factor=4, we fetch 4 candidates and
        // re-rank by combined score.
        let clock = Arc::new(AtomicI64::new(0));
        let clock_for_retriever = clock.clone();
        // Custom store: returns docs with VARYING similarity by index.
        struct VariantStore {
            rows: Mutex<Vec<Document>>,
            scores: Vec<f32>,
        }
        #[async_trait]
        impl VectorStore for VariantStore {
            async fn add(
                &self,
                docs: Vec<Document>,
                _e: Vec<Vec<f32>>,
            ) -> LgResult<Vec<String>> {
                let mut g = self.rows.lock();
                let mut ids = Vec::new();
                for mut d in docs {
                    let id = d.id.clone().unwrap_or_else(|| format!("d{}", g.len()));
                    d.id = Some(id.clone());
                    g.push(d);
                    ids.push(id);
                }
                Ok(ids)
            }
            async fn similarity_search(
                &self,
                _q: &[f32],
                k: usize,
                _f: Option<&crate::store::Filter>,
            ) -> LgResult<Vec<Document>> {
                let g = self.rows.lock();
                let mut out = Vec::new();
                for (i, d) in g.iter().take(k).enumerate() {
                    let mut copy = d.clone();
                    copy.score = Some(self.scores.get(i).copied().unwrap_or(0.0));
                    out.push(copy);
                }
                Ok(out)
            }
            async fn delete(&self, _ids: &[String]) -> LgResult<()> { Ok(()) }
            async fn len(&self) -> usize { self.rows.lock().len() }
        }
        let store = Arc::new(VariantStore {
            rows: Mutex::new(Vec::new()),
            scores: vec![0.95, 0.50],  // first doc much higher similarity
        });
        let r = TimeWeightedRetriever::new(Arc::new(ConstEmb), store.clone())
            .with_clock(move || clock_for_retriever.load(Ordering::SeqCst));

        // Add the high-sim doc 1000h ago.
        clock.store(0, Ordering::SeqCst);
        r.add_documents(vec![doc_with_id("stale", "x")], vec![vec![0.0; 4]])
            .await.unwrap();
        // Add the low-sim doc now.
        clock.store(1000 * MS_PER_HOUR as i64, Ordering::SeqCst);
        r.add_documents(vec![doc_with_id("fresh", "y")], vec![vec![0.0; 4]])
            .await.unwrap();

        // Retrieve k=1. With over-fetch, we fetch 4 candidates → rerank →
        // fresh wins because (1-0.01)^1000 ≈ 4.3e-5, while fresh has
        // recency = 1.0. Combined: stale 0.95 + 0 ≈ 0.95; fresh 0.5 + 1.0 = 1.5.
        let hits = r.retrieve("q", 1).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id.as_deref(), Some("fresh"));
    }

    #[tokio::test]
    async fn empty_store_returns_empty() {
        let r = TimeWeightedRetriever::new(
            Arc::new(ConstEmb),
            Arc::new(FixedStore::new(0.5)),
        );
        let hits = r.retrieve("q", 5).await.unwrap();
        assert!(hits.is_empty());
    }
}

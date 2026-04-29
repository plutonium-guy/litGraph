//! Example selection for FewShot prompts.
//!
//! `FewShotChatPromptTemplate` (iter 92) takes a static `Vec<Value>` of
//! examples — every render uses ALL of them, regardless of relevance to
//! the current input. That works for small example pools but burns tokens
//! and dilutes attention as the pool grows. The fix is to PICK the top-K
//! most-relevant examples per call.
//!
//! `SemanticSimilarityExampleSelector` is the canonical strategy from
//! LangChain: embed each example once (cached), embed the query per call,
//! return top-K by cosine similarity. The selector returns a `Vec<Value>`
//! shaped exactly like `FewShotChatPromptTemplate.examples`, so plumbing
//! is one line:
//!
//! ```ignore
//! let selected = selector.select("how do I fix a borrow error?", 3).await?;
//! let prompt = few_shot.clone().with_examples(selected).format(&vars)?;
//! ```
//!
//! # Why standalone, not baked into `FewShotChatPromptTemplate`
//!
//! `FewShotChatPromptTemplate.format()` is sync. Selection is async (calls
//! out to an embedding provider). Wiring async into the prompt API would
//! cascade `.await` through every call site. Keeping the selector
//! standalone — caller awaits it once, passes the result to the sync
//! template — is cleaner.
//!
//! # Cache semantics
//!
//! The pool's embeddings are computed lazily on the first `select` call
//! and cached forever. Calling `warmup()` pre-computes them at startup
//! to avoid a first-call latency spike. The cache is per-instance — clone
//! the selector to share an embedded pool across threads (the inner Arc
//! makes this cheap).

use std::sync::Arc;

use parking_lot::RwLock;
use serde_json::Value;

use crate::{Embeddings, Error, Result};

pub struct SemanticSimilarityExampleSelector {
    pool: Vec<Value>,
    embeddings: Arc<dyn Embeddings>,
    /// Which JSON field of each example to embed. Common: "input",
    /// "question". Field must exist + be a string in every pool entry,
    /// else `select`/`warmup` errors with `Error::InvalidInput`.
    key_field: String,
    cached_embeds: Arc<RwLock<Option<Vec<Vec<f32>>>>>,
}

impl SemanticSimilarityExampleSelector {
    pub fn new(
        pool: Vec<Value>,
        embeddings: Arc<dyn Embeddings>,
        key_field: impl Into<String>,
    ) -> Self {
        Self {
            pool,
            embeddings,
            key_field: key_field.into(),
            cached_embeds: Arc::new(RwLock::new(None)),
        }
    }

    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }

    /// Pre-compute pool embeddings. Idempotent — second call is a no-op.
    /// Use to avoid first-`select` latency in latency-sensitive code paths.
    pub async fn warmup(&self) -> Result<()> {
        if self.cached_embeds.read().is_some() {
            return Ok(());
        }
        let texts = self.extract_pool_texts()?;
        if texts.is_empty() {
            *self.cached_embeds.write() = Some(Vec::new());
            return Ok(());
        }
        let embeds = self.embeddings.embed_documents(&texts).await?;
        if embeds.len() != self.pool.len() {
            return Err(Error::other(format!(
                "embeddings provider returned {} vectors for {} pool entries",
                embeds.len(),
                self.pool.len()
            )));
        }
        *self.cached_embeds.write() = Some(embeds);
        Ok(())
    }

    /// Top-K most-similar pool examples to `query`. Lazily warms up on
    /// first call. Returns up to `k` entries; fewer if the pool is smaller.
    /// Returns empty Vec if pool is empty (no error).
    pub async fn select(&self, query: &str, k: usize) -> Result<Vec<Value>> {
        if self.pool.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        // Warm up if needed (first call).
        if self.cached_embeds.read().is_none() {
            self.warmup().await?;
        }
        let q_vec = self.embeddings.embed_query(query).await?;
        // Hold a read-only snapshot of the cache for scoring — avoid
        // borrowing across the await above (we already awaited).
        let cache = self.cached_embeds.read();
        let pool_embeds: &Vec<Vec<f32>> = cache
            .as_ref()
            .expect("cache populated by warmup");

        let mut scored: Vec<(usize, f32)> = pool_embeds
            .iter()
            .enumerate()
            .map(|(i, ev)| (i, cosine_sim(&q_vec, ev)))
            .collect();
        // Stable sort (ties → first-occurrence) descending by score.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(scored
            .into_iter()
            .take(k)
            .map(|(idx, _)| self.pool[idx].clone())
            .collect())
    }

    fn extract_pool_texts(&self) -> Result<Vec<String>> {
        let mut out = Vec::with_capacity(self.pool.len());
        for (i, ex) in self.pool.iter().enumerate() {
            let v = ex
                .get(&self.key_field)
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "example {i} missing field `{}` (key_field for selector)",
                        self.key_field
                    ))
                })?;
            let s = v
                .as_str()
                .ok_or_else(|| {
                    Error::invalid(format!(
                        "example {i} field `{}` is not a string",
                        self.key_field
                    ))
                })?
                .to_string();
            out.push(s);
        }
        Ok(out)
    }
}

/// Cosine similarity. Returns 0.0 if either vector is all-zero (avoids NaN).
/// Mismatched lengths are zipped to the shorter — caller's responsibility
/// to keep dims consistent (it's a configuration bug, not runtime data).
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na.sqrt() * nb.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Embeddings;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Mutex;

    /// Toy embedder: hashes each text into a fixed-dim vector based on
    /// keyword presence. Deterministic + lets us craft tests where we
    /// know which examples will be most similar to which queries.
    struct KeywordEmbedder {
        keywords: Vec<&'static str>,
        embed_calls: Mutex<usize>,
    }

    impl KeywordEmbedder {
        fn new(keywords: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self {
                keywords,
                embed_calls: Mutex::new(0),
            })
        }
        fn vec_for(&self, text: &str) -> Vec<f32> {
            let lower = text.to_lowercase();
            self.keywords
                .iter()
                .map(|kw| if lower.contains(kw) { 1.0 } else { 0.0 })
                .collect()
        }
    }

    #[async_trait]
    impl Embeddings for KeywordEmbedder {
        fn name(&self) -> &str { "keyword" }
        fn dimensions(&self) -> usize { self.keywords.len() }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            *self.embed_calls.lock().unwrap() += 1;
            Ok(self.vec_for(text))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            *self.embed_calls.lock().unwrap() += texts.len();
            Ok(texts.iter().map(|t| self.vec_for(t)).collect())
        }
    }

    fn pool() -> Vec<Value> {
        vec![
            json!({"input": "fix borrow checker error in rust", "output": "use clone or rework lifetimes"}),
            json!({"input": "css flexbox alignment", "output": "use justify-content / align-items"}),
            json!({"input": "rust lifetime annotation syntax", "output": "use 'a notation"}),
            json!({"input": "javascript promise chaining", "output": "use .then or async/await"}),
            json!({"input": "python list comprehension syntax", "output": "use [x for x in ...]"}),
        ]
    }

    #[tokio::test]
    async fn select_top_k_by_keyword_overlap() {
        let embedder = KeywordEmbedder::new(vec!["rust", "borrow", "lifetime", "css", "flexbox", "javascript", "promise", "python", "list"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder, "input");
        let picked = sel.select("how do I fix a rust borrow problem", 2).await.unwrap();
        // Top 2 should both contain "rust".
        assert_eq!(picked.len(), 2);
        for ex in &picked {
            assert!(ex["input"].as_str().unwrap().contains("rust"),
                    "expected rust example, got {ex}");
        }
    }

    #[tokio::test]
    async fn select_caps_at_pool_size() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder, "input");
        let picked = sel.select("anything", 100).await.unwrap();
        assert_eq!(picked.len(), 5, "k larger than pool returns full pool");
    }

    #[tokio::test]
    async fn select_k_zero_returns_empty() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder, "input");
        let picked = sel.select("rust", 0).await.unwrap();
        assert!(picked.is_empty());
    }

    #[tokio::test]
    async fn empty_pool_returns_empty_no_error() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let sel = SemanticSimilarityExampleSelector::new(vec![], embedder, "input");
        let picked = sel.select("anything", 5).await.unwrap();
        assert!(picked.is_empty());
    }

    #[tokio::test]
    async fn missing_key_field_errors_with_invalid_input() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let bad_pool = vec![json!({"output": "no input field"})];
        let sel = SemanticSimilarityExampleSelector::new(bad_pool, embedder, "input");
        let err = sel.select("anything", 1).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert!(err.to_string().contains("input"));
    }

    #[tokio::test]
    async fn non_string_key_field_errors_with_invalid_input() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let bad_pool = vec![json!({"input": 42})];
        let sel = SemanticSimilarityExampleSelector::new(bad_pool, embedder, "input");
        let err = sel.select("anything", 1).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn pool_embeddings_cached_across_calls() {
        let embedder = KeywordEmbedder::new(vec!["rust", "css"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder.clone(), "input");

        // First call: 5 doc-embeds + 1 query-embed = 6
        let _ = sel.select("rust", 1).await.unwrap();
        let after_first = *embedder.embed_calls.lock().unwrap();
        assert_eq!(after_first, 6);

        // Second call: only 1 query-embed (pool is cached).
        let _ = sel.select("css", 1).await.unwrap();
        let after_second = *embedder.embed_calls.lock().unwrap();
        assert_eq!(after_second, 7, "expected one extra query-embed, got {after_second} - {after_first}");
    }

    #[tokio::test]
    async fn warmup_pre_embeds_pool() {
        let embedder = KeywordEmbedder::new(vec!["rust"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder.clone(), "input");
        sel.warmup().await.unwrap();
        assert_eq!(*embedder.embed_calls.lock().unwrap(), 5);
        // Warmup again is a no-op.
        sel.warmup().await.unwrap();
        assert_eq!(*embedder.embed_calls.lock().unwrap(), 5);
    }

    #[tokio::test]
    async fn select_orders_by_descending_similarity() {
        let embedder = KeywordEmbedder::new(vec!["rust", "lifetime", "borrow"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder, "input");
        let picked = sel.select("rust borrow lifetime", 3).await.unwrap();
        assert_eq!(picked.len(), 3);
        // First two should be the rust-themed examples (both match all 3
        // keywords); third is something else (probably first remaining).
        let inputs: Vec<&str> = picked.iter().map(|p| p["input"].as_str().unwrap()).collect();
        assert!(inputs[0].contains("rust"));
        assert!(inputs[1].contains("rust"));
    }

    #[tokio::test]
    async fn cosine_zero_vector_returns_zero_not_nan() {
        let s = cosine_sim(&[0.0, 0.0, 0.0], &[1.0, 1.0, 1.0]);
        assert_eq!(s, 0.0);
        let s = cosine_sim(&[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(s, 0.0);
    }

    #[tokio::test]
    async fn pool_size_reports_pool_length() {
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let sel = SemanticSimilarityExampleSelector::new(pool(), embedder, "input");
        assert_eq!(sel.pool_size(), 5);
    }
}

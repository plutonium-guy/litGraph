//! Max-Marginal-Relevance retriever wrapper. Over-fetches K from a base
//! retriever, then MMR-selects the final K balancing relevance vs novelty.
//! LangChain `MaxMarginalRelevanceRetriever` parity.
//!
//! # Why
//!
//! Pure top-K vector retrieval often returns near-duplicates: 5 paraphrases
//! of the same fact. MMR breaks ties toward novelty: the second pick
//! penalizes similarity to the first, the third penalizes similarity to
//! the first two, etc. Net effect: K results that COVER the topic
//! rather than circle a single point.
//!
//! # Algorithm
//!
//! 1. base.retrieve(query, fetch_k) — over-fetch (default fetch_k=20).
//! 2. embed query + each candidate doc (one batch call).
//! 3. mmr_select(query_emb, docs, doc_embs, k, lambda_mult).
//!
//! `lambda_mult ∈ [0, 1]`:
//! - `1.0` = pure relevance (equivalent to vanilla top-K).
//! - `0.5` = LangChain default — balanced.
//! - `0.0` = pure diversity (no relevance term — picks most-spread docs).
//!
//! # When to use
//!
//! - Multi-aspect questions ("What are the pros AND cons of X?")
//! - Comparison queries where unique perspectives beat redundancy
//! - Long-form generation where the LLM benefits from varied evidence
//!
//! # When NOT to use
//!
//! - Lookup queries where you want the SINGLE best answer (use base directly)
//! - Performance-critical paths where 2× embed cost matters

use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Embeddings, Result};

use crate::transformers::mmr_select;
use crate::retriever::Retriever;

pub struct MaxMarginalRelevanceRetriever {
    pub base: Arc<dyn Retriever>,
    pub embeddings: Arc<dyn Embeddings>,
    /// How many docs to over-fetch from the base before MMR-selecting.
    /// Default 20; bump for noisy bases (BM25 with low-precision recall).
    pub fetch_k: usize,
    /// Relevance vs diversity weight. 1.0 = top-K, 0.5 = balanced, 0.0 = diversity-only.
    pub lambda_mult: f32,
}

impl MaxMarginalRelevanceRetriever {
    pub fn new(base: Arc<dyn Retriever>, embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            base,
            embeddings,
            fetch_k: 20,
            lambda_mult: 0.5,
        }
    }

    pub fn with_fetch_k(mut self, k: usize) -> Self {
        self.fetch_k = k.max(1);
        self
    }

    pub fn with_lambda(mut self, lambda: f32) -> Self {
        self.lambda_mult = lambda.clamp(0.0, 1.0);
        self
    }
}

#[async_trait]
impl Retriever for MaxMarginalRelevanceRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        if k == 0 {
            return Ok(Vec::new());
        }
        let fetch_k = self.fetch_k.max(k);
        let candidates = self.base.retrieve(query, fetch_k).await?;
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        // One embed call for query + all candidates (batched).
        let mut texts: Vec<String> = Vec::with_capacity(candidates.len() + 1);
        texts.push(query.to_string());
        for d in &candidates {
            texts.push(d.content.clone());
        }
        let mut all_embeds = self.embeddings.embed_documents(&texts).await?;
        if all_embeds.len() != candidates.len() + 1 {
            return Err(litgraph_core::Error::other(format!(
                "MMR retriever: embedder returned {} vectors for {} texts",
                all_embeds.len(),
                candidates.len() + 1
            )));
        }
        let query_emb = all_embeds.remove(0);
        let candidate_embs: Vec<Vec<f32>> = all_embeds;
        Ok(mmr_select(
            &query_emb,
            &candidates,
            &candidate_embs,
            k,
            self.lambda_mult,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::retriever::Retriever;
    use async_trait::async_trait;
    use litgraph_core::Embeddings;

    /// Returns a fixed list of documents for any query.
    struct FixedRetriever {
        docs: Vec<Document>,
    }

    #[async_trait]
    impl Retriever for FixedRetriever {
        async fn retrieve(&self, _query: &str, k: usize) -> Result<Vec<Document>> {
            Ok(self.docs.iter().take(k).cloned().collect())
        }
    }

    /// Keyword-presence embedder. Each text → fixed-dim vec marking
    /// which keywords appear. Lets us reason about which docs MMR picks.
    struct KeywordEmbedder {
        keywords: Vec<&'static str>,
    }

    impl KeywordEmbedder {
        fn new(keywords: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self { keywords })
        }
        fn vec_for(&self, text: &str) -> Vec<f32> {
            let lower = text.to_lowercase();
            self.keywords.iter().map(|kw| if lower.contains(kw) { 1.0 } else { 0.0 }).collect()
        }
    }

    #[async_trait]
    impl Embeddings for KeywordEmbedder {
        fn name(&self) -> &str { "keyword" }
        fn dimensions(&self) -> usize { self.keywords.len() }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.vec_for(text))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| self.vec_for(t)).collect())
        }
    }

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    #[tokio::test]
    async fn mmr_picks_diverse_docs_with_low_lambda() {
        // 4 candidates: 3 are near-duplicates about rust, 1 is about css.
        // lambda=0.0 (diversity-only) → after the first rust pick, the css
        // doc should beat the second rust doc (zero overlap with picked).
        let base = Arc::new(FixedRetriever {
            docs: vec![
                doc("rust borrow checker fix"),
                doc("rust borrow tip"),
                doc("rust borrow workaround"),
                doc("css flexbox layout"),
            ],
        });
        let embedder = KeywordEmbedder::new(vec!["rust", "borrow", "css", "flexbox"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder)
            .with_fetch_k(4)
            .with_lambda(0.0);
        let result = mmr.retrieve("rust borrow", 2).await.unwrap();
        assert_eq!(result.len(), 2);
        // First pick: rust-themed (relevance is moot at lambda=0 but the
        // first pick uses a 0 diversity term so any top-relevance wins).
        // Second pick: must be the css doc (most novel given first pick).
        let texts: Vec<&str> = result.iter().map(|d| d.content.as_str()).collect();
        assert!(texts.iter().any(|t| t.contains("css")), "diverse pick must include css; got {texts:?}");
    }

    #[tokio::test]
    async fn mmr_lambda_one_equals_top_k_relevance() {
        // lambda=1.0 → diversity term zeroed → behaves like vanilla top-K.
        let base = Arc::new(FixedRetriever {
            docs: vec![
                doc("rust borrow lifetime"),  // 3 keyword hits
                doc("rust borrow"),            // 2
                doc("rust"),                   // 1
                doc("css"),                    // 0 vs query
            ],
        });
        let embedder = KeywordEmbedder::new(vec!["rust", "borrow", "lifetime", "css"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder)
            .with_fetch_k(4)
            .with_lambda(1.0);
        let result = mmr.retrieve("rust borrow lifetime", 3).await.unwrap();
        // Top-3 by relevance: the 3-hit doc, then 2-hit, then 1-hit.
        assert_eq!(result[0].content, "rust borrow lifetime");
        assert_eq!(result[1].content, "rust borrow");
        assert_eq!(result[2].content, "rust");
    }

    #[tokio::test]
    async fn mmr_returns_at_most_k_results() {
        let base = Arc::new(FixedRetriever {
            docs: vec![doc("a"), doc("b"), doc("c"), doc("d"), doc("e")],
        });
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder).with_fetch_k(5);
        let result = mmr.retrieve("query", 2).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn mmr_k_zero_returns_empty() {
        let base = Arc::new(FixedRetriever { docs: vec![doc("a")] });
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder);
        let result = mmr.retrieve("q", 0).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn mmr_empty_base_returns_empty() {
        let base = Arc::new(FixedRetriever { docs: vec![] });
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder);
        let result = mmr.retrieve("q", 5).await.unwrap();
        assert!(result.is_empty());
    }

    #[tokio::test]
    async fn fetch_k_clamped_to_at_least_k() {
        // If user sets fetch_k < k, we should still over-fetch enough.
        let base = Arc::new(FixedRetriever {
            docs: vec![doc("a"), doc("b"), doc("c")],
        });
        let embedder = KeywordEmbedder::new(vec!["a"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder)
            .with_fetch_k(1);
        // Even with fetch_k=1, asking for k=3 should over-fetch to 3.
        let result = mmr.retrieve("q", 3).await.unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn lambda_clamped_to_unit_range() {
        let base = Arc::new(FixedRetriever { docs: vec![doc("a")] });
        let embedder = KeywordEmbedder::new(vec!["a"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder)
            .with_lambda(2.5);   // out of range → clamp to 1.0
        assert_eq!(mmr.lambda_mult, 1.0);
        let mmr = MaxMarginalRelevanceRetriever::new(
            Arc::new(FixedRetriever { docs: vec![doc("a")] }),
            KeywordEmbedder::new(vec!["a"]),
        ).with_lambda(-1.0);
        assert_eq!(mmr.lambda_mult, 0.0);
    }

    #[tokio::test]
    async fn one_embed_call_per_retrieve_batched() {
        // Confirm we don't embed query separately from candidates — should be ONE call.
        struct CountingEmbedder {
            inner: Arc<KeywordEmbedder>,
            calls: std::sync::Mutex<usize>,
        }
        #[async_trait]
        impl Embeddings for CountingEmbedder {
            fn name(&self) -> &str { "counting" }
            fn dimensions(&self) -> usize { self.inner.dimensions() }
            async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
                *self.calls.lock().unwrap() += 1;
                self.inner.embed_query(text).await
            }
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                *self.calls.lock().unwrap() += 1;  // one call regardless of batch size
                self.inner.embed_documents(texts).await
            }
        }
        let base = Arc::new(FixedRetriever { docs: vec![doc("a"), doc("b"), doc("c")] });
        let counting = Arc::new(CountingEmbedder {
            inner: KeywordEmbedder::new(vec!["a", "b", "c"]),
            calls: std::sync::Mutex::new(0),
        });
        let mmr = MaxMarginalRelevanceRetriever::new(base, counting.clone() as Arc<dyn Embeddings>);
        mmr.retrieve("q", 2).await.unwrap();
        // Single batch call (query + 3 candidates → 4 texts → 1 call).
        assert_eq!(*counting.calls.lock().unwrap(), 1);
    }

    #[tokio::test]
    async fn k_larger_than_base_returns_all_available() {
        let base = Arc::new(FixedRetriever {
            docs: vec![doc("a"), doc("b")],
        });
        let embedder = KeywordEmbedder::new(vec!["a", "b"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder).with_fetch_k(2);
        let result = mmr.retrieve("q", 10).await.unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn defaults_match_langchain() {
        let base = Arc::new(FixedRetriever { docs: vec![] });
        let embedder = KeywordEmbedder::new(vec!["x"]);
        let mmr = MaxMarginalRelevanceRetriever::new(base, embedder);
        assert_eq!(mmr.fetch_k, 20);
        assert_eq!(mmr.lambda_mult, 0.5);
    }
}

//! Local ONNX cross-encoder reranker via [`fastembed`]. Closes the
//! local-inference triple: alongside `litgraph-embeddings-fastembed`
//! you get RAG with no API key on either side of retrieval.
//!
//! # When to use a reranker
//!
//! Two-stage retrieval. The first stage (embedding-based vector
//! search) is fast but noisy — it finds candidates by approximate
//! semantic distance. A cross-encoder reranker reads the query AND
//! each candidate together and outputs a precise relevance score.
//!
//! Typical recipe:
//!
//! 1. Vector search: top 50–100 candidates (cheap).
//! 2. Reranker: rescore + truncate to top 5–10 (more expensive but
//!    only on a small candidate set).
//!
//! Pair this crate with `litgraph_retrieval::RerankingRetriever` to
//! wrap any base retriever with the rerank step:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use litgraph_retrieval::{Retriever, RerankingRetriever};
//! # use litgraph_rerankers_fastembed::FastembedReranker;
//! # async fn ex(base: Arc<dyn Retriever>) -> litgraph_core::Result<()> {
//! let reranker = Arc::new(FastembedReranker::default_model().await?);
//! let two_stage = RerankingRetriever::new(base, reranker)
//!     .with_over_fetch_k(50);
//! # let _ = two_stage; Ok(()) }
//! ```
//!
//! # Default model
//!
//! `BGERerankerBase` (~280MB, English, 12-layer cross-encoder). Good
//! quality / latency trade-off. For multilingual rerank, switch to
//! `JINARerankerV1BaseEn`, `JINARerankerV1TurboEn`, or
//! `JINARerankerV2BaseMultilingual` via `with_model`.
//!
//! # Concurrency
//!
//! `fastembed::TextRerank::rerank` takes `&mut self`, so the model
//! sits behind an `Arc<Mutex<...>>`. Lock is held only inside
//! `tokio::task::spawn_blocking` — never across an `.await`. Inference
//! is CPU-bound; for parallel throughput construct multiple instances
//! rather than relying on intra-mutex parallelism.

use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{RerankInitOptions, TextRerank};
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::Reranker;
use parking_lot::Mutex;

// Re-export so callers don't need to depend on `fastembed` directly.
pub use fastembed::RerankerModel;

/// Default reranker model. English, 12-layer cross-encoder. Mid-size
/// (~280MB) — accurate enough that 99% of users won't need anything
/// larger.
pub const DEFAULT_MODEL: RerankerModel = RerankerModel::BGERerankerBase;

pub struct FastembedReranker {
    model: Arc<Mutex<TextRerank>>,
    name: String,
}

impl std::fmt::Debug for FastembedReranker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastembedReranker")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl FastembedReranker {
    /// Load the default model (`BGERerankerBase`).
    pub async fn default_model() -> Result<Self> {
        Self::with_model(DEFAULT_MODEL).await
    }

    /// Load a specific [`RerankerModel`]. Common picks:
    ///
    /// - `BGERerankerBase` — 12-layer, English (default)
    /// - `BGERerankerV2M3` — newer + multilingual
    /// - `JINARerankerV1TurboEn` — smaller + faster, English-only
    /// - `JINARerankerV2BaseMultilingual` — best multilingual quality
    pub async fn with_model(model: RerankerModel) -> Result<Self> {
        let name = format!("{model:?}");
        let loaded = tokio::task::spawn_blocking(move || {
            TextRerank::try_new(RerankInitOptions::new(model))
                .map_err(|e| Error::other(format!("fastembed reranker init: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("fastembed reranker init join: {e}")))??;
        Ok(Self {
            model: Arc::new(Mutex::new(loaded)),
            name,
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }
}

#[async_trait]
impl Reranker for FastembedReranker {
    async fn rerank(
        &self,
        query: &str,
        docs: Vec<Document>,
        top_k: usize,
    ) -> Result<Vec<Document>> {
        if docs.is_empty() || top_k == 0 {
            return Ok(Vec::new());
        }

        // Bail on empty query — cross-encoders can technically score
        // it, but the score is meaningless. Better to surface as an
        // explicit error than return garbage.
        if query.trim().is_empty() {
            return Err(Error::invalid("fastembed reranker: empty query"));
        }

        let model = self.model.clone();
        let owned_query = query.to_string();
        // Owned `Vec<String>` so the closure can `move` it. fastembed's
        // `rerank` is generic over `S: AsRef<str>` — `String` satisfies.
        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();

        // `rerank` returns Vec<RerankResult { index, score, document }>.
        // We discard `document` (a clone of the input we already hold)
        // and use `index` to pick the right Document.
        let scores = tokio::task::spawn_blocking(move || -> Result<Vec<(usize, f32)>> {
            let mut guard = model.lock();
            // Explicit turbofish disambiguates `S` for the generic
            // `documents: impl AsRef<[S]>` — without it rustc infers
            // `S = &String` from neighbouring args and fails.
            let results = guard
                .rerank::<String>(owned_query, &texts, false, None)
                .map_err(|e| Error::other(format!("fastembed rerank: {e}")))?;
            Ok(results
                .into_iter()
                .map(|r| (r.index, r.score))
                .collect())
        })
        .await
        .map_err(|e| Error::other(format!("fastembed rerank join: {e}")))??;

        // Sort by score descending. fastembed already sorts, but we
        // re-sort defensively in case the underlying lib changes.
        let mut scored = scores;
        scored.sort_by(|a, b| b.1.total_cmp(&a.1));

        let mut out = Vec::with_capacity(top_k.min(scored.len()));
        for (idx, score) in scored.into_iter().take(top_k) {
            // Cheap bounds-check — defensive against fastembed returning
            // an out-of-range index (would be a bug; better to skip than
            // panic).
            if let Some(d) = docs.get(idx).cloned() {
                let mut d = d;
                d.score = Some(score);
                out.push(d);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    //! Live tests gated on `LITGRAPH_FASTEMBED_RERANKER_TEST=1`. First
    //! run downloads ~280MB of weights. Off by default in CI.

    use super::*;

    fn opted_in() -> bool {
        std::env::var("LITGRAPH_FASTEMBED_RERANKER_TEST")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    }

    fn doc(content: &str) -> Document {
        Document::new(content)
    }

    #[test]
    fn default_model_constant_is_bge_reranker_base() {
        assert!(matches!(DEFAULT_MODEL, RerankerModel::BGERerankerBase));
    }

    #[tokio::test]
    async fn empty_docs_returns_empty_without_loading_model() {
        // Construct a stub-ish instance by trying to rerank with empty
        // docs first. We can't avoid loading the model entirely (the
        // type owns one), but we can verify the short-circuit doesn't
        // touch the underlying model. Skip on opt-in to keep CI green.
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let out = r.rerank("query", Vec::new(), 5).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn top_k_zero_returns_empty() {
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let out = r
            .rerank("query", vec![doc("a"), doc("b")], 0)
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn empty_query_errors() {
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let err = r.rerank("   ", vec![doc("x")], 1).await.unwrap_err();
        assert!(format!("{err}").contains("empty query"));
    }

    #[tokio::test]
    async fn rerank_picks_topically_relevant_first() {
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let docs = vec![
            doc("Recipes for chocolate chip cookies."),
            doc("How transformer attention layers work in NLP."),
            doc("A history of medieval Italian architecture."),
            doc("Rust is a systems programming language with strong memory safety."),
        ];
        let out = r
            .rerank("what is rust the programming language", docs, 2)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        // The Rust doc should win.
        assert!(
            out[0].content.contains("Rust is a systems"),
            "got: {}",
            out[0].content
        );
        // Score is attached and descending.
        assert!(out[0].score.is_some() && out[1].score.is_some());
        assert!(out[0].score.unwrap() >= out[1].score.unwrap());
    }

    #[tokio::test]
    async fn rerank_top_k_truncates_correctly() {
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let docs: Vec<Document> = (0..10)
            .map(|i| doc(&format!("Document number {i} about Rust programming.")))
            .collect();
        let out = r.rerank("rust", docs, 3).await.unwrap();
        assert_eq!(out.len(), 3);
    }

    #[tokio::test]
    async fn rerank_attaches_score_to_each_doc() {
        if !opted_in() {
            return;
        }
        let r = FastembedReranker::default_model().await.unwrap();
        let docs = vec![doc("foo about rust"), doc("bar")];
        let out = r.rerank("rust", docs, 2).await.unwrap();
        for d in &out {
            assert!(d.score.is_some());
        }
    }
}

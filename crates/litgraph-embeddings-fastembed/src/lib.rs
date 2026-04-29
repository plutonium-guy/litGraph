//! Local ONNX-backed embeddings via [`fastembed`]. Production-grade
//! air-gap RAG: no API key, no network at inference time, models
//! cached on disk after first download.
//!
//! # When to reach for this vs hosted embeddings
//!
//! | Need                                | Pick                           |
//! |-------------------------------------|--------------------------------|
//! | Air-gapped / regulated env          | This crate                     |
//! | Latency < 10ms / batch              | This crate (CPU is plenty)     |
//! | Best quality across many languages  | Hosted (Cohere, OpenAI, Voyage)|
//! | Per-call cost matters               | This crate (zero marginal)     |
//! | First-call must be < 100ms always   | Hosted (no model download)     |
//!
//! Defaults to a small, fast English model (`bge-small-en-v1.5`,
//! 384-dim, ~30MB). Override via [`FastembedEmbeddings::with_model`].
//!
//! # First-call cost
//!
//! First instantiation downloads the model + tokenizer via `hf-hub`
//! into `~/.cache/huggingface`. This is a one-time cost; subsequent
//! starts are ms-fast. For air-gap deployments, pre-populate the cache
//! during your image build:
//!
//! ```bash
//! # During Docker build, with network access:
//! cargo run --example warm_fastembed_cache
//! # Then ship `~/.cache/huggingface` into the runtime image.
//! ```
//!
//! # Concurrency
//!
//! `fastembed::TextEmbedding::embed` runs CPU-bound inference. We wrap
//! every call in `tokio::task::spawn_blocking` so the async runtime
//! stays responsive. The underlying ONNX session is internally
//! thread-safe (mutex'd by ort) so a single `FastembedEmbeddings`
//! instance can be cloned via `Arc` and used concurrently.
//!
//! # Example
//!
//! ```no_run
//! # use std::sync::Arc;
//! use litgraph_core::Embeddings;
//! use litgraph_embeddings_fastembed::FastembedEmbeddings;
//!
//! # async fn ex() -> litgraph_core::Result<()> {
//! let emb = FastembedEmbeddings::default_model().await?;
//! let v = emb.embed_query("what is rust?").await?;
//! assert_eq!(v.len(), emb.dimensions());
//! # Ok(()) }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use fastembed::{InitOptions, TextEmbedding};
use litgraph_core::{Embeddings, Error, Result};
use parking_lot::Mutex;

// Re-export so callers don't need to depend on `fastembed` directly.
pub use fastembed::EmbeddingModel;

/// Default model — small, fast, English-only, 384-dim. Trade-off:
/// lower quality than `bge-base-en-v1.5` (768-dim) but ~3× faster
/// and 2× smaller on disk.
pub const DEFAULT_MODEL: EmbeddingModel = EmbeddingModel::BGESmallENV15;

/// ONNX-backed local embeddings.
///
/// `fastembed::TextEmbedding::embed` takes `&mut self`, so we hold the
/// model behind an `Arc<Mutex<...>>`. Inference is CPU-bound and runs
/// inside `spawn_blocking`, so the lock is held only for the duration
/// of the actual embed call — never across an `.await`. Callers that
/// want concurrent throughput should construct two instances rather
/// than expecting parallelism through one mutex.
pub struct FastembedEmbeddings {
    model: Arc<Mutex<TextEmbedding>>,
    /// Display name returned by `Embeddings::name`.
    name: String,
    /// Cached dimension — computed once at construction by embedding a
    /// sentinel string and reading the resulting vector length.
    dim: usize,
}

impl std::fmt::Debug for FastembedEmbeddings {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastembedEmbeddings")
            .field("name", &self.name)
            .field("dim", &self.dim)
            .finish_non_exhaustive()
    }
}

impl FastembedEmbeddings {
    /// Load the default model (`bge-small-en-v1.5`).
    pub async fn default_model() -> Result<Self> {
        Self::with_model(DEFAULT_MODEL).await
    }

    /// Load a specific [`EmbeddingModel`]. The full enum is re-exported
    /// from `fastembed` — common picks:
    ///
    /// - `BGESmallENV15` — 384-dim, English (default)
    /// - `BGEBaseENV15` — 768-dim, English (~3× compute, better quality)
    /// - `BGELargeENV15` — 1024-dim, English (best EN quality)
    /// - `MultilingualE5Small` — 384-dim, 100+ languages
    /// - `AllMiniLML6V2` — 384-dim, English, the LangChain default
    pub async fn with_model(model: EmbeddingModel) -> Result<Self> {
        let name = format!("{model:?}");
        // Loading touches the network on first run + does ONNX session
        // setup — both blocking; run on a blocking thread so we don't
        // stall the runtime.
        let loaded = tokio::task::spawn_blocking(move || {
            TextEmbedding::try_new(InitOptions::new(model)).map_err(|e| {
                Error::other(format!("fastembed init: {e}"))
            })
        })
        .await
        .map_err(|e| Error::other(format!("fastembed init join: {e}")))??;
        let model = Arc::new(Mutex::new(loaded));

        // Discover dim by embedding a sentinel. Cheaper than parsing
        // model metadata + works for every model fastembed supports.
        let probe_model = model.clone();
        let probe = tokio::task::spawn_blocking(move || {
            probe_model
                .lock()
                .embed(vec!["x"], None)
                .map_err(|e| Error::other(format!("fastembed probe: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("fastembed probe join: {e}")))??;
        let dim = probe
            .first()
            .map(|v| v.len())
            .ok_or_else(|| Error::other("fastembed probe returned no vectors"))?;

        Ok(Self { model, name, dim })
    }
}

#[async_trait]
impl Embeddings for FastembedEmbeddings {
    fn name(&self) -> &str {
        &self.name
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let model = self.model.clone();
        let owned = text.to_string();
        let mut out = tokio::task::spawn_blocking(move || {
            model
                .lock()
                .embed(vec![owned], None)
                .map_err(|e| Error::other(format!("fastembed embed_query: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("fastembed embed_query join: {e}")))??;
        out.pop()
            .ok_or_else(|| Error::other("fastembed embed_query returned no vectors"))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let model = self.model.clone();
        let owned = texts.to_vec();
        tokio::task::spawn_blocking(move || {
            model
                .lock()
                .embed(owned, None)
                .map_err(|e| Error::other(format!("fastembed embed_documents: {e}")))
        })
        .await
        .map_err(|e| Error::other(format!("fastembed embed_documents join: {e}")))?
    }
}

#[cfg(test)]
mod tests {
    //! Live tests are gated on `LITGRAPH_FASTEMBED_TEST=1` because the
    //! first run downloads ~30MB of model weights. Set it locally
    //! when iterating; CI can opt-in once cache is warm.
    //!
    //! ```sh
    //! LITGRAPH_FASTEMBED_TEST=1 cargo test -p litgraph-embeddings-fastembed
    //! ```
    //!
    //! All tests bail cleanly when the env var is unset, so the suite
    //! is a no-op on CI by default.

    use super::*;

    fn opted_in() -> bool {
        std::env::var("LITGRAPH_FASTEMBED_TEST")
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    }

    #[tokio::test]
    async fn default_model_dimensions_match_bge_small() {
        if !opted_in() {
            return;
        }
        let emb = FastembedEmbeddings::default_model().await.unwrap();
        assert_eq!(emb.dimensions(), 384);
        assert!(emb.name().contains("BGESmall"), "name={}", emb.name());
    }

    #[tokio::test]
    async fn embed_query_returns_dim_length_vector() {
        if !opted_in() {
            return;
        }
        let emb = FastembedEmbeddings::default_model().await.unwrap();
        let v = emb.embed_query("hello world").await.unwrap();
        assert_eq!(v.len(), emb.dimensions());
        // Embeddings are L2-normalized for BGE-style models; the norm
        // should be close to 1.
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.05, "norm={norm}");
    }

    #[tokio::test]
    async fn embed_documents_batches_in_one_call() {
        if !opted_in() {
            return;
        }
        let emb = FastembedEmbeddings::default_model().await.unwrap();
        let docs = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];
        let vs = emb.embed_documents(&docs).await.unwrap();
        assert_eq!(vs.len(), 3);
        for v in &vs {
            assert_eq!(v.len(), emb.dimensions());
        }
    }

    #[tokio::test]
    async fn embed_documents_empty_input_short_circuits() {
        // Doesn't need the live model — short-circuits before init.
        // We construct a stub-like instance via direct field init…
        // actually we can't without loading the model. Fine: gate this
        // on opt-in and hit the real model.
        if !opted_in() {
            return;
        }
        let emb = FastembedEmbeddings::default_model().await.unwrap();
        let vs = emb.embed_documents(&[]).await.unwrap();
        assert!(vs.is_empty());
    }

    #[tokio::test]
    async fn semantic_similar_pairs_score_higher_than_random() {
        if !opted_in() {
            return;
        }
        let emb = FastembedEmbeddings::default_model().await.unwrap();
        let queen = emb.embed_query("queen of england").await.unwrap();
        let king = emb.embed_query("king of england").await.unwrap();
        let truck = emb.embed_query("how to fix a flat tire").await.unwrap();

        // BGE outputs are pre-normalized → dot product ≈ cosine.
        let sim_kq = dot(&queen, &king);
        let sim_qt = dot(&queen, &truck);
        assert!(
            sim_kq > sim_qt + 0.05,
            "expected queen↔king > queen↔truck, got {sim_kq} vs {sim_qt}"
        );
    }

    fn dot(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }

    #[test]
    fn default_model_constant_is_bge_small_en_v15() {
        assert!(matches!(DEFAULT_MODEL, EmbeddingModel::BGESmallENV15));
    }
}

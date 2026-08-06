use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result};
use tracing::{debug, warn};

use crate::retry::is_transient;


/// Chat model wrapper that tries a chain of inner models in order. On
/// transient failure (rate-limit / timeout / 5xx) — or any error if
/// `fall_through_on_all` is true — moves to the next model. The LAST
/// model's error is propagated to the caller.
///
/// LangChain `Runnable.with_fallbacks([backup1, backup2])` parity. Real
/// prod patterns:
/// - **Provider failover**: GPT-4 primary, Claude backup, Gemini tertiary.
///   When OpenAI has an outage, requests transparently route to Anthropic.
/// - **Cost shedding**: GPT-4 primary, GPT-3.5 backup. On rate-limit,
///   degrade to the cheaper model rather than block the user.
/// - **Region failover**: us-east primary, us-west backup. On region
///   outage, re-route within minutes.
///
/// # When to use vs `RetryingChatModel`
///
/// - `RetryingChatModel`: same provider, retry transient errors with
///   exponential backoff. Use for "OpenAI 429s — try again in 500ms".
/// - `FallbackChatModel`: DIFFERENT provider, immediate switch. Use for
///   "OpenAI is down — try Anthropic right now". Compose them: wrap
///   each inner provider in `RetryingChatModel`, then wrap the chain in
///   `FallbackChatModel`.
///
/// # Streaming
///
/// `stream()` only tries the FIRST inner model — token streams can't
/// gracefully fail-over mid-stream once the first chunk arrives. For
/// streaming with fallback, capture the failure pre-stream-start at the
/// consumer layer and re-call.
pub struct FallbackChatModel {
    /// Ordered list of providers. First is primary; rest are backups.
    pub inners: Vec<Arc<dyn ChatModel>>,
    /// If true, fall through on ANY error (not just transient ones).
    /// Default false — preserves the "bad request → fail fast" semantics
    /// of `RetryingChatModel`.
    pub fall_through_on_all: bool,
}

impl FallbackChatModel {
    /// Build a fallback chain. Panics if `inners` is empty (a chain with
    /// no providers can't satisfy any request).
    pub fn new(inners: Vec<Arc<dyn ChatModel>>) -> Self {
        assert!(
            !inners.is_empty(),
            "FallbackChatModel: chain must have at least one model"
        );
        Self {
            inners,
            fall_through_on_all: false,
        }
    }

    /// Configure to fall through on every error (4xx and parse failures
    /// included). Use when the backup providers are TRULY equivalent
    /// substitutes; default `false` is safer because a malformed request
    /// against provider A will likely fail the same way against provider B.
    pub fn fall_through_on_all(mut self) -> Self {
        self.fall_through_on_all = true;
        self
    }
}

#[async_trait]
impl ChatModel for FallbackChatModel {
    fn name(&self) -> &str {
        // Names of every backed model would be churn; use a stable label.
        "fallback"
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let last_idx = self.inners.len() - 1;
        let mut last_err: Option<Error> = None;
        for (i, inner) in self.inners.iter().enumerate() {
            match inner.invoke(messages.clone(), opts).await {
                Ok(resp) => {
                    if i > 0 {
                        debug!(
                            primary_failed = %last_err.as_ref().map(|e| e.to_string()).unwrap_or_default(),
                            fallback_idx = i,
                            "fallback succeeded"
                        );
                    }
                    return Ok(resp);
                }
                Err(e) => {
                    let should_fall = self.fall_through_on_all || is_transient(&e);
                    if i == last_idx || !should_fall {
                        // Last model failed OR error is terminal — propagate.
                        warn!(model_idx = i, error = %e, "FallbackChatModel exhausted or terminal");
                        return Err(e);
                    }
                    debug!(model_idx = i, error = %e, "FallbackChatModel falling through");
                    last_err = Some(e);
                }
            }
        }
        unreachable!("loop returns or sets last_err on every iteration");
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Only first model. See module doc.
        self.inners[0].stream(messages, opts).await
    }
}

/// Parallel to `FallbackChatModel` but for the `Embeddings` trait. Tries
/// providers in order; on transient failure (rate-limit / timeout / 5xx)
/// routes to the next. Last provider's error propagates.
///
/// # Dimension invariant
///
/// ALL inner providers must produce same-dimension vectors. Silently
/// switching from a 1536-dim embedder to a 768-dim one would corrupt
/// your vector index. `new()` validates this at construction and panics
/// (programmer error — catches the config bug at startup, not 10k docs
/// into production). For same-family models (all OpenAI small variants,
/// for example), dimensions match naturally.
///
/// # When to use
///
/// - **Provider failover**: OpenAI embed primary, Voyage backup. When
///   OpenAI has an outage, embedding calls transparently route.
/// - **Cost shedding**: expensive flagship primary, cheap small backup
///   for non-critical batch embed.
/// - **Regional failover**: us-east primary, us-west backup.
///
/// Each variant MUST produce the same dim — OpenAI `text-embedding-3-small`
/// (1536) ↔ Voyage `voyage-3-lite` (512) → don't mix. Use same-family
/// models or pad downstream.
pub struct FallbackEmbeddings {
    pub inners: Vec<Arc<dyn Embeddings>>,
    pub fall_through_on_all: bool,
    /// Cached at construction; all inners agree on this.
    dim: usize,
    name_label: String,
}

impl FallbackEmbeddings {
    /// Build a fallback chain. Panics if `inners` is empty or if any
    /// two inners disagree on `dimensions()`.
    pub fn new(inners: Vec<Arc<dyn Embeddings>>) -> Self {
        assert!(
            !inners.is_empty(),
            "FallbackEmbeddings: chain must have at least one provider"
        );
        let dim = inners[0].dimensions();
        for (i, e) in inners.iter().enumerate().skip(1) {
            assert_eq!(
                e.dimensions(),
                dim,
                "FallbackEmbeddings: inner #{} has dim {} but inner #0 has dim {} — \
                 silent dimension mismatch would corrupt your vector index",
                i,
                e.dimensions(),
                dim
            );
        }
        let labels: Vec<String> = inners.iter().map(|e| e.name().to_string()).collect();
        Self {
            inners,
            fall_through_on_all: false,
            dim,
            name_label: format!("fallback({})", labels.join(", ")),
        }
    }

    /// Fall through on EVERY error, not just transient. Default is
    /// conservative — 4xx errors (malformed input) will fail the same way
    /// on a backup provider, so we fail fast.
    pub fn fall_through_on_all(mut self) -> Self {
        self.fall_through_on_all = true;
        self
    }
}

#[async_trait]
impl Embeddings for FallbackEmbeddings {
    fn name(&self) -> &str {
        &self.name_label
    }

    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let last_idx = self.inners.len() - 1;
        let mut last_err: Option<Error> = None;
        for (i, inner) in self.inners.iter().enumerate() {
            match inner.embed_query(text).await {
                Ok(v) => {
                    if i > 0 {
                        debug!(
                            fallback_idx = i,
                            "FallbackEmbeddings.embed_query recovered on backup"
                        );
                    }
                    return Ok(v);
                }
                Err(e) => {
                    let should_fall = self.fall_through_on_all || is_transient(&e);
                    if i == last_idx || !should_fall {
                        warn!(
                            provider_idx = i,
                            error = %e,
                            "FallbackEmbeddings.embed_query exhausted or terminal"
                        );
                        return Err(e);
                    }
                    debug!(
                        provider_idx = i,
                        error = %e,
                        "FallbackEmbeddings.embed_query falling through"
                    );
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::other("FallbackEmbeddings.embed_query: no result")))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let last_idx = self.inners.len() - 1;
        let mut last_err: Option<Error> = None;
        for (i, inner) in self.inners.iter().enumerate() {
            match inner.embed_documents(texts).await {
                Ok(v) => {
                    if i > 0 {
                        debug!(
                            fallback_idx = i,
                            n_texts = texts.len(),
                            "FallbackEmbeddings.embed_documents recovered on backup"
                        );
                    }
                    return Ok(v);
                }
                Err(e) => {
                    let should_fall = self.fall_through_on_all || is_transient(&e);
                    if i == last_idx || !should_fall {
                        return Err(e);
                    }
                    last_err = Some(e);
                }
            }
        }
        Err(last_err.unwrap_or_else(|| {
            Error::other("FallbackEmbeddings.embed_documents: no result")
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    #[allow(unused_imports)]
    use litgraph_core::tool::Tool as _;
    #[allow(unused_imports)]
    use litgraph_core::{ContentPart, Message, Role};
    #[allow(unused_imports)]
    use std::sync::atomic::{AtomicU32, Ordering};
    #[allow(unused_imports)]
    use litgraph_core::Error;
    #[allow(unused_imports)]
    use std::time::Duration;

    // ---- FallbackChatModel tests ----------------------------------------

    /// Deterministic model that records its name on every call and either
    /// errors or returns success based on the constructor.
    struct CannedModel {
        label: &'static str,
        result: CannedResult,
        called: AtomicU32,
    }

    enum CannedResult {
        Ok,
        RateLimited,
        Provider5xx,
        BadRequest,
    }

    #[async_trait]
    impl ChatModel for CannedModel {
        fn name(&self) -> &str {
            self.label
        }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.called.fetch_add(1, Ordering::SeqCst);
            match self.result {
                CannedResult::Ok => Ok(ChatResponse {
                    message: Message::assistant(self.label.to_string()),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: self.label.to_string(),
                }),
                CannedResult::RateLimited => Err(Error::RateLimited { retry_after_ms: None }),
                CannedResult::Provider5xx => Err(Error::provider("503 service unavailable")),
                CannedResult::BadRequest => Err(Error::invalid("malformed prompt")),
            }
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn canned(label: &'static str, result: CannedResult) -> Arc<CannedModel> {
        Arc::new(CannedModel {
            label,
            result,
            called: AtomicU32::new(0),
        })
    }

    #[tokio::test]
    async fn fallback_uses_primary_when_it_succeeds() {
        let primary = canned("primary", CannedResult::Ok);
        let backup = canned("backup", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
        ]);
        let resp = chain.invoke(vec![], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "primary");
        assert_eq!(primary.called.load(Ordering::SeqCst), 1);
        assert_eq!(backup.called.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fallback_falls_through_on_rate_limit() {
        let primary = canned("primary", CannedResult::RateLimited);
        let backup = canned("backup", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
        ]);
        let resp = chain.invoke(vec![], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "backup");
        assert_eq!(primary.called.load(Ordering::SeqCst), 1);
        assert_eq!(backup.called.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_falls_through_on_5xx() {
        let primary = canned("primary", CannedResult::Provider5xx);
        let backup = canned("backup", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
        ]);
        let resp = chain.invoke(vec![], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "backup");
    }

    #[tokio::test]
    async fn fallback_propagates_terminal_error_by_default() {
        // Bad request — same prompt would fail on backup too. Default is
        // fail-fast (don't waste tokens trying provider B).
        let primary = canned("primary", CannedResult::BadRequest);
        let backup = canned("backup", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
        ]);
        let err = chain.invoke(vec![], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert_eq!(primary.called.load(Ordering::SeqCst), 1);
        assert_eq!(backup.called.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fallback_with_fall_through_on_all_tries_backup_on_terminal_too() {
        let primary = canned("primary", CannedResult::BadRequest);
        let backup = canned("backup", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            primary.clone() as Arc<dyn ChatModel>,
            backup.clone() as Arc<dyn ChatModel>,
        ])
        .fall_through_on_all();
        let resp = chain.invoke(vec![], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "backup");
        assert_eq!(primary.called.load(Ordering::SeqCst), 1);
        assert_eq!(backup.called.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_propagates_last_error_when_all_models_fail() {
        let p = canned("p", CannedResult::RateLimited);
        let b1 = canned("b1", CannedResult::Provider5xx);
        let b2 = canned("b2", CannedResult::RateLimited);
        let chain = FallbackChatModel::new(vec![
            p.clone() as Arc<dyn ChatModel>,
            b1.clone() as Arc<dyn ChatModel>,
            b2.clone() as Arc<dyn ChatModel>,
        ]);
        let err = chain.invoke(vec![], &ChatOptions::default()).await.unwrap_err();
        // Last error (b2's RateLimited) is what surfaces.
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(p.called.load(Ordering::SeqCst), 1);
        assert_eq!(b1.called.load(Ordering::SeqCst), 1);
        assert_eq!(b2.called.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_walks_chain_until_one_succeeds() {
        let p = canned("p", CannedResult::RateLimited);
        let b1 = canned("b1", CannedResult::Provider5xx);
        let b2 = canned("b2", CannedResult::Ok);
        let chain = FallbackChatModel::new(vec![
            p.clone() as Arc<dyn ChatModel>,
            b1.clone() as Arc<dyn ChatModel>,
            b2.clone() as Arc<dyn ChatModel>,
        ]);
        let resp = chain.invoke(vec![], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "b2");
        assert_eq!(p.called.load(Ordering::SeqCst), 1);
        assert_eq!(b1.called.load(Ordering::SeqCst), 1);
        assert_eq!(b2.called.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    #[should_panic(expected = "chain must have at least one model")]
    async fn fallback_panics_on_empty_chain() {
        let _ = FallbackChatModel::new(vec![]);
    }


    // ---- FallbackEmbeddings tests --------------------------------------

    struct CannedEmbed {
        label: &'static str,
        dim: usize,
        result: CannedEmbedResult,
        call_count: AtomicU32,
    }

    #[derive(Clone)]
    enum CannedEmbedResult {
        Ok,
        RateLimited,
        Provider5xx,
        BadRequest,
    }

    #[async_trait]
    impl Embeddings for CannedEmbed {
        fn name(&self) -> &str {
            self.label
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            match self.result {
                CannedEmbedResult::Ok => Ok(vec![0.1; self.dim]),
                CannedEmbedResult::RateLimited => {
                    Err(Error::RateLimited { retry_after_ms: None })
                }
                CannedEmbedResult::Provider5xx => {
                    Err(Error::provider("503 service unavailable"))
                }
                CannedEmbedResult::BadRequest => Err(Error::invalid("malformed input")),
            }
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            match self.result {
                CannedEmbedResult::Ok => Ok(vec![vec![0.1; self.dim]; texts.len()]),
                CannedEmbedResult::RateLimited => {
                    Err(Error::RateLimited { retry_after_ms: None })
                }
                CannedEmbedResult::Provider5xx => {
                    Err(Error::provider("502 bad gateway"))
                }
                CannedEmbedResult::BadRequest => Err(Error::invalid("malformed batch")),
            }
        }
    }

    fn embed(label: &'static str, dim: usize, r: CannedEmbedResult) -> Arc<CannedEmbed> {
        Arc::new(CannedEmbed {
            label,
            dim,
            result: r,
            call_count: AtomicU32::new(0),
        })
    }

    #[tokio::test]
    async fn fallback_embed_primary_succeeds_no_backup_called() {
        let primary = embed("primary", 1536, CannedEmbedResult::Ok);
        let backup = embed("backup", 1536, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
        ]);
        let v = chain.embed_query("hi").await.unwrap();
        assert_eq!(v.len(), 1536);
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(backup.call_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fallback_embed_rate_limit_falls_through() {
        let primary = embed("primary", 512, CannedEmbedResult::RateLimited);
        let backup = embed("backup", 512, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
        ]);
        let v = chain.embed_query("hi").await.unwrap();
        assert_eq!(v.len(), 512);
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(backup.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_embed_5xx_falls_through() {
        let primary = embed("p", 768, CannedEmbedResult::Provider5xx);
        let backup = embed("b", 768, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup as Arc<dyn Embeddings>,
        ]);
        chain.embed_query("hi").await.unwrap();
    }

    #[tokio::test]
    async fn fallback_embed_terminal_error_propagates_by_default() {
        // Bad request would fail on backup too → fail-fast by default.
        let primary = embed("p", 1024, CannedEmbedResult::BadRequest);
        let backup = embed("b", 1024, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
        ]);
        let err = chain.embed_query("hi").await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        assert_eq!(backup.call_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn fallback_embed_fall_through_on_all_bypasses_terminal_check() {
        let primary = embed("p", 1024, CannedEmbedResult::BadRequest);
        let backup = embed("b", 1024, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
        ])
        .fall_through_on_all();
        chain.embed_query("hi").await.unwrap();
        assert_eq!(backup.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_embed_exhausted_surfaces_last_error() {
        let p1 = embed("p1", 1536, CannedEmbedResult::RateLimited);
        let p2 = embed("p2", 1536, CannedEmbedResult::Provider5xx);
        let p3 = embed("p3", 1536, CannedEmbedResult::RateLimited);
        let chain = FallbackEmbeddings::new(vec![
            p1.clone() as Arc<dyn Embeddings>,
            p2.clone() as Arc<dyn Embeddings>,
            p3.clone() as Arc<dyn Embeddings>,
        ]);
        let err = chain.embed_query("hi").await.unwrap_err();
        // Last-error wins (p3's RateLimited).
        assert!(matches!(err, Error::RateLimited { .. }));
        assert_eq!(p1.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(p2.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(p3.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_embed_documents_path_also_falls_through() {
        let primary = embed("p", 256, CannedEmbedResult::RateLimited);
        let backup = embed("b", 256, CannedEmbedResult::Ok);
        let chain = FallbackEmbeddings::new(vec![
            primary.clone() as Arc<dyn Embeddings>,
            backup.clone() as Arc<dyn Embeddings>,
        ]);
        let vecs = chain
            .embed_documents(&["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(vecs.len(), 3);
        assert_eq!(vecs[0].len(), 256);
        assert_eq!(primary.call_count.load(Ordering::SeqCst), 1);
        assert_eq!(backup.call_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn fallback_embed_exposes_shared_dimensions() {
        let chain = FallbackEmbeddings::new(vec![
            embed("p", 1024, CannedEmbedResult::Ok) as Arc<dyn Embeddings>,
            embed("b", 1024, CannedEmbedResult::Ok) as Arc<dyn Embeddings>,
        ]);
        assert_eq!(chain.dimensions(), 1024);
        assert!(chain.name().contains("p"));
        assert!(chain.name().contains("b"));
    }

    #[tokio::test]
    #[should_panic(expected = "must have at least one provider")]
    async fn fallback_embed_empty_chain_panics() {
        let _ = FallbackEmbeddings::new(vec![]);
    }

    #[tokio::test]
    #[should_panic(expected = "silent dimension mismatch")]
    async fn fallback_embed_dim_mismatch_panics_at_construction() {
        // 1536 vs 768 → would silently corrupt a vector index. Refuse.
        let _ = FallbackEmbeddings::new(vec![
            embed("p", 1536, CannedEmbedResult::Ok) as Arc<dyn Embeddings>,
            embed("b", 768, CannedEmbedResult::Ok) as Arc<dyn Embeddings>,
        ]);
    }

}

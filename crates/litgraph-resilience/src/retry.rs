use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result};
use tracing::{debug, warn};

#[derive(Debug, Clone)]
pub struct RetryConfig {
    pub min_delay: Duration,
    pub max_delay: Duration,
    pub factor: f32,
    pub max_times: usize,
    /// If true, jitter the delay (recommended in production to avoid thundering herds).
    pub jitter: bool,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            min_delay: Duration::from_millis(200),
            max_delay: Duration::from_secs(30),
            factor: 2.0,
            max_times: 5,
            jitter: true,
        }
    }
}

impl RetryConfig {
    fn to_builder(&self) -> ExponentialBuilder {
        let mut b = ExponentialBuilder::default()
            .with_min_delay(self.min_delay)
            .with_max_delay(self.max_delay)
            .with_factor(self.factor)
            .with_max_times(self.max_times);
        if self.jitter { b = b.with_jitter(); }
        b
    }
}

/// Classify an `Error` as transient (retry) vs terminal (give up).
pub(crate) fn is_transient(e: &Error) -> bool {
    match e {
        Error::RateLimited { .. } => true,
        Error::Timeout => true,
        Error::Provider(msg) => {
            // Match common 5xx / connection-reset patterns. Conservative: only
            // retry when we're confident the upstream might be at fault.
            let m = msg.to_ascii_lowercase();
            m.contains("500 ")
                || m.contains("502 ")
                || m.contains("503 ")
                || m.contains("504 ")
                || m.contains("connection reset")
                || m.contains("connection closed")
                || m.contains("connect error")
                || m.contains("send: ")  // reqwest send failure pre-status
        }
        _ => false,
    }
}

pub struct RetryingChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub cfg: RetryConfig,
}

impl RetryingChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, cfg: RetryConfig) -> Self {
        Self { inner, cfg }
    }
}

#[async_trait]
impl ChatModel for RetryingChatModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        let inner = self.inner.clone();
        let backoff = self.cfg.to_builder();
        let messages = messages;
        let result = (|| {
            let inner = inner.clone();
            let messages = messages.clone();
            let opts = opts.clone();
            async move { inner.invoke(messages, &opts).await }
        })
        .retry(&backoff)
        .when(|e: &Error| {
            let retry = is_transient(e);
            if retry {
                debug!(error = %e, "retrying transient error");
            } else {
                warn!(error = %e, "terminal error — not retrying");
            }
            retry
        })
        .await;
        result
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        // Don't retry streams. See module doc.
        self.inner.stream(messages, opts).await
    }
}

/// Retry wrapper for `Embeddings`. Same retry semantics as
/// `RetryingChatModel` — exponential backoff on transient failures
/// (rate-limit / timeout / 5xx), terminal errors (4xx, parse) propagate.
///
/// Applies to BOTH `embed_query` and `embed_documents`. `embed_documents`
/// retries the whole batch on failure — do NOT retry per-element since
/// that masks provider-side partial failures. If you need per-element
/// resilience, chunk before calling.
pub struct RetryingEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub cfg: RetryConfig,
}

impl RetryingEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, cfg: RetryConfig) -> Self {
        Self { inner, cfg }
    }
}

#[async_trait]
impl Embeddings for RetryingEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let inner = self.inner.clone();
        let backoff = self.cfg.to_builder();
        let text = text.to_string();
        (|| {
            let inner = inner.clone();
            let text = text.clone();
            async move { inner.embed_query(&text).await }
        })
        .retry(&backoff)
        .when(|e: &Error| {
            let retry = is_transient(e);
            if retry {
                debug!(error = %e, "RetryingEmbeddings.embed_query retry");
            } else {
                warn!(error = %e, "RetryingEmbeddings.embed_query terminal");
            }
            retry
        })
        .await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        let backoff = self.cfg.to_builder();
        let texts = texts.to_vec();
        (|| {
            let inner = inner.clone();
            let texts = texts.clone();
            async move { inner.embed_documents(&texts).await }
        })
        .retry(&backoff)
        .when(|e: &Error| is_transient(e))
        .await
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

    /// Errors first N calls, then succeeds.
    struct FlakyModel {
        fails_remaining: AtomicU32,
        kind: ErrKind,
    }

    enum ErrKind {
        RateLimited,
        Provider5xx,
        BadRequest,
    }

    #[async_trait]
    impl ChatModel for FlakyModel {
        fn name(&self) -> &str { "flaky" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    ErrKind::RateLimited => Error::RateLimited { retry_after_ms: None },
                    ErrKind::Provider5xx => Error::provider("502 bad gateway"),
                    ErrKind::BadRequest  => Error::invalid("bad request"),
                });
            }
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "ok".into() }],
                    tool_calls: vec![], tool_call_id: None, name: None, cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "flaky".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn quick_cfg() -> RetryConfig {
        RetryConfig { min_delay: Duration::from_millis(1), max_delay: Duration::from_millis(10),
                      factor: 2.0, max_times: 5, jitter: false }
    }

    #[tokio::test]
    async fn retries_rate_limited_then_succeeds() {
        let inner: Arc<dyn ChatModel> = Arc::new(FlakyModel {
            fails_remaining: AtomicU32::new(2), kind: ErrKind::RateLimited,
        });
        let r = RetryingChatModel::new(inner, quick_cfg());
        let resp = r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "ok");
    }

    #[tokio::test]
    async fn retries_5xx_then_succeeds() {
        let inner: Arc<dyn ChatModel> = Arc::new(FlakyModel {
            fails_remaining: AtomicU32::new(3), kind: ErrKind::Provider5xx,
        });
        let r = RetryingChatModel::new(inner, quick_cfg());
        let resp = r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "ok");
    }

    #[tokio::test]
    async fn does_not_retry_bad_request() {
        let inner: Arc<dyn ChatModel> = Arc::new(FlakyModel {
            fails_remaining: AtomicU32::new(10), kind: ErrKind::BadRequest,
        });
        let r = RetryingChatModel::new(inner, quick_cfg());
        let err = r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn gives_up_after_max_attempts() {
        let inner: Arc<dyn ChatModel> = Arc::new(FlakyModel {
            fails_remaining: AtomicU32::new(100), kind: ErrKind::RateLimited,
        });
        let r = RetryingChatModel::new(inner, quick_cfg());
        let err = r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }));
    }


    // ---- RetryingEmbeddings tests --------------------------------------

    /// Embed provider that fails the first N calls (transient) then succeeds.
    struct FlakyEmbed {
        fails_remaining: AtomicU32,
        kind: EmbedFlakyKind,
        dim: usize,
        total_calls: AtomicU32,
    }

    enum EmbedFlakyKind {
        RateLimited,
        Provider5xx,
        BadRequest,
    }

    #[async_trait]
    impl Embeddings for FlakyEmbed {
        fn name(&self) -> &str {
            "flaky-embed"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    EmbedFlakyKind::RateLimited => {
                        Error::RateLimited { retry_after_ms: None }
                    }
                    EmbedFlakyKind::Provider5xx => Error::provider("502 bad gateway"),
                    EmbedFlakyKind::BadRequest => Error::invalid("bad request"),
                });
            }
            Ok(vec![0.1; self.dim])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.total_calls.fetch_add(1, Ordering::SeqCst);
            let n = self.fails_remaining.load(Ordering::SeqCst);
            if n > 0 {
                self.fails_remaining.fetch_sub(1, Ordering::SeqCst);
                return Err(match self.kind {
                    EmbedFlakyKind::RateLimited => {
                        Error::RateLimited { retry_after_ms: None }
                    }
                    EmbedFlakyKind::Provider5xx => Error::provider("503 service unavailable"),
                    EmbedFlakyKind::BadRequest => Error::invalid("malformed batch"),
                });
            }
            Ok(vec![vec![0.1; self.dim]; texts.len()])
        }
    }

    fn flaky_embed(fails: u32, kind: EmbedFlakyKind, dim: usize) -> Arc<FlakyEmbed> {
        Arc::new(FlakyEmbed {
            fails_remaining: AtomicU32::new(fails),
            kind,
            dim,
            total_calls: AtomicU32::new(0),
        })
    }

    fn fast_retry() -> RetryConfig {
        RetryConfig {
            min_delay: Duration::from_millis(1),
            max_delay: Duration::from_millis(5),
            factor: 2.0,
            max_times: 5,
            jitter: false,
        }
    }

    #[tokio::test]
    async fn retry_embed_recovers_from_rate_limit() {
        let flaky = flaky_embed(2, EmbedFlakyKind::RateLimited, 512);
        let r = RetryingEmbeddings::new(flaky.clone() as Arc<dyn Embeddings>, fast_retry());
        let v = r.embed_query("hi").await.unwrap();
        assert_eq!(v.len(), 512);
        assert_eq!(flaky.total_calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_embed_recovers_from_5xx() {
        let flaky = flaky_embed(1, EmbedFlakyKind::Provider5xx, 256);
        let r = RetryingEmbeddings::new(flaky.clone() as Arc<dyn Embeddings>, fast_retry());
        r.embed_query("hi").await.unwrap();
        assert_eq!(flaky.total_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_embed_gives_up_after_max_attempts() {
        let flaky = flaky_embed(99, EmbedFlakyKind::RateLimited, 128);
        let r = RetryingEmbeddings::new(
            flaky.clone() as Arc<dyn Embeddings>,
            RetryConfig {
                min_delay: Duration::from_millis(1),
                max_delay: Duration::from_millis(2),
                factor: 1.5,
                max_times: 2,
                jitter: false,
            },
        );
        let err = r.embed_query("hi").await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }));
        // initial + 2 retries = 3 attempts total
        assert_eq!(flaky.total_calls.load(Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn retry_embed_terminal_bad_request_does_not_retry() {
        let flaky = flaky_embed(99, EmbedFlakyKind::BadRequest, 512);
        let r = RetryingEmbeddings::new(flaky.clone() as Arc<dyn Embeddings>, fast_retry());
        let err = r.embed_query("hi").await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        // 1 attempt only — no retries on terminal error
        assert_eq!(flaky.total_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn retry_embed_documents_path_also_retries() {
        let flaky = flaky_embed(1, EmbedFlakyKind::Provider5xx, 256);
        let r = RetryingEmbeddings::new(flaky.clone() as Arc<dyn Embeddings>, fast_retry());
        let v = r
            .embed_documents(&["a".into(), "b".into(), "c".into()])
            .await
            .unwrap();
        assert_eq!(v.len(), 3);
        assert_eq!(v[0].len(), 256);
        assert_eq!(flaky.total_calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn retry_embed_exposes_inner_dim_and_name() {
        let flaky = flaky_embed(0, EmbedFlakyKind::RateLimited, 1024);
        let r = RetryingEmbeddings::new(flaky as Arc<dyn Embeddings>, fast_retry());
        assert_eq!(r.dimensions(), 1024);
        assert_eq!(r.name(), "flaky-embed");
    }

}

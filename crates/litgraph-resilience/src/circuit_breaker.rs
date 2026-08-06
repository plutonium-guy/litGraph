use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, CircuitBreaker, CircuitCallError, Embeddings, Error,
    Message, Result,
};

/// Wrap any [`ChatModel`] with a [`CircuitBreaker`] so persistent
/// upstream failures stop bleeding load against a sick service.
///
/// Distinct from [`RetryingChatModel`]: that retries individual
/// transient errors (right when failures are momentary). When an
/// upstream is *down*, retrying just hammers it harder. The
/// breaker stops calls cold for a cooldown window, gives the
/// upstream room to heal, and emits an `Error::Provider("circuit
/// breaker open")` from waiting callers so they can fall through
/// (e.g. via [`FallbackChatModel`]) immediately.
///
/// # Composition
///
/// Stacks naturally with the existing wrapper toolkit. A common
/// prod chain — outer to inner:
///
/// 1. `CircuitBreakerChatModel` — fail-fast on persistent outage
/// 2. `FallbackChatModel` — switch provider on circuit-open
/// 3. `RetryingChatModel` — retry the *primary* on transient errors
/// 4. `RateLimitedChatModel` — local rate cap
/// 5. real provider
///
/// # Streaming
///
/// `stream()` is wrapped at the **handshake**: if the breaker is
/// open, returns `Error::Provider("circuit breaker open")` without
/// invoking the inner stream. If the inner `stream()` returns Ok,
/// the breaker records success and the caller drives the stream
/// normally — mid-stream failures are not visible to the breaker
/// (consumer's responsibility). This matches the breaker's
/// admission-control semantics.
pub struct CircuitBreakerChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub breaker: Arc<CircuitBreaker>,
}

impl CircuitBreakerChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }
}

#[async_trait]
impl ChatModel for CircuitBreakerChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let inner = self.inner.clone();
        let opts = opts.clone();
        let r = self
            .breaker
            .call(move || async move { inner.invoke(messages, &opts).await })
            .await;
        match r {
            Ok(resp) => Ok(resp),
            Err(CircuitCallError::CircuitOpen) => {
                Err(Error::Provider("circuit breaker open".into()))
            }
            Err(CircuitCallError::Inner(e)) => Err(e),
        }
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        let inner = self.inner.clone();
        let opts = opts.clone();
        let r = self
            .breaker
            .call(move || async move { inner.stream(messages, &opts).await })
            .await;
        match r {
            Ok(s) => Ok(s),
            Err(CircuitCallError::CircuitOpen) => {
                Err(Error::Provider("circuit breaker open".into()))
            }
            Err(CircuitCallError::Inner(e)) => Err(e),
        }
    }
}

/// Wrap any [`Embeddings`] with a [`CircuitBreaker`] so persistent
/// upstream failures stop bleeding load against a sick service.
/// Embed-axis mirror of [`CircuitBreakerChatModel`].
///
/// Same composition story: stack with [`FallbackEmbeddings`] so
/// circuit-open routes immediately to the secondary embedder.
/// Both `embed_query` and `embed_documents` admission-gate
/// through the breaker; one shared breaker covers both call
/// shapes so a flapping query path also opens the breaker for
/// document indexing (and vice-versa).
pub struct CircuitBreakerEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub breaker: Arc<CircuitBreaker>,
}

impl CircuitBreakerEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, breaker: Arc<CircuitBreaker>) -> Self {
        Self { inner, breaker }
    }
}

#[async_trait]
impl Embeddings for CircuitBreakerEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let inner = self.inner.clone();
        let text = text.to_owned();
        let r = self
            .breaker
            .call(move || async move { inner.embed_query(&text).await })
            .await;
        match r {
            Ok(v) => Ok(v),
            Err(CircuitCallError::CircuitOpen) => {
                Err(Error::Provider("circuit breaker open".into()))
            }
            Err(CircuitCallError::Inner(e)) => Err(e),
        }
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let inner = self.inner.clone();
        let texts = texts.to_vec();
        let r = self
            .breaker
            .call(move || async move { inner.embed_documents(&texts).await })
            .await;
        match r {
            Ok(v) => Ok(v),
            Err(CircuitCallError::CircuitOpen) => {
                Err(Error::Provider("circuit breaker open".into()))
            }
            Err(CircuitCallError::Inner(e)) => Err(e),
        }
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

    /// Embed provider that fails the first N calls (transient) then succeeds.
    struct FlakyEmbed {
        fails_remaining: AtomicU32,
        kind: EmbedFlakyKind,
        dim: usize,
        total_calls: AtomicU32,
    }

    #[allow(dead_code)]
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


    // ---- CircuitBreakerChatModel tests ----------------------------------

    /// Always errs with provider("502 ...") so the breaker counts every call
    /// as a failure. Counts invocations so we can verify the breaker
    /// short-circuits without invoking inner.
    struct AlwaysFailModel {
        seen: AtomicU32,
    }

    #[async_trait]
    impl ChatModel for AlwaysFailModel {
        fn name(&self) -> &str {
            "always-fail"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Err(Error::provider("502 sick upstream"))
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            self.seen.fetch_add(1, Ordering::SeqCst);
            Err(Error::provider("502 sick upstream"))
        }
    }

    #[tokio::test]
    async fn circuit_breaker_passes_through_inner_errors_until_threshold() {
        let inner = Arc::new(AlwaysFailModel {
            seen: AtomicU32::new(0),
        });
        let breaker = Arc::new(CircuitBreaker::new(3, Duration::from_secs(60)));
        let cb = CircuitBreakerChatModel::new(
            inner.clone() as Arc<dyn ChatModel>,
            breaker.clone(),
        );
        // Two failures: should be passed-through provider errors.
        for _ in 0..2 {
            let err = cb
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Provider(ref m) if m.contains("502")),
                "expected pass-through, got {err:?}",
            );
        }
        assert_eq!(inner.seen.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn circuit_breaker_short_circuits_after_threshold() {
        let inner = Arc::new(AlwaysFailModel {
            seen: AtomicU32::new(0),
        });
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerChatModel::new(
            inner.clone() as Arc<dyn ChatModel>,
            breaker.clone(),
        );
        // Trip the breaker.
        for _ in 0..2 {
            let _ = cb
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await;
        }
        // Subsequent calls fail-fast WITHOUT invoking inner.
        for _ in 0..3 {
            let err = cb
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap_err();
            assert!(
                matches!(err, Error::Provider(ref m) if m.contains("circuit breaker open")),
                "expected circuit-open error, got {err:?}",
            );
        }
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            2,
            "inner was invoked while breaker was open",
        );
    }

    #[tokio::test]
    async fn circuit_breaker_streams_also_short_circuit() {
        let inner = Arc::new(AlwaysFailModel {
            seen: AtomicU32::new(0),
        });
        let breaker = Arc::new(CircuitBreaker::new(1, Duration::from_secs(60)));
        let cb = CircuitBreakerChatModel::new(
            inner.clone() as Arc<dyn ChatModel>,
            breaker.clone(),
        );
        // First stream() error trips it.
        let _ = cb
            .stream(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        let baseline = inner.seen.load(Ordering::SeqCst);
        // Now stream() must fail-fast.
        let r = cb
            .stream(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        match r {
            Ok(_) => panic!("stream should have failed-fast"),
            Err(Error::Provider(m)) => {
                assert!(m.contains("circuit breaker open"), "got: {m}")
            }
            Err(other) => panic!("expected circuit-open error, got {other:?}"),
        }
        assert_eq!(
            inner.seen.load(Ordering::SeqCst),
            baseline,
            "stream invoked inner despite open breaker",
        );
    }

    /// Always-succeeds model for closed-path test.
    struct AlwaysOkModel;

    #[async_trait]
    impl ChatModel for AlwaysOkModel {
        fn name(&self) -> &str {
            "always-ok"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "hi".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "always-ok".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn circuit_breaker_closed_passes_successes_through() {
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerChatModel::new(inner, breaker);
        for _ in 0..5 {
            let resp = cb
                .invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap();
            assert_eq!(resp.message.text_content(), "hi");
        }
    }

    #[tokio::test]
    async fn circuit_breaker_name_proxies_inner() {
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerChatModel::new(inner, breaker);
        assert_eq!(cb.name(), "always-ok");
    }

    // ---- CircuitBreakerEmbeddings tests ---------------------------------

    #[tokio::test]
    async fn circuit_breaker_embeddings_short_circuits_after_threshold() {
        // u32::MAX fails => effectively always-fail; lets us observe
        // that inner is NOT invoked while the breaker is open.
        let inner = flaky_embed(u32::MAX, EmbedFlakyKind::Provider5xx, 4);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerEmbeddings::new(
            inner.clone() as Arc<dyn Embeddings>,
            breaker,
        );
        // Two failures pass through and trip the breaker.
        for _ in 0..2 {
            let err = cb.embed_query("hi").await.unwrap_err();
            assert!(matches!(err, Error::Provider(ref m) if m.contains("502")));
        }
        let baseline = inner.total_calls.load(Ordering::SeqCst);
        // Subsequent calls fail-fast WITHOUT invoking inner.
        for _ in 0..3 {
            let err = cb.embed_query("hi").await.unwrap_err();
            assert!(matches!(err, Error::Provider(ref m) if m.contains("circuit breaker open")));
        }
        // embed_documents path also short-circuits via the SAME breaker.
        let err = cb
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Provider(ref m) if m.contains("circuit breaker open")));
        assert_eq!(
            inner.total_calls.load(Ordering::SeqCst),
            baseline,
            "inner was invoked while breaker was open",
        );
    }

    #[tokio::test]
    async fn circuit_breaker_embeddings_closed_passes_successes() {
        // Zero fails => always succeeds; verifies the closed path.
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 3);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerEmbeddings::new(
            inner as Arc<dyn Embeddings>,
            breaker,
        );
        let v = cb.embed_query("hi").await.unwrap();
        assert_eq!(v.len(), 3);
        let docs = cb
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].len(), 3);
    }

    #[tokio::test]
    async fn circuit_breaker_embeddings_query_failures_open_breaker_for_documents() {
        // One shared breaker spans both call shapes: a flapping
        // embed_query path opens the breaker so a subsequent
        // embed_documents call also fails fast.
        let inner = flaky_embed(u32::MAX, EmbedFlakyKind::Provider5xx, 2);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerEmbeddings::new(
            inner.clone() as Arc<dyn Embeddings>,
            breaker,
        );
        for _ in 0..2 {
            let _ = cb.embed_query("hi").await;
        }
        let baseline = inner.total_calls.load(Ordering::SeqCst);
        let err = cb
            .embed_documents(&["a".into()])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Provider(ref m) if m.contains("circuit breaker open")));
        assert_eq!(
            inner.total_calls.load(Ordering::SeqCst),
            baseline,
            "embed_documents invoked inner despite the query path opening the breaker",
        );
    }

    #[tokio::test]
    async fn circuit_breaker_embeddings_dimensions_proxy_inner() {
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 7);
        let breaker = Arc::new(CircuitBreaker::new(2, Duration::from_secs(60)));
        let cb = CircuitBreakerEmbeddings::new(
            inner as Arc<dyn Embeddings>,
            breaker,
        );
        assert_eq!(cb.dimensions(), 7);
        assert_eq!(cb.name(), "flaky-embed");
    }

}

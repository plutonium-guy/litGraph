use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result};

/// Wrap any `ChatModel` so each `invoke` call must complete within
/// `timeout`. On timeout, returns `Error::Timeout`; the inner future
/// is cancelled (`tokio::time::timeout` drops it, releasing whatever
/// HTTP connection / parse state it held).
///
/// # Why this exists
///
/// Mirrors `litgraph_tools_utils::TimeoutTool` but for the chat-model
/// side. Real production patterns:
///
/// - **SLA enforcement**: an interactive UI can't wait 60s for a
///   pathological model response — wrap the call with a 10s deadline
///   and let `RetryingChatModel` (with backoff) or `FallbackChatModel`
///   pick a different provider.
/// - **Circuit-breaker preconditions**: pair with a counter that
///   trips when N timeouts hit in a window.
/// - **Budgeted batches**: prevent a single slow request from
///   dragging out a `batch_concurrent` (iter 182) call.
///
/// # Parallelism
///
/// `tokio::time::timeout` runs the inner future and a timer future
/// **concurrently** under a `select!`-style poller. First to complete
/// wins. Distinct from iter 184's `RaceChatModel` (which races N
/// inner futures via `JoinSet`+`abort_all`); here the "competitor"
/// is a deadline timer, not another provider.
///
/// # Streaming
///
/// `stream()` is **not** wrapped — once chunks start arriving the
/// per-call deadline is meaningless (you'd need a per-chunk deadline
/// instead). Stream consumers should run their own per-chunk
/// `tokio::time::timeout` if they need that.
pub struct TimeoutChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub timeout: Duration,
}

impl TimeoutChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

#[async_trait]
impl ChatModel for TimeoutChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        match tokio::time::timeout(self.timeout, self.inner.invoke(messages, opts)).await {
            Ok(r) => r,
            Err(_) => Err(Error::Timeout),
        }
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Per-call timeout doesn't compose cleanly with token streams;
        // pass through. See module doc.
        self.inner.stream(messages, opts).await
    }
}

/// Wrap any `Embeddings` so each `embed_query` / `embed_documents`
/// must complete within `timeout`. On timeout, returns
/// `Error::Timeout`; the inner future is cancelled.
///
/// Embeddings analogue of `TimeoutChatModel`. Real prod use cases:
///
/// - **Tail-latency cap on the embed-query critical path** — every
///   retrieval blocks on this call; if a slow provider stalls,
///   trip and let `RetryingEmbeddings` / `FallbackEmbeddings`
///   pick up.
/// - **Bounded `embed_documents_concurrent`** — wrap the inner
///   provider so a single hung chunk doesn't drag out a 10k-text
///   ingestion.
///
/// `tokio::time::timeout` runs the inner + a timer concurrently —
/// same primitive as `TimeoutChatModel`, just over the embeddings
/// trait.
pub struct TimeoutEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub timeout: Duration,
}

impl TimeoutEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, timeout: Duration) -> Self {
        Self { inner, timeout }
    }
}

#[async_trait]
impl Embeddings for TimeoutEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        match tokio::time::timeout(self.timeout, self.inner.embed_query(text)).await {
            Ok(r) => r,
            Err(_) => Err(Error::Timeout),
        }
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        match tokio::time::timeout(self.timeout, self.inner.embed_documents(texts)).await {
            Ok(r) => r,
            Err(_) => Err(Error::Timeout),
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

    // ---- TimeoutChatModel tests ----------------------------------------

    /// Sleeps `delay_ms` then returns a fixed response.
    struct SlowChat {
        delay_ms: u64,
    }

    #[async_trait]
    impl ChatModel for SlowChat {
        fn name(&self) -> &str {
            "slow-chat"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "ok".into() }],
                    tool_calls: vec![],
                    tool_call_id: None,
                    name: None,
                    cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "slow-chat".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn timeout_chat_passes_through_when_under_deadline() {
        let inner: Arc<dyn ChatModel> = Arc::new(SlowChat { delay_ms: 5 });
        let t = TimeoutChatModel::new(inner, Duration::from_millis(100));
        let r = t
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(r.message.text_content(), "ok");
    }

    #[tokio::test]
    async fn timeout_chat_returns_timeout_error_when_inner_too_slow() {
        let inner: Arc<dyn ChatModel> = Arc::new(SlowChat { delay_ms: 200 });
        let t = TimeoutChatModel::new(inner, Duration::from_millis(20));
        let err = t
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout), "got: {err:?}");
    }

    #[tokio::test]
    async fn timeout_chat_aborts_inner_when_it_times_out() {
        // Wall-clock for the timeout call should be ~20ms even though
        // the inner sleeps 500ms — `tokio::time::timeout` drops the
        // inner future on timeout, releasing the sleep.
        let inner: Arc<dyn ChatModel> = Arc::new(SlowChat { delay_ms: 500 });
        let t = TimeoutChatModel::new(inner, Duration::from_millis(20));
        let started = std::time::Instant::now();
        let _ = t
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        let elapsed = started.elapsed().as_millis() as u64;
        assert!(
            elapsed < 200,
            "elapsed {elapsed}ms — inner future was not dropped on timeout",
        );
    }

    #[tokio::test]
    async fn timeout_chat_preserves_inner_name() {
        let inner: Arc<dyn ChatModel> = Arc::new(SlowChat { delay_ms: 0 });
        let t = TimeoutChatModel::new(inner, Duration::from_millis(100));
        assert_eq!(t.name(), "slow-chat");
    }

    // ---- TimeoutEmbeddings tests ---------------------------------------

    struct SlowEmbed {
        delay_ms: u64,
        dim: usize,
    }

    #[async_trait]
    impl Embeddings for SlowEmbed {
        fn name(&self) -> &str {
            "slow-embed"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(vec![0.1; self.dim])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            Ok(vec![vec![0.1; self.dim]; texts.len()])
        }
    }

    #[tokio::test]
    async fn timeout_embed_query_under_deadline_passes_through() {
        let inner: Arc<dyn Embeddings> = Arc::new(SlowEmbed { delay_ms: 5, dim: 4 });
        let t = TimeoutEmbeddings::new(inner, Duration::from_millis(100));
        let v = t.embed_query("hi").await.unwrap();
        assert_eq!(v.len(), 4);
    }

    #[tokio::test]
    async fn timeout_embed_query_over_deadline_errors() {
        let inner: Arc<dyn Embeddings> = Arc::new(SlowEmbed { delay_ms: 200, dim: 4 });
        let t = TimeoutEmbeddings::new(inner, Duration::from_millis(20));
        let err = t.embed_query("hi").await.unwrap_err();
        assert!(matches!(err, Error::Timeout), "got: {err:?}");
    }

    #[tokio::test]
    async fn timeout_embed_documents_over_deadline_errors() {
        let inner: Arc<dyn Embeddings> = Arc::new(SlowEmbed { delay_ms: 200, dim: 4 });
        let t = TimeoutEmbeddings::new(inner, Duration::from_millis(20));
        let err = t
            .embed_documents(&["a".into(), "b".into()])
            .await
            .unwrap_err();
        assert!(matches!(err, Error::Timeout));
    }

    #[tokio::test]
    async fn timeout_embed_preserves_dimensions() {
        let inner: Arc<dyn Embeddings> = Arc::new(SlowEmbed { delay_ms: 0, dim: 7 });
        let t = TimeoutEmbeddings::new(inner, Duration::from_millis(100));
        assert_eq!(t.dimensions(), 7);
    }

}

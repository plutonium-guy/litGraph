use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{
    Bulkhead, ChatModel, ChatOptions, ChatResponse, Embeddings, Error, Message, Result,
};

/// How an over-cap caller is handled when the [`Bulkhead`] is
/// saturated.
#[derive(Debug, Clone)]
pub enum BulkheadMode {
    /// Reject immediately. The call returns
    /// `Error::RateLimited { retry_after_ms: None }` so existing
    /// retry / fallback wrappers treat it as transient.
    Reject,
    /// Block up to `Duration` for a slot, then reject.
    WaitUpTo(Duration),
}

/// Wrap any [`ChatModel`] so concurrent calls share a [`Bulkhead`]
/// budget. Bridges the iter-248 primitive into the resilience
/// family.
///
/// # Why this exists
///
/// Many provider APIs (and self-hosted GPU servers) handle a
/// fixed number of concurrent requests gracefully and queue
/// the rest indefinitely. Indefinite queueing is the wrong
/// failure mode for an interactive UI — the user wants either
/// a fast answer or a fast "this provider is hot, switching
/// to a backup." `BulkheadChatModel` enforces the cap and
/// rejects with `Error::RateLimited`.
///
/// Composition: `Error::RateLimited` is what the existing
/// `is_transient` classifier matches, so:
/// - `RetryingChatModel` outer retries with backoff (slot may
///   open).
/// - `FallbackChatModel` outer switches to a backup provider.
/// - Both interactions just work, no extra wiring.
///
/// # Sharing across wrappers
///
/// Pass the same `Arc<Bulkhead>` to multiple
/// `BulkheadChatModel` wrappers to enforce one cap across many
/// model variants. Realistic prod pattern: gpt-4 / gpt-4o-mini /
/// gpt-4-turbo on one OpenAI key share a 10-concurrent budget.
///
/// # Streaming
///
/// `stream()` enters the bulkhead at the handshake. The slot
/// releases when this method returns (the caller drives the
/// stream itself). For per-stream-lifetime exclusion, hold the
/// guard externally around the `stream` + `drain` cycle.
pub struct BulkheadChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub bulkhead: Arc<Bulkhead>,
    pub mode: BulkheadMode,
}

impl BulkheadChatModel {
    pub fn new(
        inner: Arc<dyn ChatModel>,
        bulkhead: Arc<Bulkhead>,
        mode: BulkheadMode,
    ) -> Self {
        Self {
            inner,
            bulkhead,
            mode,
        }
    }

    async fn enter(&self) -> Option<litgraph_core::BulkheadGuard> {
        match self.mode {
            BulkheadMode::Reject => self.bulkhead.try_enter(),
            BulkheadMode::WaitUpTo(d) => self.bulkhead.enter_with_timeout(d).await,
        }
    }
}

#[async_trait]
impl ChatModel for BulkheadChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let _guard = match self.enter().await {
            Some(g) => g,
            None => return Err(Error::RateLimited { retry_after_ms: None }),
        };
        self.inner.invoke(messages, opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        let _guard = match self.enter().await {
            Some(g) => g,
            None => return Err(Error::RateLimited { retry_after_ms: None }),
        };
        self.inner.stream(messages, opts).await
    }
}

/// Embed-axis sibling. Same shared-budget + reject-or-wait
/// semantics; one Bulkhead can span both `embed_query` and
/// `embed_documents`, and (when the same `Arc<Bulkhead>` is
/// passed) across chat + embed wrappers.
pub struct BulkheadEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub bulkhead: Arc<Bulkhead>,
    pub mode: BulkheadMode,
}

impl BulkheadEmbeddings {
    pub fn new(
        inner: Arc<dyn Embeddings>,
        bulkhead: Arc<Bulkhead>,
        mode: BulkheadMode,
    ) -> Self {
        Self {
            inner,
            bulkhead,
            mode,
        }
    }

    async fn enter(&self) -> Option<litgraph_core::BulkheadGuard> {
        match self.mode {
            BulkheadMode::Reject => self.bulkhead.try_enter(),
            BulkheadMode::WaitUpTo(d) => self.bulkhead.enter_with_timeout(d).await,
        }
    }
}

#[async_trait]
impl Embeddings for BulkheadEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let _guard = match self.enter().await {
            Some(g) => g,
            None => return Err(Error::RateLimited { retry_after_ms: None }),
        };
        self.inner.embed_query(text).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let _guard = match self.enter().await {
            Some(g) => g,
            None => return Err(Error::RateLimited { retry_after_ms: None }),
        };
        self.inner.embed_documents(texts).await
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

    /// Sleeps `delay_ms` per call and tracks peak concurrent calls.
    /// Lets keyed-mutex tests verify same-key serialization vs
    /// different-key parallelism.
    struct DelayChatModel {
        delay_ms: u64,
        in_flight: Arc<std::sync::atomic::AtomicUsize>,
        peak: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for DelayChatModel {
        fn name(&self) -> &str {
            "delay-chat"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self.peak.compare_exchange(
                    p,
                    now,
                    Ordering::SeqCst,
                    Ordering::SeqCst,
                ) {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
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
                model: "delay-chat".into(),
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


    // ---- BulkheadChatModel / BulkheadEmbeddings tests -------------------

    #[tokio::test]
    async fn bulkhead_chat_rejects_with_rate_limited_when_at_cap() {
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 30,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(2));
        let cm = Arc::new(BulkheadChatModel::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::Reject,
        ));
        // 5 concurrent calls: 2 should succeed, 3 should reject.
        let mut handles = Vec::new();
        for _ in 0..5 {
            let cm = cm.clone();
            handles.push(tokio::spawn(async move {
                cm.invoke(
                    vec![Message::user("hi")],
                    &ChatOptions::default(),
                )
                .await
            }));
        }
        let mut oks = 0;
        let mut rate_limited = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => oks += 1,
                Err(Error::RateLimited { .. }) => rate_limited += 1,
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        // Cap=2 → at most 2 concurrent in-flight. Some calls finish
        // before others start, so up to 5 may succeed depending on
        // scheduling — but rejected_count must reflect actual rejects.
        assert!(oks >= 2, "fewer than cap calls succeeded: {oks}");
        assert!(
            rate_limited >= 1,
            "expected at least one rejection: rate_limited={rate_limited}",
        );
        assert_eq!(rate_limited as u64, bulkhead.rejected_count());
        assert!(
            peak.load(std::sync::atomic::Ordering::SeqCst) <= 2,
            "peak in_flight exceeded cap",
        );
    }

    #[tokio::test]
    async fn bulkhead_chat_wait_mode_blocks_then_succeeds() {
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 20,
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(1));
        let cm = Arc::new(BulkheadChatModel::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::WaitUpTo(Duration::from_millis(100)),
        ));
        // 3 concurrent calls under cap=1 with 20ms delay each.
        // wait window 100ms is enough for all to succeed.
        let mut handles = Vec::new();
        for _ in 0..3 {
            let cm = cm.clone();
            handles.push(tokio::spawn(async move {
                cm.invoke(
                    vec![Message::user("hi")],
                    &ChatOptions::default(),
                )
                .await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        // Strict cap=1 → peak in_flight must be 1.
        assert_eq!(peak.load(std::sync::atomic::Ordering::SeqCst), 1);
        // Within window, no rejects.
        assert_eq!(bulkhead.rejected_count(), 0);
    }

    #[tokio::test]
    async fn bulkhead_chat_wait_mode_rejects_after_deadline() {
        use std::sync::atomic::AtomicUsize;
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let inner: Arc<dyn ChatModel> = Arc::new(DelayChatModel {
            delay_ms: 100, // long-held slot
            in_flight: in_flight.clone(),
            peak: peak.clone(),
        });
        let bulkhead = Arc::new(Bulkhead::new(1));
        let cm = Arc::new(BulkheadChatModel::new(
            inner,
            bulkhead.clone(),
            BulkheadMode::WaitUpTo(Duration::from_millis(15)),
        ));
        // First call holds the slot 100ms.
        let cm1 = cm.clone();
        let h1 = tokio::spawn(async move {
            cm1.invoke(
                vec![Message::user("hi")],
                &ChatOptions::default(),
            )
            .await
        });
        // Give it a beat to enter.
        tokio::time::sleep(Duration::from_millis(5)).await;
        // Second call: waits 15ms, then rejects.
        let r = cm
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        assert!(matches!(r, Err(Error::RateLimited { .. })));
        h1.await.unwrap().unwrap();
        assert_eq!(bulkhead.rejected_count(), 1);
    }

    #[tokio::test]
    async fn bulkhead_chat_succeeds_when_under_cap() {
        let bulkhead = Arc::new(Bulkhead::new(5));
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let cm = BulkheadChatModel::new(inner, bulkhead.clone(), BulkheadMode::Reject);
        for _ in 0..5 {
            cm.invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap();
        }
        assert_eq!(bulkhead.rejected_count(), 0);
    }

    #[tokio::test]
    async fn bulkhead_embed_rejects_when_at_cap() {
        let bulkhead = Arc::new(Bulkhead::new(1));
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 3);
        let cm = Arc::new(BulkheadEmbeddings::new(
            inner as Arc<dyn Embeddings>,
            bulkhead.clone(),
            BulkheadMode::Reject,
        ));
        // Hold a slot via try_enter directly so we can verify the
        // wrapper rejects under cap.
        let _held = bulkhead.try_enter().unwrap();
        let r = cm.embed_query("hi").await;
        assert!(matches!(r, Err(Error::RateLimited { .. })));
        let r = cm.embed_documents(&["a".into()]).await;
        assert!(matches!(r, Err(Error::RateLimited { .. })));
        assert_eq!(bulkhead.rejected_count(), 2);
    }

    #[tokio::test]
    async fn bulkhead_chat_and_embed_share_one_budget() {
        // One Bulkhead spans both axes.
        let bulkhead = Arc::new(Bulkhead::new(1));
        let chat: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let embed = flaky_embed(0, EmbedFlakyKind::Provider5xx, 3);
        let cm = BulkheadChatModel::new(
            chat,
            bulkhead.clone(),
            BulkheadMode::Reject,
        );
        let em = BulkheadEmbeddings::new(
            embed as Arc<dyn Embeddings>,
            bulkhead.clone(),
            BulkheadMode::Reject,
        );
        // Hold the slot.
        let _held = bulkhead.try_enter().unwrap();
        // Both axes reject because the shared bulkhead is full.
        let r1 = cm
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await;
        let r2 = em.embed_query("hi").await;
        assert!(matches!(r1, Err(Error::RateLimited { .. })));
        assert!(matches!(r2, Err(Error::RateLimited { .. })));
        assert_eq!(bulkhead.rejected_count(), 2);
    }

    #[tokio::test]
    async fn bulkhead_chat_name_proxies_inner() {
        let bulkhead = Arc::new(Bulkhead::new(5));
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let cm = BulkheadChatModel::new(inner, bulkhead, BulkheadMode::Reject);
        assert_eq!(cm.name(), "always-ok");
    }

}

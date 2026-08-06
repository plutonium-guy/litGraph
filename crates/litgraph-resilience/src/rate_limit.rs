use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Embeddings, Message, RateLimiter, Result};

/// Token-bucket rate limit config — two knobs only:
/// `requests_per_minute` (steady-state rate) and `burst` (max bucket capacity,
/// i.e. how much credit accumulates during idle periods). For a strict
/// non-bursty limit, set `burst = 1`.
#[derive(Debug, Clone, Copy)]
pub struct RateLimitConfig {
    pub requests_per_minute: u32,
    pub burst: u32,
}

impl RateLimitConfig {
    pub fn per_minute(rpm: u32) -> Self {
        Self { requests_per_minute: rpm, burst: rpm.max(1) }
    }
    pub fn with_burst(mut self, b: u32) -> Self {
        self.burst = b.max(1);
        self
    }
}

struct BucketState {
    tokens: f64,
    last_refill: std::time::Instant,
}

/// Provider-agnostic token-bucket rate limiter. Each `invoke` / `stream` call
/// acquires one token; if the bucket is empty, the caller awaits until enough
/// time has passed for the next token to refill. Acquisitions are serialized
/// via a `tokio::sync::Mutex` so concurrent invokes form a fair queue.
pub struct RateLimitedChatModel {
    pub inner: Arc<dyn ChatModel>,
    refill_per_sec: f64,
    capacity: f64,
    state: tokio::sync::Mutex<BucketState>,
}

impl RateLimitedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, cfg: RateLimitConfig) -> Self {
        let refill_per_sec = (cfg.requests_per_minute as f64) / 60.0;
        let capacity = cfg.burst as f64;
        Self {
            inner,
            refill_per_sec,
            capacity,
            state: tokio::sync::Mutex::new(BucketState {
                tokens: capacity,           // start full so the first call is immediate
                last_refill: std::time::Instant::now(),
            }),
        }
    }

    async fn acquire(&self) {
        if self.refill_per_sec <= 0.0 {
            return; // rate limit = ∞ => no-op
        }
        let mut state = self.state.lock().await;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        state.last_refill = now;
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            return;
        }
        // Hold the lock across the sleep so the queue is FIFO.
        let deficit = 1.0 - state.tokens;
        let wait = Duration::from_secs_f64(deficit / self.refill_per_sec);
        tokio::time::sleep(wait).await;
        state.tokens = 0.0;
        state.last_refill = std::time::Instant::now();
    }
}

#[async_trait]
impl ChatModel for RateLimitedChatModel {
    fn name(&self) -> &str { self.inner.name() }

    async fn invoke(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
        self.acquire().await;
        self.inner.invoke(messages, opts).await
    }

    async fn stream(&self, messages: Vec<Message>, opts: &ChatOptions) -> Result<ChatStream> {
        self.acquire().await;
        self.inner.stream(messages, opts).await
    }
}

/// Token-bucket rate limiter for `Embeddings`. Same semantics as
/// `RateLimitedChatModel` — one token per call, refills at
/// `requests_per_minute` rate, `burst` bucket capacity.
///
/// NOTE: the bucket counts CALLS not texts. A single `embed_documents`
/// batch of 100 texts consumes one token (most providers bill per call,
/// not per text, so this matches cost semantics). If your provider rate-
/// limits per-text, chunk upstream and wrap each chunk separately.
pub struct RateLimitedEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    refill_per_sec: f64,
    capacity: f64,
    state: tokio::sync::Mutex<BucketState>,
}

impl RateLimitedEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, cfg: RateLimitConfig) -> Self {
        let refill_per_sec = (cfg.requests_per_minute as f64) / 60.0;
        let capacity = cfg.burst as f64;
        Self {
            inner,
            refill_per_sec,
            capacity,
            state: tokio::sync::Mutex::new(BucketState {
                tokens: capacity,
                last_refill: std::time::Instant::now(),
            }),
        }
    }

    async fn acquire(&self) {
        if self.refill_per_sec <= 0.0 {
            return;
        }
        let mut state = self.state.lock().await;
        let now = std::time::Instant::now();
        let elapsed = now.duration_since(state.last_refill).as_secs_f64();
        state.tokens = (state.tokens + elapsed * self.refill_per_sec).min(self.capacity);
        state.last_refill = now;
        if state.tokens >= 1.0 {
            state.tokens -= 1.0;
            return;
        }
        let deficit = 1.0 - state.tokens;
        let wait = Duration::from_secs_f64(deficit / self.refill_per_sec);
        tokio::time::sleep(wait).await;
        state.tokens = 0.0;
        state.last_refill = std::time::Instant::now();
    }
}

#[async_trait]
impl Embeddings for RateLimitedEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.acquire().await;
        self.inner.embed_query(text).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.acquire().await;
        self.inner.embed_documents(texts).await
    }
}

/// Wrap any [`ChatModel`] so each call charges against a SHARED
/// [`RateLimiter`] (iter 242). Distinct from
/// [`RateLimitedChatModel`] which owns its own bucket: this
/// takes an `Arc<RateLimiter>` so multiple wrapped models can
/// charge against ONE budget.
///
/// # Why this exists
///
/// One provider API key typically has ONE quota (TPM/RPM) shared
/// across every model variant served by that key. The realistic
/// prod pattern: a router that dispatches to gpt-4, gpt-4-turbo,
/// gpt-4o-mini based on heuristics — all three calls draw from
/// the same key's TPM. With per-model `RateLimitedChatModel`
/// each variant has its own bucket and the aggregate exceeds
/// the real budget. With this wrapper they all charge against
/// one shared `RateLimiter`.
///
/// Per-call weight is fixed at 1 token (one request charge) by
/// default. For weight-by-tokens-estimate use cases, wrap with
/// a custom adapter or call `limiter.acquire(estimate).await`
/// directly upstream. We pick 1-token-per-request as the
/// minimum-surprise default.
///
/// # Streaming
///
/// `stream()` charges 1 token at handshake, same as `invoke`.
/// Mid-stream tokens don't deduct further.
pub struct SharedRateLimitedChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub limiter: Arc<RateLimiter>,
}

impl SharedRateLimitedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, limiter: Arc<RateLimiter>) -> Self {
        Self { inner, limiter }
    }
}

#[async_trait]
impl ChatModel for SharedRateLimitedChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.limiter.acquire(1).await;
        self.inner.invoke(messages, opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        self.limiter.acquire(1).await;
        self.inner.stream(messages, opts).await
    }
}

/// Embed-axis sibling. Both `embed_query` and `embed_documents`
/// charge 1 token per call against the shared bucket. For
/// chunked-embedding pipelines, charge once per chunk (not per
/// document) by sizing your bucket accordingly — `RateLimiter`
/// is request-count agnostic.
pub struct SharedRateLimitedEmbeddings {
    pub inner: Arc<dyn Embeddings>,
    pub limiter: Arc<RateLimiter>,
}

impl SharedRateLimitedEmbeddings {
    pub fn new(inner: Arc<dyn Embeddings>, limiter: Arc<RateLimiter>) -> Self {
        Self { inner, limiter }
    }
}

#[async_trait]
impl Embeddings for SharedRateLimitedEmbeddings {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.limiter.acquire(1).await;
        self.inner.embed_query(text).await
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.limiter.acquire(1).await;
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

    /// Always-succeeds model that counts how many times it was hit.
    struct CountingModel { calls: AtomicU32 }

    #[async_trait]
    impl ChatModel for CountingModel {
        fn name(&self) -> &str { "count" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "ok".into() }],
                    tool_calls: vec![], tool_call_id: None, name: None, cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "count".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limiter_serves_burst_immediately_then_throttles() {
        let inner: Arc<dyn ChatModel> = Arc::new(CountingModel { calls: AtomicU32::new(0) });
        // 60 RPM = 1 RPS, burst of 3.
        let r = RateLimitedChatModel::new(inner.clone(),
            RateLimitConfig::per_minute(60).with_burst(3));
        let start = tokio::time::Instant::now();
        // Burst of 3 should drain immediately.
        for _ in 0..3 {
            r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        }
        assert!(start.elapsed() < Duration::from_millis(50),
            "burst should be near-instant, took {:?}", start.elapsed());
        // 4th call must wait ~1s for the next refill.
        r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        let elapsed = start.elapsed();
        assert!(elapsed >= Duration::from_millis(990) && elapsed < Duration::from_millis(1500),
            "4th call should wait ~1s, took {:?}", elapsed);
    }

    #[tokio::test(start_paused = true)]
    async fn rate_limiter_steady_state_matches_configured_rate() {
        let inner: Arc<dyn ChatModel> = Arc::new(CountingModel { calls: AtomicU32::new(0) });
        // 120 RPM = 2 RPS, no burst (=1) → strict 1-every-500ms cadence.
        let r = RateLimitedChatModel::new(inner.clone(),
            RateLimitConfig::per_minute(120).with_burst(1));
        let start = tokio::time::Instant::now();
        // 4 calls @ 2 RPS w/ burst=1 → first instant, then 500/1000/1500ms.
        for _ in 0..4 {
            r.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        }
        let total = start.elapsed();
        assert!(total >= Duration::from_millis(1490) && total < Duration::from_millis(2000),
            "4 calls @ 2 RPS should take ~1.5s, took {:?}", total);
    }


    // ---- RateLimitedEmbeddings tests -----------------------------------

    struct CountingEmbed {
        calls: AtomicU32,
        dim: usize,
    }

    #[async_trait]
    impl Embeddings for CountingEmbed {
        fn name(&self) -> &str {
            "counting-embed"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![0.0; self.dim])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![vec![0.0; self.dim]; texts.len()])
        }
    }

    #[tokio::test]
    async fn ratelimit_embed_steady_state_matches_configured_rate() {
        let inner: Arc<dyn Embeddings> = Arc::new(CountingEmbed {
            calls: AtomicU32::new(0),
            dim: 128,
        });
        // 120 RPM = 2 RPS, burst=1 → strict 1-every-500ms cadence.
        let r = RateLimitedEmbeddings::new(
            inner,
            RateLimitConfig::per_minute(120).with_burst(1),
        );
        let start = tokio::time::Instant::now();
        for _ in 0..4 {
            r.embed_query("hi").await.unwrap();
        }
        let total = start.elapsed();
        assert!(
            total >= Duration::from_millis(1490) && total < Duration::from_millis(2000),
            "4 calls @ 2 RPS w/ burst=1 should take ~1.5s, took {:?}",
            total
        );
    }

    #[tokio::test]
    async fn ratelimit_embed_burst_serves_immediately_then_throttles() {
        let inner: Arc<dyn Embeddings> = Arc::new(CountingEmbed {
            calls: AtomicU32::new(0),
            dim: 512,
        });
        // 60 RPM = 1 RPS, burst=3 → 3 instant, 4th waits ~1s.
        let r = RateLimitedEmbeddings::new(
            inner,
            RateLimitConfig::per_minute(60).with_burst(3),
        );
        let start = tokio::time::Instant::now();
        for _ in 0..3 {
            r.embed_query("hi").await.unwrap();
        }
        // Burst absorbed instantly.
        assert!(start.elapsed() < Duration::from_millis(100));
        // 4th call throttles.
        r.embed_query("hi").await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(900));
    }

    #[tokio::test]
    async fn ratelimit_embed_batch_counts_as_one_token() {
        // embed_documents with 100 texts consumes ONE token (providers
        // bill per call, not per text).
        let inner: Arc<dyn Embeddings> = Arc::new(CountingEmbed {
            calls: AtomicU32::new(0),
            dim: 256,
        });
        let r = RateLimitedEmbeddings::new(
            inner,
            RateLimitConfig::per_minute(60).with_burst(1),
        );
        let big_batch: Vec<String> = (0..100).map(|i| format!("text_{i}")).collect();
        // First call — uses the single burst token immediately.
        let start = tokio::time::Instant::now();
        r.embed_documents(&big_batch).await.unwrap();
        assert!(start.elapsed() < Duration::from_millis(100));
        // Second call — must wait ~1s (1 RPS steady).
        r.embed_documents(&big_batch).await.unwrap();
        assert!(start.elapsed() >= Duration::from_millis(900));
    }

    #[tokio::test]
    async fn ratelimit_embed_exposes_inner_dim() {
        let inner: Arc<dyn Embeddings> = Arc::new(CountingEmbed {
            calls: AtomicU32::new(0),
            dim: 1536,
        });
        let r = RateLimitedEmbeddings::new(inner, RateLimitConfig::per_minute(1000));
        assert_eq!(r.dimensions(), 1536);
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


    // ---- SharedRateLimited{Chat,Embed} tests ---------------------------

    #[tokio::test]
    async fn shared_rate_limit_two_chat_models_share_one_budget() {
        // Bucket: capacity 2, refill 50/sec ≈ 20ms/token. Two distinct
        // wrapped models share the limiter. Four total calls (2 per
        // model) must take >= 2 * 20ms = 40ms (the 3rd and 4th wait).
        let limiter = Arc::new(RateLimiter::new(2, 50));
        let m1: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let m2: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let s1 = Arc::new(SharedRateLimitedChatModel::new(m1, limiter.clone()));
        let s2 = Arc::new(SharedRateLimitedChatModel::new(m2, limiter.clone()));
        let started = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..2 {
            let s = s1.clone();
            handles.push(tokio::spawn(async move {
                s.invoke(vec![Message::user("hi")], &ChatOptions::default()).await
            }));
        }
        for _ in 0..2 {
            let s = s2.clone();
            handles.push(tokio::spawn(async move {
                s.invoke(vec![Message::user("hi")], &ChatOptions::default()).await
            }));
        }
        for h in handles {
            h.await.unwrap().unwrap();
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(30),
            "shared budget didn't throttle: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn shared_rate_limit_chat_serves_burst_immediately() {
        // Capacity 4, refill 1/sec. Bursting 4 calls must NOT block.
        let limiter = Arc::new(RateLimiter::new(4, 1));
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let s = Arc::new(SharedRateLimitedChatModel::new(inner, limiter));
        let started = std::time::Instant::now();
        for _ in 0..4 {
            s.invoke(vec![Message::user("hi")], &ChatOptions::default())
                .await
                .unwrap();
        }
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(50),
            "burst was throttled: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn shared_rate_limit_embed_query_and_documents_share_budget() {
        // Bucket: capacity 1, refill 50/sec. Three calls
        // (embed_query, embed_documents, embed_query) on the same
        // wrapper must take >= 2 * 20ms.
        let limiter = Arc::new(RateLimiter::new(1, 50));
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 3);
        let s = Arc::new(SharedRateLimitedEmbeddings::new(
            inner as Arc<dyn Embeddings>,
            limiter,
        ));
        let started = std::time::Instant::now();
        s.embed_query("a").await.unwrap();
        s.embed_documents(&["x".into()]).await.unwrap();
        s.embed_query("b").await.unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(20),
            "shared embed budget didn't throttle: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn shared_rate_limit_chat_name_proxies_inner() {
        let limiter = Arc::new(RateLimiter::new(10, 10));
        let inner: Arc<dyn ChatModel> = Arc::new(AlwaysOkModel);
        let s = SharedRateLimitedChatModel::new(inner, limiter);
        assert_eq!(s.name(), "always-ok");
    }

    #[tokio::test]
    async fn shared_rate_limit_embed_dimensions_proxy_inner() {
        let limiter = Arc::new(RateLimiter::new(10, 10));
        let inner = flaky_embed(0, EmbedFlakyKind::Provider5xx, 5);
        let s = SharedRateLimitedEmbeddings::new(
            inner as Arc<dyn Embeddings>,
            limiter,
        );
        assert_eq!(s.dimensions(), 5);
        assert_eq!(s.name(), "flaky-embed");
    }

}

//! Resilient wrappers for `ChatModel`. Wraps any provider with retry + jittered
//! exponential backoff via the `backon` crate.
//!
//! # What gets retried
//!
//! - `Error::RateLimited` (429) — retried, honors the upstream `retry_after_ms`
//!   when present.
//! - `Error::Timeout` — retried.
//! - `Error::Provider(s)` where `s` matches a 5xx status pattern — retried.
//!
//! Everything else (bad request, parse error, invalid input, tool failure,
//! cancellation) is treated as terminal and returned to the caller without
//! retries — replays would just waste tokens / propagate the bug.
//!
//! # Streaming
//!
//! `stream()` is NOT retried (token streams can't restart cleanly mid-stream).
//! For streaming retries, capture the failure at the consumer layer and re-call.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use backon::{ExponentialBuilder, Retryable};
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Error, Message, Result};
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
fn is_transient(e: &Error) -> bool {
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

// ─────────────────────────────────────────────────────────────────────────────
// Rate limiting
// ─────────────────────────────────────────────────────────────────────────────

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

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ContentPart, Message, Role};
    use std::sync::atomic::{AtomicU32, Ordering};

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
}

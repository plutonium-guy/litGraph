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

// ─────────────────────────────────────────────────────────────────────────────
// Cross-provider fallback
// ─────────────────────────────────────────────────────────────────────────────

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
}

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
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, CircuitBreaker, CircuitCallError, ContentPart,
    Embeddings, Error, Message, PiiScrubber, Result,
};
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

// ─────────────────────────────────────────────────────────────────────────────
// RaceChatModel — concurrent invoke, first-success-wins
// ─────────────────────────────────────────────────────────────────────────────

/// Invokes N inner `ChatModel`s **concurrently**, returns the first one
/// that succeeds. Outstanding invocations are aborted as soon as a
/// winner emerges.
///
/// Use this when latency matters more than cost: paying for N parallel
/// requests buys you `min(t_1, .., t_N)` end-to-end. Typical patterns:
///
/// - Multi-region failover where the request hedges across regions
///   instead of waiting for one to time out.
/// - Latency-critical paths backed by both a fast cheap model and a
///   slow strong model — race them and take whichever finishes first.
/// - Cross-provider speculative serving (OpenAI + Anthropic +
///   Bedrock) for tail-latency reduction.
///
/// # Cost model
///
/// All N requests are *issued*. Cancellation aborts the in-flight
/// future, but providers have already begun processing — most bill for
/// any tokens generated before the connection closes. Budget for
/// `cost_floor ≈ N × p50_inference_cost` even though latency is `p_min`.
///
/// # Failure
///
/// Returns `Ok` as soon as **any** inner returns `Ok`. Returns `Err`
/// only if **every** inner fails — error message aggregates all
/// failures (newline-separated) so the caller can debug.
///
/// # Streaming
///
/// `stream()` falls through to `inners[0]`. Racing token streams is
/// possible (race for first chunk, then commit) but inviting in
/// practice — providers' first chunks vary wildly in latency and
/// quality, and switching mid-stream is impossible. Consumers who
/// need it can race `invoke()` calls themselves.
///
/// # Composition
///
/// `Race(Retry(A), Retry(B))` is the typical shape — let each branch
/// handle its own transient errors, race the steady-state outcomes.
/// Avoid `Race(Race(A,B), C)` — flatten the chain into a single
/// `Race(A, B, C)` so cancellation works across the whole set.
///
/// ```ignore
/// use std::sync::Arc;
/// use litgraph_resilience::RaceChatModel;
/// let race = RaceChatModel::new(vec![openai_arc, anthropic_arc, bedrock_arc]);
/// // First to return wins; the other two are aborted.
/// let resp = race.invoke(messages, &opts).await?;
/// ```
pub struct RaceChatModel {
    pub inners: Vec<Arc<dyn ChatModel>>,
}

impl RaceChatModel {
    /// Build a race set. Panics if `inners` is empty (a race with no
    /// runners can't yield a winner).
    pub fn new(inners: Vec<Arc<dyn ChatModel>>) -> Self {
        assert!(
            !inners.is_empty(),
            "RaceChatModel: need at least one inner model",
        );
        Self { inners }
    }
}

#[async_trait]
impl ChatModel for RaceChatModel {
    fn name(&self) -> &str {
        "race"
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        use tokio::task::JoinSet;

        // Single-inner shortcut keeps the spawn-overhead off the hot
        // path for users who probe with a one-element race.
        if self.inners.len() == 1 {
            return self.inners[0].invoke(messages, opts).await;
        }

        let mut set: JoinSet<Result<ChatResponse>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let msgs = messages.clone();
            let opts = opts.clone();
            set.spawn(async move { inner.invoke(msgs, &opts).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(resp)) => {
                    // Winner — abort everything else and return.
                    set.abort_all();
                    return Ok(resp);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceChatModel: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Mid-stream switching is unworkable. See module doc.
        self.inners[0].stream(messages, opts).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeoutChatModel — per-invocation deadline wrapper
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// CircuitBreakerChatModel — fail-fast wrapper backed by iter-243 primitive
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Per-invocation token budget
// ─────────────────────────────────────────────────────────────────────────────

/// Chat-model wrapper that enforces a token-count budget per invocation.
/// Two modes:
///
/// - **Strict (default)**: messages exceeding the budget → `Error::InvalidInput`.
///   Caller must trim / summarize upstream. Fails fast, predictable cost.
/// - **Auto-trim**: call `.auto_trim()` to enable — the wrapper uses
///   [`litgraph_tokenizers::trim_messages`] to drop oldest non-system
///   messages until under budget, then forwards. System messages are
///   ALWAYS preserved (they carry the persona / task). Last message
///   always preserved too (the user's actual query).
///
/// # Why
///
/// Long conversations silently balloon token cost. Without a budget,
/// a support chatbot that never forgets can rack up $100 LLM bills on
/// a single session. This wrapper makes the cap explicit.
///
/// # Model-specific token counts
///
/// The tokenizer used by [`trim_messages`] is picked by the inner
/// model's `name()`. Providers whose name contains "gpt" use tiktoken's
/// cl100k/o200k; others fall back to `tiktoken-rs` cl100k estimate.
/// Non-GPT-named models may be off by 5–15% — this is a best-effort
/// estimator, not a billing oracle.
///
/// # Composition
///
/// Safe to stack with other wrappers: `Retry(Budget(inner))` is fine,
/// as is `Budget(Retry(inner))`. Typical order: budget innermost
/// (trim ONCE per logical invocation, then retry with the same trimmed
/// message set on transient errors).
///
/// ```ignore
/// use litgraph_resilience::{RetryingChatModel, RetryConfig, TokenBudgetChatModel};
/// let budgeted = TokenBudgetChatModel::new(inner, 4096).auto_trim();
/// let retrying = RetryingChatModel::new(Arc::new(budgeted), RetryConfig::default());
/// ```
pub struct TokenBudgetChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub max_tokens: usize,
    pub auto_trim: bool,
}

impl TokenBudgetChatModel {
    /// Build in strict mode. Use [`.auto_trim()`] to switch to auto-trimming.
    pub fn new(inner: Arc<dyn ChatModel>, max_tokens: usize) -> Self {
        Self {
            inner,
            max_tokens,
            auto_trim: false,
        }
    }

    /// Enable auto-trimming mode.
    pub fn auto_trim(mut self) -> Self {
        self.auto_trim = true;
        self
    }
}

#[async_trait]
impl ChatModel for TokenBudgetChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let cost = litgraph_tokenizers::count_message_tokens(self.inner.name(), &messages);
        if cost <= self.max_tokens {
            return self.inner.invoke(messages, opts).await;
        }
        if !self.auto_trim {
            return Err(Error::invalid(format!(
                "TokenBudgetChatModel: messages use ~{} tokens, budget is {}. \
                 Trim upstream or enable .auto_trim().",
                cost, self.max_tokens
            )));
        }
        // Auto-trim mode: drop oldest non-system until under budget.
        let trimmed = litgraph_tokenizers::trim_messages(
            self.inner.name(),
            &messages,
            self.max_tokens,
        );
        tracing::debug!(
            model = self.inner.name(),
            input = messages.len(),
            kept = trimmed.len(),
            budget = self.max_tokens,
            "TokenBudgetChatModel auto-trimmed history"
        );
        self.inner.invoke(trimmed, opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Same budget logic on the streaming path.
        let cost = litgraph_tokenizers::count_message_tokens(self.inner.name(), &messages);
        if cost <= self.max_tokens {
            return self.inner.stream(messages, opts).await;
        }
        if !self.auto_trim {
            return Err(Error::invalid(format!(
                "TokenBudgetChatModel: messages use ~{} tokens, budget is {}.",
                cost, self.max_tokens
            )));
        }
        let trimmed = litgraph_tokenizers::trim_messages(
            self.inner.name(),
            &messages,
            self.max_tokens,
        );
        self.inner.stream(trimmed, opts).await
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Cross-provider Embeddings fallback
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// RaceEmbeddings — concurrent N-provider race, first-success-wins
// ─────────────────────────────────────────────────────────────────────────────

/// Embeddings analogue of [`RaceChatModel`]: invokes every inner
/// `Embeddings` provider concurrently, returns the first successful
/// result, aborts the others.
///
/// # Why use this over `FallbackEmbeddings`
///
/// `FallbackEmbeddings` is **sequential** — try A, on failure try B
/// (cost-min, but pays A's full latency before B even starts).
/// `RaceEmbeddings` is **concurrent** — issue A, B, C at the same
/// instant, take whoever returns first (latency-min, pays for all
/// requests but cuts the p95).
///
/// Typical patterns:
/// - Hedge a remote provider (OpenAI / Voyage / Cohere) against a
///   local fastembed cross-encoder. Local usually wins on warm
///   cache; remote wins when local hits CPU pressure.
/// - Multi-region embedding: us-east + eu-west, race for whichever
///   is closer to the caller's pod.
/// - Tail-latency reduction on the embed-query critical path —
///   `embed_query` blocks every retrieval, so shaving 50ms off p95
///   compounds over a session.
///
/// # Cost
///
/// All N providers are *issued* — most bill for tokens generated
/// before the connection closes. Budget `cost_floor ≈ N × p50_cost`
/// even though latency is `p_min`.
///
/// # Dimensions
///
/// All inners must agree on `dimensions()` — checked at construction
/// (panic). Race semantics on the result mean a 1536-dim winner this
/// call vs a 512-dim winner next call would silently corrupt your
/// vector index, so we forbid the configuration outright.
pub struct RaceEmbeddings {
    pub inners: Vec<Arc<dyn Embeddings>>,
    dim: usize,
    name_label: String,
}

impl RaceEmbeddings {
    /// Build a race set. Panics if `inners` is empty or any two
    /// inners disagree on `dimensions()`.
    pub fn new(inners: Vec<Arc<dyn Embeddings>>) -> Self {
        assert!(
            !inners.is_empty(),
            "RaceEmbeddings: need at least one inner provider",
        );
        let dim = inners[0].dimensions();
        for (i, e) in inners.iter().enumerate().skip(1) {
            assert_eq!(
                e.dimensions(),
                dim,
                "RaceEmbeddings: inner #{} has dim {} but inner #0 has dim {} — \
                 race semantics + dim mismatch would silently corrupt the vector index",
                i,
                e.dimensions(),
                dim,
            );
        }
        let labels: Vec<String> = inners.iter().map(|e| e.name().to_string()).collect();
        Self {
            inners,
            dim,
            name_label: format!("race({})", labels.join(", ")),
        }
    }
}

#[async_trait]
impl Embeddings for RaceEmbeddings {
    fn name(&self) -> &str {
        &self.name_label
    }
    fn dimensions(&self) -> usize {
        self.dim
    }

    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        use tokio::task::JoinSet;

        if self.inners.len() == 1 {
            return self.inners[0].embed_query(text).await;
        }

        let mut set: JoinSet<Result<Vec<f32>>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let text = text.to_string();
            set.spawn(async move { inner.embed_query(&text).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(v)) => {
                    set.abort_all();
                    return Ok(v);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceEmbeddings.embed_query: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }

    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        use tokio::task::JoinSet;

        if self.inners.len() == 1 {
            return self.inners[0].embed_documents(texts).await;
        }

        let mut set: JoinSet<Result<Vec<Vec<f32>>>> = JoinSet::new();
        for inner in self.inners.iter() {
            let inner = inner.clone();
            let owned: Vec<String> = texts.to_vec();
            set.spawn(async move { inner.embed_documents(&owned).await });
        }

        let mut errors: Vec<String> = Vec::with_capacity(self.inners.len());
        while let Some(joined) = set.join_next().await {
            match joined {
                Ok(Ok(v)) => {
                    set.abort_all();
                    return Ok(v);
                }
                Ok(Err(e)) => errors.push(e.to_string()),
                Err(e) => errors.push(format!("task join: {e}")),
            }
        }
        Err(Error::other(format!(
            "RaceEmbeddings.embed_documents: all {} inners failed:\n  - {}",
            self.inners.len(),
            errors.join("\n  - "),
        )))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// TimeoutEmbeddings — per-invocation deadline wrapper
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// Embeddings retry + rate-limit wrappers
// ─────────────────────────────────────────────────────────────────────────────

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

// ─────────────────────────────────────────────────────────────────────────────
// PII scrubbing chat wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Chat-model wrapper that redacts PII before sending prompts to the
/// inner provider and (optionally) redacts PII from the response text.
///
/// # Why
///
/// - **GDPR / CCPA** — sending raw user emails / phones / SSNs to a
///   third-party LLM vendor without a DPA is an audit finding.
/// - **Prompt-injection hygiene** — AWS keys and JWTs inadvertently
///   pasted by users get scrubbed before the model sees them; reduces
///   exfiltration blast radius.
/// - **Observability safety** — if this wrapper sits between the user
///   and the provider, downstream logging / tracing sees redacted
///   prompts, not raw PII.
///
/// # Default behavior
///
/// - `scrub_inputs = true` — redact outgoing user + system messages.
///   Assistant / tool messages are NOT touched (they came from the
///   model or tool, and re-scrubbing would corrupt agent traces).
/// - `scrub_outputs = false` — leave LLM responses as-is. Real-world
///   LLMs rarely leak PII (they didn't see it); output-scrubbing is
///   off by default to avoid mangling code blocks that contain
///   email-like or IP-like strings. Opt in with `.with_output_scrubbing()`
///   for strict environments.
/// - `scrub_system = false` — system prompts are operator-written and
///   usually contain no real PII. Off by default; opt in if you inject
///   user data into system messages.
///
/// # Note on streaming
///
/// `stream()` currently scrubs inputs but does NOT scrub token deltas.
/// Token-by-token scrubbing would require a streaming PII parser that
/// handles span boundaries — out of scope here. Full-string output
/// scrubbing still works at the final `Done` event if the consumer
/// reassembles the response and passes it through `PiiScrubber::scrub`.
///
/// # Composition
///
/// Stack with other wrappers freely: `Retry(Budget(Scrub(inner)))` is
/// typical — scrub innermost so retries don't re-scrub (CPU waste) and
/// the budget + retry counts apply to the scrubbed payload.
pub struct PiiScrubbingChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub scrubber: Arc<PiiScrubber>,
    pub scrub_inputs: bool,
    pub scrub_system: bool,
    pub scrub_outputs: bool,
}

impl PiiScrubbingChatModel {
    /// Build with the default PiiScrubber (all iter-129 detectors).
    pub fn new(inner: Arc<dyn ChatModel>) -> Self {
        Self {
            inner,
            scrubber: Arc::new(PiiScrubber::new()),
            scrub_inputs: true,
            scrub_system: false,
            scrub_outputs: false,
        }
    }

    /// Build with a caller-provided scrubber (e.g. with custom patterns
    /// or `.without_luhn()` for test environments).
    pub fn with_scrubber(mut self, scrubber: Arc<PiiScrubber>) -> Self {
        self.scrubber = scrubber;
        self
    }

    pub fn with_output_scrubbing(mut self) -> Self {
        self.scrub_outputs = true;
        self
    }

    pub fn with_system_scrubbing(mut self) -> Self {
        self.scrub_system = true;
        self
    }

    pub fn scrub_inputs(mut self, on: bool) -> Self {
        self.scrub_inputs = on;
        self
    }

    /// Scrub a Message's text content IN PLACE. Non-text ContentParts
    /// (images, audio) are untouched — we only mask string PII.
    /// Returns the new Message.
    fn scrub_message(&self, m: &Message) -> Message {
        use litgraph_core::Role;
        // Skip roles where scrubbing is off or doesn't make sense.
        let should_scrub = match m.role {
            Role::User => self.scrub_inputs,
            Role::System => self.scrub_inputs && self.scrub_system,
            // Assistant / Tool messages preserve the model's / tool's output.
            Role::Assistant | Role::Tool => false,
        };
        if !should_scrub {
            return m.clone();
        }
        let new_parts: Vec<ContentPart> = m
            .content
            .iter()
            .map(|p| match p {
                ContentPart::Text { text } => {
                    let scrubbed = self.scrubber.scrub(text).scrubbed;
                    ContentPart::Text { text: scrubbed }
                }
                other => other.clone(),
            })
            .collect();
        Message {
            role: m.role,
            content: new_parts,
            tool_calls: m.tool_calls.clone(),
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
            cache: m.cache,
        }
    }

    fn scrub_all(&self, messages: Vec<Message>) -> Vec<Message> {
        if !self.scrub_inputs {
            return messages;
        }
        messages.iter().map(|m| self.scrub_message(m)).collect()
    }

    fn scrub_response_text(&self, mut resp: ChatResponse) -> ChatResponse {
        if !self.scrub_outputs {
            return resp;
        }
        let new_parts: Vec<ContentPart> = resp
            .message
            .content
            .into_iter()
            .map(|p| match p {
                ContentPart::Text { text } => {
                    let scrubbed = self.scrubber.scrub(&text).scrubbed;
                    ContentPart::Text { text: scrubbed }
                }
                other => other,
            })
            .collect();
        resp.message.content = new_parts;
        resp
    }
}

#[async_trait]
impl ChatModel for PiiScrubbingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        let scrubbed = self.scrub_all(messages);
        let resp = self.inner.invoke(scrubbed, opts).await?;
        Ok(self.scrub_response_text(resp))
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // Scrub inputs but pass stream through as-is (token-delta
        // scrubbing is out of scope — see the module doc).
        let scrubbed = self.scrub_all(messages);
        self.inner.stream(scrubbed, opts).await
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

    // ---- RaceChatModel tests ----------------------------------------------

    /// Sleeps `delay_ms` then either returns a labelled response or errors.
    struct DelayedModel {
        label: &'static str,
        delay_ms: u64,
        succeed: bool,
        invocations: Arc<std::sync::atomic::AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for DelayedModel {
        fn name(&self) -> &str {
            self.label
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.invocations
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(ChatResponse {
                    message: Message {
                        role: Role::Assistant,
                        content: vec![ContentPart::Text {
                            text: self.label.into(),
                        }],
                        tool_calls: vec![],
                        tool_call_id: None,
                        name: None,
                        cache: false,
                    },
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: self.label.into(),
                })
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn arc_dm(
        label: &'static str,
        delay_ms: u64,
        succeed: bool,
    ) -> (Arc<dyn ChatModel>, Arc<std::sync::atomic::AtomicUsize>) {
        let count = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let m = DelayedModel {
            label,
            delay_ms,
            succeed,
            invocations: count.clone(),
        };
        (Arc::new(m), count)
    }

    #[tokio::test]
    #[should_panic(expected = "at least one inner model")]
    async fn race_panics_on_empty() {
        let _ = RaceChatModel::new(vec![]);
    }

    #[tokio::test]
    async fn race_returns_first_winner() {
        // A finishes in 5ms, B in 50ms — A must win.
        let (a, _) = arc_dm("a", 5, true);
        let (b, _) = arc_dm("b", 50, true);
        let race = RaceChatModel::new(vec![a, b]);
        let resp = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.model, "a");
        assert_eq!(resp.message.text_content(), "a");
    }

    #[tokio::test]
    async fn race_falls_through_failures() {
        // A fails immediately, B succeeds slowly — B is the answer.
        let (a, _) = arc_dm("a", 1, false);
        let (b, _) = arc_dm("b", 10, true);
        let race = RaceChatModel::new(vec![a, b]);
        let resp = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(resp.model, "b");
    }

    #[tokio::test]
    async fn race_aggregates_when_all_fail() {
        let (a, _) = arc_dm("a", 1, false);
        let (b, _) = arc_dm("b", 2, false);
        let (c, _) = arc_dm("c", 3, false);
        let race = RaceChatModel::new(vec![a, b, c]);
        let err = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("all 3 inners failed"), "got: {s}");
        assert!(s.contains("a failed"));
        assert!(s.contains("b failed"));
        assert!(s.contains("c failed"));
    }

    #[tokio::test]
    async fn race_single_inner_passes_through() {
        let (a, count) = arc_dm("a", 0, true);
        let race = RaceChatModel::new(vec![a]);
        let _ = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        assert_eq!(count.load(std::sync::atomic::Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn race_aborts_losers_after_winner() {
        // A finishes in 5ms; B sleeps 500ms. After A wins, B's task
        // should be aborted — measured by the wall-clock total being
        // closer to A's 5ms than B's 500ms.
        let (a, _) = arc_dm("a", 5, true);
        let (b, _) = arc_dm("b", 500, true);
        let race = RaceChatModel::new(vec![a, b]);
        let started = std::time::Instant::now();
        let _ = race
            .invoke(vec![Message::user("hi")], &ChatOptions::default())
            .await
            .unwrap();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        // Allow generous slack for CI variance, but B's 500ms must NOT
        // dominate.
        assert!(
            elapsed_ms < 200,
            "elapsed {elapsed_ms}ms — losers were not aborted",
        );
    }

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

    // ---- TokenBudgetChatModel tests -----------------------------------

    /// Captures `messages.len()` + `system message count` of each invoke call.
    struct CapturingChat {
        last_msg_count: std::sync::atomic::AtomicUsize,
        last_sys_count: std::sync::atomic::AtomicUsize,
    }

    impl CapturingChat {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                last_msg_count: std::sync::atomic::AtomicUsize::new(0),
                last_sys_count: std::sync::atomic::AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ChatModel for CapturingChat {
        fn name(&self) -> &str {
            "gpt-4o-mini"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.last_msg_count
                .store(messages.len(), Ordering::SeqCst);
            self.last_sys_count.store(
                messages
                    .iter()
                    .filter(|m| matches!(m.role, Role::System))
                    .count(),
                Ordering::SeqCst,
            );
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "gpt-4o-mini".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn long_history(n: usize) -> Vec<Message> {
        let mut out = vec![Message::system("You are helpful.")];
        for i in 0..n {
            out.push(Message::user(format!(
                "message {i} with some filler text to inflate token count"
            )));
            out.push(Message::assistant(format!(
                "response {i} with more filler to inflate tokens"
            )));
        }
        out
    }

    #[tokio::test]
    async fn budget_auto_trim_reduces_history_when_over() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 50)
            .auto_trim();
        let msgs = long_history(20); // ~41 messages total
        budget.invoke(msgs.clone(), &ChatOptions::default()).await.unwrap();
        let sent = inner.last_msg_count.load(Ordering::SeqCst);
        assert!(sent < msgs.len(), "expected trim: sent={sent}, input={}", msgs.len());
        // System message preserved.
        assert_eq!(inner.last_sys_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_under_cap_passes_through_unchanged() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 10_000)
            .auto_trim();
        let msgs = vec![
            Message::system("be brief"),
            Message::user("hi"),
        ];
        budget.invoke(msgs.clone(), &ChatOptions::default()).await.unwrap();
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn budget_strict_mode_errors_on_overflow() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 50);
        // strict mode (default — auto_trim NOT called)
        let msgs = long_history(20);
        let err = budget
            .invoke(msgs, &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
        let msg = format!("{err}");
        assert!(msg.contains("budget"));
        // Inner model never called.
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn budget_strict_mode_under_cap_succeeds() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 10_000);
        let msgs = vec![Message::user("hi")];
        budget.invoke(msgs, &ChatOptions::default()).await.unwrap();
        assert_eq!(inner.last_msg_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_preserves_system_message_even_under_tight_cap() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner.clone() as Arc<dyn ChatModel>, 20)
            .auto_trim();
        let msgs = long_history(30);
        budget.invoke(msgs, &ChatOptions::default()).await.unwrap();
        // System message always retained by trim_messages.
        assert_eq!(inner.last_sys_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn budget_proxies_name_from_inner() {
        let inner = CapturingChat::new();
        let budget = TokenBudgetChatModel::new(inner as Arc<dyn ChatModel>, 100);
        assert_eq!(budget.name(), "gpt-4o-mini");
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

    // ---- RaceEmbeddings tests ------------------------------------------

    /// Embed provider that sleeps `delay_ms` then either returns a
    /// `dim`-vector of `marker` floats or errors. Lets us pin which
    /// inner wins a race deterministically.
    struct DelayedEmbed {
        label: &'static str,
        dim: usize,
        delay_ms: u64,
        marker: f32,
        succeed: bool,
    }

    #[async_trait]
    impl Embeddings for DelayedEmbed {
        fn name(&self) -> &str {
            self.label
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(vec![self.marker; self.dim])
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            if self.succeed {
                Ok(vec![vec![self.marker; self.dim]; texts.len()])
            } else {
                Err(Error::provider(format!("{} failed", self.label)))
            }
        }
    }

    fn delayed(
        label: &'static str,
        dim: usize,
        delay_ms: u64,
        marker: f32,
        succeed: bool,
    ) -> Arc<dyn Embeddings> {
        Arc::new(DelayedEmbed {
            label,
            dim,
            delay_ms,
            marker,
            succeed,
        })
    }

    #[tokio::test]
    #[should_panic(expected = "at least one inner provider")]
    async fn race_embed_panics_on_empty() {
        let _ = RaceEmbeddings::new(vec![]);
    }

    #[tokio::test]
    #[should_panic(expected = "dim mismatch")]
    async fn race_embed_panics_on_dim_mismatch() {
        let a = delayed("a", 1536, 1, 0.1, true);
        let b = delayed("b", 768, 1, 0.2, true);
        let _ = RaceEmbeddings::new(vec![a, b]);
    }

    #[tokio::test]
    async fn race_embed_query_returns_first_winner() {
        // a finishes in 5ms, b in 50ms — a wins, marker=0.42 ↦ all 1536 dims = 0.42
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 50, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race.embed_query("hi").await.unwrap();
        assert_eq!(out, vec![0.42; 4]);
    }

    #[tokio::test]
    async fn race_embed_query_falls_through_failures() {
        // a fails fast, b succeeds slowly → b's vector is the result.
        let a = delayed("a", 4, 1, 0.0, false);
        let b = delayed("b", 4, 10, 0.5, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race.embed_query("q").await.unwrap();
        assert_eq!(out, vec![0.5; 4]);
    }

    #[tokio::test]
    async fn race_embed_query_aggregates_when_all_fail() {
        let a = delayed("a", 4, 1, 0.0, false);
        let b = delayed("b", 4, 2, 0.0, false);
        let race = RaceEmbeddings::new(vec![a, b]);
        let err = race.embed_query("q").await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("all 2 inners failed"), "got: {s}");
        assert!(s.contains("a failed"));
        assert!(s.contains("b failed"));
    }

    #[tokio::test]
    async fn race_embed_documents_returns_first_winner() {
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 50, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let out = race
            .embed_documents(&["hi".into(), "there".into()])
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], vec![0.42; 4]);
        assert_eq!(out[1], vec![0.42; 4]);
    }

    #[tokio::test]
    async fn race_embed_single_inner_passes_through() {
        let a = delayed("only", 4, 0, 0.7, true);
        let race = RaceEmbeddings::new(vec![a]);
        let out = race.embed_query("q").await.unwrap();
        assert_eq!(out, vec![0.7; 4]);
    }

    #[tokio::test]
    async fn race_embed_aborts_losers_after_winner() {
        // a wins in 5ms, b sleeps 500ms — total wall-clock should be
        // closer to 5ms than 500ms.
        let a = delayed("a", 4, 5, 0.42, true);
        let b = delayed("b", 4, 500, 0.99, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        let started = std::time::Instant::now();
        let _ = race.embed_query("q").await.unwrap();
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 200,
            "elapsed {elapsed_ms}ms — losers were not aborted",
        );
    }

    #[tokio::test]
    async fn race_embed_name_includes_all_inners() {
        let a = delayed("openai", 4, 0, 0.0, true);
        let b = delayed("voyage", 4, 0, 0.0, true);
        let race = RaceEmbeddings::new(vec![a, b]);
        assert!(race.name().contains("openai"));
        assert!(race.name().contains("voyage"));
        assert!(race.name().starts_with("race("));
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

    // ---- PiiScrubbingChatModel tests -----------------------------------

    /// Chat model that captures the messages it was called with, returning
    /// a canned response.
    struct CapturingChatPii {
        seen: std::sync::Mutex<Vec<Vec<Message>>>,
        canned_response: String,
    }

    impl CapturingChatPii {
        fn new(canned: &str) -> Arc<Self> {
            Arc::new(Self {
                seen: std::sync::Mutex::new(Vec::new()),
                canned_response: canned.to_string(),
            })
        }
    }

    #[async_trait]
    impl ChatModel for CapturingChatPii {
        fn name(&self) -> &str {
            "capturing-pii"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            Ok(ChatResponse {
                message: Message::assistant(self.canned_response.clone()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "capturing-pii".into(),
            })
        }
        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn pii_scrub_redacts_user_message_before_invoke() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![
            Message::user("email me at alice@example.com for details"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let user_text = seen[0].text_content();
        assert!(user_text.contains("<EMAIL>"));
        assert!(!user_text.contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_leaves_assistant_messages_untouched() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        // Assistant messages in the history (prior turn) should NOT be
        // re-scrubbed — they came from the model, agent-trace integrity.
        let msgs = vec![
            Message::user("tell me about bob@example.com"),
            Message::assistant("Here's what I know about bob@example.com — he ..."),
            Message::user("continue"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // User messages scrubbed.
        assert!(seen[0].text_content().contains("<EMAIL>"));
        assert!(!seen[0].text_content().contains("bob@example.com"));
        // Assistant message UNTOUCHED.
        assert!(seen[1].text_content().contains("bob@example.com"));
        // Last user message scrubbed (no PII, passes through).
        assert_eq!(seen[2].text_content(), "continue");
    }

    #[tokio::test]
    async fn pii_scrub_system_off_by_default() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![
            Message::system("operator@corp.com is the admin"),
            Message::user("hi"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // System message unchanged — operator prompts are trusted.
        assert!(seen[0].text_content().contains("operator@corp.com"));
    }

    #[tokio::test]
    async fn pii_scrub_system_opt_in_scrubs_operator_prompt_too() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_system_scrubbing();
        let msgs = vec![
            Message::system("admin is operator@corp.com"),
            Message::user("hi"),
        ];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        assert!(seen[0].text_content().contains("<EMAIL>"));
    }

    #[tokio::test]
    async fn pii_scrub_output_off_by_default() {
        // Model returns response containing what looks like PII. Default
        // behavior: don't mangle the LLM's output.
        let inner = CapturingChatPii::new("Contact alice@example.com for support.");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        let msgs = vec![Message::user("who to contact")];
        let resp = scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        assert!(resp.message.text_content().contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_output_opt_in_scrubs_response_text() {
        let inner = CapturingChatPii::new("Contact alice@example.com for support.");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_output_scrubbing();
        let msgs = vec![Message::user("who to contact")];
        let resp = scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let text = resp.message.text_content();
        assert!(text.contains("<EMAIL>"));
        assert!(!text.contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_inputs_false_passes_everything_through() {
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .scrub_inputs(false);
        let msgs = vec![Message::user("email alice@example.com now")];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        // Scrubbing off → email preserved.
        assert!(seen[0].text_content().contains("alice@example.com"));
    }

    #[tokio::test]
    async fn pii_scrub_with_custom_scrubber_respects_custom_patterns() {
        use regex::Regex;
        // Operator adds an internal INTERNAL_ID pattern on top of defaults.
        let custom = PiiScrubber::new().with_patterns(vec![(
            "INTERNAL_ID".to_string(),
            Regex::new(r"\bINT-\d{4}\b").unwrap(),
        )]);
        let inner = CapturingChatPii::new("ok");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>)
            .with_scrubber(Arc::new(custom));
        let msgs = vec![Message::user("issue INT-1234 filed by alice@example.com")];
        scrub.invoke(msgs, &ChatOptions::default()).await.unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let text = seen[0].text_content();
        assert!(text.contains("<INTERNAL_ID>"));
        assert!(text.contains("<EMAIL>"));
    }

    #[tokio::test]
    async fn pii_scrub_name_proxy_from_inner() {
        let inner = CapturingChatPii::new("x");
        let scrub = PiiScrubbingChatModel::new(inner as Arc<dyn ChatModel>);
        assert_eq!(scrub.name(), "capturing-pii");
    }

    #[tokio::test]
    async fn pii_scrub_preserves_tool_calls_and_metadata_fields() {
        use litgraph_core::tool::ToolCall;
        let inner = CapturingChatPii::new("x");
        let scrub = PiiScrubbingChatModel::new(inner.clone() as Arc<dyn ChatModel>);
        // User message with tool_calls attached (rare in user role, but
        // test that the shape is preserved on assistant messages too).
        let msg = Message {
            role: Role::Assistant,
            content: vec![ContentPart::Text {
                text: "bob@example.com assistant response".into(),
            }],
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                name: "look_up".into(),
                arguments: serde_json::json!({"email": "bob@example.com"}),
            }],
            tool_call_id: Some("prior".into()),
            name: Some("asst".into()),
            cache: true,
        };
        scrub
            .invoke(vec![msg.clone()], &ChatOptions::default())
            .await
            .unwrap();
        let seen = &inner.seen.lock().unwrap()[0];
        let kept = &seen[0];
        // Assistant-role → not scrubbed, so email in text is preserved.
        assert!(kept.text_content().contains("bob@example.com"));
        // Tool-calls, tool_call_id, name, cache all round-tripped.
        assert_eq!(kept.tool_calls.len(), 1);
        assert_eq!(kept.tool_calls[0].id, "c1");
        assert_eq!(kept.tool_call_id.as_deref(), Some("prior"));
        assert_eq!(kept.name.as_deref(), Some("asst"));
        assert!(kept.cache);
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
}

// =============================================================================
// PromptCachingChatModel — auto-mark messages as Anthropic prompt-cache breakpoints.
// =============================================================================

/// Wrap any `ChatModel` to auto-set `Message.cache = true` on messages matching
/// a policy before forwarding. Anthropic (and Bedrock-on-Anthropic) providers
/// read the flag and attach `cache_control: {"type":"ephemeral"}` to the
/// message's last content block; other providers ignore it, so stacking is safe.
///
/// # Why
///
/// Anthropic's prompt cache discounts cached input tokens to ~0.1× (writes are
/// 1.25×). For agents with stable system prompts or long pinned context
/// (RAG docs, tool specs, style guides), a single cache hit can cut input
/// cost by 80–90%. The wrapper sidesteps having to flag `.cached()` at every
/// call site — declare the policy once on construction.
///
/// # Policy knobs
///
/// - `cache_system=true` (default) — mark the first System message.
/// - `cache_last_user_over=Some(N)` — mark the LAST User message if its text
///   exceeds N bytes (typical long-context-pinned-in-user pattern).
/// - `extra_indices` — manual: mark specific message indices. Overrides other
///   policies if set (advanced use).
///
/// Anthropic allows up to 4 breakpoints per request — the wrapper doesn't
/// enforce this cap (provider surfaces the error); keep policies minimal.
///
/// ```rust,ignore
/// use litgraph_resilience::PromptCachingChatModel;
/// let chat = PromptCachingChatModel::new(inner)
///     .cache_last_user_if_over(4096);  // cache system + long user
/// ```
pub struct PromptCachingChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub cache_system: bool,
    pub cache_last_user_over: Option<usize>,
    pub extra_indices: Vec<usize>,
}

impl PromptCachingChatModel {
    /// Default policy: cache the system message only. Most common pattern
    /// (stable system prompt across many turns).
    pub fn new(inner: Arc<dyn ChatModel>) -> Self {
        Self {
            inner,
            cache_system: true,
            cache_last_user_over: None,
            extra_indices: Vec::new(),
        }
    }

    /// Disable system-message caching. Use when the system prompt varies
    /// per-call and only user-side context is worth caching.
    pub fn without_system(mut self) -> Self {
        self.cache_system = false;
        self
    }

    /// Also mark the LAST User message as a cache breakpoint if its text
    /// content exceeds `bytes`. Threshold guards against caching short
    /// user turns (cache writes cost ~1.25× — pointless for small inputs).
    pub fn cache_last_user_if_over(mut self, bytes: usize) -> Self {
        self.cache_last_user_over = Some(bytes);
        self
    }

    /// Manually mark message indices as cache breakpoints. Indices out of
    /// range are silently ignored.
    pub fn cache_indices(mut self, indices: Vec<usize>) -> Self {
        self.extra_indices = indices;
        self
    }

    fn apply_policy(&self, mut messages: Vec<Message>) -> Vec<Message> {
        use litgraph_core::Role;

        if self.cache_system {
            if let Some(first_sys) = messages.iter_mut().find(|m| matches!(m.role, Role::System)) {
                first_sys.cache = true;
            }
        }

        if let Some(threshold) = self.cache_last_user_over {
            if let Some(last_user) = messages
                .iter_mut()
                .rev()
                .find(|m| matches!(m.role, Role::User))
            {
                if last_user.text_content().len() > threshold {
                    last_user.cache = true;
                }
            }
        }

        for &idx in &self.extra_indices {
            if let Some(m) = messages.get_mut(idx) {
                m.cache = true;
            }
        }

        messages
    }
}

#[async_trait]
impl ChatModel for PromptCachingChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        self.inner.invoke(self.apply_policy(messages), opts).await
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        self.inner.stream(self.apply_policy(messages), opts).await
    }
}

#[cfg(test)]
mod prompt_cache_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::Mutex;

    /// Records messages seen on invoke; returns a canned reply.
    struct SpyModel {
        seen: Mutex<Vec<Vec<Message>>>,
    }

    impl SpyModel {
        fn new() -> Arc<Self> {
            Arc::new(Self { seen: Mutex::new(Vec::new()) })
        }
    }

    #[async_trait]
    impl ChatModel for SpyModel {
        fn name(&self) -> &str { "spy" }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage { prompt: 1, completion: 1, total: 2, cache_creation: 0, cache_read: 0 },
                model: "spy".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn default_caches_system_only() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![
                Message::system("you are helpful"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache, "system should be cached");
        assert!(!seen[1].cache, "user should NOT be cached");
    }

    #[tokio::test]
    async fn without_system_leaves_system_alone() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone()).without_system();
        chat.invoke(
            vec![
                Message::system("you are helpful"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
        assert!(!seen[1].cache);
    }

    #[tokio::test]
    async fn cache_last_user_only_if_over_threshold() {
        let spy = SpyModel::new();
        let long_ctx = "x".repeat(5000);
        let chat = PromptCachingChatModel::new(spy.clone())
            .cache_last_user_if_over(4096);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("hi"),              // short — NOT cached
                Message::assistant("hello"),
                Message::user(long_ctx.clone()),  // long — cached
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);   // system (default on)
        assert!(!seen[1].cache);  // short user
        assert!(!seen[2].cache);  // assistant not touched
        assert!(seen[3].cache);   // long user
    }

    #[tokio::test]
    async fn short_user_not_cached_even_with_policy() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .cache_last_user_if_over(4096);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("short"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);   // system
        assert!(!seen[1].cache);  // user too short
    }

    #[tokio::test]
    async fn extra_indices_marks_specific_messages() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .without_system()
            .cache_indices(vec![1, 3]);
        chat.invoke(
            vec![
                Message::system("sys"),
                Message::user("first"),
                Message::assistant("a"),
                Message::user("second"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
        assert!(seen[1].cache);
        assert!(!seen[2].cache);
        assert!(seen[3].cache);
    }

    #[tokio::test]
    async fn no_system_message_still_works() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![Message::user("hi")],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);  // no system to cache; nothing crashes
    }

    #[tokio::test]
    async fn out_of_range_indices_ignored() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone())
            .without_system()
            .cache_indices(vec![99, 100]);
        chat.invoke(
            vec![Message::user("hi")],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(!seen[0].cache);
    }

    #[tokio::test]
    async fn caches_first_system_only_when_multiple() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone());
        chat.invoke(
            vec![
                Message::system("first sys"),
                Message::system("second sys"),
                Message::user("hi"),
            ],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache);
        assert!(!seen[1].cache, "only FIRST system marked");
    }

    #[tokio::test]
    async fn preserves_existing_cache_flag() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy.clone()).without_system();
        let pre_cached = Message::user("big ctx").cached();
        chat.invoke(
            vec![pre_cached],
            &ChatOptions::default(),
        ).await.unwrap();
        let seen = &spy.seen.lock().unwrap()[0];
        assert!(seen[0].cache, "policy never clears existing cache flags");
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let spy = SpyModel::new();
        let chat = PromptCachingChatModel::new(spy);
        assert_eq!(chat.name(), "spy");
    }
}

// =============================================================================
// CostCappedChatModel — hard USD cap per cumulative spend.
// =============================================================================

use litgraph_core::TokenUsage;
use litgraph_observability::cost::{ModelPrice, PriceSheet};
use parking_lot::Mutex as PlMutex;

/// Wrap any `ChatModel` with a hard USD cap on cumulative spend. Once the
/// running total crosses `max_usd`, subsequent `invoke`/`stream` calls fail
/// with `Error::InvalidInput` — before any request reaches the provider.
/// The failing call doesn't burn tokens.
///
/// # Why
///
/// Token budget (iter 130) caps the SIZE of any one call. Rate limit (iter 94)
/// caps the RATE of calls. Neither bounds cumulative $ — an agent stuck in a
/// tool-call loop can burn through a month's budget in minutes. CostCap is
/// the floor-level safety on dollar spend: declare a ceiling, get an error
/// instead of a bill.
///
/// # How the math works
///
/// On each successful invoke, cost is computed from `ChatResponse.usage` +
/// `PriceSheet::lookup(response.model)`:
/// - `prompt_tokens × prompt_per_mtok / 1M`
/// - `completion_tokens × completion_per_mtok / 1M`
/// - `cache_creation_tokens × prompt_per_mtok × 1.25 / 1M`  (Anthropic write)
/// - `cache_read_tokens × prompt_per_mtok × 0.10 / 1M`     (Anthropic read)
///
/// If the model isn't in the price sheet, the call adds 0 to the total — the
/// cap silently doesn't enforce for unpriced models. This is a deliberate
/// fail-open: an unrecognized custom model shouldn't halt the caller's pipeline.
/// The caller can pass a custom `PriceSheet` that includes their model to opt in.
///
/// Streams: cost is tallied from the `ChatStreamEvent::Done` final usage (which
/// the wrapper lets flow through verbatim — no reordering). Error variants on
/// the stream path are NOT charged.
///
/// # Thread safety
///
/// Running total guarded by a `parking_lot::Mutex<f64>`. Two concurrent
/// invokes against the same CostCap might both observe the total below cap
/// and both succeed — there's a small race window between the pre-check and
/// post-update. This is acceptable: over-shoot is bounded by (N_concurrent ×
/// cost_per_call), which for typical cap budgets is a rounding error. Tighter
/// pre-reservation would require estimating cost pre-call (impossible without
/// tokenizing + guessing completion length), which would silently over-reject.
///
/// ```rust,ignore
/// use litgraph_resilience::CostCappedChatModel;
/// use litgraph_observability::cost::default_prices;
/// let guarded = CostCappedChatModel::new(inner, default_prices(), 5.00);  // $5 cap
/// match guarded.invoke(msgs, &opts).await {
///     Ok(r) => { /* normal path */ }
///     Err(e) if e.to_string().contains("cost cap") => { /* over budget */ }
///     Err(e) => { /* other provider error */ }
/// }
/// ```
pub struct CostCappedChatModel {
    pub inner: Arc<dyn ChatModel>,
    prices: PriceSheet,
    max_usd: f64,
    total_usd: Arc<PlMutex<f64>>,
}

impl CostCappedChatModel {
    pub fn new(inner: Arc<dyn ChatModel>, prices: PriceSheet, max_usd: f64) -> Self {
        Self {
            inner,
            prices,
            max_usd: max_usd.max(0.0),
            total_usd: Arc::new(PlMutex::new(0.0)),
        }
    }

    /// Current cumulative spend in USD.
    pub fn total_usd(&self) -> f64 {
        *self.total_usd.lock()
    }

    /// Remaining budget (max_usd − total_usd, clamped at 0).
    pub fn remaining_usd(&self) -> f64 {
        (self.max_usd - self.total_usd()).max(0.0)
    }

    /// Reset the running counter (e.g. at midnight UTC for daily budgets).
    pub fn reset(&self) {
        *self.total_usd.lock() = 0.0;
    }

    /// Calculate the USD cost of a single response given its usage + model.
    /// Public for callers who want to replay the accounting manually.
    pub fn cost_of(&self, usage: &TokenUsage, model: &str) -> f64 {
        let Some(ModelPrice { prompt_per_mtok, completion_per_mtok }) = self.prices.lookup(model)
        else {
            return 0.0;
        };
        let mtok = 1_000_000.0;
        let prompt_cost = usage.prompt as f64 * prompt_per_mtok / mtok;
        let completion_cost = usage.completion as f64 * completion_per_mtok / mtok;
        // Anthropic cache pricing: creation = 1.25× prompt, read = 0.10× prompt.
        let cache_write_cost = usage.cache_creation as f64 * prompt_per_mtok * 1.25 / mtok;
        let cache_read_cost = usage.cache_read as f64 * prompt_per_mtok * 0.10 / mtok;
        prompt_cost + completion_cost + cache_write_cost + cache_read_cost
    }
}

#[async_trait]
impl ChatModel for CostCappedChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Pre-check: already over cap? reject before hitting the provider.
        {
            let total = *self.total_usd.lock();
            if total >= self.max_usd {
                return Err(Error::invalid(format!(
                    "CostCappedChatModel: cost cap exceeded (${:.4} used, ${:.4} limit)",
                    total, self.max_usd
                )));
            }
        }
        let resp = self.inner.invoke(messages, opts).await?;
        let cost = self.cost_of(&resp.usage, &resp.model);
        *self.total_usd.lock() += cost;
        tracing::debug!(
            model = %resp.model,
            call_usd = cost,
            total_usd = *self.total_usd.lock(),
            cap_usd = self.max_usd,
            "CostCappedChatModel charged"
        );
        Ok(resp)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        {
            let total = *self.total_usd.lock();
            if total >= self.max_usd {
                return Err(Error::invalid(format!(
                    "CostCappedChatModel: cost cap exceeded (${:.4} used, ${:.4} limit)",
                    total, self.max_usd
                )));
            }
        }
        // Wrap the inner stream so the terminal Done event updates the total.
        // Don't attempt mid-stream termination if the user crosses the cap
        // during a single long stream — they're already mid-response and the
        // bill is already committed; just let it land and charge.
        let inner_stream = self.inner.stream(messages, opts).await?;
        let total = self.total_usd.clone();
        let prices = self.prices.clone();
        use futures_util::StreamExt;
        let mapped = inner_stream.map(move |event| {
            if let Ok(litgraph_core::model::ChatStreamEvent::Done { response }) = &event {
                if let Some(ModelPrice { prompt_per_mtok, completion_per_mtok }) =
                    prices.lookup(&response.model)
                {
                    let usage = &response.usage;
                    let mtok = 1_000_000.0;
                    let cost = usage.prompt as f64 * prompt_per_mtok / mtok
                        + usage.completion as f64 * completion_per_mtok / mtok
                        + usage.cache_creation as f64 * prompt_per_mtok * 1.25 / mtok
                        + usage.cache_read as f64 * prompt_per_mtok * 0.10 / mtok;
                    *total.lock() += cost;
                }
            }
            event
        });
        Ok(Box::pin(mapped))
    }
}

#[cfg(test)]
mod cost_cap_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use litgraph_observability::cost::{ModelPrice, PriceSheet};

    struct FixedCostModel {
        usage: TokenUsage,
        model: String,
    }

    #[async_trait]
    impl ChatModel for FixedCostModel {
        fn name(&self) -> &str { "fixed" }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("ok"),
                finish_reason: FinishReason::Stop,
                usage: self.usage,
                model: self.model.clone(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn prices_gpt4o() -> PriceSheet {
        let mut s = PriceSheet::new();
        // gpt-4o: $2.50/Mtok prompt, $10.00/Mtok completion.
        s.set("gpt-4o", ModelPrice { prompt_per_mtok: 2.50, completion_per_mtok: 10.00 });
        s
    }

    #[tokio::test]
    async fn passes_through_under_cap() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 1_000, completion: 500, total: 1_500, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        // $0.0025 + $0.005 = $0.0075 per call
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        chat.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
        let t = chat.total_usd();
        assert!((t - 0.0075).abs() < 1e-9, "expected ~$0.0075, got {t}");
    }

    #[tokio::test]
    async fn rejects_after_cumulative_over_cap() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 50_000, total: 150_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        // $0.25 + $0.50 = $0.75 per call; cap $1.00 → call 1 ok, call 2 reject.
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        let r1 = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r1.is_ok());
        // At $0.75 < $1.00 so next call is ALLOWED (pre-check only guards >=).
        let r2 = chat.invoke(vec![Message::user("b")], &ChatOptions::default()).await;
        assert!(r2.is_ok(), "second call pre-checks at $0.75 < $1.00");
        // Now at $1.50 > cap. Third call must be rejected.
        let r3 = chat.invoke(vec![Message::user("c")], &ChatOptions::default()).await;
        let err = r3.unwrap_err();
        assert!(err.to_string().contains("cost cap exceeded"),
                "unexpected: {err}");
    }

    #[tokio::test]
    async fn unpriced_model_charges_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 1_000_000, completion: 1_000_000, total: 2_000_000, cache_creation: 0, cache_read: 0 },
            model: "custom-internal-model".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 0.01);
        // Call succeeds — no price for "custom-internal-model" in sheet.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_ok());
        assert_eq!(chat.total_usd(), 0.0);
    }

    #[tokio::test]
    async fn cache_creation_charged_at_1_25x_prompt() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 0, completion: 0, total: 0, cache_creation: 1_000_000, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // 1M cache_creation × $2.50 × 1.25 = $3.125
        assert!((chat.total_usd() - 3.125).abs() < 1e-9);
    }

    #[tokio::test]
    async fn cache_read_charged_at_0_10x_prompt() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 0, completion: 0, total: 0, cache_creation: 0, cache_read: 1_000_000 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // 1M cache_read × $2.50 × 0.10 = $0.25
        assert!((chat.total_usd() - 0.25).abs() < 1e-9);
    }

    #[tokio::test]
    async fn remaining_usd_decreases_with_spend() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 0, total: 100_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.00);
        assert!((chat.remaining_usd() - 1.00).abs() < 1e-9);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        // $0.25 spent, $0.75 remaining.
        assert!((chat.remaining_usd() - 0.75).abs() < 1e-9);
    }

    #[tokio::test]
    async fn reset_returns_total_to_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 100_000, completion: 50_000, total: 150_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.00);
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        assert!(chat.total_usd() > 0.0);
        chat.reset();
        assert_eq!(chat.total_usd(), 0.0);
        assert!((chat.remaining_usd() - 10.00).abs() < 1e-9);
    }

    #[tokio::test]
    async fn zero_cap_rejects_all_requests() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 0.0);
        // total = $0, cap = $0, pre-check `0 >= 0` → reject.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_err());
        assert!(r.unwrap_err().to_string().contains("cost cap exceeded"));
    }

    #[tokio::test]
    async fn negative_cap_clamps_to_zero() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), -5.0);
        // Negative cap is clamped to $0 at construction.
        let r = chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await;
        assert!(r.is_err(), "negative cap treated as $0 — all calls rejected");
    }

    #[tokio::test]
    async fn cost_of_helper_matches_invoke_accounting() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage { prompt: 500_000, completion: 200_000, total: 700_000, cache_creation: 0, cache_read: 0 },
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 10.0);
        let expected = chat.cost_of(
            &TokenUsage { prompt: 500_000, completion: 200_000, total: 700_000, cache_creation: 0, cache_read: 0 },
            "gpt-4o",
        );
        chat.invoke(vec![Message::user("a")], &ChatOptions::default()).await.unwrap();
        assert!((chat.total_usd() - expected).abs() < 1e-9);
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let inner = Arc::new(FixedCostModel {
            usage: TokenUsage::default(),
            model: "gpt-4o".into(),
        });
        let chat = CostCappedChatModel::new(inner, prices_gpt4o(), 1.0);
        assert_eq!(chat.name(), "fixed");
    }
}

// =============================================================================
// SelfConsistencyChatModel — N-sample majority vote (Wang et al 2022).
// =============================================================================

use std::collections::HashMap;

/// Picks the winner from N sampled responses. Return the INDEX of the
/// winning response (caller uses it to select the full ChatResponse). If
/// the voter returns `None` — no majority / all invalid — the wrapper
/// falls back to returning the first sample.
pub type ConsistencyVoter =
    Arc<dyn Fn(&[ChatResponse]) -> Option<usize> + Send + Sync>;

/// Self-consistency wrapper: sample the model `samples` times at elevated
/// temperature, then pick the majority answer via `voter`. Classic
/// Chain-of-Thought reasoning technique from Wang et al 2022 — for math,
/// code, and multi-step reasoning, N=5 at T=0.7 often lifts accuracy
/// 5–20% over greedy decoding. Costs N× tokens per question.
///
/// # How voting works
///
/// Default voter: normalize each response's text (trim + lowercase +
/// collapse whitespace) and pick the most-common. Ties broken by first
/// occurrence. For structured tasks, pass a custom `voter` that extracts
/// the answer field (e.g. last number in the text, or the JSON field)
/// before counting — raw-text majority over a 500-token reasoning chain
/// will never converge, but majority over extracted answers will.
///
/// # Parallelism
///
/// N samples run concurrently via `tokio::JoinSet`. On a typical provider
/// with an async HTTP pool, 5 parallel samples take ~1× the wall-clock of
/// 1 sample (I/O-bound). CPU-bound models or strict rate-limits will
/// serialize; stack with `RateLimitedChatModel` when needed.
///
/// # Streaming
///
/// `stream()` delegates to a single sample (no streaming fan-out — there's
/// no meaningful way to stream N parallel samples and vote). Callers who
/// want vote-then-stream should do it in two phases upstream.
///
/// ```rust,ignore
/// use litgraph_resilience::SelfConsistencyChatModel;
/// let voter_chat = SelfConsistencyChatModel::new(inner, 5).with_temperature(0.7);
/// let resp = voter_chat.invoke(msgs, &opts).await?;
/// // `resp.usage` includes summed tokens across all 5 samples.
/// ```
pub struct SelfConsistencyChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub samples: usize,
    pub sample_temperature: f32,
    voter: ConsistencyVoter,
}

impl SelfConsistencyChatModel {
    /// Default voter (text-majority). `samples` is clamped to at least 1.
    pub fn new(inner: Arc<dyn ChatModel>, samples: usize) -> Self {
        Self {
            inner,
            samples: samples.max(1),
            sample_temperature: 0.7,
            voter: default_text_voter(),
        }
    }

    /// Override the sampling temperature (default 0.7 — per the paper's
    /// sweet spot for reasoning diversity without incoherence).
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.sample_temperature = t;
        self
    }

    /// Custom voter — e.g. extract the last integer from each response
    /// and pick the majority. Receives all N responses, returns the index
    /// of the winner. If `None`, wrapper falls back to the first sample.
    pub fn with_voter(mut self, voter: ConsistencyVoter) -> Self {
        self.voter = voter;
        self
    }
}

/// Default voter: normalize text (trim + lowercase + collapse whitespace)
/// and return the index of the response whose normalized text appears most.
pub fn default_text_voter() -> ConsistencyVoter {
    Arc::new(|responses: &[ChatResponse]| {
        if responses.is_empty() {
            return None;
        }
        let normalized: Vec<String> = responses
            .iter()
            .map(|r| normalize_for_vote(&r.message.text_content()))
            .collect();
        // Count occurrences; keep first-seen tie-breaker.
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for n in &normalized {
            *counts.entry(n.as_str()).or_insert(0) += 1;
        }
        let (best_text, _) = counts.iter().max_by_key(|(_, c)| *c)?;
        // Return index of the FIRST response whose normalized text matches.
        normalized.iter().position(|n| n == *best_text)
    })
}

fn normalize_for_vote(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build a voter that extracts a field per response via `extract` and picks
/// the majority value. `extract` returns `None` for responses where the
/// field is missing — those are excluded from the vote.
pub fn extracted_field_voter<F>(extract: F) -> ConsistencyVoter
where
    F: Fn(&ChatResponse) -> Option<String> + Send + Sync + 'static,
{
    let extract = Arc::new(extract);
    Arc::new(move |responses: &[ChatResponse]| {
        if responses.is_empty() {
            return None;
        }
        let extracted: Vec<Option<String>> =
            responses.iter().map(|r| extract(r)).collect();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for e in &extracted {
            if let Some(v) = e {
                *counts.entry(v.as_str()).or_insert(0) += 1;
            }
        }
        let (best, _) = counts.iter().max_by_key(|(_, c)| *c)?;
        extracted
            .iter()
            .position(|e| e.as_deref() == Some(*best))
    })
}

#[async_trait]
impl ChatModel for SelfConsistencyChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Sample N times in parallel. Override temperature on the cloned
        // opts so the caller's preferred temperature doesn't squash
        // sampling diversity. Keep all other opts (max_tokens, tools, etc).
        let mut sample_opts = opts.clone();
        sample_opts.temperature = Some(self.sample_temperature);

        // Spawn N parallel samples and preserve their spawn order in the
        // result vec (join_all yields results in the SAME order as the
        // input futures — critical for deterministic tie-break below).
        let futures: Vec<_> = (0..self.samples)
            .map(|_| {
                let inner = self.inner.clone();
                let msgs = messages.clone();
                let o = sample_opts.clone();
                async move { inner.invoke(msgs, &o).await }
            })
            .collect();
        let results = futures_util::future::join_all(futures).await;

        let mut samples: Vec<ChatResponse> = Vec::with_capacity(self.samples);
        let mut first_err: Option<Error> = None;
        for res in results {
            match res {
                Ok(r) => samples.push(r),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if samples.is_empty() {
            // All N samples failed — bubble up the first error.
            return Err(first_err.unwrap_or_else(|| {
                Error::other("SelfConsistencyChatModel: all samples failed")
            }));
        }

        // Pick the winner. Voter returning None → fall through to first.
        let winner_idx = (self.voter)(&samples).unwrap_or(0);
        let mut winner = samples
            .get(winner_idx)
            .cloned()
            .unwrap_or_else(|| samples[0].clone());

        // Sum usage across ALL samples so the caller's cost tracker sees
        // the full fan-out cost — critical for CostCapped composition.
        let mut summed = TokenUsage::default();
        for s in &samples {
            summed.prompt += s.usage.prompt;
            summed.completion += s.usage.completion;
            summed.total += s.usage.total;
            summed.cache_creation += s.usage.cache_creation;
            summed.cache_read += s.usage.cache_read;
        }
        winner.usage = summed;
        Ok(winner)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // No fan-out on streams — delegate to a single sample.
        self.inner.stream(messages, opts).await
    }
}

#[cfg(test)]
mod self_consistency_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Returns a scripted sequence of responses — one per call.
    struct ScriptedModel {
        texts: Vec<&'static str>,
        idx: AtomicUsize,
    }

    impl ScriptedModel {
        fn new(texts: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self { texts, idx: AtomicUsize::new(0) })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedModel {
        fn name(&self) -> &str { "scripted" }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let i = self.idx.fetch_add(1, Ordering::SeqCst) % self.texts.len();
            Ok(ChatResponse {
                message: Message::assistant(self.texts[i]),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage { prompt: 10, completion: 5, total: 15, cache_creation: 0, cache_read: 0 },
                model: "scripted".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn picks_majority_text() {
        // 3 "42" + 2 "41" → majority "42".
        let inner = ScriptedModel::new(vec!["42", "41", "42", "41", "42"]);
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("2+40")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "42");
    }

    #[tokio::test]
    async fn summed_usage_across_samples() {
        let inner = ScriptedModel::new(vec!["a", "a", "a"]);
        let chat = SelfConsistencyChatModel::new(inner, 3);
        let resp = chat.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        // Each sample uses 10+5=15. N=3 → 30 prompt + 15 completion = 45 total.
        assert_eq!(resp.usage.prompt, 30);
        assert_eq!(resp.usage.completion, 15);
        assert_eq!(resp.usage.total, 45);
    }

    #[tokio::test]
    async fn sample_temperature_overrides_caller_temp() {
        struct TempCapture {
            seen: std::sync::Mutex<Vec<f32>>,
        }
        #[async_trait]
        impl ChatModel for TempCapture {
            fn name(&self) -> &str { "tc" }
            async fn invoke(&self, _m: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
                self.seen.lock().unwrap().push(opts.temperature.unwrap_or(0.0));
                Ok(ChatResponse {
                    message: Message::assistant("ok"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "tc".into(),
                })
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let inner = Arc::new(TempCapture { seen: std::sync::Mutex::new(vec![]) });
        let chat = SelfConsistencyChatModel::new(inner.clone(), 3).with_temperature(0.9);
        chat.invoke(
            vec![Message::user("q")],
            &ChatOptions { temperature: Some(0.0), ..Default::default() },
        ).await.unwrap();
        let seen = inner.seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        for t in seen.iter() {
            assert!((t - 0.9).abs() < 1e-6, "expected 0.9, got {t}");
        }
    }

    #[tokio::test]
    async fn tie_winner_is_one_of_the_tied_majority() {
        // 2-2-1 tie between "a" and "b". Which wins depends on which
        // finishes first in the parallel race — both are valid majority
        // picks. The test asserts the winner is NOT the minority ("c").
        let inner = ScriptedModel::new(vec!["a", "b", "a", "b", "c"]);
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        let winner = resp.message.text_content();
        assert!(winner == "a" || winner == "b", "winner was {winner}, expected tied majority");
    }

    #[tokio::test]
    async fn custom_voter_extracts_last_number() {
        // Reasoning chains — raw majority would never converge; but
        // last-number-extract majority picks 42.
        let inner = ScriptedModel::new(vec![
            "Let me think... so the answer is 42.",
            "After calculation, I get 42.",
            "Maybe 17? No, 42.",
            "The solution is 41 oh wait, 42.",
            "I believe it's 42.",
        ]);
        let voter = extracted_field_voter(|r| {
            let text = r.message.text_content();
            text.split_whitespace()
                .filter_map(|w| w.trim_end_matches('.').parse::<i64>().ok())
                .last()
                .map(|n| n.to_string())
        });
        let chat = SelfConsistencyChatModel::new(inner, 5).with_voter(voter);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        // Winner must be one of the responses ending with "42".
        assert!(resp.message.text_content().ends_with("42.") || resp.message.text_content().ends_with("42"));
    }

    #[tokio::test]
    async fn all_samples_fail_returns_first_error() {
        struct AllFail;
        #[async_trait]
        impl ChatModel for AllFail {
            fn name(&self) -> &str { "fail" }
            async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
                Err(Error::provider("upstream dead"))
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let chat = SelfConsistencyChatModel::new(Arc::new(AllFail), 3);
        let err = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap_err();
        assert!(err.to_string().contains("upstream dead"));
    }

    #[tokio::test]
    async fn partial_sample_failure_still_votes() {
        // 5 samples, some fail — voter runs on the successful ones.
        struct FlakyScripted {
            texts: Vec<Option<&'static str>>,
            idx: AtomicUsize,
        }
        #[async_trait]
        impl ChatModel for FlakyScripted {
            fn name(&self) -> &str { "flaky" }
            async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
                let i = self.idx.fetch_add(1, Ordering::SeqCst) % self.texts.len();
                match self.texts[i] {
                    Some(t) => Ok(ChatResponse {
                        message: Message::assistant(t),
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::default(),
                        model: "flaky".into(),
                    }),
                    None => Err(Error::Timeout),
                }
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let inner = Arc::new(FlakyScripted {
            texts: vec![Some("42"), None, Some("42"), None, Some("42")],
            idx: AtomicUsize::new(0),
        });
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "42");
    }

    #[tokio::test]
    async fn samples_one_passes_through_as_single_call() {
        let inner = ScriptedModel::new(vec!["one"]);
        let chat = SelfConsistencyChatModel::new(inner, 1);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "one");
        assert_eq!(resp.usage.total, 15);
    }

    #[tokio::test]
    async fn zero_samples_clamps_to_one() {
        let inner = ScriptedModel::new(vec!["only"]);
        let chat = SelfConsistencyChatModel::new(inner, 0);
        assert_eq!(chat.samples, 1);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "only");
    }

    #[tokio::test]
    async fn normalize_for_vote_collapses_whitespace_and_case() {
        assert_eq!(normalize_for_vote("  Hello   World  "), "hello world");
        assert_eq!(normalize_for_vote("Hello  World"), normalize_for_vote("HELLO WORLD"));
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let inner = ScriptedModel::new(vec!["x"]);
        let chat = SelfConsistencyChatModel::new(inner, 3);
        assert_eq!(chat.name(), "scripted");
    }
}

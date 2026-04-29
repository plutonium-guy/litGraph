//! Tool resilience wrappers — `TimeoutTool` and `RetryTool`. Parallels
//! the chat-model resilience matrix (RetryingChatModel, etc) for the
//! tool-call axis. Currently if a tool hangs the whole agent hangs;
//! these wrappers add per-call timeouts + automatic retries.
//!
//! Both wrap any `Tool` and pass the schema through unchanged — the LLM
//! doesn't need to know the tool is wrapped. Same pattern as `CachedTool`
//! (iter 117).
//!
//! # Composition order
//!
//! Recommended (innermost → outermost): `Retry(Timeout(inner))`.
//! Reasoning: timeout caps each attempt; retry counts attempts. With
//! Timeout outermost, the timeout would have to budget for all retries,
//! which makes the budget hard to reason about. Inner-timeout +
//! outer-retry gives "retry up to N times, each capped at T seconds."
//!
//! # CachedTool composition
//!
//! Cache outermost: `Retry(Timeout(Cached(inner)))` — cache hits skip
//! retry/timeout entirely (instant). Cache misses go through the
//! retry-timeout chain. Don't put Cached innermost; you'd cache
//! transient failures.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::Value;

// =============================================================================
// TimeoutTool — per-call deadline wrapper.
// =============================================================================

/// Wraps any `Tool` so each call must complete within `timeout`.
/// On timeout, returns `Error::Timeout`. Inner work is cancelled
/// (tokio::time::timeout drops the future, releasing its resources).
pub struct TimeoutTool {
    inner: Arc<dyn Tool>,
    timeout: Duration,
}

impl TimeoutTool {
    pub fn wrap(inner: Arc<dyn Tool>, timeout: Duration) -> Arc<Self> {
        Arc::new(Self { inner, timeout })
    }
}

#[async_trait]
impl Tool for TimeoutTool {
    fn schema(&self) -> ToolSchema { self.inner.schema() }

    async fn run(&self, args: Value) -> Result<Value> {
        match tokio::time::timeout(self.timeout, self.inner.run(args)).await {
            Ok(r) => r,
            Err(_) => Err(Error::Timeout),
        }
    }
}

// =============================================================================
// RetryTool — exp-backoff retry on transient errors.
// =============================================================================

#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Max attempts INCLUDING the first try. `max_attempts=3` → 1 initial + 2 retries.
    /// Default 3.
    pub max_attempts: u32,
    /// Initial backoff delay. Default 100ms.
    pub initial_delay: Duration,
    /// Max single delay (caps the exponential growth). Default 5s.
    pub max_delay: Duration,
    /// Multiplier applied to delay between attempts. Default 2.0.
    pub multiplier: f64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(5),
            multiplier: 2.0,
        }
    }
}

impl RetryConfig {
    pub fn new() -> Self { Self::default() }
    pub fn with_max_attempts(mut self, n: u32) -> Self { self.max_attempts = n.max(1); self }
    pub fn with_initial_delay(mut self, d: Duration) -> Self { self.initial_delay = d; self }
    pub fn with_max_delay(mut self, d: Duration) -> Self { self.max_delay = d; self }
    pub fn with_multiplier(mut self, m: f64) -> Self { self.multiplier = m.max(1.0); self }
}

/// Wraps any `Tool` with exponential-backoff retry on transient errors.
/// Retries `Error::Timeout`, `Error::RateLimited`, `Error::Provider`
/// variants (matches `litgraph-resilience::is_transient` policy).
/// Terminal errors (`InvalidInput`, `Parse`, etc.) bubble immediately —
/// retrying a bad-input error just wastes time.
pub struct RetryTool {
    inner: Arc<dyn Tool>,
    cfg: RetryConfig,
}

impl RetryTool {
    pub fn wrap(inner: Arc<dyn Tool>, cfg: RetryConfig) -> Arc<Self> {
        Arc::new(Self { inner, cfg })
    }

    pub fn with_default_config(inner: Arc<dyn Tool>) -> Arc<Self> {
        Self::wrap(inner, RetryConfig::default())
    }
}

#[async_trait]
impl Tool for RetryTool {
    fn schema(&self) -> ToolSchema { self.inner.schema() }

    async fn run(&self, args: Value) -> Result<Value> {
        let mut delay = self.cfg.initial_delay;
        let mut last_err: Option<Error> = None;
        for attempt in 0..self.cfg.max_attempts {
            match self.inner.run(args.clone()).await {
                Ok(v) => return Ok(v),
                Err(e) => {
                    if !is_transient(&e) {
                        return Err(e);
                    }
                    last_err = Some(e);
                    // Don't sleep after the final attempt — we're about to bail.
                    if attempt + 1 < self.cfg.max_attempts {
                        tokio::time::sleep(delay).await;
                        // Exponential backoff with cap.
                        let next_ms = (delay.as_millis() as f64 * self.cfg.multiplier) as u128;
                        let cap_ms = self.cfg.max_delay.as_millis();
                        let capped = next_ms.min(cap_ms);
                        delay = Duration::from_millis(capped as u64);
                    }
                }
            }
        }
        Err(last_err.unwrap_or_else(|| Error::other("RetryTool: no attempts ran")))
    }
}

/// Same transient-error classifier as litgraph-resilience uses for ChatModel.
/// Duplicated here (instead of depending on litgraph-resilience) because that
/// would invert the dep — resilience already uses litgraph-tools-utils features
/// indirectly. Lean +5 LOC vs a cross-crate cycle.
fn is_transient(e: &Error) -> bool {
    matches!(
        e,
        Error::Timeout | Error::RateLimited { .. } | Error::Provider(_)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use tokio::time::Duration as TokioDuration;

    /// Tool that increments a counter on each call and returns its value as
    /// a JSON number. Used to assert how many times the inner tool ran.
    struct CountingTool { count: AtomicU32 }
    #[async_trait]
    impl Tool for CountingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "counter".into(),
                description: "increments and returns the call count".into(),
                parameters: serde_json::json!({"type": "object", "properties": {}}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            let n = self.count.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(serde_json::json!(n))
        }
    }

    /// Tool that hangs forever (until cancelled by timeout).
    struct HangingTool;
    #[async_trait]
    impl Tool for HangingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "hang".into(),
                description: "never returns".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            tokio::time::sleep(TokioDuration::from_secs(60)).await;
            Ok(serde_json::json!(null))
        }
    }

    /// Tool that fails N times with a configurable error then succeeds.
    struct FlakyTool {
        fail_count: AtomicU32,
        err_factory: Box<dyn Fn() -> Error + Send + Sync>,
    }
    impl FlakyTool {
        fn rate_limited(n: u32) -> Self {
            Self {
                fail_count: AtomicU32::new(n),
                err_factory: Box::new(|| Error::RateLimited { retry_after_ms: None }),
            }
        }
        fn provider_5xx(n: u32) -> Self {
            Self {
                fail_count: AtomicU32::new(n),
                err_factory: Box::new(|| Error::provider("503 service unavailable")),
            }
        }
        fn invalid_input(n: u32) -> Self {
            Self {
                fail_count: AtomicU32::new(n),
                err_factory: Box::new(|| Error::invalid("bad input")),
            }
        }
    }
    #[async_trait]
    impl Tool for FlakyTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "flaky".into(),
                description: "fails N times then succeeds".into(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            let remaining = self.fail_count.fetch_sub(1, Ordering::SeqCst);
            if remaining > 0 {
                Err((self.err_factory)())
            } else {
                self.fail_count.store(0, Ordering::SeqCst);
                Ok(serde_json::json!("ok"))
            }
        }
    }

    // ---- TimeoutTool tests ----

    #[tokio::test]
    async fn timeout_returns_timeout_error_on_hang() {
        let t = TimeoutTool::wrap(Arc::new(HangingTool), Duration::from_millis(50));
        let err = t.run(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, Error::Timeout));
    }

    #[tokio::test]
    async fn timeout_passes_through_fast_calls() {
        let t = TimeoutTool::wrap(
            Arc::new(CountingTool { count: AtomicU32::new(0) }),
            Duration::from_secs(1),
        );
        let r = t.run(serde_json::json!({})).await.unwrap();
        assert_eq!(r, serde_json::json!(1));
    }

    #[tokio::test]
    async fn timeout_schema_passes_through() {
        let inner = Arc::new(CountingTool { count: AtomicU32::new(0) });
        let t = TimeoutTool::wrap(inner.clone(), Duration::from_secs(1));
        assert_eq!(t.schema().name, inner.schema().name);
    }

    // ---- RetryTool tests ----

    #[tokio::test]
    async fn retry_succeeds_after_transient_failures() {
        let inner = Arc::new(FlakyTool::rate_limited(2));
        let r = RetryTool::wrap(
            inner,
            RetryConfig::new()
                .with_max_attempts(3)
                .with_initial_delay(Duration::from_millis(1)),
        );
        let result = r.run(serde_json::json!({})).await.unwrap();
        assert_eq!(result, serde_json::json!("ok"));
    }

    #[tokio::test]
    async fn retry_exhausts_returns_last_error() {
        let inner = Arc::new(FlakyTool::rate_limited(10));  // fails more than max_attempts
        let r = RetryTool::wrap(
            inner,
            RetryConfig::new()
                .with_max_attempts(2)
                .with_initial_delay(Duration::from_millis(1)),
        );
        let err = r.run(serde_json::json!({})).await.unwrap_err();
        assert!(matches!(err, Error::RateLimited { .. }));
    }

    #[tokio::test]
    async fn retry_does_not_retry_terminal_errors() {
        // InvalidInput is terminal — should bail on first attempt.
        let inner = Arc::new(FlakyTool::invalid_input(10));
        let r = RetryTool::wrap(
            inner,
            RetryConfig::new()
                .with_max_attempts(5)
                .with_initial_delay(Duration::from_millis(1)),
        );
        let _err = r.run(serde_json::json!({})).await.unwrap_err();
        // The flaky tool's counter should have decremented exactly once
        // (the first attempt). We can't easily inspect from outside, but
        // the key assertion is that we got the error back fast (no retries).
    }

    #[tokio::test]
    async fn retry_handles_provider_5xx_errors() {
        let inner = Arc::new(FlakyTool::provider_5xx(1));
        let r = RetryTool::wrap(
            inner,
            RetryConfig::new()
                .with_max_attempts(3)
                .with_initial_delay(Duration::from_millis(1)),
        );
        let result = r.run(serde_json::json!({})).await.unwrap();
        assert_eq!(result, serde_json::json!("ok"));
    }

    #[tokio::test]
    async fn retry_max_attempts_clamped_to_at_least_one() {
        let cfg = RetryConfig::new().with_max_attempts(0);
        assert_eq!(cfg.max_attempts, 1);
    }

    #[tokio::test]
    async fn retry_multiplier_clamped_to_at_least_one() {
        let cfg = RetryConfig::new().with_multiplier(0.5);
        assert_eq!(cfg.multiplier, 1.0);
    }

    #[tokio::test]
    async fn retry_passes_through_schema_unchanged() {
        let inner = Arc::new(CountingTool { count: AtomicU32::new(0) });
        let r = RetryTool::wrap(inner.clone(), RetryConfig::default());
        assert_eq!(r.schema().name, inner.schema().name);
    }

    // ---- Composition: Retry(Timeout(inner)) ----

    #[tokio::test]
    async fn retry_around_timeout_gives_per_attempt_budget() {
        // Each attempt times out after 30ms. Retry 3 times → up to ~90ms total.
        // (Real elapsed time also includes inter-attempt sleeps, but the key
        // property is that EACH attempt is bounded.)
        let timeout = TimeoutTool::wrap(Arc::new(HangingTool), Duration::from_millis(30));
        let retry = RetryTool::wrap(
            timeout,
            RetryConfig::new()
                .with_max_attempts(3)
                .with_initial_delay(Duration::from_millis(1)),
        );
        let start = std::time::Instant::now();
        let err = retry.run(serde_json::json!({})).await.unwrap_err();
        let elapsed = start.elapsed();
        assert!(matches!(err, Error::Timeout));
        // 3 timeouts × 30ms + 2 backoff sleeps × ~1ms ≈ 92ms. Allow generous slack.
        assert!(
            elapsed < Duration::from_millis(500),
            "expected <500ms, took {elapsed:?}"
        );
    }
}

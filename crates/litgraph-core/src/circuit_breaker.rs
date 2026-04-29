//! `CircuitBreaker` — fail-fast wrapper around flaky upstreams.
//!
//! Three-state machine:
//!
//! | State              | Behavior                                                |
//! |--------------------|---------------------------------------------------------|
//! | `Closed`           | Calls go through. Track consecutive failures.           |
//! | `Open`             | Reject calls immediately with `CallError::CircuitOpen`. |
//! | `HalfOpenProbing`  | Allow exactly one in-flight probe; reject the rest.     |
//!
//! Transitions:
//!
//! - `Closed → Open` when `failure_threshold` consecutive failures
//!   are recorded.
//! - `Open → HalfOpenProbing` when a call arrives after the
//!   cooldown elapses.
//! - `HalfOpenProbing → Closed` if the probe succeeds.
//! - `HalfOpenProbing → Open` if the probe fails (resets
//!   cooldown).
//!
//! # Why a separate primitive
//!
//! `RetryingChatModel` retries on individual call errors. That
//! works when failures are transient. When an upstream is *down*,
//! retrying just amplifies load against a sick service and
//! delays recovery. A circuit breaker stops the bleeding: after
//! N consecutive failures, every subsequent call fails fast for
//! `cooldown`, giving the upstream room to heal.
//!
//! Composable. Wrap any `Future`-returning closure (chat invoke,
//! tool call, retriever, embedder) in `breaker.call(|| f()).await`.
//!
//! # Real prod use
//!
//! - **Provider outage**: third-party LLM provider returns 503 in
//!   bursts. After 5 consecutive failures, the breaker opens for
//!   30s; agents fail fast (and fall back to a secondary
//!   provider via `FallbackChatModel`) instead of spending 30s
//!   retrying each call.
//! - **Vector store quarantine**: pgvector connection pool
//!   exhausted. Open the breaker, route reads to an HNSW
//!   in-memory replica until pgvector recovers.
//! - **Tool blast-radius limiter**: external API tool flapping.
//!   Open the breaker so the agent reasons about an unavailable
//!   tool rather than waiting on timeouts.

use std::fmt;
use std::future::Future;
use std::time::{Duration, Instant};

use parking_lot::Mutex as PlMutex;

/// Wrapping error from [`CircuitBreaker::call`].
#[derive(Debug)]
pub enum CallError<E> {
    /// The breaker is open; the inner closure was NOT invoked.
    CircuitOpen,
    /// The inner closure returned `Err(_)`.
    Inner(E),
}

impl<E: fmt::Display> fmt::Display for CallError<E> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CallError::CircuitOpen => write!(f, "circuit breaker open"),
            CallError::Inner(e) => write!(f, "{e}"),
        }
    }
}

impl<E: std::error::Error + 'static> std::error::Error for CallError<E> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CallError::CircuitOpen => None,
            CallError::Inner(e) => Some(e),
        }
    }
}

/// Public read-only view of the breaker state. Useful for
/// telemetry / dashboards. Returned by [`CircuitBreaker::state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CircuitState {
    Closed,
    Open,
    HalfOpenProbing,
}

#[derive(Debug, Clone, Copy)]
enum InnerState {
    Closed { consecutive_failures: usize },
    Open { until: Instant },
    HalfOpenProbing,
}

/// Async circuit breaker. Cheap to clone (`Arc` semantics through
/// `Arc<CircuitBreaker>` — the type itself is intended to be
/// wrapped in an `Arc` if shared across tasks).
pub struct CircuitBreaker {
    failure_threshold: usize,
    cooldown: Duration,
    state: PlMutex<InnerState>,
}

impl CircuitBreaker {
    /// Construct a closed breaker. `failure_threshold = 0` is
    /// normalised to 1 (one failure trips it).
    pub fn new(failure_threshold: usize, cooldown: Duration) -> Self {
        Self {
            failure_threshold: failure_threshold.max(1),
            cooldown,
            state: PlMutex::new(InnerState::Closed {
                consecutive_failures: 0,
            }),
        }
    }

    /// Snapshot the current state. Race-prone for decision-making;
    /// use only for telemetry.
    pub fn state(&self) -> CircuitState {
        match *self.state.lock() {
            InnerState::Closed { .. } => CircuitState::Closed,
            InnerState::Open { until } if Instant::now() < until => CircuitState::Open,
            InnerState::Open { .. } => CircuitState::Open, // still Open until next call
            InnerState::HalfOpenProbing => CircuitState::HalfOpenProbing,
        }
    }

    /// Force the breaker open immediately. Intended for ops
    /// runbooks: "we know provider X is down, open the breaker
    /// for 60s while we cut over."
    pub fn trip(&self, cooldown: Duration) {
        *self.state.lock() = InnerState::Open {
            until: Instant::now() + cooldown,
        };
    }

    /// Force the breaker closed (resets failure counter).
    /// Counterpart to [`Self::trip`] for ops use.
    pub fn reset(&self) {
        *self.state.lock() = InnerState::Closed {
            consecutive_failures: 0,
        };
    }

    /// Run `f` through the breaker. If the breaker is open, returns
    /// `Err(CircuitOpen)` *without* invoking `f`. Otherwise invokes
    /// `f`, records the outcome, and updates state.
    ///
    /// In `HalfOpenProbing` state: exactly one call is allowed
    /// through as the probe. Concurrent callers see
    /// `Err(CircuitOpen)` until the probe completes. The probe's
    /// outcome decides whether the breaker re-closes (success) or
    /// re-opens with a fresh cooldown (failure).
    pub async fn call<F, Fut, T, E>(&self, f: F) -> Result<T, CallError<E>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<T, E>>,
    {
        // ---- Admission ----
        {
            let mut s = self.state.lock();
            match *s {
                InnerState::Closed { .. } => { /* go */ }
                InnerState::Open { until } => {
                    if Instant::now() >= until {
                        // Cooldown elapsed: take the probe slot.
                        *s = InnerState::HalfOpenProbing;
                    } else {
                        return Err(CallError::CircuitOpen);
                    }
                }
                InnerState::HalfOpenProbing => {
                    // Another caller is already probing.
                    return Err(CallError::CircuitOpen);
                }
            }
        }

        let result = f().await;

        // ---- Outcome ----
        {
            let mut s = self.state.lock();
            match &result {
                Ok(_) => {
                    // Success on probe or in-closed state both
                    // collapse to Closed { failures = 0 }.
                    *s = InnerState::Closed {
                        consecutive_failures: 0,
                    };
                }
                Err(_) => {
                    let new_state = match *s {
                        InnerState::Closed {
                            consecutive_failures,
                        } => {
                            let n = consecutive_failures + 1;
                            if n >= self.failure_threshold {
                                InnerState::Open {
                                    until: Instant::now() + self.cooldown,
                                }
                            } else {
                                InnerState::Closed {
                                    consecutive_failures: n,
                                }
                            }
                        }
                        InnerState::HalfOpenProbing => InnerState::Open {
                            until: Instant::now() + self.cooldown,
                        },
                        InnerState::Open { .. } => *s, // unchanged (concurrent late arrival)
                    };
                    *s = new_state;
                }
            }
        }

        result.map_err(CallError::Inner)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use tokio::time::sleep;

    fn ok_result() -> Result<u32, &'static str> {
        Ok(7)
    }

    fn err_result() -> Result<u32, &'static str> {
        Err("upstream blew up")
    }

    #[tokio::test]
    async fn closed_breaker_passes_calls_through() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));
        let r = cb.call(|| async { ok_result() }).await.unwrap();
        assert_eq!(r, 7);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn opens_after_threshold_consecutive_failures() {
        let cb = CircuitBreaker::new(3, Duration::from_millis(50));
        for _ in 0..2 {
            let r: Result<u32, CallError<&'static str>> =
                cb.call(|| async { err_result() }).await;
            assert!(matches!(r, Err(CallError::Inner(_))));
        }
        assert_eq!(cb.state(), CircuitState::Closed);
        // Third failure trips the breaker.
        let _: Result<u32, CallError<&'static str>> =
            cb.call(|| async { err_result() }).await;
        assert_eq!(cb.state(), CircuitState::Open);
    }

    #[tokio::test]
    async fn rejects_when_open_without_invoking_closure() {
        let cb = CircuitBreaker::new(1, Duration::from_secs(10));
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        assert_eq!(cb.state(), CircuitState::Open);
        let invoked = Arc::new(AtomicUsize::new(0));
        let i = invoked.clone();
        let r: Result<u32, _> = cb
            .call(|| async move {
                i.fetch_add(1, Ordering::SeqCst);
                ok_result()
            })
            .await;
        assert!(matches!(r, Err(CallError::CircuitOpen)));
        assert_eq!(invoked.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn success_resets_failure_counter() {
        let cb = CircuitBreaker::new(3, Duration::from_secs(1));
        for _ in 0..2 {
            let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        }
        // 2 failures recorded; one success resets.
        let _ = cb.call(|| async { ok_result() }).await.unwrap();
        // Two more failures should NOT trip — counter was reset.
        for _ in 0..2 {
            let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        }
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn half_open_probe_success_closes_breaker() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(20));
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        assert_eq!(cb.state(), CircuitState::Open);
        sleep(Duration::from_millis(30)).await;
        // Probe succeeds → Closed.
        let r = cb.call(|| async { ok_result() }).await.unwrap();
        assert_eq!(r, 7);
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn half_open_probe_failure_reopens_breaker() {
        let cb = CircuitBreaker::new(1, Duration::from_millis(20));
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        sleep(Duration::from_millis(30)).await;
        // Probe fails → reopened.
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        assert_eq!(cb.state(), CircuitState::Open);
        // Should also reject immediately (not within cooldown).
        let r: Result<u32, _> = cb.call(|| async { ok_result() }).await;
        assert!(matches!(r, Err(CallError::CircuitOpen)));
    }

    #[tokio::test]
    async fn concurrent_callers_in_half_open_only_one_probes() {
        let cb = Arc::new(CircuitBreaker::new(1, Duration::from_millis(20)));
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        sleep(Duration::from_millis(30)).await;
        // Three concurrent callers; only one runs the probe.
        let probe_count = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..3 {
            let cb = cb.clone();
            let pc = probe_count.clone();
            handles.push(tokio::spawn(async move {
                cb.call(|| async {
                    pc.fetch_add(1, Ordering::SeqCst);
                    sleep(Duration::from_millis(10)).await;
                    ok_result()
                })
                .await
            }));
        }
        let mut oks = 0;
        let mut rejects = 0;
        for h in handles {
            match h.await.unwrap() {
                Ok(_) => oks += 1,
                Err(CallError::CircuitOpen) => rejects += 1,
                Err(CallError::Inner(_)) => panic!("unexpected inner err"),
            }
        }
        assert_eq!(probe_count.load(Ordering::SeqCst), 1, "more than one probe ran");
        assert_eq!(oks, 1);
        assert_eq!(rejects, 2);
    }

    #[tokio::test]
    async fn manual_trip_and_reset() {
        let cb = CircuitBreaker::new(10, Duration::from_secs(1));
        cb.trip(Duration::from_secs(60));
        assert_eq!(cb.state(), CircuitState::Open);
        cb.reset();
        assert_eq!(cb.state(), CircuitState::Closed);
    }

    #[tokio::test]
    async fn threshold_zero_normalised_to_one() {
        let cb = CircuitBreaker::new(0, Duration::from_secs(1));
        let _: Result<u32, _> = cb.call(|| async { err_result() }).await;
        // One failure trips it.
        assert_eq!(cb.state(), CircuitState::Open);
    }
}

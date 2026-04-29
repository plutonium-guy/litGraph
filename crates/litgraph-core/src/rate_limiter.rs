//! `RateLimiter` — async token-bucket rate limiter.
//!
//! Distinct from [`crate::resilience::RateLimitedChatModel`]
//! (which wraps a single `ChatModel`) — this is a *reusable
//! primitive* any caller can charge against. Useful when a single
//! quota covers heterogeneous calls (chat + embed + tool) sharing
//! one provider's API budget.
//!
//! # Token-bucket semantics
//!
//! - Bucket holds at most `capacity` tokens.
//! - Tokens refill continuously at `refill_per_sec` per second.
//! - `acquire(n)` deducts `n` tokens; if fewer are available,
//!   blocks until enough have accumulated.
//! - `try_acquire(n)` is non-blocking; returns `false` if the
//!   bucket is short.
//! - Bursting up to `capacity` is allowed; sustained rate is
//!   capped at `refill_per_sec`.
//!
//! Implementation is **lazy-refill**: no background task is
//! spawned. Each `acquire`/`try_acquire` call recomputes the
//! current token count from `last_refill_instant`. Cheap
//! (one mutex acquisition + a few f64 ops) and correct under
//! arbitrary clock-tick patterns.
//!
//! # Real prod use
//!
//! - **Shared provider quota**: one OpenAI key serving 5
//!   concurrent agents — every agent calls
//!   `limiter.acquire(tokens_estimate).await` before its HTTP
//!   request, so the cluster never exceeds the per-minute TPM
//!   budget.
//! - **Egress traffic shaping**: outbound HTTP fan-out limited
//!   to N requests/sec across all loaders to avoid overwhelming
//!   a target.
//! - **Per-user fairness**: paired with [`crate::keyed_mutex`]'s
//!   key-keyed registry pattern, one limiter per `user_id` in a
//!   `HashMap<UserId, Arc<RateLimiter>>` enforces "X requests
//!   per minute per user."

use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex as PlMutex;

use crate::ShutdownSignal;

struct State {
    /// Tokens currently in the bucket. `f64` so refill can
    /// accumulate fractional tokens between calls without
    /// integer-rounding loss at high rates.
    tokens: f64,
    /// Wall-clock instant of the last refill computation.
    last_refill: Instant,
}

/// Async token-bucket rate limiter. Cheap to clone (`Arc`
/// inside) so multiple consumers can share one budget.
#[derive(Clone)]
pub struct RateLimiter {
    capacity: f64,
    refill_per_sec: f64,
    state: Arc<PlMutex<State>>,
}

impl RateLimiter {
    /// Construct a new bucket. `capacity = 0` is normalised to
    /// 1.0; `refill_per_sec = 0` is normalised to 1.0 (so the
    /// bucket isn't permanently dry).
    pub fn new(capacity: u64, refill_per_sec: u64) -> Self {
        let cap = (capacity.max(1)) as f64;
        let rate = (refill_per_sec.max(1)) as f64;
        Self {
            capacity: cap,
            refill_per_sec: rate,
            state: Arc::new(PlMutex::new(State {
                // Bucket starts full so first burst doesn't pause.
                tokens: cap,
                last_refill: Instant::now(),
            })),
        }
    }

    /// Recompute the current token count. Called on every
    /// `acquire` / `try_acquire` so the bucket is always
    /// up-to-date when consulted.
    fn refill_locked(&self, s: &mut State) {
        let now = Instant::now();
        let dt = now.duration_since(s.last_refill).as_secs_f64();
        if dt > 0.0 {
            s.tokens = (s.tokens + dt * self.refill_per_sec).min(self.capacity);
            s.last_refill = now;
        }
    }

    /// Snapshot the current token count. Useful for telemetry /
    /// dashboard counters; not for decision-making (race-prone).
    pub fn available(&self) -> f64 {
        let mut s = self.state.lock();
        self.refill_locked(&mut s);
        s.tokens
    }

    /// Capacity (max burst size).
    pub fn capacity(&self) -> u64 {
        self.capacity as u64
    }

    /// Sustained rate.
    pub fn refill_per_sec(&self) -> u64 {
        self.refill_per_sec as u64
    }

    /// Try to deduct `n` tokens without blocking. Returns `true`
    /// on success, `false` if the bucket is short. `n = 0`
    /// trivially succeeds.
    pub fn try_acquire(&self, n: u64) -> bool {
        let n = n as f64;
        if n <= 0.0 {
            return true;
        }
        let mut s = self.state.lock();
        self.refill_locked(&mut s);
        if s.tokens >= n {
            s.tokens -= n;
            true
        } else {
            false
        }
    }

    /// Block until `n` tokens are available, then deduct them.
    /// `n` larger than `capacity` is **clamped** to `capacity` —
    /// otherwise the call would block forever (the bucket can
    /// never accumulate more than capacity).
    pub async fn acquire(&self, n: u64) {
        let n = (n as f64).min(self.capacity);
        if n <= 0.0 {
            return;
        }
        loop {
            let wait = {
                let mut s = self.state.lock();
                self.refill_locked(&mut s);
                if s.tokens >= n {
                    s.tokens -= n;
                    return;
                }
                let needed = n - s.tokens;
                Duration::from_secs_f64(needed / self.refill_per_sec)
            };
            tokio::time::sleep(wait).await;
        }
    }

    /// Race the acquire against a [`ShutdownSignal`]. Returns
    /// `Some(())` if the deduction completed, `None` if shutdown
    /// fired first. **Tokens are not deducted on the shutdown
    /// path** — the budget is preserved for surviving callers.
    pub async fn acquire_with_shutdown(
        &self,
        n: u64,
        shutdown: &ShutdownSignal,
    ) -> Option<()> {
        let n = (n as f64).min(self.capacity);
        if n <= 0.0 {
            return Some(());
        }
        if shutdown.is_signaled() {
            return None;
        }
        loop {
            let wait = {
                let mut s = self.state.lock();
                self.refill_locked(&mut s);
                if s.tokens >= n {
                    s.tokens -= n;
                    return Some(());
                }
                let needed = n - s.tokens;
                Duration::from_secs_f64(needed / self.refill_per_sec)
            };
            tokio::select! {
                _ = tokio::time::sleep(wait) => continue,
                _ = shutdown.wait() => return None,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::Instant;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn fresh_bucket_is_full() {
        let l = RateLimiter::new(10, 1);
        assert_eq!(l.available() as u64, 10);
        assert!(l.try_acquire(10));
        assert_eq!(l.available() as u64, 0);
    }

    #[tokio::test]
    async fn try_acquire_returns_false_when_short() {
        let l = RateLimiter::new(2, 1);
        assert!(l.try_acquire(2));
        assert!(!l.try_acquire(1));
    }

    #[tokio::test]
    async fn capacity_zero_normalised_to_one() {
        let l = RateLimiter::new(0, 0);
        assert_eq!(l.capacity(), 1);
        assert_eq!(l.refill_per_sec(), 1);
    }

    #[tokio::test]
    async fn acquire_blocks_then_succeeds_after_refill() {
        let l = RateLimiter::new(1, 100); // refills 100/sec ≈ 10ms/token
        assert!(l.try_acquire(1));
        // Empty now; acquire(1) should wait ~10ms.
        let started = Instant::now();
        timeout(Duration::from_millis(100), l.acquire(1))
            .await
            .expect("acquire blocked too long");
        let elapsed = started.elapsed();
        assert!(
            elapsed >= Duration::from_millis(5),
            "acquire returned too fast: {elapsed:?}",
        );
        assert!(
            elapsed <= Duration::from_millis(50),
            "acquire took too long: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn n_larger_than_capacity_clamped() {
        let l = RateLimiter::new(5, 100);
        // Asking for 50 tokens against capacity 5: clamps to 5.
        // Bucket is full → returns immediately with tokens=0.
        timeout(Duration::from_millis(100), l.acquire(50))
            .await
            .expect("clamp didn't engage; acquire blocked forever");
        assert_eq!(l.available() as u64, 0);
    }

    #[tokio::test]
    async fn concurrent_acquire_serializes() {
        // Bucket capacity 1, refill 50/sec (20ms/token). 4 concurrent
        // acquires of 1 → must take at least ~3 * 20 = 60ms total.
        let l = RateLimiter::new(1, 50);
        let counter = Arc::new(AtomicU64::new(0));
        let started = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..4 {
            let lc = l.clone();
            let cc = counter.clone();
            handles.push(tokio::spawn(async move {
                lc.acquire(1).await;
                cc.fetch_add(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        let elapsed = started.elapsed();
        assert_eq!(counter.load(Ordering::SeqCst), 4);
        assert!(
            elapsed >= Duration::from_millis(50),
            "4 acquires too fast (rate not honoured): {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn refill_caps_at_capacity() {
        let l = RateLimiter::new(3, 10);
        // Drain.
        assert!(l.try_acquire(3));
        // Sleep long enough to refill 10 tokens — must cap at 3.
        sleep(Duration::from_millis(1100)).await;
        assert!(l.available() <= 3.0001);
        assert!(l.try_acquire(3));
        assert!(!l.try_acquire(1));
    }

    // ---- acquire_with_shutdown tests ----------------------------------

    #[tokio::test]
    async fn shutdown_fires_during_wait_returns_none() {
        let l = RateLimiter::new(1, 1); // 1 token/sec — slow refill
        assert!(l.try_acquire(1));
        // Now empty; acquire(1) would wait ~1s.
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        tokio::spawn(async move {
            sleep(Duration::from_millis(30)).await;
            s2.signal();
        });
        let started = Instant::now();
        let r = l.acquire_with_shutdown(1, &s).await;
        let elapsed = started.elapsed();
        assert_eq!(r, None);
        assert!(
            elapsed < Duration::from_millis(200),
            "shutdown didn't break the wait: {elapsed:?}",
        );
        // Tokens were NOT deducted — budget preserved.
        assert_eq!(l.available() as u64, 0);
    }

    #[tokio::test]
    async fn shutdown_pre_fired_returns_none_no_deduct() {
        let l = RateLimiter::new(5, 1);
        let s = ShutdownSignal::new();
        s.signal();
        let r = l.acquire_with_shutdown(2, &s).await;
        assert_eq!(r, None);
        // No deduction.
        assert_eq!(l.available() as u64, 5);
    }

    #[tokio::test]
    async fn acquire_with_shutdown_succeeds_when_tokens_available() {
        let l = RateLimiter::new(3, 1);
        let s = ShutdownSignal::new();
        let r = l.acquire_with_shutdown(2, &s).await;
        assert_eq!(r, Some(()));
        assert_eq!(l.available() as u64, 1);
    }
}

//! `Bulkhead` — concurrent-call cap with **rejection** semantics.
//!
//! Distinct from a plain `tokio::sync::Semaphore`:
//!
//! | Primitive    | At-capacity behavior                        |
//! |--------------|---------------------------------------------|
//! | `Semaphore`  | Caller queues until a permit is released    |
//! | `Bulkhead`   | Caller is rejected immediately (`try_enter`) |
//!
//! Both share the same underlying mechanism (a Semaphore). The
//! difference is the *intended use*: a Semaphore queues callers
//! to enforce a degree of parallelism; a Bulkhead caps in-flight
//! concurrency to protect a downstream resource and surfaces the
//! rejection back to the caller as a *signal* — "this dependency
//! is saturated, you may want to shed load."
//!
//! Named after the "Release It!" pattern: separate failure
//! domains (like a ship's bulkheads) so a single saturated
//! resource doesn't drown the whole process.
//!
//! # API shapes
//!
//! - [`Bulkhead::try_enter`] — non-blocking. Returns
//!   `Some(BulkheadGuard)` if a slot is free, `None` otherwise
//!   (and bumps the rejected counter for telemetry).
//! - [`Bulkhead::enter`] — block until a slot opens.
//!   Equivalent to a Semaphore acquire; provided so the
//!   wrapper is useful when the caller actually does want to
//!   queue.
//! - [`Bulkhead::enter_with_timeout`] — block up to `timeout`,
//!   then give up with `None` (and bump the counter).
//!
//! # Real prod use
//!
//! - **Per-tool concurrent cap**: a tool that wraps a flaky
//!   third-party API can take at most 5 concurrent calls; the
//!   6th caller gets `BulkheadFull` immediately so the agent
//!   can pick a different action rather than waiting.
//! - **Vector-store connection budget**: pgvector pool of 20
//!   connections; cap concurrent retrievers at 18 to leave
//!   headroom for writes.
//! - **Outbound HTTP fan-out**: at most N concurrent crawler
//!   requests against a target; reject the (N+1)th to skip the
//!   slow one rather than queue forever.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore};

/// In-flight slot guard. Released on drop. Cheap (one
/// `Arc<Semaphore>` Arc-decrement on drop via the inner permit).
pub struct BulkheadGuard {
    _permit: OwnedSemaphorePermit,
}

/// Concurrent-call cap with rejection semantics. Cheap to clone
/// (`Arc` inside).
#[derive(Clone)]
pub struct Bulkhead {
    sem: Arc<Semaphore>,
    cap: usize,
    rejected: Arc<AtomicU64>,
}

impl Bulkhead {
    /// Construct a new bulkhead with capacity `cap`. `cap = 0`
    /// is normalised to 1 (otherwise no caller could ever
    /// enter, which is rarely the intent).
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            sem: Arc::new(Semaphore::new(cap)),
            cap,
            rejected: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Maximum concurrent slots.
    pub fn cap(&self) -> usize {
        self.cap
    }

    /// Approximate count of currently-held slots. Saturates at
    /// `cap`; race-prone for decision-making, fine for
    /// telemetry.
    pub fn in_flight(&self) -> usize {
        self.cap.saturating_sub(self.sem.available_permits())
    }

    /// Total rejection count since construction. Bumped by
    /// `try_enter` / `enter_with_timeout` when the cap is
    /// reached.
    pub fn rejected_count(&self) -> u64 {
        self.rejected.load(Ordering::Relaxed)
    }

    /// Try to enter without blocking. `Some(guard)` if a slot
    /// was free; `None` otherwise (and the rejected counter is
    /// incremented).
    pub fn try_enter(&self) -> Option<BulkheadGuard> {
        match self.sem.clone().try_acquire_owned() {
            Ok(permit) => Some(BulkheadGuard { _permit: permit }),
            Err(_) => {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }

    /// Block until a slot opens. Equivalent to a Semaphore
    /// acquire — useful when the caller actually wants
    /// queueing semantics. Cancel-safe.
    ///
    /// # Panics
    ///
    /// If the inner Semaphore is closed (we never close it, so
    /// this is unreachable in practice).
    pub async fn enter(&self) -> BulkheadGuard {
        let permit = self
            .sem
            .clone()
            .acquire_owned()
            .await
            .expect("bulkhead semaphore closed");
        BulkheadGuard { _permit: permit }
    }

    /// Block up to `timeout` for a slot. `Some(guard)` if a
    /// slot opened within the window; `None` if it timed out
    /// (and the rejected counter is incremented).
    pub async fn enter_with_timeout(&self, timeout: Duration) -> Option<BulkheadGuard> {
        match tokio::time::timeout(timeout, self.sem.clone().acquire_owned()).await {
            Ok(Ok(permit)) => Some(BulkheadGuard { _permit: permit }),
            Ok(Err(_)) => None, // semaphore closed (unreachable in practice)
            Err(_) => {
                self.rejected.fetch_add(1, Ordering::Relaxed);
                None
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as O};
    use tokio::time::{sleep, Instant};

    #[tokio::test]
    async fn try_enter_succeeds_when_under_cap() {
        let b = Bulkhead::new(2);
        let g1 = b.try_enter().expect("first slot should be free");
        let g2 = b.try_enter().expect("second slot should be free");
        // At cap; third must reject.
        assert!(b.try_enter().is_none());
        assert_eq!(b.rejected_count(), 1);
        drop(g1);
        // One slot is back.
        let _g3 = b.try_enter().expect("slot should reopen after drop");
        drop(g2);
    }

    #[tokio::test]
    async fn cap_zero_normalised_to_one() {
        let b = Bulkhead::new(0);
        assert_eq!(b.cap(), 1);
        assert!(b.try_enter().is_some());
    }

    #[tokio::test]
    async fn in_flight_tracks_held_slots() {
        let b = Bulkhead::new(3);
        assert_eq!(b.in_flight(), 0);
        let g1 = b.try_enter().unwrap();
        assert_eq!(b.in_flight(), 1);
        let g2 = b.try_enter().unwrap();
        assert_eq!(b.in_flight(), 2);
        drop(g1);
        assert_eq!(b.in_flight(), 1);
        drop(g2);
        assert_eq!(b.in_flight(), 0);
    }

    #[tokio::test]
    async fn enter_blocks_until_slot_opens() {
        let b = Bulkhead::new(1);
        let _g = b.try_enter().unwrap();
        let b2 = b.clone();
        let h = tokio::spawn(async move {
            let started = Instant::now();
            let _g = b2.enter().await;
            started.elapsed()
        });
        sleep(Duration::from_millis(20)).await;
        // Drop the held guard — the spawned task should unblock.
        drop(_g);
        let waited = h.await.unwrap();
        assert!(
            waited >= Duration::from_millis(15),
            "enter returned too fast: {waited:?}",
        );
    }

    #[tokio::test]
    async fn enter_with_timeout_returns_none_after_deadline() {
        let b = Bulkhead::new(1);
        let _held = b.try_enter().unwrap();
        let started = Instant::now();
        let r = b.enter_with_timeout(Duration::from_millis(20)).await;
        let elapsed = started.elapsed();
        assert!(r.is_none());
        assert!(elapsed >= Duration::from_millis(15));
        assert!(elapsed < Duration::from_millis(100));
        assert_eq!(b.rejected_count(), 1);
    }

    #[tokio::test]
    async fn enter_with_timeout_succeeds_if_slot_opens_in_time() {
        let b = Bulkhead::new(1);
        let g = b.try_enter().unwrap();
        let b2 = b.clone();
        let h = tokio::spawn(async move {
            sleep(Duration::from_millis(15)).await;
            drop(g);
        });
        let r = b.enter_with_timeout(Duration::from_millis(100)).await;
        assert!(r.is_some());
        h.await.unwrap();
        // No rejection because the wait succeeded.
        assert_eq!(b.rejected_count(), 0);
    }

    #[tokio::test]
    async fn rejected_count_accumulates_across_modes() {
        let b = Bulkhead::new(1);
        let _g = b.try_enter().unwrap();
        // try_enter rejection.
        assert!(b.try_enter().is_none());
        // enter_with_timeout rejection.
        assert!(b
            .enter_with_timeout(Duration::from_millis(5))
            .await
            .is_none());
        assert_eq!(b.rejected_count(), 2);
    }

    #[tokio::test]
    async fn concurrent_callers_observe_strict_cap() {
        // 10 spawned tasks; cap=3. Peak in_flight must equal 3.
        let b = Bulkhead::new(3);
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let b = b.clone();
            let inf = in_flight.clone();
            let pk = peak.clone();
            handles.push(tokio::spawn(async move {
                let _g = b.enter().await;
                let now = inf.fetch_add(1, O::SeqCst) + 1;
                let mut p = pk.load(O::SeqCst);
                while now > p {
                    match pk.compare_exchange(p, now, O::SeqCst, O::SeqCst) {
                        Ok(_) => break,
                        Err(actual) => p = actual,
                    }
                }
                sleep(Duration::from_millis(10)).await;
                inf.fetch_sub(1, O::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(peak.load(O::SeqCst), 3);
    }
}

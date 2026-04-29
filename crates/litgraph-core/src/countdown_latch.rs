//! `CountDownLatch` — wait-until-zero coordination primitive.
//!
//! Sister to [`crate::barrier::Barrier`] but with a different
//! shape:
//!
//! | Primitive          | Who calls what                             |
//! |--------------------|--------------------------------------------|
//! | `Barrier`          | every participant calls `wait` (arrives + waits in one op) |
//! | `CountDownLatch`   | producers call `count_down` (no wait), observers call `wait` (no decrement) |
//!
//! # Why a separate primitive
//!
//! `Barrier` couples arrival and waiting in the same call. That's
//! right when every cohort member is symmetric (all run, all
//! synchronize). `CountDownLatch` is right when **producers and
//! observers are different roles**:
//!
//! - 10 background workers race off, call `latch.count_down()`
//!   when each finishes.
//! - 1 main task awaits `latch.wait()` to know when ALL 10 are
//!   done — without holding any `JoinHandle`s, without iterating
//!   a `JoinSet`.
//!
//! The decoupling matters when workers are spawned by *other*
//! code (a scheduler, a different module) and the observer just
//! has a clone of the latch.
//!
//! # Real prod use
//!
//! - **Fan-out completion gate**: spawn N retrievers / N tools;
//!   await `latch` to know everyone returned (regardless of who
//!   spawned them, regardless of `JoinSet` lifetime).
//! - **Initialization barrier**: 5 caches start filling on
//!   different threads; main task waits for all 5 to report
//!   ready before serving traffic.
//! - **Cleanup synchronization**: on shutdown, every worker
//!   `count_down()`s as it drains; supervisor waits for the
//!   latch to confirm graceful exit.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

use crate::ShutdownSignal;

/// Wait-until-zero counter. Cheap to clone (`Arc` inside).
#[derive(Clone)]
pub struct CountDownLatch {
    remaining: Arc<AtomicUsize>,
    notify: Arc<Notify>,
}

impl CountDownLatch {
    /// Construct a latch with `count` initial outstanding events.
    /// `count = 0` is allowed and means the latch is born already
    /// open — `wait` resolves instantly.
    pub fn new(count: usize) -> Self {
        let s = Self {
            remaining: Arc::new(AtomicUsize::new(count)),
            notify: Arc::new(Notify::new()),
        };
        if count == 0 {
            // Pre-fire so any future `wait` finds the flag-equiv
            // (counter at 0) and short-circuits.
            s.notify.notify_waiters();
        }
        s
    }

    /// Current outstanding count. `0` once the latch has opened.
    pub fn count(&self) -> usize {
        self.remaining.load(Ordering::SeqCst)
    }

    /// `true` once the count has reached zero.
    pub fn is_zero(&self) -> bool {
        self.remaining.load(Ordering::SeqCst) == 0
    }

    /// Decrement the outstanding count by one. When the count
    /// transitions from 1 → 0, every pending `wait` resolves.
    /// Decrementing past zero is a no-op (saturates at 0).
    ///
    /// Returns the new count after the decrement.
    pub fn count_down(&self) -> usize {
        // saturating_sub via compare_exchange so we never wrap.
        loop {
            let cur = self.remaining.load(Ordering::SeqCst);
            if cur == 0 {
                return 0;
            }
            let next = cur - 1;
            if self
                .remaining
                .compare_exchange(cur, next, Ordering::SeqCst, Ordering::SeqCst)
                .is_ok()
            {
                if next == 0 {
                    self.notify.notify_waiters();
                }
                return next;
            }
        }
    }

    /// Block until the count reaches zero. Cancel-safe — drop the
    /// future to stop waiting (other observers and counters are
    /// unaffected).
    pub async fn wait(&self) {
        if self.remaining.load(Ordering::SeqCst) == 0 {
            return;
        }
        // Same flag-after-register pattern as Barrier / ShutdownSignal:
        // register the notified future before re-checking so we don't
        // miss a count_down that races between our load and our sleep.
        let notified = self.notify.notified();
        if self.remaining.load(Ordering::SeqCst) == 0 {
            return;
        }
        notified.await;
    }

    /// Race the wait against a [`ShutdownSignal`]. Returns
    /// `Some(())` if the count reached zero, `None` if shutdown
    /// fired first.
    ///
    /// If the latch already opened before this call, returns
    /// `Some(())` even if shutdown also fired (the work is
    /// already done, no point reporting cancellation).
    pub async fn wait_with_shutdown(&self, shutdown: &ShutdownSignal) -> Option<()> {
        if self.remaining.load(Ordering::SeqCst) == 0 {
            return Some(());
        }
        if shutdown.is_signaled() {
            return None;
        }
        let notified = self.notify.notified();
        // Re-check both after registration so we don't miss either
        // edge that races between the loads above and the await.
        if self.remaining.load(Ordering::SeqCst) == 0 {
            return Some(());
        }
        if shutdown.is_signaled() {
            return None;
        }
        tokio::select! {
            _ = notified => Some(()),
            _ = shutdown.wait() => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::time::timeout;

    #[tokio::test]
    async fn count_zero_is_open_immediately() {
        let l = CountDownLatch::new(0);
        assert!(l.is_zero());
        timeout(Duration::from_millis(20), l.wait()).await.unwrap();
    }

    #[tokio::test]
    async fn count_down_to_zero_releases_waiter() {
        let l = CountDownLatch::new(3);
        let l1 = l.clone();
        let h = tokio::spawn(async move { l1.wait().await });
        // Two count downs — still pending.
        l.count_down();
        l.count_down();
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert_eq!(l.count(), 1);
        // Third releases the waiter.
        l.count_down();
        timeout(Duration::from_millis(50), h)
            .await
            .expect("waiter blocked past third count_down")
            .unwrap();
        assert!(l.is_zero());
    }

    #[tokio::test]
    async fn count_down_returns_remaining() {
        let l = CountDownLatch::new(3);
        assert_eq!(l.count_down(), 2);
        assert_eq!(l.count_down(), 1);
        assert_eq!(l.count_down(), 0);
        // Saturating: extra count_down stays at 0.
        assert_eq!(l.count_down(), 0);
    }

    #[tokio::test]
    async fn many_waiters_all_release() {
        let l = CountDownLatch::new(1);
        let mut handles = Vec::new();
        for _ in 0..5 {
            let lc = l.clone();
            handles.push(tokio::spawn(async move { lc.wait().await }));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        l.count_down();
        for h in handles {
            timeout(Duration::from_millis(50), h)
                .await
                .expect("waiter didn't unblock")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn late_wait_after_open_returns_immediately() {
        let l = CountDownLatch::new(1);
        l.count_down();
        timeout(Duration::from_millis(20), l.wait())
            .await
            .expect("late wait should be instant");
    }

    #[tokio::test]
    async fn parallel_count_down_from_many_tasks() {
        let l = CountDownLatch::new(10);
        let mut producers = Vec::new();
        for _ in 0..10 {
            let lc = l.clone();
            producers.push(tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(5)).await;
                lc.count_down();
            }));
        }
        timeout(Duration::from_millis(200), l.wait())
            .await
            .expect("latch never opened despite 10 count_downs");
        for p in producers {
            p.await.unwrap();
        }
    }

    // ---- wait_with_shutdown tests -------------------------------------

    #[tokio::test]
    async fn shutdown_fires_before_open_returns_none() {
        let l = CountDownLatch::new(3);
        let s = ShutdownSignal::new();
        let lc = l.clone();
        let sc = s.clone();
        let h = tokio::spawn(async move { lc.wait_with_shutdown(&sc).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        s.signal();
        let r = timeout(Duration::from_millis(50), h).await.unwrap().unwrap();
        assert_eq!(r, None);
        assert!(!l.is_zero());
    }

    #[tokio::test]
    async fn shutdown_pre_fired_returns_none_without_blocking() {
        let l = CountDownLatch::new(3);
        let s = ShutdownSignal::new();
        s.signal();
        let r = timeout(Duration::from_millis(20), l.wait_with_shutdown(&s))
            .await
            .unwrap();
        assert_eq!(r, None);
    }

    #[tokio::test]
    async fn open_before_shutdown_returns_some() {
        let l = CountDownLatch::new(2);
        let s = ShutdownSignal::new();
        let lc = l.clone();
        let sc = s.clone();
        let h = tokio::spawn(async move { lc.wait_with_shutdown(&sc).await });
        l.count_down();
        l.count_down();
        let r = timeout(Duration::from_millis(50), h).await.unwrap().unwrap();
        assert_eq!(r, Some(()));
    }

    #[tokio::test]
    async fn already_open_returns_some_even_if_shutdown_fired() {
        let l = CountDownLatch::new(1);
        l.count_down();
        let s = ShutdownSignal::new();
        s.signal();
        // Open state wins over fired shutdown — the work is done.
        let r = l.wait_with_shutdown(&s).await;
        assert_eq!(r, Some(()));
    }
}

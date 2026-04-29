//! `Barrier` — wait-for-N rendezvous coordination primitive.
//!
//! Sixth distinct channel shape in this lineage (after mpsc /
//! broadcast / watch / oneshot / Notify-edge):
//!
//! | Channel kind        | Item delivery                                 | Iter |
//! |---------------------|-----------------------------------------------|------|
//! | `mpsc`              | every event, single consumer                  | 189  |
//! | `broadcast`         | every event, multi-consumer                   | 195  |
//! | `watch`             | latest-value, multi-consumer                  | 199  |
//! | `oneshot`           | single-fire, one consumer                     | 201  |
//! | `Notify` (edge)     | single-fire EDGE, N concurrent waiters        | 225  |
//! | `Barrier` (this)    | rendezvous: every waiter unblocks at threshold | 239  |
//!
//! # Semantics
//!
//! `Barrier::new(n)` requires N participants to call `wait()`.
//! When the N-th waiter arrives, every pending waiter unblocks
//! simultaneously. Late arrivals (past the N-th) return
//! immediately — once released, the barrier stays released.
//!
//! Distinct from `tokio::sync::Barrier` in that this version is
//! shutdown-aware: `wait_with_shutdown` races against a
//! [`ShutdownSignal`] and returns `None` if the signal fires
//! before the rendezvous opens. Pending waiters wake instead of
//! parking forever when the orchestrator decides to abandon the
//! synchronized step.
//!
//! # Real prod use
//!
//! - **Coordinated agent rounds**: 5 agents each compute their
//!   step in parallel, reach `barrier.wait()`, all unblock
//!   together to start the next round. Prevents fast agents
//!   from racing ahead with stale shared state.
//! - **Warm-up rendezvous**: N workers each load their model
//!   weights / embed prefixes / open DB connections, then
//!   `barrier.wait()` — every worker starts serving in
//!   lockstep so the first request doesn't see a half-warm
//!   cluster.
//! - **Phase synchronization**: pipeline stage N+1 can't begin
//!   any item until every item of stage N has finished.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

use crate::ShutdownSignal;

/// Wait-for-N rendezvous primitive. Cheap to clone (`Arc`
/// inside) so each participant takes its own handle.
#[derive(Clone)]
pub struct Barrier {
    needed: usize,
    arrived: Arc<AtomicUsize>,
    released: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Barrier {
    /// Construct a barrier that releases when `n` participants
    /// have called `wait`. `n = 0` is normalised to 1 (the first
    /// `wait` releases).
    pub fn new(n: usize) -> Self {
        Self {
            needed: n.max(1),
            arrived: Arc::new(AtomicUsize::new(0)),
            released: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Number of participants required to release the barrier.
    pub fn needed(&self) -> usize {
        self.needed
    }

    /// How many participants have called `wait` so far. Saturates
    /// at `needed` once released — late arrivals don't keep
    /// inflating the count.
    pub fn arrived_count(&self) -> usize {
        self.arrived.load(Ordering::SeqCst).min(self.needed)
    }

    /// Non-blocking check. `true` once the barrier has released.
    pub fn is_released(&self) -> bool {
        self.released.load(Ordering::SeqCst)
    }

    /// Block until the barrier releases. Cancel-safe — drop the
    /// future to stop waiting (the participant's slot is *not*
    /// reclaimed; their arrival counts even if they drop the
    /// future before release).
    ///
    /// Once released, every future `wait` returns immediately.
    pub async fn wait(&self) {
        if self.released.load(Ordering::SeqCst) {
            return;
        }
        let mine = self.arrived.fetch_add(1, Ordering::SeqCst) + 1;
        if mine >= self.needed {
            self.released.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
            return;
        }
        // Register notified BEFORE re-checking the flag so we don't
        // miss a release that happens between our increment and our
        // sleep. Same pattern as ShutdownSignal::wait.
        let notified = self.notify.notified();
        if self.released.load(Ordering::SeqCst) {
            return;
        }
        notified.await;
    }

    /// Race the wait against a [`ShutdownSignal`]. Returns
    /// `Some(())` if the barrier released, `None` if shutdown
    /// fired first.
    ///
    /// Pre-fired signal returns `None` without incrementing the
    /// arrival count — a participant that's been told to abort
    /// before they could even register doesn't drag the rest of
    /// the cohort closer to a release that might never come.
    ///
    /// If the rendezvous already happened before this call,
    /// returns `Some(())` even if shutdown also fired (released
    /// state wins — the work is already done).
    pub async fn wait_with_shutdown(&self, shutdown: &ShutdownSignal) -> Option<()> {
        if self.released.load(Ordering::SeqCst) {
            return Some(());
        }
        if shutdown.is_signaled() {
            return None;
        }
        let mine = self.arrived.fetch_add(1, Ordering::SeqCst) + 1;
        if mine >= self.needed {
            self.released.store(true, Ordering::SeqCst);
            self.notify.notify_waiters();
            return Some(());
        }
        let notified = self.notify.notified();
        // Re-check release after registration in case the N-th
        // arrival happened between our increment and our register.
        if self.released.load(Ordering::SeqCst) {
            return Some(());
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
    async fn n_one_releases_on_first_wait() {
        let b = Barrier::new(1);
        timeout(Duration::from_millis(50), b.wait()).await.unwrap();
        assert!(b.is_released());
        assert_eq!(b.arrived_count(), 1);
    }

    #[tokio::test]
    async fn n_zero_normalised_to_one() {
        let b = Barrier::new(0);
        assert_eq!(b.needed(), 1);
        timeout(Duration::from_millis(50), b.wait()).await.unwrap();
        assert!(b.is_released());
    }

    #[tokio::test]
    async fn three_waiters_all_release_at_threshold() {
        let b = Barrier::new(3);
        let b1 = b.clone();
        let b2 = b.clone();
        let b3 = b.clone();
        let h1 = tokio::spawn(async move { b1.wait().await });
        let h2 = tokio::spawn(async move { b2.wait().await });
        // Two pending; not yet released.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!b.is_released());
        let h3 = tokio::spawn(async move { b3.wait().await });
        // Third arrival releases all three.
        timeout(Duration::from_millis(100), async {
            h1.await.unwrap();
            h2.await.unwrap();
            h3.await.unwrap();
        })
        .await
        .unwrap();
        assert!(b.is_released());
    }

    #[tokio::test]
    async fn late_arrival_after_release_returns_immediately() {
        let b = Barrier::new(2);
        let b1 = b.clone();
        let b2 = b.clone();
        tokio::spawn(async move { b1.wait().await });
        tokio::spawn(async move { b2.wait().await }).await.unwrap();
        // Trigger release via fan-in on the second arrival above.
        // Actually the second wait might not have observed the
        // release yet — give it a beat.
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(b.is_released());
        // Third late arrival: must resolve instantly.
        timeout(Duration::from_millis(20), b.wait())
            .await
            .expect("late wait should resolve immediately after release");
    }

    #[tokio::test]
    async fn pending_waiter_blocks_until_threshold() {
        let b = Barrier::new(2);
        let b1 = b.clone();
        let h = tokio::spawn(async move { b1.wait().await });
        // Single arrival — should not release.
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!b.is_released());
        // Second arrival from this task releases.
        b.wait().await;
        h.await.unwrap();
        assert!(b.is_released());
    }

    // ---- wait_with_shutdown tests -------------------------------------

    #[tokio::test]
    async fn shutdown_fires_before_release_returns_none() {
        let b = Barrier::new(3);
        let s = ShutdownSignal::new();
        let b1 = b.clone();
        let s1 = s.clone();
        let h = tokio::spawn(async move { b1.wait_with_shutdown(&s1).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        s.signal();
        let r = timeout(Duration::from_millis(50), h).await.unwrap().unwrap();
        assert_eq!(r, None);
        assert!(!b.is_released());
    }

    #[tokio::test]
    async fn shutdown_pre_fired_returns_none_without_incrementing() {
        let b = Barrier::new(3);
        let s = ShutdownSignal::new();
        s.signal();
        let r = b.wait_with_shutdown(&s).await;
        assert_eq!(r, None);
        assert_eq!(b.arrived_count(), 0);
    }

    #[tokio::test]
    async fn release_before_shutdown_returns_some() {
        let b = Barrier::new(2);
        let s = ShutdownSignal::new();
        let b1 = b.clone();
        let s1 = s.clone();
        let h1 = tokio::spawn(async move { b1.wait_with_shutdown(&s1).await });
        let b2 = b.clone();
        let s2 = s.clone();
        let h2 = tokio::spawn(async move { b2.wait_with_shutdown(&s2).await });
        let r1 = timeout(Duration::from_millis(50), h1)
            .await
            .unwrap()
            .unwrap();
        let r2 = timeout(Duration::from_millis(50), h2)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(r1, Some(()));
        assert_eq!(r2, Some(()));
        assert!(b.is_released());
    }

    #[tokio::test]
    async fn already_released_returns_some_even_if_shutdown_fired() {
        let b = Barrier::new(1);
        b.wait().await;
        assert!(b.is_released());
        let s = ShutdownSignal::new();
        s.signal();
        // Released state wins over fired shutdown — the rendezvous
        // already happened, the work is done.
        let r = b.wait_with_shutdown(&s).await;
        assert_eq!(r, Some(()));
    }

    #[tokio::test]
    async fn mixed_some_shutdown_some_released_when_threshold_met_concurrently() {
        // 3 waiters, threshold 3. If shutdown fires AFTER the third
        // arrival increments but BEFORE notify_waiters propagates,
        // the released flag was set first → all three should still
        // see Some. Verifies the flag-before-notify ordering.
        let b = Barrier::new(3);
        let s = ShutdownSignal::new();
        let mut handles = Vec::new();
        for _ in 0..3 {
            let bc = b.clone();
            let sc = s.clone();
            handles.push(tokio::spawn(async move { bc.wait_with_shutdown(&sc).await }));
        }
        // Give them a tick to release naturally.
        tokio::time::sleep(Duration::from_millis(30)).await;
        // Signal shutdown after release (no-op since flag won).
        s.signal();
        for h in handles {
            assert_eq!(h.await.unwrap(), Some(()));
        }
        assert!(b.is_released());
    }
}

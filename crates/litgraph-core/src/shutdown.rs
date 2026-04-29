//! `ShutdownSignal` — coordination primitive for graceful shutdown.
//!
//! Backed by `tokio::sync::Notify`, this is the **fifth** distinct
//! channel shape in this lineage:
//!
//! | Channel kind        | Item delivery                           | Iter |
//! |---------------------|-----------------------------------------|------|
//! | `mpsc`              | every event, single consumer            | 189  |
//! | `broadcast`         | every event, multi-consumer             | 195  |
//! | `watch`             | latest-value, multi-consumer            | 199  |
//! | `oneshot`           | single-fire, one consumer               | 201  |
//! | `Notify` (this)     | single-fire EDGE, **N concurrent waiters** | 225  |
//!
//! # Why a separate primitive
//!
//! `tokio::sync::Notify` exposes "wake any one waiter" /
//! "wake all waiters" primitives but no built-in "did this fire
//! yet?" state. A bare `Notify` makes *late* waiters (those that
//! call `notified().await` after `notify_waiters()` already fired)
//! park forever — there's no replay. For shutdown semantics we
//! want the opposite: a late `wait()` should resolve immediately
//! if shutdown already happened.
//!
//! `ShutdownSignal` adds an `AtomicBool` "fired" flag alongside the
//! `Notify`. `signal()` sets the flag then wakes every waiter;
//! `wait()` checks the flag first and returns instantly if set,
//! otherwise sleeps on the `Notify`.
//!
//! # Real prod use
//!
//! - **Graceful shutdown**: every worker task takes a clone of the
//!   signal; main task flips it on Ctrl+C and every worker wakes
//!   to drain in-flight work and exit.
//! - **Cache invalidation**: producer task signals; N consumer
//!   tasks each wake to refetch.
//! - **Single-edge "go" signal**: warm up workers, then signal
//!   all-at-once to start coordinated work.

use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use tokio::sync::Notify;

/// One-shot edge signal that **every** waiter sees, including
/// ones that join after the signal fires. Cheap to clone (`Arc`
/// inside) so worker tasks each take their own handle.
#[derive(Clone)]
pub struct ShutdownSignal {
    fired: Arc<AtomicBool>,
    notify: Arc<Notify>,
}

impl Default for ShutdownSignal {
    fn default() -> Self {
        Self::new()
    }
}

impl ShutdownSignal {
    /// Construct an un-fired signal.
    pub fn new() -> Self {
        Self {
            fired: Arc::new(AtomicBool::new(false)),
            notify: Arc::new(Notify::new()),
        }
    }

    /// Fire the signal — every current and future `wait()` resolves.
    /// Idempotent: signaling twice is a no-op (the flag stays set).
    pub fn signal(&self) {
        // Set flag BEFORE waking — guarantees that any waiter that
        // was about to call `wait()` and check the flag sees it as
        // true and skips the notified path entirely.
        self.fired.store(true, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    /// Block until the signal fires. Returns immediately if it
    /// already has. Cancel-safe — drop the future to stop waiting.
    pub async fn wait(&self) {
        if self.fired.load(Ordering::SeqCst) {
            return;
        }
        // Race: the producer might fire between our check above and
        // the registration below. `Notify::notified()` returns a
        // future that registers immediately on first poll, then
        // resolves on the next `notify_waiters` call OR if a wake
        // is already pending. The fast-path re-check below handles
        // the case where signal fired between the load and the await.
        let notified = self.notify.notified();
        if self.fired.load(Ordering::SeqCst) {
            return;
        }
        notified.await;
    }

    /// Non-blocking check. `true` once `signal()` has been called.
    pub fn is_signaled(&self) -> bool {
        self.fired.load(Ordering::SeqCst)
    }
}

/// Race any future against a [`ShutdownSignal`]. Returns
/// `Some(T)` if the future completed first, `None` if shutdown
/// won. Drops the inner future on shutdown so any HTTP / DB /
/// `tokio::time::sleep` resources held inside get released
/// promptly — no orphaned in-flight work.
///
/// Ergonomic graceful-cancel for any await call:
///
/// ```ignore
/// let resp = until_shutdown(model.invoke(msgs, &opts), &shutdown).await;
/// match resp {
///     Some(Ok(r)) => { /* normal */ }
///     Some(Err(e)) => { /* model failure */ }
///     None => { /* user hit Ctrl+C */ }
/// }
/// ```
///
/// If the signal already fired before this call, returns `None`
/// instantly without polling `fut` at all.
pub async fn until_shutdown<F, T>(
    fut: F,
    shutdown: &ShutdownSignal,
) -> Option<T>
where
    F: Future<Output = T>,
{
    // Fast-path: signal already fired — drop fut without polling.
    if shutdown.is_signaled() {
        return None;
    }
    tokio::select! {
        v = fut => Some(v),
        _ = shutdown.wait() => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn unsignaled_wait_blocks() {
        let s = ShutdownSignal::new();
        // Wait with a short timeout — must time out (i.e., wait
        // never resolved on its own).
        let r = tokio::time::timeout(Duration::from_millis(20), s.wait()).await;
        assert!(r.is_err(), "wait() should not resolve before signal");
    }

    #[tokio::test]
    async fn signal_wakes_pending_waiter() {
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        let h = tokio::spawn(async move { s2.wait().await });
        // Yield once so the spawned task parks on the Notify.
        tokio::task::yield_now().await;
        s.signal();
        // wait() should resolve quickly now.
        let r = tokio::time::timeout(Duration::from_millis(50), h)
            .await
            .expect("waiter timed out after signal");
        r.unwrap();
    }

    #[tokio::test]
    async fn late_waiter_resolves_immediately_after_signal() {
        let s = ShutdownSignal::new();
        s.signal();
        // Wait AFTER signal — must resolve instantly, no Notify
        // replay needed.
        let r = tokio::time::timeout(Duration::from_millis(10), s.wait()).await;
        assert!(r.is_ok(), "wait() should be instant after signal");
    }

    #[tokio::test]
    async fn many_concurrent_waiters_all_wake() {
        let s = ShutdownSignal::new();
        let mut handles = Vec::new();
        for _ in 0..10 {
            let s2 = s.clone();
            handles.push(tokio::spawn(async move { s2.wait().await }));
        }
        // Yield to let all 10 tasks park.
        tokio::task::yield_now().await;
        s.signal();
        // Every handle must complete quickly.
        for h in handles {
            tokio::time::timeout(Duration::from_millis(100), h)
                .await
                .expect("waiter timed out")
                .unwrap();
        }
    }

    #[tokio::test]
    async fn is_signaled_reflects_state() {
        let s = ShutdownSignal::new();
        assert!(!s.is_signaled());
        s.signal();
        assert!(s.is_signaled());
    }

    #[tokio::test]
    async fn double_signal_is_idempotent() {
        let s = ShutdownSignal::new();
        s.signal();
        s.signal(); // must not panic / deadlock
        assert!(s.is_signaled());
        // wait() still resolves instantly.
        let r = tokio::time::timeout(Duration::from_millis(10), s.wait()).await;
        assert!(r.is_ok());
    }

    #[tokio::test]
    async fn cancelled_wait_does_not_consume_signal_for_others() {
        // Drop one waiter mid-wait; others must still wake on signal.
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        let other = tokio::spawn(async move { s2.wait().await });
        // Spawn a waiter and drop it without resolving.
        let s3 = s.clone();
        let dropped = tokio::spawn(async move {
            let _ = tokio::time::timeout(
                Duration::from_millis(5),
                s3.wait(),
            )
            .await;
        });
        let _ = dropped.await;
        // Now signal — `other` must wake.
        tokio::task::yield_now().await;
        s.signal();
        let r = tokio::time::timeout(Duration::from_millis(50), other).await;
        assert!(r.is_ok(), "remaining waiter didn't wake after cancel of peer");
    }

    #[tokio::test]
    async fn default_constructs_unsignaled() {
        let s = ShutdownSignal::default();
        assert!(!s.is_signaled());
    }

    // ---- until_shutdown tests -----------------------------------------

    #[tokio::test]
    async fn until_shutdown_returns_some_when_future_wins() {
        let s = ShutdownSignal::new();
        let r = until_shutdown(async { 42_u32 }, &s).await;
        assert_eq!(r, Some(42));
    }

    #[tokio::test]
    async fn until_shutdown_returns_none_on_pre_fired_signal() {
        let s = ShutdownSignal::new();
        s.signal();
        // The inner future would never resolve, but fast-path
        // returns None without polling it.
        let never = std::future::pending::<u32>();
        let r = until_shutdown(never, &s).await;
        assert_eq!(r, None);
    }

    #[tokio::test]
    async fn until_shutdown_returns_none_when_signal_wins() {
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        let h = tokio::spawn(async move {
            until_shutdown(
                tokio::time::sleep(Duration::from_secs(60)),
                &s2,
            )
            .await
        });
        // Yield once so the spawned task parks on the select.
        tokio::task::yield_now().await;
        s.signal();
        let r = tokio::time::timeout(Duration::from_millis(100), h)
            .await
            .expect("until_shutdown didn't return after signal")
            .unwrap();
        assert!(r.is_none());
    }

    #[tokio::test]
    async fn until_shutdown_drops_inner_future_on_signal() {
        // The inner future holds a sleep; on signal, the sleep
        // should be dropped and not actually run for 60s. We
        // measure wall-clock to confirm.
        let s = ShutdownSignal::new();
        let s2 = s.clone();
        let started = std::time::Instant::now();
        let h = tokio::spawn(async move {
            until_shutdown(
                tokio::time::sleep(Duration::from_secs(60)),
                &s2,
            )
            .await
        });
        tokio::task::yield_now().await;
        s.signal();
        let _ = h.await.unwrap();
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "wall-clock {elapsed:?} — inner future wasn't dropped",
        );
    }

    #[tokio::test]
    async fn until_shutdown_passes_through_future_value() {
        let s = ShutdownSignal::new();
        let r: Option<Result<&str, &str>> =
            until_shutdown(async { Ok::<&str, &str>("done") }, &s).await;
        assert_eq!(r, Some(Ok("done")));
    }

    #[tokio::test]
    async fn clone_shares_state() {
        let s1 = ShutdownSignal::new();
        let s2 = s1.clone();
        // Wait on s2 in a task; signal via s1.
        let h = tokio::spawn(async move { s2.wait().await });
        tokio::task::yield_now().await;
        s1.signal();
        let r = tokio::time::timeout(Duration::from_millis(50), h).await;
        assert!(r.is_ok());
        // Both views see the flag.
        assert!(s1.is_signaled());
    }
}

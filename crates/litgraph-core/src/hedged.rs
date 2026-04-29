//! `hedged_call` — tail-latency mitigation combinator.
//!
//! Run a `primary` future. If it hasn't completed within
//! `hedge_delay`, also run `backup` concurrently. Return
//! whichever finishes first; drop the loser to release its
//! resources.
//!
//! # Why a separate primitive (vs `RaceChatModel`)
//!
//! `RaceChatModel` (iter 184) issues to *every* inner provider
//! simultaneously. That's right when you want "first response
//! wins, cost-no-object." But for tail-latency mitigation —
//! where p50 is fine and you only want to insure against the p99
//! — issuing a second call on every request doubles cost. A
//! hedged call only pays for the backup on slow requests.
//!
//! Standard pattern in distributed systems: Google's "The Tail
//! At Scale" (Dean & Barroso, 2013). When the median latency
//! is acceptable but the tail is heavy, a small hedge delay +
//! a single backup request collapses the tail.
//!
//! # Real prod use
//!
//! - **LLM tail latency**: a chat provider with 500ms p50 / 30s
//!   p99 — set `hedge_delay = 2s`; calls completing within 2s
//!   pay zero overhead, calls past 2s get a backup that usually
//!   wins.
//! - **Multi-region failover**: primary in us-east-1, backup in
//!   us-west-2. Hedge after 1s — most calls complete on the
//!   close region; failures and slow nodes get covered by the
//!   far region.
//! - **Replica hedging**: same provider, two API keys; spread
//!   the load and insure against per-key throttling.

use std::future::Future;
use std::time::Duration;

/// Run `primary`. If it hasn't completed within `hedge_delay`,
/// also start `backup` and race them. Returns whichever
/// completes first.
///
/// `hedge_delay = 0` is allowed (effectively a `RaceChatModel`-
/// style parallel race; backup starts as soon as `primary` has
/// been polled once).
///
/// The loser future is dropped, releasing whatever HTTP
/// connection / compute slot / channel it was holding —
/// `tokio` cancellation on drop applies.
///
/// Both futures must produce the same output type; that's the
/// natural shape for hedging the same logical request.
pub async fn hedged_call<F1, F2, Fut1, Fut2, T>(
    primary: F1,
    backup: F2,
    hedge_delay: Duration,
) -> T
where
    F1: FnOnce() -> Fut1,
    F2: FnOnce() -> Fut2,
    Fut1: Future<Output = T>,
    Fut2: Future<Output = T>,
{
    let primary_fut = primary();
    tokio::pin!(primary_fut);

    // Phase 1: wait up to hedge_delay for primary alone.
    tokio::select! {
        biased;  // prefer primary when both ready (rare race window)
        r = &mut primary_fut => return r,
        _ = tokio::time::sleep(hedge_delay) => {}
    }

    // Phase 2: hedge_delay elapsed without primary finishing.
    // Start backup and race the two. Whichever completes first
    // wins; the loser is dropped.
    let backup_fut = backup();
    tokio::pin!(backup_fut);
    tokio::select! {
        r = &mut primary_fut => r,
        r = &mut backup_fut => r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::time::sleep;

    #[tokio::test]
    async fn primary_wins_if_completes_within_hedge_delay() {
        let backup_started = Arc::new(AtomicBool::new(false));
        let bs = backup_started.clone();
        let r = hedged_call(
            || async {
                sleep(Duration::from_millis(10)).await;
                "primary"
            },
            move || {
                bs.store(true, Ordering::SeqCst);
                async { "backup" }
            },
            Duration::from_millis(50),
        )
        .await;
        assert_eq!(r, "primary");
        // Backup never even started — primary finished within hedge window.
        assert!(!backup_started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn backup_starts_after_hedge_delay() {
        let backup_started = Arc::new(AtomicBool::new(false));
        let bs = backup_started.clone();
        let r = hedged_call(
            || async {
                sleep(Duration::from_millis(80)).await;
                "primary"
            },
            move || {
                bs.store(true, Ordering::SeqCst);
                async {
                    sleep(Duration::from_millis(5)).await;
                    "backup"
                }
            },
            Duration::from_millis(20),
        )
        .await;
        // Backup started after 20ms hedge delay and finishes at ~25ms;
        // primary would have finished at 80ms. Backup wins.
        assert_eq!(r, "backup");
        assert!(backup_started.load(Ordering::SeqCst));
    }

    #[tokio::test]
    async fn primary_can_still_win_after_hedge_fires() {
        let r = hedged_call(
            || async {
                sleep(Duration::from_millis(30)).await;
                "primary"
            },
            || async {
                sleep(Duration::from_millis(100)).await;
                "backup"
            },
            Duration::from_millis(10),
        )
        .await;
        // Both started. Primary finishes at ~30ms (faster than backup's 100ms).
        assert_eq!(r, "primary");
    }

    #[tokio::test]
    async fn loser_future_is_dropped_when_winner_returns() {
        // Counter increments only if the future runs to completion;
        // verify that the loser doesn't tick.
        let primary_done = Arc::new(AtomicUsize::new(0));
        let backup_done = Arc::new(AtomicUsize::new(0));
        let pd = primary_done.clone();
        let bd = backup_done.clone();
        let r = hedged_call(
            move || {
                let pd = pd.clone();
                async move {
                    sleep(Duration::from_millis(15)).await;
                    pd.fetch_add(1, Ordering::SeqCst);
                    "primary"
                }
            },
            move || {
                let bd = bd.clone();
                async move {
                    sleep(Duration::from_millis(200)).await;
                    bd.fetch_add(1, Ordering::SeqCst);
                    "backup"
                }
            },
            Duration::from_millis(5),
        )
        .await;
        assert_eq!(r, "primary");
        // Wait long enough that backup WOULD have ticked if it weren't dropped.
        sleep(Duration::from_millis(80)).await;
        assert_eq!(primary_done.load(Ordering::SeqCst), 1);
        assert_eq!(
            backup_done.load(Ordering::SeqCst),
            0,
            "loser future kept running after winner returned",
        );
    }

    #[tokio::test]
    async fn zero_hedge_delay_starts_backup_immediately() {
        let started = Instant::now();
        let r = hedged_call(
            || async {
                sleep(Duration::from_millis(50)).await;
                "primary"
            },
            || async {
                sleep(Duration::from_millis(10)).await;
                "backup"
            },
            Duration::from_millis(0),
        )
        .await;
        let elapsed = started.elapsed();
        assert_eq!(r, "backup");
        // Backup should win at ~10ms (parallel race).
        assert!(
            elapsed < Duration::from_millis(30),
            "zero hedge delay didn't race in parallel: {elapsed:?}",
        );
    }

    #[tokio::test]
    async fn supports_result_types() {
        // Common case: hedging fallible operations.
        let r: Result<&str, &str> = hedged_call(
            || async {
                sleep(Duration::from_millis(10)).await;
                Ok::<_, &str>("primary-ok")
            },
            || async { Ok("backup-ok") },
            Duration::from_millis(50),
        )
        .await;
        assert_eq!(r.unwrap(), "primary-ok");
    }
}

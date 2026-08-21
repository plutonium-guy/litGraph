//! Tenant-scoped policy: rate limiting and spend accounting.
//!
//! These are deliberately NOT the `litgraph-resilience` decorators. Those
//! wrap a model instance and hold one bucket shared by every caller; a
//! gateway needs the opposite — per-key limits over shared deployments.
//! Wrapping deployments per key would mean N-keys × M-deployments wrapper
//! instances for the same effect.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

use crate::keys::VirtualKey;

/// Milliseconds since an arbitrary epoch. A trait so tests never sleep.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the unix epoch")
            .as_millis() as u64
    }
}

/// Manually-advanced clock for tests.
#[derive(Debug, Default)]
pub struct TestClock {
    ms: AtomicU64,
}

impl TestClock {
    pub fn new() -> Self {
        Self { ms: AtomicU64::new(0) }
    }
    pub fn advance_ms(&self, delta: u64) {
        self.ms.fetch_add(delta, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.ms.load(Ordering::SeqCst)
    }
}

/// Token bucket with millisecond-granularity lazy refill.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    state: RwLock<(f64, u64)>, // (tokens, last_refill_ms)
}

impl TokenBucket {
    pub fn per_minute(rpm: u32, now_ms: u64) -> Self {
        let capacity = rpm as f64;
        Self {
            capacity,
            refill_per_ms: capacity / 60_000.0,
            state: RwLock::new((capacity, now_ms)),
        }
    }

    /// Take one token if available. Returns the retry delay when empty.
    pub fn try_acquire(&self, now_ms: u64) -> Result<(), u64> {
        let mut st = self.state.write();
        let (ref mut tokens, ref mut last) = *st;
        let elapsed = now_ms.saturating_sub(*last);
        if elapsed > 0 {
            *tokens = (*tokens + elapsed as f64 * self.refill_per_ms).min(self.capacity);
            *last = now_ms;
        }
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            Ok(())
        } else {
            let deficit = 1.0 - *tokens;
            if self.refill_per_ms == 0.0 {
                // rpm = 0 means reject everything with a plausible retry hint
                Err(1000)
            } else {
                Err((deficit / self.refill_per_ms).ceil() as u64)
            }
        }
    }
}

/// Where per-tenant spend lives. In-memory in v1; the seam that lets a
/// Postgres implementation land without reshaping the edge.
pub trait SpendStore: Send + Sync {
    fn record(&self, key_id: &str, usd: f64);
    fn spent_today(&self, key_id: &str) -> f64;
}

pub struct MemorySpendStore {
    clock: Arc<dyn Clock>,
    // (day_index, usd) per key, where day_index = now_ms / 86_400_000
    inner: RwLock<HashMap<String, (u64, f64)>>,
}

impl std::fmt::Debug for MemorySpendStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemorySpendStore").finish_non_exhaustive()
    }
}

impl MemorySpendStore {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self { clock, inner: RwLock::new(HashMap::new()) }
    }

    fn current_day_index(&self) -> u64 {
        self.clock.now_ms() / 86_400_000
    }
}

impl SpendStore for MemorySpendStore {
    fn record(&self, key_id: &str, usd: f64) {
        let day_idx = self.current_day_index();
        let mut inner = self.inner.write();
        let entry = inner.entry(key_id.to_string()).or_insert((day_idx, 0.0));
        if entry.0 < day_idx {
            entry.0 = day_idx;
            entry.1 = 0.0;
        }
        entry.1 += usd;
    }

    fn spent_today(&self, key_id: &str) -> f64 {
        let day_idx = self.current_day_index();
        let inner = self.inner.read();
        match inner.get(key_id) {
            Some((stored_day_idx, usd)) if *stored_day_idx == day_idx => *usd,
            _ => 0.0,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum PolicyDecision {
    Allow,
    RateLimited { retry_after_ms: u64 },
    BudgetExhausted { spent_usd: f64, cap_usd: f64 },
}

/// Per-key buckets plus the spend store.
///
/// The map is read-mostly (keys change only on reload) and the values carry
/// their own interior mutability, so the hot path takes a read lock. That
/// is why this does not need `dashmap`.
pub struct TenantPolicy {
    clock: Arc<dyn Clock>,
    spend: Arc<dyn SpendStore>,
    buckets: RwLock<HashMap<String, Arc<TokenBucket>>>,
}

impl TenantPolicy {
    pub fn new(clock: Arc<dyn Clock>, spend: Arc<dyn SpendStore>) -> Self {
        Self { clock, spend, buckets: RwLock::new(HashMap::new()) }
    }

    pub fn check(&self, key: &VirtualKey) -> PolicyDecision {
        if let Some(cap) = key.max_usd_per_day {
            let spent = self.spend.spent_today(&key.id);
            if spent >= cap {
                return PolicyDecision::BudgetExhausted { spent_usd: spent, cap_usd: cap };
            }
        }
        let Some(rpm) = key.rpm else {
            return PolicyDecision::Allow;
        };
        let now = self.clock.now_ms();
        // Double-checked locking: try read-only first, then upgrade if needed
        let bucket = {
            // Check if bucket exists under read lock
            let maybe_bucket = self.buckets.read().get(&key.id).cloned();
            if let Some(b) = maybe_bucket {
                b
            } else {
                // Bucket doesn't exist, acquire write lock to create it
                let mut w = self.buckets.write();
                w.entry(key.id.clone())
                    .or_insert_with(|| Arc::new(TokenBucket::per_minute(rpm, now)))
                    .clone()
            }
        };
        match bucket.try_acquire(now) {
            Ok(()) => PolicyDecision::Allow,
            Err(retry_after_ms) => PolicyDecision::RateLimited { retry_after_ms },
        }
    }

    pub fn record_spend(&self, key_id: &str, usd: f64) {
        self.spend.record(key_id, usd);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::VirtualKey;

    fn key(id: &str, rpm: Option<u32>, cap: Option<f64>) -> VirtualKey {
        VirtualKey { id: id.into(), groups: vec!["g".into()], rpm, max_usd_per_day: cap }
    }

    #[test]
    fn bucket_allows_burst_then_refuses_until_refill() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock.clone())));
        let k = key("team-a", Some(60), None);

        // 60 rpm => capacity 60, refill 1/sec.
        for i in 0..60 {
            assert_eq!(policy.check(&k), PolicyDecision::Allow, "request {i} should pass");
        }
        assert!(matches!(policy.check(&k), PolicyDecision::RateLimited { .. }));

        clock.advance_ms(1_000);
        assert_eq!(policy.check(&k), PolicyDecision::Allow, "one token should have refilled");
    }

    #[test]
    fn rate_limits_are_per_key_and_never_shared() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock.clone())));
        let a = key("team-a", Some(1), None);
        let b = key("team-b", Some(1), None);

        assert_eq!(policy.check(&a), PolicyDecision::Allow);
        assert!(matches!(policy.check(&a), PolicyDecision::RateLimited { .. }));
        // Exhausting A must not affect B.
        assert_eq!(policy.check(&b), PolicyDecision::Allow);
    }

    #[test]
    fn no_rpm_configured_means_unlimited() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock)));
        let k = key("team-a", None, None);
        for _ in 0..1_000 {
            assert_eq!(policy.check(&k), PolicyDecision::Allow);
        }
    }

    #[test]
    fn spend_cap_rejects_only_once_the_ceiling_is_already_crossed() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));
        let policy = TenantPolicy::new(clock, store.clone());
        let k = key("team-a", None, Some(1.0));

        assert_eq!(policy.check(&k), PolicyDecision::Allow);
        // Cost is only knowable after the tokens exist, so a single request
        // may overshoot. The guarantee is "reject once over", not a hard cap.
        policy.record_spend("team-a", 1.50);
        assert!(matches!(policy.check(&k), PolicyDecision::BudgetExhausted { .. }));
    }

    #[test]
    fn spend_is_per_key_and_never_bleeds() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock));
        let policy = TenantPolicy::new(Arc::new(TestClock::new()), store.clone());

        policy.record_spend("team-a", 5.0);
        assert!((store.spent_today("team-a") - 5.0).abs() < 1e-9);
        assert_eq!(store.spent_today("team-b"), 0.0);
    }

    #[test]
    fn spend_recorded_on_day_n_is_visible_on_day_n() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));

        // Start at day 0
        store.record("team-a", 2.5);
        assert!((store.spent_today("team-a") - 2.5).abs() < 1e-9);

        // Record more on same day
        store.record("team-a", 1.5);
        assert!((store.spent_today("team-a") - 4.0).abs() < 1e-9);
    }

    #[test]
    fn advancing_clock_past_day_boundary_resets_spent_today() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));

        // Day 0
        store.record("team-a", 5.0);
        assert!((store.spent_today("team-a") - 5.0).abs() < 1e-9);

        // Advance to day 1 (86_400_000 ms per day)
        clock.advance_ms(86_400_000);
        assert_eq!(store.spent_today("team-a"), 0.0, "spend should reset on new day");
    }

    #[test]
    fn budget_exhausted_becomes_allow_after_rollover() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));
        let policy = TenantPolicy::new(clock.clone(), store.clone());
        let k = key("team-a", None, Some(1.0));

        // Day 0: exhaust budget
        assert_eq!(policy.check(&k), PolicyDecision::Allow);
        policy.record_spend("team-a", 1.50);
        assert!(matches!(policy.check(&k), PolicyDecision::BudgetExhausted { .. }));

        // Day 1: budget resets
        clock.advance_ms(86_400_000);
        assert_eq!(policy.check(&k), PolicyDecision::Allow, "should allow after day rollover");
    }

    #[test]
    fn rollover_is_per_key() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));

        // Day 0: record for both keys
        store.record("team-a", 5.0);
        store.record("team-b", 3.0);
        assert!((store.spent_today("team-a") - 5.0).abs() < 1e-9);
        assert!((store.spent_today("team-b") - 3.0).abs() < 1e-9);

        // Day 1: advance clock
        clock.advance_ms(86_400_000);

        // Both should reset independently
        assert_eq!(store.spent_today("team-a"), 0.0);
        assert_eq!(store.spent_today("team-b"), 0.0);

        // Record only for team-b
        store.record("team-b", 2.0);
        assert_eq!(store.spent_today("team-a"), 0.0, "team-a should still be 0");
        assert!((store.spent_today("team-b") - 2.0).abs() < 1e-9, "team-b should have 2.0");
    }

    #[test]
    fn rpm_zero_rejects_everything() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock, Arc::new(MemorySpendStore::new(Arc::new(TestClock::new()))));
        let k = key("team-a", Some(0), None);

        // First request should be rejected
        assert!(matches!(policy.check(&k), PolicyDecision::RateLimited { .. }));
        // Every request should be rejected
        assert!(matches!(policy.check(&k), PolicyDecision::RateLimited { .. }));
    }

    #[test]
    fn rpm_zero_retry_after_is_plausible() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock, Arc::new(MemorySpendStore::new(Arc::new(TestClock::new()))));
        let k = key("team-a", Some(0), None);

        if let PolicyDecision::RateLimited { retry_after_ms } = policy.check(&k) {
            assert!(retry_after_ms > 0 && retry_after_ms < 10_000, "retry_after_ms should be plausible, not u64::MAX");
        } else {
            panic!("expected RateLimited");
        }
    }
}

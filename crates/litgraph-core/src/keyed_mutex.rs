//! `KeyedMutex<K>` — per-key async serialization.
//!
//! A `tokio::sync::Mutex` serializes *every* caller. A
//! `KeyedMutex<K>` serializes only callers that share the same
//! key — different keys run in parallel.
//!
//! # Why a separate primitive
//!
//! Common prod pattern in agent systems: "only one task may be
//! mutating the conversation state for `thread_id` at a time" —
//! BUT thousands of different `thread_id`s should run in parallel.
//! A single global mutex serializes everyone (wrong); spawning a
//! `Mutex` per key in user code requires a registry with bounded
//! memory and correct cleanup (tedious).
//!
//! `KeyedMutex` is that registry. It uses `Weak` references so
//! entries clean themselves up: when no caller holds the lock and
//! no caller is waiting, the inner `Arc<Mutex>` is dropped and
//! the `Weak` in the map becomes stale. A subsequent `lock(same
//! key)` notices the stale `Weak` and creates a fresh `Mutex`.
//!
//! # Real prod use
//!
//! - **Per-thread agent serialization**: ReAct agent step for
//!   `thread_id=X` must finish before the next step for `X` runs;
//!   different threads are independent.
//! - **Per-user rate-coupling**: tool that mutates a user-scoped
//!   resource (Notion page, GitHub issue) must serialize per
//!   `user_id`.
//! - **Per-resource exclusivity**: only one writer per
//!   vector-store collection / table partition / shard at a time.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::{Arc, Weak};

use parking_lot::Mutex as PlMutex;
use tokio::sync::{Mutex as TokioMutex, OwnedMutexGuard};

/// Per-key async mutex registry. Cheap to clone (`Arc` inside).
pub struct KeyedMutex<K: Hash + Eq + Clone> {
    map: PlMutex<HashMap<K, Weak<TokioMutex<()>>>>,
}

impl<K: Hash + Eq + Clone> Default for KeyedMutex<K> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone> KeyedMutex<K> {
    /// Construct an empty registry.
    pub fn new() -> Self {
        Self {
            map: PlMutex::new(HashMap::new()),
        }
    }

    /// Acquire the per-key lock. Returns an `OwnedMutexGuard<()>`
    /// — drop the guard to release.
    ///
    /// Different keys never block each other. Same-key callers
    /// queue in arrival order.
    pub async fn lock(&self, key: K) -> OwnedMutexGuard<()> {
        let arc = {
            let mut map = self.map.lock();
            match map.get(&key).and_then(|w| w.upgrade()) {
                Some(arc) => arc,
                None => {
                    let arc = Arc::new(TokioMutex::new(()));
                    map.insert(key, Arc::downgrade(&arc));
                    arc
                }
            }
        };
        // Hold the Arc until lock_owned acquires; the OwnedMutexGuard
        // it returns keeps the Arc alive for the duration of the
        // critical section.
        arc.lock_owned().await
    }

    /// Drop stale `Weak` entries (where no one holds the lock and
    /// no one is waiting). Cheap; safe to call from a periodic
    /// cleanup task. Without this, stale entries accumulate at
    /// roughly 16 bytes each — fine for bounded key sets, worth
    /// running for unbounded ones (e.g. ephemeral request IDs).
    pub fn cleanup(&self) {
        let mut map = self.map.lock();
        map.retain(|_, w| w.strong_count() > 0);
    }

    /// Approximate number of keys tracked, including stale `Weak`
    /// entries. Use [`Self::cleanup`] first if you want only
    /// active keys. Mostly a debugging / observability hook.
    pub fn len(&self) -> usize {
        self.map.lock().len()
    }

    /// `true` when `len() == 0`.
    pub fn is_empty(&self) -> bool {
        self.map.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};
    use tokio::time::sleep;

    #[tokio::test]
    async fn same_key_serializes() {
        let km: Arc<KeyedMutex<&'static str>> = Arc::new(KeyedMutex::new());
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let km = km.clone();
            let inf = in_flight.clone();
            let pk = peak.clone();
            handles.push(tokio::spawn(async move {
                let _g = km.lock("user_a").await;
                let now = inf.fetch_add(1, Ordering::SeqCst) + 1;
                let cur = pk.load(Ordering::SeqCst);
                if now > cur {
                    pk.store(now, Ordering::SeqCst);
                }
                sleep(Duration::from_millis(10)).await;
                inf.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 1, "same-key didn't serialize");
    }

    #[tokio::test]
    async fn different_keys_run_in_parallel() {
        let km: Arc<KeyedMutex<usize>> = Arc::new(KeyedMutex::new());
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for i in 0..5 {
            let km = km.clone();
            let inf = in_flight.clone();
            let pk = peak.clone();
            handles.push(tokio::spawn(async move {
                let _g = km.lock(i).await;
                let now = inf.fetch_add(1, Ordering::SeqCst) + 1;
                let cur = pk.load(Ordering::SeqCst);
                if now > cur {
                    pk.store(now, Ordering::SeqCst);
                }
                sleep(Duration::from_millis(20)).await;
                inf.fetch_sub(1, Ordering::SeqCst);
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        assert!(
            peak.load(Ordering::SeqCst) >= 2,
            "different keys never ran concurrently",
        );
    }

    #[tokio::test]
    async fn lock_releases_when_guard_drops() {
        let km: Arc<KeyedMutex<&'static str>> = Arc::new(KeyedMutex::new());
        // Hold a guard for 30ms, then a second lock acquires and
        // we measure that the second lock did not start before the
        // first finished.
        let km1 = km.clone();
        let started_a = Instant::now();
        let h_a = tokio::spawn(async move {
            let _g = km1.lock("k").await;
            sleep(Duration::from_millis(30)).await;
            started_a.elapsed()
        });
        sleep(Duration::from_millis(5)).await;
        let km2 = km.clone();
        let started_b = Instant::now();
        let h_b = tokio::spawn(async move {
            let _g = km2.lock("k").await;
            started_b.elapsed()
        });
        let _ = h_a.await.unwrap();
        let elapsed_b_acquire = h_b.await.unwrap();
        assert!(
            elapsed_b_acquire >= Duration::from_millis(20),
            "second lock acquired too fast: {elapsed_b_acquire:?}",
        );
    }

    #[tokio::test]
    async fn cleanup_drops_stale_weaks() {
        let km: KeyedMutex<usize> = KeyedMutex::new();
        // Acquire-and-drop a few guards; weaks remain in the map.
        for i in 0..10 {
            let _g = km.lock(i).await;
        }
        assert_eq!(km.len(), 10);
        // After all guards have dropped, every weak is stale.
        km.cleanup();
        assert_eq!(km.len(), 0);
    }

    #[tokio::test]
    async fn cleanup_keeps_active_entries() {
        let km: Arc<KeyedMutex<usize>> = Arc::new(KeyedMutex::new());
        // Hold guard 0 across the cleanup; drop guard 1 before.
        let _g0 = km.lock(0).await;
        {
            let _g1 = km.lock(1).await;
        }
        km.cleanup();
        assert_eq!(km.len(), 1, "active entry was incorrectly evicted");
    }

    #[tokio::test]
    async fn lock_after_release_creates_fresh_entry() {
        let km: Arc<KeyedMutex<&'static str>> = Arc::new(KeyedMutex::new());
        {
            let _g = km.lock("k").await;
        }
        // Weak is now stale; lock again must succeed (creates fresh).
        let _g = km.lock("k").await;
        assert_eq!(km.len(), 1);
    }
}

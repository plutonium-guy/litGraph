//! `Singleflight<K, V>` — request-coalescing primitive.
//!
//! When N concurrent callers ask for the same key, only ONE
//! inner computation runs. The N-1 other callers await the
//! leader's result; every caller gets the same value (cloned
//! from the leader's output).
//!
//! # Why a separate primitive
//!
//! Cache-miss thundering herd: 100 agent requests all want the
//! same embedding for "default system prompt"; without
//! coalescing, 100 HTTP calls go out and the upstream rate-limits
//! everyone. A `tokio::sync::Mutex` serializes everyone but
//! still runs the compute N times. `Singleflight` runs it once
//! and broadcasts the result.
//!
//! Same shape as Go's `golang.org/x/sync/singleflight`: keyed
//! deduplication of in-flight async work.
//!
//! # Real prod use
//!
//! - **Embedding cache priming**: 50 agents starting up all want
//!   the embedding of the same long system prompt. One HTTP call
//!   serves all 50.
//! - **Tool-result coalescing**: a tool that wraps an idempotent
//!   query (latest stock price for `AAPL`) gets called by 10
//!   agent steps in the same window. One upstream call.
//! - **Lazy initialization**: 5 worker tasks all want to load the
//!   same model weights on first access. One load.

use std::collections::HashMap;
use std::hash::Hash;
#[allow(unused_imports)]
use std::sync::Arc;

use parking_lot::Mutex as PlMutex;
use tokio::sync::broadcast;

/// Request-coalescing primitive. `V` must be `Clone` because the
/// leader's result is broadcast to every concurrent caller.
/// `Arc<T>` is the typical choice when `T` itself is expensive
/// to clone.
pub struct Singleflight<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + Send + Sync + 'static,
{
    map: PlMutex<HashMap<K, broadcast::Sender<V>>>,
}

impl<K, V> Default for Singleflight<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<K, V> Singleflight<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            map: PlMutex::new(HashMap::new()),
        }
    }

    /// Number of in-flight compute calls. Useful for telemetry.
    pub fn in_flight(&self) -> usize {
        self.map.lock().len()
    }

    /// Run `compute` and return the result. If another caller
    /// is already computing for `key`, wait for that result
    /// instead of running `compute`.
    ///
    /// Errors are *not* deduped — each caller's `compute`
    /// closure is independent. To dedup error paths too, return
    /// `Result<V, ArcError>` from the closure.
    pub async fn get_or_compute<F, Fut>(&self, key: K, compute: F) -> V
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = V>,
    {
        // Either join an in-flight compute or become the leader.
        let role = {
            let mut map = self.map.lock();
            if let Some(tx) = map.get(&key) {
                Role::Follower(tx.subscribe())
            } else {
                let (tx, _rx) = broadcast::channel::<V>(1);
                map.insert(key.clone(), tx.clone());
                Role::Leader(tx)
            }
        };

        match role {
            Role::Leader(tx) => {
                let result = compute().await;
                // Remove the entry so future callers start a fresh
                // compute. Send the result to whoever subscribed
                // before we removed the entry.
                {
                    let mut map = self.map.lock();
                    map.remove(&key);
                }
                // `send` returns Err if there are no receivers; that's
                // fine — leader still has the value to return.
                let _ = tx.send(result.clone());
                result
            }
            Role::Follower(mut rx) => {
                match rx.recv().await {
                    Ok(v) => v,
                    Err(broadcast::error::RecvError::Closed) => {
                        // Leader was dropped (cancelled) before sending.
                        // Run compute ourselves — we lose dedup for this
                        // window, but we can't otherwise return.
                        compute().await
                    }
                    Err(broadcast::error::RecvError::Lagged(_)) => {
                        // Channel has capacity 1 and we're the only
                        // pending receiver per leader broadcast — Lagged
                        // shouldn't happen here, but treat it as
                        // "leader's value isn't observable to us, run
                        // compute ourselves."
                        compute().await
                    }
                }
            }
        }
    }
}

enum Role<V: Clone + Send + Sync + 'static> {
    Leader(broadcast::Sender<V>),
    Follower(broadcast::Receiver<V>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::time::sleep;

    #[tokio::test]
    async fn single_caller_runs_compute_once() {
        let sf: Singleflight<&'static str, u64> = Singleflight::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let r = sf
            .get_or_compute("k", || async move {
                c.fetch_add(1, Ordering::SeqCst);
                7
            })
            .await;
        assert_eq!(r, 7);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(sf.in_flight(), 0); // entry cleaned up after return
    }

    #[tokio::test]
    async fn concurrent_callers_share_one_compute() {
        let sf: Arc<Singleflight<&'static str, u64>> = Arc::new(Singleflight::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..10 {
            let sf = sf.clone();
            let c = calls.clone();
            handles.push(tokio::spawn(async move {
                sf.get_or_compute("shared", move || async move {
                    sleep(Duration::from_millis(20)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    42
                })
                .await
            }));
        }
        let mut got = Vec::new();
        for h in handles {
            got.push(h.await.unwrap());
        }
        assert!(got.iter().all(|&v| v == 42));
        // All 10 callers got the same result, but compute ran ONCE.
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "compute ran more than once",
        );
    }

    #[tokio::test]
    async fn different_keys_run_independently() {
        let sf: Arc<Singleflight<u64, u64>> = Arc::new(Singleflight::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for i in 0..5 {
            let sf = sf.clone();
            let c = calls.clone();
            handles.push(tokio::spawn(async move {
                sf.get_or_compute(i, move || async move {
                    sleep(Duration::from_millis(10)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    i * 10
                })
                .await
            }));
        }
        let mut got = Vec::new();
        for h in handles {
            got.push(h.await.unwrap());
        }
        got.sort();
        assert_eq!(got, vec![0, 10, 20, 30, 40]);
        // Each distinct key ran its own compute.
        assert_eq!(calls.load(Ordering::SeqCst), 5);
    }

    #[tokio::test]
    async fn second_call_after_first_completes_runs_fresh_compute() {
        let sf: Singleflight<&'static str, u64> = Singleflight::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let c = calls.clone();
        let _ = sf
            .get_or_compute("k", || async move {
                c.fetch_add(1, Ordering::SeqCst);
                1
            })
            .await;
        let c2 = calls.clone();
        let _ = sf
            .get_or_compute("k", || async move {
                c2.fetch_add(1, Ordering::SeqCst);
                2
            })
            .await;
        // Each call runs because the in-flight window closed between them.
        assert_eq!(calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn followers_get_leader_result_even_after_subscribing_late() {
        // Leader: 50ms compute. Two followers: subscribe at 5ms and 30ms.
        let sf: Arc<Singleflight<&'static str, u64>> = Arc::new(Singleflight::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let sf2 = sf.clone();
        let c = calls.clone();
        let leader = tokio::spawn(async move {
            sf2.get_or_compute("k", move || async move {
                sleep(Duration::from_millis(50)).await;
                c.fetch_add(1, Ordering::SeqCst);
                99
            })
            .await
        });
        sleep(Duration::from_millis(5)).await;
        let sf2 = sf.clone();
        let f1 = tokio::spawn(async move {
            sf2.get_or_compute("k", || async { 0 }).await
        });
        sleep(Duration::from_millis(25)).await;
        let sf2 = sf.clone();
        let f2 = tokio::spawn(async move {
            sf2.get_or_compute("k", || async { 0 }).await
        });
        let lr = leader.await.unwrap();
        let r1 = f1.await.unwrap();
        let r2 = f2.await.unwrap();
        assert_eq!(lr, 99);
        assert_eq!(r1, 99);
        assert_eq!(r2, 99);
        // Compute ran only once.
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn arc_v_for_expensive_clone_types() {
        // Common pattern: V = Arc<BigThing> so the broadcast clone
        // is just an Arc-clone, not a deep copy.
        type Big = Arc<Vec<u8>>;
        let sf: Arc<Singleflight<&'static str, Big>> = Arc::new(Singleflight::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..5 {
            let sf = sf.clone();
            let c = calls.clone();
            handles.push(tokio::spawn(async move {
                sf.get_or_compute("buf", move || async move {
                    sleep(Duration::from_millis(10)).await;
                    c.fetch_add(1, Ordering::SeqCst);
                    Arc::new(vec![1, 2, 3, 4, 5])
                })
                .await
            }));
        }
        for h in handles {
            let r = h.await.unwrap();
            assert_eq!(&*r, &vec![1, 2, 3, 4, 5]);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

//! `PriorityQueue<T>` — async priority work queue.
//!
//! Push items with a `u64` priority; pop returns the highest-
//! priority item. Within the same priority, FIFO order is
//! preserved via an insertion sequence number.
//!
//! # Distinct from `tokio::sync::mpsc`
//!
//! `mpsc` is FIFO across the whole channel — a high-priority
//! task pushed late waits behind every earlier-pushed task. A
//! priority queue lets schedulers reorder work after-the-fact:
//!
//! - **Urgent retries first**: a graph node that failed and is
//!   being re-scheduled jumps ahead of fresh work.
//! - **Hard cases first**: an eval harness scores the
//!   most-likely-to-fail rows first so a cancelled run
//!   surfaces failures fastest.
//! - **Latency budget**: a UI-driven request gets `priority=10`
//!   and jumps the batch.
//!
//! # Real prod use
//!
//! Pair with [`crate::shutdown::ShutdownSignal`] for graceful
//! drain — `pop_with_shutdown` returns `None` when the signal
//! fires, so worker tasks can exit cleanly. Combine with
//! [`crate::Bulkhead`] (iter 248) to enforce a concurrent-call
//! cap on top of the priority queue.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};
use std::sync::Arc;

use parking_lot::Mutex as PlMutex;
use tokio::sync::Notify;

use crate::ShutdownSignal;

/// Internal entry — public-via-`Send` so multi-thread uses are
/// fine. Higher priority sorts first; on ties, lower seq sorts
/// first (FIFO).
struct Entry<T> {
    priority: u64,
    seq: u64,
    item: T,
}

impl<T> Eq for Entry<T> {}

impl<T> PartialEq for Entry<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl<T> Ord for Entry<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Max-heap on priority: higher priority returns Greater so
        // `BinaryHeap::pop` returns it first. On ties, lower seq is
        // Greater (so it pops first → FIFO within priority).
        match self.priority.cmp(&other.priority) {
            Ordering::Equal => other.seq.cmp(&self.seq),
            ord => ord,
        }
    }
}

impl<T> PartialOrd for Entry<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Async priority queue. Cheap to clone (`Arc` inside).
pub struct PriorityQueue<T: Send + 'static> {
    inner: PlMutex<BinaryHeap<Entry<T>>>,
    seq: AtomicU64,
    notify: Notify,
}

impl<T: Send + 'static> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> PriorityQueue<T> {
    pub fn new() -> Self {
        Self {
            inner: PlMutex::new(BinaryHeap::new()),
            seq: AtomicU64::new(0),
            notify: Notify::new(),
        }
    }

    /// Push an item. Higher `priority` pops first. Within the
    /// same priority, items pop in push order (FIFO).
    pub fn push(self: &Arc<Self>, priority: u64, item: T) {
        let seq = self.seq.fetch_add(1, AtomicOrd::SeqCst);
        self.inner.lock().push(Entry {
            priority,
            seq,
            item,
        });
        // Wake one waiter; if there are none, this is a no-op.
        self.notify.notify_one();
    }

    /// Try to pop without blocking. Returns the highest-priority
    /// item, or `None` if empty.
    pub fn try_pop(&self) -> Option<T> {
        self.inner.lock().pop().map(|e| e.item)
    }

    /// Block until an item is available, then pop it. Cancel-safe.
    pub async fn pop(self: &Arc<Self>) -> T {
        loop {
            if let Some(item) = self.try_pop() {
                return item;
            }
            // Register notified BEFORE the next try_pop — same
            // pattern as ShutdownSignal::wait, so we don't miss a
            // push that races between our load and our sleep.
            let notified = self.notify.notified();
            if let Some(item) = self.try_pop() {
                return item;
            }
            notified.await;
        }
    }

    /// Race the pop against a [`ShutdownSignal`]. Returns
    /// `Some(item)` if an item became available, `None` if
    /// shutdown fired first.
    pub async fn pop_with_shutdown(
        self: &Arc<Self>,
        shutdown: &ShutdownSignal,
    ) -> Option<T> {
        loop {
            if let Some(item) = self.try_pop() {
                return Some(item);
            }
            if shutdown.is_signaled() {
                return None;
            }
            let notified = self.notify.notified();
            if let Some(item) = self.try_pop() {
                return Some(item);
            }
            if shutdown.is_signaled() {
                return None;
            }
            tokio::select! {
                _ = notified => continue,
                _ = shutdown.wait() => return None,
            }
        }
    }

    pub fn len(&self) -> usize {
        self.inner.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::time::{sleep, timeout};

    #[tokio::test]
    async fn try_pop_returns_none_when_empty() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        assert!(q.try_pop().is_none());
    }

    #[tokio::test]
    async fn higher_priority_pops_first() {
        let q: Arc<PriorityQueue<&'static str>> = Arc::new(PriorityQueue::new());
        q.push(1, "low");
        q.push(10, "high");
        q.push(5, "mid");
        assert_eq!(q.try_pop(), Some("high"));
        assert_eq!(q.try_pop(), Some("mid"));
        assert_eq!(q.try_pop(), Some("low"));
        assert!(q.try_pop().is_none());
    }

    #[tokio::test]
    async fn fifo_within_same_priority() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        for i in 0..5 {
            q.push(7, i);
        }
        let got: Vec<_> = (0..5).map(|_| q.try_pop().unwrap()).collect();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn pop_blocks_until_push() {
        let q: Arc<PriorityQueue<&'static str>> = Arc::new(PriorityQueue::new());
        let q2 = q.clone();
        let h = tokio::spawn(async move { q2.pop().await });
        sleep(Duration::from_millis(20)).await;
        // Worker is waiting; push.
        q.push(1, "x");
        let r = timeout(Duration::from_millis(50), h)
            .await
            .expect("pop didn't unblock")
            .unwrap();
        assert_eq!(r, "x");
    }

    #[tokio::test]
    async fn many_concurrent_poppers_each_get_one_item() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        let mut handles = Vec::new();
        for _ in 0..5 {
            let q = q.clone();
            handles.push(tokio::spawn(async move { q.pop().await }));
        }
        sleep(Duration::from_millis(20)).await;
        for i in 0..5 {
            q.push(0, i);
        }
        let mut got = Vec::new();
        for h in handles {
            got.push(timeout(Duration::from_millis(100), h).await.unwrap().unwrap());
        }
        got.sort();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn pop_with_shutdown_returns_none_on_shutdown() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        let s = ShutdownSignal::new();
        let q2 = q.clone();
        let s2 = s.clone();
        let h = tokio::spawn(async move { q2.pop_with_shutdown(&s2).await });
        sleep(Duration::from_millis(20)).await;
        s.signal();
        let r = timeout(Duration::from_millis(50), h)
            .await
            .expect("pop_with_shutdown didn't unblock")
            .unwrap();
        assert_eq!(r, None);
    }

    #[tokio::test]
    async fn pop_with_shutdown_pre_fired_returns_none_immediately() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        let s = ShutdownSignal::new();
        s.signal();
        let r = timeout(Duration::from_millis(50), q.pop_with_shutdown(&s))
            .await
            .unwrap();
        assert_eq!(r, None);
    }

    #[tokio::test]
    async fn pop_with_shutdown_returns_item_when_pushed_first() {
        let q: Arc<PriorityQueue<&'static str>> = Arc::new(PriorityQueue::new());
        let s = ShutdownSignal::new();
        q.push(1, "ready");
        let r = q.pop_with_shutdown(&s).await;
        assert_eq!(r, Some("ready"));
    }

    #[tokio::test]
    async fn priority_overrides_insertion_order() {
        // Three items pushed: low, low, then high. Pop order: high, low, low.
        let q: Arc<PriorityQueue<&'static str>> = Arc::new(PriorityQueue::new());
        q.push(1, "low_a");
        q.push(1, "low_b");
        q.push(99, "urgent");
        assert_eq!(q.try_pop(), Some("urgent"));
        assert_eq!(q.try_pop(), Some("low_a"));
        assert_eq!(q.try_pop(), Some("low_b"));
    }

    #[tokio::test]
    async fn len_and_is_empty_track_state() {
        let q: Arc<PriorityQueue<u32>> = Arc::new(PriorityQueue::new());
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
        q.push(1, 42);
        assert_eq!(q.len(), 1);
        assert!(!q.is_empty());
        let _ = q.try_pop();
        assert_eq!(q.len(), 0);
        assert!(q.is_empty());
    }
}

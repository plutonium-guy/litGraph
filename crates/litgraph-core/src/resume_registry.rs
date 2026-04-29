//! `ResumeRegistry` — coordination primitive that pairs paused work
//! with externally-signalable resume values via
//! `tokio::sync::oneshot`.
//!
//! # Why this exists
//!
//! LangGraph's interrupt-resume pattern: an agent run hits a
//! `Command(interrupt)` and parks. Some external event (a Slack
//! click, a human approval, a webhook callback) eventually decides
//! the resume value. The runtime needs a way to:
//!
//! 1. Park an agent thread on a resume signal **without busy-polling**.
//! 2. Let an HTTP handler (or any other event source) **signal**
//!    that thread asynchronously, with a JSON payload.
//! 3. Detect cancellation (the agent gave up waiting) cleanly.
//!
//! `tokio::sync::oneshot` is the textbook fit: one sender, one
//! receiver, fires once. `ResumeRegistry` is the dispatch table that
//! pairs senders with receivers by thread id.
//!
//! # First oneshot primitive in the iter 180-200 lineage
//!
//! Channel shapes shipped before this:
//! - `mpsc` (iter 189 fan-in, iter 196 multi-stage pipeline)
//! - `broadcast` (iter 195 1→N fan-out)
//! - `watch` (iter 199 latest-value)
//! - `oneshot` (this — single-fire signal)

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tokio::sync::oneshot;

use crate::error::{Error, Result};

/// Future returned by [`ResumeRegistry::register`]. Awaiting it
/// yields the resume value when [`ResumeRegistry::resume`] is called
/// for the same thread id, or `None` when the registry cancels the
/// wait (via `cancel` or by being dropped).
pub struct ResumeFuture {
    rx: oneshot::Receiver<Value>,
}

impl ResumeFuture {
    /// Await the resume value. Returns `Some(v)` on `resume`,
    /// `None` on `cancel` / sender drop.
    pub async fn await_resume(self) -> Option<Value> {
        self.rx.await.ok()
    }
}

/// Per-thread oneshot dispatch. Cheap to clone (`Arc` inside) so a
/// single registry can be shared across an axum router, the runtime,
/// and the agent loop.
#[derive(Clone, Default)]
pub struct ResumeRegistry {
    inner: Arc<Mutex<HashMap<String, oneshot::Sender<Value>>>>,
}

impl ResumeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a paused thread expecting a resume. Returns the
    /// future to await on. Re-registering an already-pending thread
    /// returns `Err` — caller is expected to use a fresh thread id
    /// or `cancel` first.
    pub fn register(&self, thread_id: &str) -> Result<ResumeFuture> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| Error::other("resume registry poisoned"))?;
        if guard.contains_key(thread_id) {
            return Err(Error::other(format!(
                "thread `{thread_id}` is already pending a resume"
            )));
        }
        let (tx, rx) = oneshot::channel();
        guard.insert(thread_id.to_string(), tx);
        Ok(ResumeFuture { rx })
    }

    /// Deliver a resume `value` to the thread that registered for
    /// `thread_id`. Returns `Err` if no thread is registered (or it
    /// already resolved). The waiting future resolves with the value.
    pub fn resume(&self, thread_id: &str, value: Value) -> Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| Error::other("resume registry poisoned"))?;
        let tx = guard
            .remove(thread_id)
            .ok_or_else(|| Error::other(format!("no pending resume for `{thread_id}`")))?;
        // `send` returns Err only if the receiver was dropped — we
        // surface that as Ok since the caller can't do anything
        // about it (the receiver gave up waiting).
        let _ = tx.send(value);
        Ok(())
    }

    /// Cancel a pending resume. The waiting future resolves with
    /// `None`. Returns `true` if a pending entry was cancelled,
    /// `false` if no thread was registered.
    pub fn cancel(&self, thread_id: &str) -> bool {
        let Ok(mut guard) = self.inner.lock() else {
            return false;
        };
        guard.remove(thread_id).is_some()
    }

    /// Number of currently-pending threads.
    pub fn pending_count(&self) -> usize {
        self.inner.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Currently-pending thread ids (snapshot). Useful for admin
    /// UIs that list which threads are waiting.
    pub fn pending_ids(&self) -> Vec<String> {
        self.inner
            .lock()
            .map(|g| g.keys().cloned().collect())
            .unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn register_then_resume_resolves_with_value() {
        let r = ResumeRegistry::new();
        let fut = r.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        // Yield once so the spawned task parks on the receiver.
        tokio::task::yield_now().await;
        r.resume("t1", json!({"reply": "hi"})).unwrap();
        let got = h.await.unwrap();
        assert_eq!(got, Some(json!({"reply": "hi"})));
    }

    #[tokio::test]
    async fn cancel_resolves_future_with_none() {
        let r = ResumeRegistry::new();
        let fut = r.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;
        let cancelled = r.cancel("t1");
        assert!(cancelled);
        let got = h.await.unwrap();
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn resume_without_register_errors() {
        let r = ResumeRegistry::new();
        let err = r.resume("t-unknown", json!({})).unwrap_err();
        assert!(format!("{err}").contains("no pending resume"));
    }

    #[tokio::test]
    async fn double_register_same_id_errors() {
        let r = ResumeRegistry::new();
        let _f1 = r.register("t1").unwrap();
        match r.register("t1") {
            Ok(_) => panic!("expected double-register to error"),
            Err(e) => assert!(format!("{e}").contains("already pending")),
        }
    }

    #[tokio::test]
    async fn double_resume_second_errors() {
        let r = ResumeRegistry::new();
        let fut = r.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;
        r.resume("t1", json!(1)).unwrap();
        let _ = h.await;
        // Second resume: thread no longer registered.
        let err = r.resume("t1", json!(2)).unwrap_err();
        assert!(format!("{err}").contains("no pending resume"));
    }

    #[tokio::test]
    async fn cancel_unknown_returns_false() {
        let r = ResumeRegistry::new();
        assert!(!r.cancel("never-registered"));
    }

    #[tokio::test]
    async fn multiple_threads_isolated() {
        let r = ResumeRegistry::new();
        let f_a = r.register("a").unwrap();
        let f_b = r.register("b").unwrap();
        let f_c = r.register("c").unwrap();
        let h_a = tokio::spawn(async move { f_a.await_resume().await });
        let h_b = tokio::spawn(async move { f_b.await_resume().await });
        let h_c = tokio::spawn(async move { f_c.await_resume().await });
        tokio::task::yield_now().await;
        r.resume("b", json!("B-val")).unwrap();
        r.cancel("c");
        r.resume("a", json!("A-val")).unwrap();
        let (a, b, c) = tokio::join!(h_a, h_b, h_c);
        assert_eq!(a.unwrap(), Some(json!("A-val")));
        assert_eq!(b.unwrap(), Some(json!("B-val")));
        assert_eq!(c.unwrap(), None);
    }

    #[tokio::test]
    async fn pending_count_and_ids_track_state() {
        let r = ResumeRegistry::new();
        assert_eq!(r.pending_count(), 0);
        let _f1 = r.register("t1").unwrap();
        let _f2 = r.register("t2").unwrap();
        assert_eq!(r.pending_count(), 2);
        let mut ids = r.pending_ids();
        ids.sort();
        assert_eq!(ids, vec!["t1".to_string(), "t2".to_string()]);
        r.cancel("t1");
        assert_eq!(r.pending_count(), 1);
    }

    #[tokio::test]
    async fn registry_clone_shares_state() {
        let r = ResumeRegistry::new();
        let r2 = r.clone();
        let fut = r.register("t1").unwrap();
        let h = tokio::spawn(async move { fut.await_resume().await });
        tokio::task::yield_now().await;
        // Resume from the cloned handle.
        r2.resume("t1", json!("from-clone")).unwrap();
        let got = h.await.unwrap();
        assert_eq!(got, Some(json!("from-clone")));
    }

    #[tokio::test]
    async fn dropped_receiver_makes_resume_succeed_silently() {
        // If the agent gives up waiting (drops the future), the
        // sender's `send` returns Err, but `resume()` swallows it
        // — caller has nothing useful to do with that signal.
        let r = ResumeRegistry::new();
        let fut = r.register("t1").unwrap();
        drop(fut); // simulate agent giving up
        // resume() should succeed (returns Ok), even though no one
        // is listening.
        let res = r.resume("t1", json!({}));
        assert!(res.is_ok());
    }

    #[test]
    fn default_registry_is_empty() {
        let r = ResumeRegistry::default();
        assert_eq!(r.pending_count(), 0);
        assert!(r.pending_ids().is_empty());
    }
}

//! Callback bus — subscribers register once, receive batched event slices.
//!
//! # Why batched?
//!
//! When the consumer is Python, each individual `on_event` call costs one GIL
//! acquisition. A streaming LLM may emit 100+ token events per second per thread;
//! acquiring the GIL that often is enough to stall everything. We buffer events on
//! the Rust side and hand Python a `Vec<Event>` per tick (default 16ms). Python
//! code gets called ~60×/s instead of per-event.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use parking_lot::Mutex;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio::time::interval;
use tracing::warn;

use crate::event::Event;

#[async_trait]
pub trait Callback: Send + Sync {
    /// Receive a batch of events. May be called concurrently from multiple tasks.
    async fn on_events(&self, events: &[Event]);
}

/// Broadcast handle returned to event producers. Cheap to clone.
#[derive(Clone)]
pub struct CallbackHandle {
    tx: mpsc::UnboundedSender<Event>,
}

impl CallbackHandle {
    pub fn emit(&self, ev: Event) {
        // Unbounded so producers never block; the bus task drains in batches.
        // If the receiver is gone (bus shut down), we silently drop.
        let _ = self.tx.send(ev);
    }
}

/// The event bus. Holds a set of `Callback`s; a single task drains the channel
/// and forwards to every subscriber in batches.
pub struct CallbackBus {
    subscribers: Arc<Mutex<Vec<Arc<dyn Callback>>>>,
    flush_every: Duration,
    max_batch: usize,
}

impl CallbackBus {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(Vec::new())),
            flush_every: Duration::from_millis(16),
            max_batch: 256,
        }
    }

    pub fn with_flush_interval(mut self, d: Duration) -> Self { self.flush_every = d; self }
    pub fn with_max_batch(mut self, n: usize) -> Self { self.max_batch = n; self }

    pub fn subscribe(&self, cb: Arc<dyn Callback>) {
        self.subscribers.lock().push(cb);
    }

    /// Start the drain task. Returns a handle used to emit events, plus a join
    /// handle so the caller can await shutdown.
    pub fn start(self) -> (CallbackHandle, JoinHandle<()>) {
        let (tx, rx) = mpsc::unbounded_channel::<Event>();
        let subs = self.subscribers.clone();
        let flush_every = self.flush_every;
        let max_batch = self.max_batch;

        let handle = tokio::spawn(async move {
            drain(rx, subs, flush_every, max_batch).await;
        });
        (CallbackHandle { tx }, handle)
    }
}

impl Default for CallbackBus {
    fn default() -> Self { Self::new() }
}

async fn drain(
    mut rx: mpsc::UnboundedReceiver<Event>,
    subs: Arc<Mutex<Vec<Arc<dyn Callback>>>>,
    flush_every: Duration,
    max_batch: usize,
) {
    let mut tick = interval(flush_every);
    tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut buf: Vec<Event> = Vec::with_capacity(max_batch);

    loop {
        tokio::select! {
            biased;
            // Drain any immediately-available events first, up to max_batch.
            evt = rx.recv() => {
                match evt {
                    Some(e) => {
                        buf.push(e);
                        while buf.len() < max_batch {
                            match rx.try_recv() {
                                Ok(e) => buf.push(e),
                                Err(_) => break,
                            }
                        }
                        // If we've hit the batch cap, flush now instead of waiting for tick.
                        if buf.len() >= max_batch {
                            flush(&subs, &mut buf).await;
                        }
                    }
                    None => {
                        // Channel closed — final flush then exit.
                        if !buf.is_empty() { flush(&subs, &mut buf).await; }
                        return;
                    }
                }
            }
            _ = tick.tick() => {
                if !buf.is_empty() { flush(&subs, &mut buf).await; }
            }
        }
    }
}

async fn flush(
    subs: &Arc<Mutex<Vec<Arc<dyn Callback>>>>,
    buf: &mut Vec<Event>,
) {
    // Clone the subscriber list under the mutex, then drop the lock before awaiting.
    let list: Vec<Arc<dyn Callback>> = subs.lock().iter().cloned().collect();
    let events = std::mem::take(buf);
    for s in list {
        let slice = events.clone(); // subscribers may run concurrently — give each an owned copy
        // Swallow per-subscriber panics so one bad callback doesn't take out the bus.
        let res = std::panic::AssertUnwindSafe(async move { s.on_events(&slice).await });
        use futures::FutureExt;
        if let Err(e) = res.catch_unwind().await {
            warn!(?e, "callback panicked — dropping events for that subscriber");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tokio::time::sleep;

    struct Counter { hits: Arc<AtomicUsize>, events: Arc<AtomicUsize> }

    #[async_trait]
    impl Callback for Counter {
        async fn on_events(&self, events: &[Event]) {
            self.hits.fetch_add(1, Ordering::SeqCst);
            self.events.fetch_add(events.len(), Ordering::SeqCst);
        }
    }

    #[tokio::test]
    async fn batches_multiple_events_into_one_call() {
        let bus = CallbackBus::new().with_flush_interval(Duration::from_millis(10));
        let hits = Arc::new(AtomicUsize::new(0));
        let events = Arc::new(AtomicUsize::new(0));
        bus.subscribe(Arc::new(Counter { hits: hits.clone(), events: events.clone() }));
        let (h, _task) = bus.start();

        for i in 0..100 {
            h.emit(Event::Custom { name: "ping".into(), payload: i.into(), ts_ms: 0 });
        }
        // Let the drain task run a few ticks.
        sleep(Duration::from_millis(60)).await;

        let total_events = events.load(Ordering::SeqCst);
        let batches = hits.load(Ordering::SeqCst);
        assert_eq!(total_events, 100);
        // Expect significantly fewer batches than events — the whole point of batching.
        assert!(batches > 0 && batches < 100, "batches={batches} events={total_events}");
    }
}

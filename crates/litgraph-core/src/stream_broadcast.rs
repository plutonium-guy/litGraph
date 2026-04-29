//! `broadcast_chat_stream` — fan ONE `ChatStream` out to N concurrent
//! subscribers via `tokio::sync::broadcast`.
//!
//! # Why this exists (vs. iter 189 multiplex_chat_streams)
//!
//! | Iter | Primitive                  | Direction      | Channel kind |
//! |------|----------------------------|----------------|--------------|
//! | 189  | `multiplex_chat_streams`   | N → 1 (fan-in) | `mpsc`       |
//! | 195  | `broadcast_chat_stream`    | 1 → N (fan-out)| `broadcast`  |
//!
//! Real use cases:
//!
//! - **Live UI + audit log**: user sees tokens; an audit subscriber
//!   logs the same chunks; both consume the stream concurrently.
//! - **Multi-pane debugger**: every connected client sees the same
//!   tokens of an agent's reply, no replay needed.
//! - **Sidecar evaluators**: a judge subscriber watches the model's
//!   tokens in flight to early-stop on policy violations.
//!
//! # Semantics of `tokio::sync::broadcast`
//!
//! - Every subscriber gets **every** event sent **after** they
//!   subscribed. Late subscribers miss earlier events — the
//!   broadcast channel doesn't replay history.
//! - Each subscriber has its own `capacity`-sized lookahead. A slow
//!   subscriber that falls behind by `capacity` events gets a
//!   `Lagged(skipped)` notice and resumes from the next available
//!   chunk; **fast subscribers are not blocked** by a slow one.
//! - When the upstream sender drops, every receiver eventually
//!   sees `Closed`.
//!
//! Lagged behaviour is exposed via the `BroadcastEvent::Lagged`
//! variant so consumers can decide whether a missed-tokens UI render
//! is acceptable for their use-case.

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use futures::{Stream, StreamExt};
use tokio::sync::broadcast;
use tokio_stream::wrappers::BroadcastStream;

use crate::error::Result;
use crate::model::{ChatStream, ChatStreamEvent};

/// One item on a subscriber stream. `Event` is the same shape as the
/// upstream `Result<ChatStreamEvent>`. `Lagged` signals that the
/// subscriber fell behind the broadcast capacity and `skipped` events
/// are gone — the stream continues from the next available chunk.
#[derive(Debug, Clone)]
pub enum BroadcastEvent {
    Event(std::result::Result<ChatStreamEvent, String>),
    Lagged { skipped: u64 },
}

/// Stream alias for one subscriber.
pub type BroadcastSubscriberStream =
    Pin<Box<dyn Stream<Item = BroadcastEvent> + Send>>;

/// Handle to a configured-but-not-yet-pumping broadcast. The first
/// call to [`BroadcastHandle::subscribe`] spawns the pump task —
/// **subscribers MUST join before the pump starts** to see all
/// events, because tokio's `broadcast` channel doesn't replay
/// events to late subscribers.
///
/// Why lazy: if the pump were spawned at `broadcast_chat_stream`
/// time, it could race subscribers. Late subscribers would miss
/// the first events, which surprises callers with simple "create
/// + subscribe + drain" code. Lazy spawn closes the race: every
/// subscriber that joins before the first `subscribe` call gets
/// every event.
#[derive(Clone)]
pub struct BroadcastHandle {
    /// `Option` so the pump can drop the inner sender when upstream
    /// is exhausted. With both pump-clone AND handle-clone dropped,
    /// `tokio::sync::broadcast`'s internal sender count hits zero and
    /// receivers see `RecvError::Closed` — without this, subscribers
    /// would block forever after the upstream finishes.
    sender: Arc<Mutex<Option<broadcast::Sender<std::result::Result<ChatStreamEvent, String>>>>>,
    /// Holds the upstream until the first subscribe call. `Mutex`
    /// for `Send + Sync`; only ever taken once.
    upstream: Arc<Mutex<Option<ChatStream>>>,
}

impl BroadcastHandle {
    /// Number of currently active receivers, or 0 if the handle's
    /// sender has already been dropped (after pump exit).
    pub fn receiver_count(&self) -> usize {
        self.sender
            .lock()
            .expect("poisoned")
            .as_ref()
            .map(|s| s.receiver_count())
            .unwrap_or(0)
    }

    /// Subscribe a new consumer. Receives every event sent **after**
    /// this subscriber was registered.
    ///
    /// First subscribe call spawns the pump task; subsequent calls
    /// just create new receivers on the existing channel. Returns an
    /// empty stream if the handle's sender has already been dropped
    /// (pump finished and no more events will ever arrive).
    pub fn subscribe(&self) -> BroadcastSubscriberStream {
        // Create the receiver FIRST so it's registered before the
        // pump produces any events.
        let rx = match self.sender.lock().expect("poisoned").as_ref() {
            Some(s) => s.subscribe(),
            None => {
                // Pump already finished and sender was dropped.
                // Return an empty stream.
                return Box::pin(futures::stream::empty());
            }
        };
        // Spawn the pump exactly once, on the first call. Take the
        // upstream out of its slot; future calls find None and skip.
        let mut guard = self.upstream.lock().expect("poisoned");
        if let Some(stream) = guard.take() {
            let sender_holder = self.sender.clone();
            // Pump holds its OWN sender clone (via the holder). When
            // upstream exhausts, the pump explicitly clears the
            // sender holder so the handle's reference is also dropped
            // — once both are gone, the broadcast channel closes and
            // every subscriber sees `RecvError::Closed`.
            let tx_pump = sender_holder
                .lock()
                .expect("poisoned")
                .as_ref()
                .expect("sender present at construction")
                .clone();
            tokio::spawn(async move {
                let mut s = stream;
                while let Some(item) = s.next().await {
                    let payload = item.map_err(|e| e.to_string());
                    let _ = tx_pump.send(payload);
                }
                // Drop the pump's sender clone first.
                drop(tx_pump);
                // Then clear the handle's slot — drops the original
                // sender, closing the channel. Subscribers waiting on
                // `recv()` now resolve to RecvError::Closed.
                sender_holder.lock().expect("poisoned").take();
            });
        }
        drop(guard);
        wrap_receiver(rx)
    }
}

fn wrap_receiver(
    rx: broadcast::Receiver<std::result::Result<ChatStreamEvent, String>>,
) -> BroadcastSubscriberStream {
    let bs = BroadcastStream::new(rx);
    let mapped = bs.map(|item| match item {
        Ok(payload) => BroadcastEvent::Event(payload),
        Err(tokio_stream::wrappers::errors::BroadcastStreamRecvError::Lagged(n)) => {
            BroadcastEvent::Lagged { skipped: n }
        }
    });
    Box::pin(mapped)
}

/// Drain `upstream` into a `tokio::sync::broadcast::channel` of
/// `capacity` slots; return a handle from which any number of
/// subscribers can be created.
///
/// The pump is spawned on the current Tokio runtime and runs until
/// `upstream` is exhausted. After upstream ends, the broadcast sender
/// is dropped, signalling `Closed` to all current and future
/// subscribers.
///
/// # Picking `capacity`
///
/// `capacity` is the per-subscriber lookahead before a slow consumer
/// starts dropping events with `Lagged`. For typical token streams
/// (one delta = one chunk), `capacity = 1024` is generous — a
/// subscriber would have to be ~30s behind a 30 tok/sec model before
/// losing data. Lower if memory is tight; higher if you have known-slow
/// downstream consumers (e.g., disk-flushing audit log) and want them
/// never to drop events.
pub fn broadcast_chat_stream(
    upstream: ChatStream,
    capacity: usize,
) -> Result<BroadcastHandle> {
    let cap = capacity.max(1);
    let (tx, _rx0) = broadcast::channel::<std::result::Result<ChatStreamEvent, String>>(cap);
    drop(_rx0);
    Ok(BroadcastHandle {
        sender: Arc::new(Mutex::new(Some(tx))),
        upstream: Arc::new(Mutex::new(Some(upstream))),
    })
}

/// Convenience: broadcast a `ChatStream` AND immediately produce one
/// subscriber stream. The returned subscriber starts from the first
/// upstream event (it's created before the pump is spawned). Useful
/// when the caller is the "main" consumer and the broadcast handle
/// is a shared sidecar.
pub fn broadcast_chat_stream_with_main(
    upstream: ChatStream,
    capacity: usize,
) -> Result<(BroadcastHandle, BroadcastSubscriberStream)> {
    let cap = capacity.max(1);
    let (tx, rx_main) = broadcast::channel::<std::result::Result<ChatStreamEvent, String>>(cap);
    let main_stream: BroadcastSubscriberStream = wrap_receiver(rx_main);
    let sender_holder: Arc<Mutex<Option<broadcast::Sender<_>>>> =
        Arc::new(Mutex::new(Some(tx)));
    // Pump can run immediately because `rx_main` is already registered.
    let pump_holder = sender_holder.clone();
    let tx_pump = pump_holder
        .lock()
        .expect("poisoned")
        .as_ref()
        .expect("sender present at construction")
        .clone();
    tokio::spawn(async move {
        let mut s = upstream;
        while let Some(item) = s.next().await {
            let payload = item.map_err(|e| e.to_string());
            let _ = tx_pump.send(payload);
        }
        drop(tx_pump);
        pump_holder.lock().expect("poisoned").take();
    });
    Ok((
        BroadcastHandle {
            sender: sender_holder,
            // Pump already running; nothing to spawn lazily.
            upstream: Arc::new(Mutex::new(None)),
        },
        main_stream,
    ))
}

/// Helper for tests: drain a subscriber stream into a `Vec` of text
/// deltas, ignoring `Lagged` notices.
pub async fn collect_deltas(mut stream: BroadcastSubscriberStream) -> Vec<String> {
    let mut out = Vec::new();
    while let Some(ev) = stream.next().await {
        if let BroadcastEvent::Event(Ok(ChatStreamEvent::Delta { text })) = ev {
            out.push(text);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Error;
    use crate::model::{ChatResponse, FinishReason, TokenUsage};
    use crate::Message;
    use async_stream::stream;
    use std::time::Duration;

    fn upstream_with(deltas: Vec<&'static str>) -> ChatStream {
        let owned: Vec<String> = deltas.into_iter().map(String::from).collect();
        let s = stream! {
            for t in owned {
                tokio::time::sleep(Duration::from_millis(1)).await;
                yield Ok(ChatStreamEvent::Delta { text: t });
            }
            yield Ok(ChatStreamEvent::Done {
                response: ChatResponse {
                    message: Message::assistant("done"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "test".into(),
                },
            });
        };
        Box::pin(s)
    }

    #[tokio::test]
    async fn one_subscriber_sees_every_delta() {
        let upstream = upstream_with(vec!["hello", " ", "world"]);
        let handle = broadcast_chat_stream(upstream, 16).unwrap();
        let sub = handle.subscribe();
        let texts = collect_deltas(sub).await;
        assert_eq!(texts, vec!["hello", " ", "world"]);
    }

    #[tokio::test]
    async fn two_subscribers_each_get_full_stream() {
        let upstream = upstream_with(vec!["a", "b", "c"]);
        let handle = broadcast_chat_stream(upstream, 16).unwrap();
        let sub_a = handle.subscribe();
        let sub_b = handle.subscribe();
        // Drain both concurrently so neither lags.
        let (texts_a, texts_b) = tokio::join!(
            collect_deltas(sub_a),
            collect_deltas(sub_b),
        );
        assert_eq!(texts_a, vec!["a", "b", "c"]);
        assert_eq!(texts_b, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn late_subscriber_misses_earlier_events() {
        // Subscribe AFTER the upstream has already produced + been
        // consumed by the first subscriber. Late subscriber should
        // see Closed almost immediately.
        let upstream = upstream_with(vec!["x", "y"]);
        let handle = broadcast_chat_stream(upstream, 16).unwrap();
        let sub_main = handle.subscribe();
        let main_texts = collect_deltas(sub_main).await;
        assert_eq!(main_texts, vec!["x", "y"]);

        // Late subscriber: nothing left.
        let late = handle.subscribe();
        let late_texts = collect_deltas(late).await;
        assert!(late_texts.is_empty());
    }

    #[tokio::test]
    async fn lagged_subscriber_emits_lagged_event() {
        // capacity=2: the upstream sends 5 events into the channel
        // while the slow subscriber sleeps. By the time it reads,
        // 3+ events are gone → Lagged(3) (or more).
        let upstream = upstream_with(vec!["a", "b", "c", "d", "e"]);
        let handle = broadcast_chat_stream(upstream, 2).unwrap();
        let mut slow = handle.subscribe();
        // Wait long enough for upstream to have produced all events.
        tokio::time::sleep(Duration::from_millis(100)).await;
        let mut saw_lagged = false;
        while let Some(ev) = slow.next().await {
            if matches!(ev, BroadcastEvent::Lagged { .. }) {
                saw_lagged = true;
            }
        }
        assert!(saw_lagged, "slow subscriber should have lagged");
    }

    #[tokio::test]
    async fn slow_subscriber_does_not_block_fast_subscriber() {
        let upstream = upstream_with(vec!["a", "b", "c"]);
        let handle = broadcast_chat_stream(upstream, 64).unwrap();
        let fast = handle.subscribe();
        let slow = handle.subscribe();
        // Drain `fast` immediately; `slow` waits.
        let fast_handle = tokio::spawn(collect_deltas(fast));
        // Slow stays idle for a moment then drains.
        tokio::time::sleep(Duration::from_millis(20)).await;
        let slow_handle = tokio::spawn(collect_deltas(slow));
        let (fast_texts, slow_texts) = tokio::try_join!(fast_handle, slow_handle).unwrap();
        // Fast saw the full stream.
        assert_eq!(fast_texts, vec!["a", "b", "c"]);
        // Slow saw the full stream too (capacity=64 was generous).
        assert_eq!(slow_texts, vec!["a", "b", "c"]);
    }

    #[tokio::test]
    async fn upstream_error_propagates_as_event_err() {
        let s = stream! {
            yield Ok(ChatStreamEvent::Delta { text: "ok".into() });
            yield Err(Error::other("synthetic"));
        };
        let upstream: ChatStream = Box::pin(s);
        let handle = broadcast_chat_stream(upstream, 16).unwrap();
        let mut sub = handle.subscribe();
        let mut saw_err = false;
        let mut saw_ok = false;
        while let Some(ev) = sub.next().await {
            if let BroadcastEvent::Event(Ok(ChatStreamEvent::Delta { .. })) = ev {
                saw_ok = true;
            }
            if let BroadcastEvent::Event(Err(msg)) = ev {
                assert!(msg.contains("synthetic"), "got: {msg}");
                saw_err = true;
            }
        }
        assert!(saw_ok);
        assert!(saw_err);
    }

    #[tokio::test]
    async fn receiver_count_tracks_subscribers() {
        let upstream = upstream_with(vec!["x"]);
        let handle = broadcast_chat_stream(upstream, 8).unwrap();
        assert_eq!(handle.receiver_count(), 0);
        let _a = handle.subscribe();
        assert_eq!(handle.receiver_count(), 1);
        let _b = handle.subscribe();
        assert_eq!(handle.receiver_count(), 2);
    }

    #[tokio::test]
    async fn with_main_returns_first_subscriber_too() {
        let upstream = upstream_with(vec!["one", "two"]);
        let (handle, main) = broadcast_chat_stream_with_main(upstream, 16).unwrap();
        let extra = handle.subscribe();
        let (main_texts, extra_texts) =
            tokio::join!(collect_deltas(main), collect_deltas(extra));
        assert_eq!(main_texts, vec!["one", "two"]);
        assert_eq!(extra_texts, vec!["one", "two"]);
    }
}

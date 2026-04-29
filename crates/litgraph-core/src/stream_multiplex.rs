//! `multiplex_chat_streams` — invoke N `ChatModel::stream` calls
//! concurrently, fan their token deltas into a single tagged stream.
//!
//! This is the "see N models reply live, side-by-side" primitive.
//! Useful for:
//!
//! - **Live multi-model demos / debug UIs** — show GPT, Claude, and
//!   Gemini token streams interleaved as they arrive.
//! - **Multi-judge live evals** — three judges score the same answer
//!   token by token; UI updates the lowest-running score in real time.
//! - **A/B shadow streaming** — primary model serves the user; shadow
//!   model also streams so you can compare on a side panel.
//!
//! # Parallelism shape
//!
//! Each inner `stream()` call is wrapped in a `tokio::spawn` task.
//! Tasks forward their `Result<ChatStreamEvent>` items into a single
//! `tokio::sync::mpsc::channel`; the outer `Stream` simply yields
//! whatever lands on the channel next, with the sending model's
//! label attached. So:
//!
//! - A slow model **never blocks** a fast model's tokens.
//! - Errors on one model arrive as a tagged failure event but
//!   **do not** poison the others.
//! - When every model has emitted its final `Done` (or errored),
//!   the receiver drains and the multiplexed stream ends.
//!
//! Distinct from earlier iters (180–188) — those used JoinSet /
//! Semaphore / Rayon. This is the first channel-based fan-in pattern
//! in the codebase.

use std::pin::Pin;
use std::sync::Arc;

use async_stream::stream;
use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::error::{Error, Result};
use crate::model::{ChatModel, ChatOptions, ChatStreamEvent};
use crate::Message;

/// One chunk in the multiplexed stream — the inner `event` is whatever
/// the originating model produced (delta, tool-call partial, done),
/// tagged with the caller-supplied `model_label` so consumers can
/// route per-model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiplexEvent {
    /// Label the caller assigned to the originating model (e.g.
    /// `"openai-gpt-4o"`). Free-form — multiplex doesn't interpret it.
    pub model_label: String,
    pub event: ChatStreamEvent,
}

/// Stream alias for the multiplexer's output. Items are
/// `Result<MultiplexEvent>` so a per-model error surfaces as a
/// `Some(Err(...))` and the stream continues with the rest.
pub type MultiplexStream =
    Pin<Box<dyn Stream<Item = Result<MultiplexEvent>> + Send>>;

/// Run N model streams concurrently, return a tagged-multiplexed
/// stream over their events.
///
/// `models` is `(label, model)` pairs — labels show up in
/// `MultiplexEvent.model_label`. The `messages` and `opts` are sent
/// to **every** inner model identically; if you need per-model prompt
/// shaping, wrap each model in a middleware first.
///
/// The buffer size of the internal channel is `8 * models.len()`,
/// chosen so that any one model can run a few chunks ahead without
/// blocking, but a runaway producer can't OOM the process.
pub fn multiplex_chat_streams(
    models: Vec<(String, Arc<dyn ChatModel>)>,
    messages: Vec<Message>,
    opts: ChatOptions,
) -> MultiplexStream {
    let cap = (models.len() * 8).max(1);
    Box::pin(stream! {
        if models.is_empty() {
            return;
        }
        let (tx, mut rx) = mpsc::channel::<Result<MultiplexEvent>>(cap);

        for (label, model) in models {
            let tx = tx.clone();
            let msgs = messages.clone();
            let opts = opts.clone();
            tokio::spawn(async move {
                let stream = match model.stream(msgs, &opts).await {
                    Ok(s) => s,
                    Err(e) => {
                        // stream() failed before yielding any chunk —
                        // tag it and forward, but don't kill the
                        // multiplex (other models keep running).
                        let _ = tx
                            .send(Err(Error::other(format!(
                                "[{label}] stream init: {e}",
                            ))))
                            .await;
                        return;
                    }
                };
                let mut s = stream;
                while let Some(item) = s.next().await {
                    let payload = match item {
                        Ok(ev) => Ok(MultiplexEvent {
                            model_label: label.clone(),
                            event: ev,
                        }),
                        Err(e) => Err(Error::other(format!("[{label}] {e}"))),
                    };
                    if tx.send(payload).await.is_err() {
                        // Receiver dropped — caller stopped consuming;
                        // exit cleanly to avoid wasting work.
                        return;
                    }
                }
            });
        }
        // Drop the cloning sender so the channel closes when all
        // worker tasks finish.
        drop(tx);

        // Yield items as they arrive — `Result` per item so a per-model
        // error is observable but does NOT short-circuit the rest.
        while let Some(item) = rx.recv().await {
            yield item;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatStream, FinishReason, TokenUsage};
    use async_stream::stream;
    use async_trait::async_trait;
    use std::time::Duration;

    /// Streams a fixed sequence of deltas, each preceded by a sleep.
    /// Lets us interleave events deterministically.
    struct ScriptedStreamModel {
        label: &'static str,
        deltas: Vec<(u64, String)>, // (sleep_ms, text)
        succeed: bool,
    }

    #[async_trait]
    impl ChatModel for ScriptedStreamModel {
        fn name(&self) -> &str {
            self.label
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<crate::ChatResponse> {
            unimplemented!("invoke not used in multiplex tests")
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            if !self.succeed {
                return Err(Error::other(format!("{} failed at stream init", self.label)));
            }
            let deltas = self.deltas.clone();
            let label = self.label;
            let s = stream! {
                for (sleep_ms, text) in deltas {
                    tokio::time::sleep(Duration::from_millis(sleep_ms)).await;
                    yield Ok(ChatStreamEvent::Delta { text });
                }
                yield Ok(ChatStreamEvent::Done {
                    response: crate::ChatResponse {
                        message: Message::assistant(format!("done from {label}")),
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::default(),
                        model: label.to_string(),
                    }
                });
            };
            Ok(Box::pin(s))
        }
    }

    fn arc_scripted(
        label: &'static str,
        deltas: Vec<(u64, &'static str)>,
        succeed: bool,
    ) -> Arc<dyn ChatModel> {
        Arc::new(ScriptedStreamModel {
            label,
            deltas: deltas
                .into_iter()
                .map(|(s, t)| (s, t.to_string()))
                .collect(),
            succeed,
        })
    }

    fn delta_text(ev: &ChatStreamEvent) -> Option<&str> {
        if let ChatStreamEvent::Delta { text } = ev {
            Some(text)
        } else {
            None
        }
    }

    fn is_done(ev: &ChatStreamEvent) -> bool {
        matches!(ev, ChatStreamEvent::Done { .. })
    }

    #[tokio::test]
    async fn empty_models_yields_empty() {
        let mut s = multiplex_chat_streams(
            vec![],
            vec![Message::user("hi")],
            ChatOptions::default(),
        );
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn single_model_passthrough() {
        let m = arc_scripted("only", vec![(0, "hi"), (0, "!")], true);
        let mut s = multiplex_chat_streams(
            vec![("only".into(), m)],
            vec![Message::user("q")],
            ChatOptions::default(),
        );
        let mut deltas: Vec<String> = Vec::new();
        let mut saw_done = false;
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            assert_eq!(ev.model_label, "only");
            if let Some(t) = delta_text(&ev.event) {
                deltas.push(t.into());
            }
            if is_done(&ev.event) {
                saw_done = true;
            }
        }
        assert_eq!(deltas, vec!["hi", "!"]);
        assert!(saw_done);
    }

    #[tokio::test]
    async fn two_models_interleave_by_arrival() {
        // A emits "a1" at 1ms, "a2" at 30ms.
        // B emits "b1" at 10ms, "b2" at 20ms.
        // Expected order on the receiver: a1, b1, b2, a2 (by arrival).
        let a = arc_scripted("A", vec![(1, "a1"), (30, "a2")], true);
        let b = arc_scripted("B", vec![(10, "b1"), (10, "b2")], true);
        let mut s = multiplex_chat_streams(
            vec![("A".into(), a), ("B".into(), b)],
            vec![Message::user("q")],
            ChatOptions::default(),
        );
        let mut order: Vec<(String, String)> = Vec::new();
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            if let Some(t) = delta_text(&ev.event) {
                order.push((ev.model_label, t.into()));
            }
        }
        // a1 must come first; a2 must come last.
        assert_eq!(order.first().unwrap().1, "a1");
        assert_eq!(order.last().unwrap().1, "a2");
        // both b's come between.
        let middle: Vec<&String> = order
            .iter()
            .skip(1)
            .take(order.len() - 2)
            .map(|(_, t)| t)
            .collect();
        assert!(middle.iter().any(|t| *t == "b1"));
        assert!(middle.iter().any(|t| *t == "b2"));
    }

    #[tokio::test]
    async fn slow_model_does_not_block_fast_model() {
        // Fast: 2 quick deltas. Slow: huge delay between deltas.
        // Fast must complete before slow's second delta even arrives.
        let fast = arc_scripted("fast", vec![(0, "f1"), (0, "f2")], true);
        let slow = arc_scripted("slow", vec![(5, "s1"), (200, "s2")], true);
        let mut s = multiplex_chat_streams(
            vec![("fast".into(), fast), ("slow".into(), slow)],
            vec![Message::user("q")],
            ChatOptions::default(),
        );
        // Drain just the first 2 fast deltas — that should happen well
        // before the 200ms slow delay.
        let started = std::time::Instant::now();
        let mut fast_seen = 0;
        while let Some(item) = s.next().await {
            let ev = item.unwrap();
            if ev.model_label == "fast" {
                if let Some(_) = delta_text(&ev.event) {
                    fast_seen += 1;
                    if fast_seen == 2 {
                        break;
                    }
                }
            }
        }
        let elapsed_ms = started.elapsed().as_millis();
        assert!(
            elapsed_ms < 100,
            "fast deltas took {elapsed_ms}ms — slow model blocked",
        );
    }

    #[tokio::test]
    async fn one_failing_model_does_not_kill_others() {
        // failing's stream() init fails; the good model still streams.
        let bad = arc_scripted("bad", vec![], false);
        let good = arc_scripted("good", vec![(0, "g1")], true);
        let mut s = multiplex_chat_streams(
            vec![("bad".into(), bad), ("good".into(), good)],
            vec![Message::user("q")],
            ChatOptions::default(),
        );
        let mut got_good = false;
        let mut got_err = false;
        while let Some(item) = s.next().await {
            match item {
                Ok(ev) if ev.model_label == "good" => {
                    if delta_text(&ev.event) == Some("g1") {
                        got_good = true;
                    }
                }
                Ok(_) => {}
                Err(e) => {
                    assert!(
                        format!("{e}").contains("[bad]"),
                        "error not tagged: {e}",
                    );
                    got_err = true;
                }
            }
        }
        assert!(got_good, "good model output missing");
        assert!(got_err, "bad model error not propagated");
    }

    #[tokio::test]
    async fn stream_terminates_when_all_done() {
        let a = arc_scripted("A", vec![(0, "x")], true);
        let b = arc_scripted("B", vec![(0, "y")], true);
        let mut s = multiplex_chat_streams(
            vec![("A".into(), a), ("B".into(), b)],
            vec![Message::user("q")],
            ChatOptions::default(),
        );
        let mut count = 0usize;
        while let Some(item) = s.next().await {
            let _ = item.unwrap();
            count += 1;
        }
        // Each model: 1 delta + 1 done = 2 events. Total = 4.
        assert_eq!(count, 4);
    }
}

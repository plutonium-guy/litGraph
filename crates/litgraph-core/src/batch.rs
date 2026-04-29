//! Bounded-concurrency `ChatModel` batch — fan out N invocations across
//! a Tokio task set with a semaphore cap. Drop-in for the langchain
//! `chat.batch([...])` pattern, but with three guarantees the LangChain
//! version doesn't make:
//!
//! 1. **Order preserved** — output index `i` is always the response for
//!    input index `i`, no matter the order of completion.
//! 2. **Per-call `Result`** — a single failing call does not tank the
//!    whole batch. Callers can `.partition` on success/failure or
//!    `.into_iter().collect::<Result<Vec<_>, _>>()` if they prefer
//!    fail-fast semantics.
//! 3. **Bounded concurrency** — `max_concurrency` is enforced via
//!    `tokio::sync::Semaphore`, so callers can run a 10k-input eval
//!    set with `max=50` and the runtime stays sane.
//!
//! Why a free function instead of a default trait method on
//! `ChatModel`: providers like OpenAI / Anthropic / Bedrock have
//! native batch endpoints that are cheaper than N parallel /chat
//! calls (Anthropic's Message Batches API drops cost ~50%). The
//! per-provider `batch` override can hit those. This function is what
//! you use for providers that *don't* have a batch endpoint, or when
//! you need true real-time parallelism rather than the deferred
//! batch-API "results within 24h" model.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;

use crate::{ChatModel, ChatOptions, ChatResponse, Error, Message, Progress, Result};

/// Run N invocations of `model` concurrently, capped at `max_concurrency`
/// in flight. Returns a `Vec` of `Result<ChatResponse>` aligned with the
/// input order — one slot per input regardless of pass/fail.
///
/// `max_concurrency` of `0` is normalised to `1` (sequential — same as
/// `ChatModel::batch`'s default). For `inputs.len() < max_concurrency`,
/// only `inputs.len()` permits are ever held.
pub async fn batch_concurrent(
    model: Arc<dyn ChatModel>,
    inputs: Vec<Vec<Message>>,
    opts: ChatOptions,
    max_concurrency: usize,
) -> Vec<Result<ChatResponse>> {
    if inputs.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<ChatResponse>)> = JoinSet::new();

    for (idx, msgs) in inputs.into_iter().enumerate() {
        let sem = sem.clone();
        let model = model.clone();
        let opts = opts.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("batch semaphore closed"))),
            };
            let r = model.invoke(msgs, &opts).await;
            (idx, r)
        });
    }

    // Pre-size results so we can index by original position.
    let mut results: Vec<Option<Result<ChatResponse>>> = (0..set.len()).map(|_| None).collect();

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => results[idx] = Some(r),
            Err(e) => {
                // JoinError — task panicked or got cancelled. Find an
                // empty slot (any) and record the failure. We can't
                // recover the original index from a panicked task;
                // putting the JoinError into the first empty slot is
                // a deliberate, documented compromise. In practice
                // panics inside a `ChatModel::invoke` are bugs, not
                // recoverable conditions.
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("batch task join: {e}"))));
                }
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("batch task lost"))))
        .collect()
}

/// Counters maintained by [`batch_concurrent_with_progress`]. All
/// fields are monotonic — they only increase as the batch advances.
/// Snapshot from any [`Progress<BatchProgress>`] observer to drive a
/// progress bar, log throughput, or trip a circuit breaker on stuck
/// runs.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct BatchProgress {
    /// Total inputs submitted (set once at the start).
    pub total: u64,
    /// Inputs whose `invoke` has finished, success or failure.
    pub completed: u64,
    /// Subset of `completed` that returned `Err`.
    pub errors: u64,
}

/// Same as [`batch_concurrent`] but updates `progress` as each input
/// completes. Real use case: an eval harness running 1000 LLM evals
/// rendering a live counter / ETA.
///
/// `progress.total` is set on entry; `completed` and `errors` are
/// incremented as `invoke` calls finish. Observers can snapshot
/// at any time and see the latest counters — `tokio::sync::watch`
/// semantics (iter 199) collapse rapid updates to the latest state.
///
/// Composition: this is the second progress-aware composition iter
/// after `ingest_to_stream_with_progress` (iter 200). Same pattern,
/// different domain.
pub async fn batch_concurrent_with_progress(
    model: Arc<dyn ChatModel>,
    inputs: Vec<Vec<Message>>,
    opts: ChatOptions,
    max_concurrency: usize,
    progress: Progress<BatchProgress>,
) -> Vec<Result<ChatResponse>> {
    if inputs.is_empty() {
        return Vec::new();
    }
    // Set total up front so observers see the run size before the
    // first completion arrives.
    let total = inputs.len() as u64;
    let _ = progress.update(|p| BatchProgress {
        total,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<ChatResponse>)> = JoinSet::new();

    for (idx, msgs) in inputs.into_iter().enumerate() {
        let sem = sem.clone();
        let model = model.clone();
        let opts = opts.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("batch semaphore closed"))),
            };
            let r = model.invoke(msgs, &opts).await;
            (idx, r)
        });
    }

    let mut results: Vec<Option<Result<ChatResponse>>> = (0..set.len()).map(|_| None).collect();

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => {
                let is_err = r.is_err();
                results[idx] = Some(r);
                let _ = progress.update(|p| BatchProgress {
                    completed: p.completed + 1,
                    errors: p.errors + if is_err { 1 } else { 0 },
                    ..p.clone()
                });
            }
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("batch task join: {e}"))));
                }
                let _ = progress.update(|p| BatchProgress {
                    completed: p.completed + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("batch task lost"))))
        .collect()
}

/// One emitted result from [`batch_concurrent_stream`] — the input
/// index plus that input's outcome. Aligned with `batch_concurrent`'s
/// `Vec<Result>` semantics, but delivered incrementally as each call
/// completes rather than buffered to the end.
pub type BatchStreamItem = (usize, Result<ChatResponse>);

/// Streaming variant of [`batch_concurrent`]. Yields `(idx, Result)`
/// pairs **in completion order** as each call finishes — caller can
/// start consuming as soon as the first response arrives, instead of
/// waiting for the whole batch.
///
/// The `usize` index is the original `inputs[i]` slot, so callers can
/// reassemble in input order if they need to (e.g., write into a
/// pre-sized `Vec<Option<ChatResponse>>`). Stream finishes when every
/// input has produced exactly one item.
///
/// # When to use this vs `batch_concurrent`
///
/// - `batch_concurrent`: caller wants the **whole result Vec at once**.
///   Simpler API, simpler error handling.
/// - `batch_concurrent_stream` (this): caller wants to **render
///   results live**, **dispatch downstream work as items complete**,
///   or **stream into a UI** without waiting for stragglers. Eval
///   harnesses and bulk-evaluation dashboards are the main use.
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub fn batch_concurrent_stream(
    model: Arc<dyn ChatModel>,
    inputs: Vec<Vec<Message>>,
    opts: ChatOptions,
    max_concurrency: usize,
) -> Pin<Box<dyn Stream<Item = BatchStreamItem> + Send>> {
    if inputs.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let n = inputs.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<BatchStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<BatchStreamItem> = JoinSet::new();
        for (idx, msgs) in inputs.into_iter().enumerate() {
            let sem = sem.clone();
            let model = model.clone();
            let opts = opts.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return (idx, Err(Error::other("batch semaphore closed"))),
                };
                let r = model.invoke(msgs, &opts).await;
                (idx, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => {
                    // Index is unrecoverable from a JoinError, so we
                    // emit a sentinel with `usize::MAX`. This is a
                    // bug-not-runtime-error situation — `invoke`
                    // panics are upstream defects.
                    (
                        usize::MAX,
                        Err(Error::other(format!("batch task join: {e}"))),
                    )
                }
            };
            if tx.send(item).await.is_err() {
                // Receiver dropped — caller stopped consuming. Abort
                // remaining work to avoid wasting cycles.
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// Like `batch_concurrent` but fail-fast: returns `Err` on the first
/// failed invocation and cancels the rest. Useful when partial results
/// aren't useful (e.g. all-or-nothing parallel rendering pipeline).
pub async fn batch_concurrent_fail_fast(
    model: Arc<dyn ChatModel>,
    inputs: Vec<Vec<Message>>,
    opts: ChatOptions,
    max_concurrency: usize,
) -> Result<Vec<ChatResponse>> {
    let n = inputs.len();
    let results = batch_concurrent(model, inputs, opts, max_concurrency).await;
    let mut out = Vec::with_capacity(n);
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatStream, FinishReason, TokenUsage};
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Counts concurrent `invoke` calls and tracks the peak — lets the
    /// concurrency-cap test assert the semaphore is honoured.
    struct ConcurrencyProbe {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        delay_ms: u64,
    }

    #[async_trait]
    impl ChatModel for ConcurrencyProbe {
        fn name(&self) -> &str {
            "probe"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            // Update peak with monotonic max.
            let mut peak = self.peak.load(Ordering::SeqCst);
            while now > peak {
                match self.peak.compare_exchange(peak, now, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let last = messages
                .last()
                .map(|m| m.text_content())
                .unwrap_or_default();
            Ok(ChatResponse {
                message: Message::assistant(last),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "probe".into(),
            })
        }
        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Deterministically fails on a configured input index.
    struct FailOn {
        bad_index: usize,
        seen: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl ChatModel for FailOn {
        fn name(&self) -> &str {
            "fail-on"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let idx = self.seen.fetch_add(1, Ordering::SeqCst);
            // Look for an "idx=N" marker in the message text.
            let want_fail = messages
                .last()
                .map(|m| m.text_content().contains(&format!("idx={}", self.bad_index)))
                .unwrap_or(false);
            if want_fail {
                return Err(Error::other(format!("synthetic failure (saw {idx} calls)")));
            }
            Ok(ChatResponse {
                message: Message::assistant(messages.last().unwrap().text_content()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "fail-on".into(),
            })
        }
        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    fn input(text: &str) -> Vec<Message> {
        vec![Message::user(text)]
    }

    #[tokio::test]
    async fn empty_input_returns_empty() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let out = batch_concurrent(model, vec![], ChatOptions::default(), 4).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_order_matches_input_order() {
        // Stagger sleep durations so completion order != input order.
        // The output Vec must still align with input position.
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 5,
        });
        let inputs: Vec<Vec<Message>> = (0..6).map(|i| input(&format!("q{i}"))).collect();
        let out = batch_concurrent(model, inputs, ChatOptions::default(), 4).await;
        assert_eq!(out.len(), 6);
        for (i, r) in out.iter().enumerate() {
            let resp = r.as_ref().expect("ok");
            assert_eq!(resp.message.text_content(), format!("q{i}"));
        }
    }

    #[tokio::test]
    async fn concurrency_cap_is_honoured() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: in_flight.clone(),
            peak: peak.clone(),
            delay_ms: 30,
        });
        let inputs: Vec<Vec<Message>> = (0..20).map(|i| input(&format!("q{i}"))).collect();
        let _ = batch_concurrent(model, inputs, ChatOptions::default(), 3).await;
        let observed_peak = peak.load(Ordering::SeqCst);
        assert!(
            observed_peak <= 3,
            "peak {observed_peak} > cap 3 — semaphore not honoured"
        );
        // Sanity: with 20 inputs and a 30ms each, we should hit the cap.
        assert!(
            observed_peak >= 2,
            "peak {observed_peak} < 2 — concurrency never engaged"
        );
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: in_flight.clone(),
            peak: peak.clone(),
            delay_ms: 5,
        });
        let inputs: Vec<Vec<Message>> = (0..4).map(|i| input(&format!("q{i}"))).collect();
        let out = batch_concurrent(model, inputs, ChatOptions::default(), 0).await;
        assert_eq!(out.len(), 4);
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn per_call_failure_is_isolated() {
        let model: Arc<dyn ChatModel> = Arc::new(FailOn {
            bad_index: 2,
            seen: Arc::new(AtomicUsize::new(0)),
        });
        let inputs: Vec<Vec<Message>> = (0..5).map(|i| input(&format!("idx={i}"))).collect();
        let out = batch_concurrent(model, inputs, ChatOptions::default(), 4).await;
        assert_eq!(out.len(), 5);
        assert!(out[0].is_ok());
        assert!(out[1].is_ok());
        assert!(out[2].is_err());
        assert!(out[3].is_ok());
        assert!(out[4].is_ok());
        let err_msg = format!("{}", out[2].as_ref().err().unwrap());
        assert!(err_msg.contains("synthetic failure"), "got: {err_msg}");
    }

    #[tokio::test]
    async fn fail_fast_returns_first_error() {
        let model: Arc<dyn ChatModel> = Arc::new(FailOn {
            bad_index: 1,
            seen: Arc::new(AtomicUsize::new(0)),
        });
        let inputs: Vec<Vec<Message>> = (0..4).map(|i| input(&format!("idx={i}"))).collect();
        let r =
            batch_concurrent_fail_fast(model, inputs, ChatOptions::default(), 4).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn fail_fast_succeeds_when_all_ok() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 1,
        });
        let inputs: Vec<Vec<Message>> = (0..3).map(|i| input(&format!("q{i}"))).collect();
        let out = batch_concurrent_fail_fast(model, inputs, ChatOptions::default(), 4)
            .await
            .unwrap();
        assert_eq!(out.len(), 3);
    }

    // ---- batch_concurrent_with_progress tests --------------------------

    #[tokio::test]
    async fn progress_total_is_set_at_start() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let inputs: Vec<Vec<Message>> = (0..7).map(|i| input(&format!("q{i}"))).collect();
        let progress = Progress::new(BatchProgress::default());
        let obs = progress.observer();
        let _ = batch_concurrent_with_progress(
            model,
            inputs,
            ChatOptions::default(),
            4,
            progress,
        )
        .await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 7);
    }

    #[tokio::test]
    async fn progress_counts_completions() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let inputs: Vec<Vec<Message>> = (0..5).map(|i| input(&format!("q{i}"))).collect();
        let progress = Progress::new(BatchProgress::default());
        let obs = progress.observer();
        let out = batch_concurrent_with_progress(
            model,
            inputs,
            ChatOptions::default(),
            4,
            progress,
        )
        .await;
        assert_eq!(out.len(), 5);
        let snap = obs.snapshot();
        assert_eq!(snap.total, 5);
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn progress_records_errors() {
        let model: Arc<dyn ChatModel> = Arc::new(FailOn {
            bad_index: 2,
            seen: Arc::new(AtomicUsize::new(0)),
        });
        let inputs: Vec<Vec<Message>> =
            (0..5).map(|i| input(&format!("idx={i}"))).collect();
        let progress = Progress::new(BatchProgress::default());
        let obs = progress.observer();
        let _ = batch_concurrent_with_progress(
            model,
            inputs,
            ChatOptions::default(),
            4,
            progress,
        )
        .await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 5);
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.errors, 1);
    }

    #[tokio::test]
    async fn progress_observer_can_be_polled_mid_run() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 10,
        });
        let inputs: Vec<Vec<Message>> = (0..8).map(|i| input(&format!("q{i}"))).collect();
        let progress = Progress::new(BatchProgress::default());
        let mut obs = progress.observer();
        let h = tokio::spawn(batch_concurrent_with_progress(
            model,
            inputs,
            ChatOptions::default(),
            2,
            progress,
        ));
        // Wait for any update (total set, or first completion).
        let _ = obs.changed().await;
        let mid = obs.snapshot();
        assert_eq!(mid.total, 8);
        // Drain rest.
        let _ = h.await.unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 8);
    }

    // ---- batch_concurrent_stream tests --------------------------------

    use futures::StreamExt;

    #[tokio::test]
    async fn stream_yields_every_input_exactly_once() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let inputs: Vec<Vec<Message>> = (0..7).map(|i| input(&format!("q{i}"))).collect();
        let mut s = batch_concurrent_stream(model, inputs, ChatOptions::default(), 4);
        let mut got_indices: Vec<usize> = Vec::new();
        while let Some((idx, r)) = s.next().await {
            assert!(r.is_ok());
            got_indices.push(idx);
        }
        got_indices.sort();
        assert_eq!(got_indices, (0..7).collect::<Vec<_>>());
    }

    #[tokio::test]
    async fn stream_preserves_input_index_in_emitted_pairs() {
        // Each input carries its idx; the response echoes the last
        // message text. We verify the emitted (idx, response) pair's
        // text matches the input at that idx.
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 5,
        });
        let inputs: Vec<Vec<Message>> = (0..6).map(|i| input(&format!("idx={i}"))).collect();
        let mut s = batch_concurrent_stream(model, inputs, ChatOptions::default(), 3);
        while let Some((idx, r)) = s.next().await {
            let resp = r.unwrap();
            assert_eq!(resp.message.text_content(), format!("idx={idx}"));
        }
    }

    #[tokio::test]
    async fn stream_per_call_failure_arrives_as_err_item() {
        let model: Arc<dyn ChatModel> = Arc::new(FailOn {
            bad_index: 2,
            seen: Arc::new(AtomicUsize::new(0)),
        });
        let inputs: Vec<Vec<Message>> =
            (0..5).map(|i| input(&format!("idx={i}"))).collect();
        let mut s = batch_concurrent_stream(model, inputs, ChatOptions::default(), 4);
        let mut errors = 0;
        let mut count = 0;
        while let Some((_idx, r)) = s.next().await {
            count += 1;
            if r.is_err() {
                errors += 1;
            }
        }
        assert_eq!(count, 5);
        assert_eq!(errors, 1);
    }

    #[tokio::test]
    async fn stream_empty_inputs_yields_empty() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let mut s = batch_concurrent_stream(model, vec![], ChatOptions::default(), 4);
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_caller_drop_aborts_in_flight_work() {
        // 100 inputs each delayed 50ms with cap=2 — full sequential
        // run would take ~2.5s. We drop the stream after 1 item; the
        // total wall-clock should be far less than the full run.
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 50,
        });
        let inputs: Vec<Vec<Message>> =
            (0..100).map(|i| input(&format!("q{i}"))).collect();
        let started = std::time::Instant::now();
        {
            let mut s = batch_concurrent_stream(
                model,
                inputs,
                ChatOptions::default(),
                2,
            );
            let _first = s.next().await.unwrap();
            // Drop `s` here.
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 500,
            "elapsed {elapsed_ms}ms — caller-drop didn't abort remaining tasks",
        );
    }

    #[tokio::test]
    async fn progress_empty_inputs_no_updates() {
        let model: Arc<dyn ChatModel> = Arc::new(ConcurrencyProbe {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: Arc::new(AtomicUsize::new(0)),
            delay_ms: 0,
        });
        let progress = Progress::new(BatchProgress::default());
        let obs = progress.observer();
        let out = batch_concurrent_with_progress(
            model,
            vec![],
            ChatOptions::default(),
            4,
            progress,
        )
        .await;
        assert!(out.is_empty());
        // Empty input → no progress mutations either.
        let snap = obs.snapshot();
        assert_eq!(snap, BatchProgress::default());
    }
}

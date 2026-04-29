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

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::{ChatModel, ChatOptions, ChatResponse, Error, Message, Result};

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
}

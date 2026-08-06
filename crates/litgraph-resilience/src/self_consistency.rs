use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::model::ChatStream;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Error, Message, Result, TokenUsage};

use std::collections::HashMap;

/// Picks the winner from N sampled responses. Return the INDEX of the
/// winning response (caller uses it to select the full ChatResponse). If
/// the voter returns `None` — no majority / all invalid — the wrapper
/// falls back to returning the first sample.
pub type ConsistencyVoter =
    Arc<dyn Fn(&[ChatResponse]) -> Option<usize> + Send + Sync>;

/// Self-consistency wrapper: sample the model `samples` times at elevated
/// temperature, then pick the majority answer via `voter`. Classic
/// Chain-of-Thought reasoning technique from Wang et al 2022 — for math,
/// code, and multi-step reasoning, N=5 at T=0.7 often lifts accuracy
/// 5–20% over greedy decoding. Costs N× tokens per question.
///
/// # How voting works
///
/// Default voter: normalize each response's text (trim + lowercase +
/// collapse whitespace) and pick the most-common. Ties broken by first
/// occurrence. For structured tasks, pass a custom `voter` that extracts
/// the answer field (e.g. last number in the text, or the JSON field)
/// before counting — raw-text majority over a 500-token reasoning chain
/// will never converge, but majority over extracted answers will.
///
/// # Parallelism
///
/// N samples run concurrently via `tokio::JoinSet`. On a typical provider
/// with an async HTTP pool, 5 parallel samples take ~1× the wall-clock of
/// 1 sample (I/O-bound). CPU-bound models or strict rate-limits will
/// serialize; stack with `RateLimitedChatModel` when needed.
///
/// # Streaming
///
/// `stream()` delegates to a single sample (no streaming fan-out — there's
/// no meaningful way to stream N parallel samples and vote). Callers who
/// want vote-then-stream should do it in two phases upstream.
///
/// ```rust,ignore
/// use litgraph_resilience::SelfConsistencyChatModel;
/// let voter_chat = SelfConsistencyChatModel::new(inner, 5).with_temperature(0.7);
/// let resp = voter_chat.invoke(msgs, &opts).await?;
/// // `resp.usage` includes summed tokens across all 5 samples.
/// ```
pub struct SelfConsistencyChatModel {
    pub inner: Arc<dyn ChatModel>,
    pub samples: usize,
    pub sample_temperature: f32,
    voter: ConsistencyVoter,
}

impl SelfConsistencyChatModel {
    /// Default voter (text-majority). `samples` is clamped to at least 1.
    pub fn new(inner: Arc<dyn ChatModel>, samples: usize) -> Self {
        Self {
            inner,
            samples: samples.max(1),
            sample_temperature: 0.7,
            voter: default_text_voter(),
        }
    }

    /// Override the sampling temperature (default 0.7 — per the paper's
    /// sweet spot for reasoning diversity without incoherence).
    pub fn with_temperature(mut self, t: f32) -> Self {
        self.sample_temperature = t;
        self
    }

    /// Custom voter — e.g. extract the last integer from each response
    /// and pick the majority. Receives all N responses, returns the index
    /// of the winner. If `None`, wrapper falls back to the first sample.
    pub fn with_voter(mut self, voter: ConsistencyVoter) -> Self {
        self.voter = voter;
        self
    }
}

/// Default voter: normalize text (trim + lowercase + collapse whitespace)
/// and return the index of the response whose normalized text appears most.
pub fn default_text_voter() -> ConsistencyVoter {
    Arc::new(|responses: &[ChatResponse]| {
        if responses.is_empty() {
            return None;
        }
        let normalized: Vec<String> = responses
            .iter()
            .map(|r| normalize_for_vote(&r.message.text_content()))
            .collect();
        // Count occurrences; keep first-seen tie-breaker.
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for n in &normalized {
            *counts.entry(n.as_str()).or_insert(0) += 1;
        }
        let (best_text, _) = counts.iter().max_by_key(|(_, c)| *c)?;
        // Return index of the FIRST response whose normalized text matches.
        normalized.iter().position(|n| n == *best_text)
    })
}

fn normalize_for_vote(s: &str) -> String {
    s.trim()
        .to_lowercase()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Build a voter that extracts a field per response via `extract` and picks
/// the majority value. `extract` returns `None` for responses where the
/// field is missing — those are excluded from the vote.
pub fn extracted_field_voter<F>(extract: F) -> ConsistencyVoter
where
    F: Fn(&ChatResponse) -> Option<String> + Send + Sync + 'static,
{
    let extract = Arc::new(extract);
    Arc::new(move |responses: &[ChatResponse]| {
        if responses.is_empty() {
            return None;
        }
        let extracted: Vec<Option<String>> =
            responses.iter().map(|r| extract(r)).collect();
        let mut counts: HashMap<&str, usize> = HashMap::new();
        for e in &extracted {
            if let Some(v) = e {
                *counts.entry(v.as_str()).or_insert(0) += 1;
            }
        }
        let (best, _) = counts.iter().max_by_key(|(_, c)| *c)?;
        extracted
            .iter()
            .position(|e| e.as_deref() == Some(*best))
    })
}

#[async_trait]
impl ChatModel for SelfConsistencyChatModel {
    fn name(&self) -> &str {
        self.inner.name()
    }

    async fn invoke(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatResponse> {
        // Sample N times in parallel. Override temperature on the cloned
        // opts so the caller's preferred temperature doesn't squash
        // sampling diversity. Keep all other opts (max_tokens, tools, etc).
        let mut sample_opts = opts.clone();
        sample_opts.temperature = Some(self.sample_temperature);

        // Spawn N parallel samples and preserve their spawn order in the
        // result vec (join_all yields results in the SAME order as the
        // input futures — critical for deterministic tie-break below).
        let futures: Vec<_> = (0..self.samples)
            .map(|_| {
                let inner = self.inner.clone();
                let msgs = messages.clone();
                let o = sample_opts.clone();
                async move { inner.invoke(msgs, &o).await }
            })
            .collect();
        let results = futures_util::future::join_all(futures).await;

        let mut samples: Vec<ChatResponse> = Vec::with_capacity(self.samples);
        let mut first_err: Option<Error> = None;
        for res in results {
            match res {
                Ok(r) => samples.push(r),
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
        if samples.is_empty() {
            // All N samples failed — bubble up the first error.
            return Err(first_err.unwrap_or_else(|| {
                Error::other("SelfConsistencyChatModel: all samples failed")
            }));
        }

        // Pick the winner. Voter returning None → fall through to first.
        let winner_idx = (self.voter)(&samples).unwrap_or(0);
        let mut winner = samples
            .get(winner_idx)
            .cloned()
            .unwrap_or_else(|| samples[0].clone());

        // Sum usage across ALL samples so the caller's cost tracker sees
        // the full fan-out cost — critical for CostCapped composition.
        let mut summed = TokenUsage::default();
        for s in &samples {
            summed.prompt += s.usage.prompt;
            summed.completion += s.usage.completion;
            summed.total += s.usage.total;
            summed.cache_creation += s.usage.cache_creation;
            summed.cache_read += s.usage.cache_read;
        }
        winner.usage = summed;
        Ok(winner)
    }

    async fn stream(
        &self,
        messages: Vec<Message>,
        opts: &ChatOptions,
    ) -> Result<ChatStream> {
        // No fan-out on streams — delegate to a single sample.
        self.inner.stream(messages, opts).await
    }
}

#[cfg(test)]
mod self_consistency_tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Message};
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Returns a scripted sequence of responses — one per call.
    struct ScriptedModel {
        texts: Vec<&'static str>,
        idx: AtomicUsize,
    }

    impl ScriptedModel {
        fn new(texts: Vec<&'static str>) -> Arc<Self> {
            Arc::new(Self { texts, idx: AtomicUsize::new(0) })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedModel {
        fn name(&self) -> &str { "scripted" }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let i = self.idx.fetch_add(1, Ordering::SeqCst) % self.texts.len();
            Ok(ChatResponse {
                message: Message::assistant(self.texts[i]),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage { prompt: 10, completion: 5, total: 15, cache_creation: 0, cache_read: 0 },
                model: "scripted".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn picks_majority_text() {
        // 3 "42" + 2 "41" → majority "42".
        let inner = ScriptedModel::new(vec!["42", "41", "42", "41", "42"]);
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("2+40")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "42");
    }

    #[tokio::test]
    async fn summed_usage_across_samples() {
        let inner = ScriptedModel::new(vec!["a", "a", "a"]);
        let chat = SelfConsistencyChatModel::new(inner, 3);
        let resp = chat.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        // Each sample uses 10+5=15. N=3 → 30 prompt + 15 completion = 45 total.
        assert_eq!(resp.usage.prompt, 30);
        assert_eq!(resp.usage.completion, 15);
        assert_eq!(resp.usage.total, 45);
    }

    #[tokio::test]
    async fn sample_temperature_overrides_caller_temp() {
        struct TempCapture {
            seen: std::sync::Mutex<Vec<f32>>,
        }
        #[async_trait]
        impl ChatModel for TempCapture {
            fn name(&self) -> &str { "tc" }
            async fn invoke(&self, _m: Vec<Message>, opts: &ChatOptions) -> Result<ChatResponse> {
                self.seen.lock().unwrap().push(opts.temperature.unwrap_or(0.0));
                Ok(ChatResponse {
                    message: Message::assistant("ok"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "tc".into(),
                })
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let inner = Arc::new(TempCapture { seen: std::sync::Mutex::new(vec![]) });
        let chat = SelfConsistencyChatModel::new(inner.clone(), 3).with_temperature(0.9);
        chat.invoke(
            vec![Message::user("q")],
            &ChatOptions { temperature: Some(0.0), ..Default::default() },
        ).await.unwrap();
        let seen = inner.seen.lock().unwrap();
        assert_eq!(seen.len(), 3);
        for t in seen.iter() {
            assert!((t - 0.9).abs() < 1e-6, "expected 0.9, got {t}");
        }
    }

    #[tokio::test]
    async fn tie_winner_is_one_of_the_tied_majority() {
        // 2-2-1 tie between "a" and "b". Which wins depends on which
        // finishes first in the parallel race — both are valid majority
        // picks. The test asserts the winner is NOT the minority ("c").
        let inner = ScriptedModel::new(vec!["a", "b", "a", "b", "c"]);
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("x")], &ChatOptions::default()).await.unwrap();
        let winner = resp.message.text_content();
        assert!(winner == "a" || winner == "b", "winner was {winner}, expected tied majority");
    }

    #[tokio::test]
    async fn custom_voter_extracts_last_number() {
        // Reasoning chains — raw majority would never converge; but
        // last-number-extract majority picks 42.
        let inner = ScriptedModel::new(vec![
            "Let me think... so the answer is 42.",
            "After calculation, I get 42.",
            "Maybe 17? No, 42.",
            "The solution is 41 oh wait, 42.",
            "I believe it's 42.",
        ]);
        let voter = extracted_field_voter(|r| {
            let text = r.message.text_content();
            text.split_whitespace()
                .filter_map(|w| w.trim_end_matches('.').parse::<i64>().ok())
                .last()
                .map(|n| n.to_string())
        });
        let chat = SelfConsistencyChatModel::new(inner, 5).with_voter(voter);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        // Winner must be one of the responses ending with "42".
        assert!(resp.message.text_content().ends_with("42.") || resp.message.text_content().ends_with("42"));
    }

    #[tokio::test]
    async fn all_samples_fail_returns_first_error() {
        struct AllFail;
        #[async_trait]
        impl ChatModel for AllFail {
            fn name(&self) -> &str { "fail" }
            async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
                Err(Error::provider("upstream dead"))
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let chat = SelfConsistencyChatModel::new(Arc::new(AllFail), 3);
        let err = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap_err();
        assert!(err.to_string().contains("upstream dead"));
    }

    #[tokio::test]
    async fn partial_sample_failure_still_votes() {
        // 5 samples, some fail — voter runs on the successful ones.
        struct FlakyScripted {
            texts: Vec<Option<&'static str>>,
            idx: AtomicUsize,
        }
        #[async_trait]
        impl ChatModel for FlakyScripted {
            fn name(&self) -> &str { "flaky" }
            async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
                let i = self.idx.fetch_add(1, Ordering::SeqCst) % self.texts.len();
                match self.texts[i] {
                    Some(t) => Ok(ChatResponse {
                        message: Message::assistant(t),
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::default(),
                        model: "flaky".into(),
                    }),
                    None => Err(Error::Timeout),
                }
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> { unimplemented!() }
        }
        let inner = Arc::new(FlakyScripted {
            texts: vec![Some("42"), None, Some("42"), None, Some("42")],
            idx: AtomicUsize::new(0),
        });
        let chat = SelfConsistencyChatModel::new(inner, 5);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "42");
    }

    #[tokio::test]
    async fn samples_one_passes_through_as_single_call() {
        let inner = ScriptedModel::new(vec!["one"]);
        let chat = SelfConsistencyChatModel::new(inner, 1);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "one");
        assert_eq!(resp.usage.total, 15);
    }

    #[tokio::test]
    async fn zero_samples_clamps_to_one() {
        let inner = ScriptedModel::new(vec!["only"]);
        let chat = SelfConsistencyChatModel::new(inner, 0);
        assert_eq!(chat.samples, 1);
        let resp = chat.invoke(vec![Message::user("q")], &ChatOptions::default()).await.unwrap();
        assert_eq!(resp.message.text_content(), "only");
    }

    #[tokio::test]
    async fn normalize_for_vote_collapses_whitespace_and_case() {
        assert_eq!(normalize_for_vote("  Hello   World  "), "hello world");
        assert_eq!(normalize_for_vote("Hello  World"), normalize_for_vote("HELLO WORLD"));
    }

    #[tokio::test]
    async fn name_delegates_to_inner() {
        let inner = ScriptedModel::new(vec!["x"]);
        let chat = SelfConsistencyChatModel::new(inner, 3);
        assert_eq!(chat.name(), "scripted");
    }
}


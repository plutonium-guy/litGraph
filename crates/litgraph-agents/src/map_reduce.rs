//! `MapReduceSummarizer` — chunk-then-summarize-then-reduce primitive
//! for distilling long documents into a single summary.
//!
//! # The classic pattern
//!
//! 1. **Map**: split the input into chunks; ask the model to summarize
//!    each chunk *independently* and *in parallel*.
//! 2. **Reduce**: concatenate the per-chunk summaries; ask the model
//!    to produce one cohesive summary over them.
//! 3. **Recurse** (optional): if the combined map output is *itself*
//!    too large to fit a single reduce call, group the summaries and
//!    recursively map-reduce. This keeps the primitive useful for
//!    very long documents (book-scale, multi-hour transcripts) where
//!    a single reduce pass would overflow the context window.
//!
//! # Why not just stuff
//!
//! "Stuff documents" — concatenate everything and ask for a summary
//! in one call — only works while the total stays under the context
//! window. Long inputs (legal docs, transcripts, codebases, research
//! papers) blow that. Map-reduce trades latency (multiple model
//! calls) and a small fidelity loss (per-chunk summaries discard
//! cross-chunk context) for the ability to handle inputs of arbitrary
//! length.
//!
//! # Why not refine
//!
//! "Refine chains" pass a running summary through chunks one at a
//! time, refining at each step. They preserve more cross-chunk
//! context but are inherently sequential — every step waits for the
//! previous. Map-reduce parallelizes the expensive map phase, which
//! matters when chunk count is high and the model is the latency
//! bottleneck.
//!
//! # Bounded concurrency
//!
//! Map-phase fan-out is `Semaphore`-bounded (default 8). Provider
//! rate limits / TPM caps mean unbounded fan-out causes 429 storms.
//! The semaphore enforces a per-call concurrency cap; pair with
//! upstream `RateLimitedChatModel` for global throttling across
//! summarizers / agents that share one budget.

use std::sync::Arc;

use litgraph_core::model::ChatOptions;
use litgraph_core::{ChatModel, Error, Message, Result};
use tokio::sync::Semaphore;
use tracing::debug;

const DEFAULT_MAP_PROMPT: &str =
    "You are a concise summarizer. Summarize the following text in a few sentences. \
     Preserve the most important facts, names, numbers, and conclusions. \
     Do not add commentary; emit only the summary.";

const DEFAULT_REDUCE_PROMPT: &str = "You are a concise summarizer. The following are summaries of \
     different sections of a single document. Combine them into one cohesive summary that \
     reads as a unified piece. Preserve key facts and avoid repetition. Do not add commentary.";

/// Map-reduce summarization over arbitrary chunk lists.
///
/// Construct via `MapReduceSummarizer::new(chat_model)`; tune via the
/// fluent `with_*` builders. Default config: 8-way concurrent map,
/// recursive reduce enabled with a 12_000-char threshold (~3k tokens
/// at 4 chars/token — comfortably under most provider context
/// windows leaving room for the reduce-prompt overhead).
pub struct MapReduceSummarizer {
    pub chat_model: Arc<dyn ChatModel>,
    pub map_prompt: String,
    pub reduce_prompt: String,
    pub max_concurrent: usize,
    pub recursive_reduce: bool,
    /// If the combined map output exceeds this many chars AND
    /// `recursive_reduce` is enabled, the reduce phase recurses
    /// instead of issuing a single (likely-overflowing) call.
    pub recurse_threshold_chars: usize,
}

impl MapReduceSummarizer {
    pub fn new(chat_model: Arc<dyn ChatModel>) -> Self {
        Self {
            chat_model,
            map_prompt: DEFAULT_MAP_PROMPT.to_string(),
            reduce_prompt: DEFAULT_REDUCE_PROMPT.to_string(),
            max_concurrent: 8,
            recursive_reduce: true,
            recurse_threshold_chars: 12_000,
        }
    }

    pub fn with_map_prompt(mut self, p: impl Into<String>) -> Self {
        self.map_prompt = p.into();
        self
    }

    pub fn with_reduce_prompt(mut self, p: impl Into<String>) -> Self {
        self.reduce_prompt = p.into();
        self
    }

    pub fn with_max_concurrent(mut self, n: usize) -> Self {
        self.max_concurrent = n.max(1);
        self
    }

    pub fn with_recursive_reduce(mut self, recurse: bool) -> Self {
        self.recursive_reduce = recurse;
        self
    }

    pub fn with_recurse_threshold_chars(mut self, n: usize) -> Self {
        self.recurse_threshold_chars = n.max(1);
        self
    }

    /// Run map-reduce over a list of pre-chunked text segments.
    ///
    /// - `chunks.is_empty()` → returns the empty string (no work to do).
    /// - `chunks.len() == 1` → skips the map phase and runs the reduce
    ///   prompt directly on the single chunk. Saves one round-trip
    ///   when the input already fits.
    pub async fn summarize_chunks(&self, chunks: &[String]) -> Result<String> {
        if chunks.is_empty() {
            return Ok(String::new());
        }
        if chunks.len() == 1 {
            // Single chunk — skip map; run reduce directly. Reduce-prompt
            // is the cleanest single-pass summarization instruction.
            return self.run_call(&self.reduce_prompt, &chunks[0]).await;
        }
        // Map phase — bounded concurrency.
        let summaries = self.map_phase(chunks).await?;
        // Reduce phase — possibly recursive.
        self.reduce_phase(summaries).await
    }

    async fn map_phase(&self, chunks: &[String]) -> Result<Vec<String>> {
        let semaphore = Arc::new(Semaphore::new(self.max_concurrent));
        let mut handles = Vec::with_capacity(chunks.len());
        for chunk in chunks {
            let permit_sem = Arc::clone(&semaphore);
            let chat = Arc::clone(&self.chat_model);
            let prompt = self.map_prompt.clone();
            let chunk = chunk.clone();
            handles.push(tokio::spawn(async move {
                let _permit = permit_sem
                    .acquire_owned()
                    .await
                    .map_err(|e| Error::Other(format!("map_reduce: semaphore closed: {e}")))?;
                run_call_static(chat, &prompt, &chunk).await
            }));
        }
        let mut summaries = Vec::with_capacity(handles.len());
        for h in handles {
            match h.await {
                Ok(Ok(s)) => summaries.push(s),
                Ok(Err(e)) => return Err(e),
                Err(join_err) => {
                    return Err(Error::Other(format!(
                        "map_reduce: map task panicked: {join_err}"
                    )))
                }
            }
        }
        debug!(
            target: "litgraph_agents::map_reduce",
            chunks = chunks.len(),
            "map phase complete"
        );
        Ok(summaries)
    }

    async fn reduce_phase(&self, summaries: Vec<String>) -> Result<String> {
        if summaries.is_empty() {
            return Ok(String::new());
        }
        if summaries.len() == 1 {
            return Ok(summaries.into_iter().next().unwrap());
        }
        let combined = combine(&summaries);
        // Only recurse when grouping actually shrinks the input — i.e. we
        // have at least 3 summaries to bucket into 2 groups. With 2
        // summaries, group_size would be 1 and recursion would never
        // converge if "R[1] --- R[1]" still exceeds the threshold.
        if self.recursive_reduce
            && combined.len() > self.recurse_threshold_chars
            && summaries.len() > 2
        {
            // Group summaries; recurse on each group; then run a final reduce.
            let group_size = summaries.len().div_ceil(2);
            let mut group_summaries = Vec::new();
            for group in summaries.chunks(group_size) {
                let group_combined = combine(group);
                let r = self.run_call(&self.reduce_prompt, &group_combined).await?;
                group_summaries.push(r);
            }
            // Tail-recurse on the group summaries.
            return Box::pin(self.reduce_phase(group_summaries)).await;
        }
        self.run_call(&self.reduce_prompt, &combined).await
    }

    async fn run_call(&self, system_prompt: &str, user_text: &str) -> Result<String> {
        run_call_static(Arc::clone(&self.chat_model), system_prompt, user_text).await
    }
}

async fn run_call_static(
    chat: Arc<dyn ChatModel>,
    system_prompt: &str,
    user_text: &str,
) -> Result<String> {
    let messages = vec![
        Message::system(system_prompt),
        Message::user(user_text),
    ];
    let opts = ChatOptions::default();
    let resp = chat.invoke(messages, &opts).await?;
    Ok(resp.message.text_content())
}

fn combine(summaries: &[String]) -> String {
    summaries.join("\n\n---\n\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::ChatResponse;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Counts calls; returns `"M[<chunk-marker>]"` when the system
    /// prompt looks like a map prompt, `"R[<n-summaries>]"` when it
    /// looks like a reduce prompt. Detection: substring match on
    /// "Combine" (only present in the default reduce prompt) lets
    /// the test check map-vs-reduce behavior independently of LLM
    /// reasoning.
    struct ScriptedSummarizer {
        calls: AtomicUsize,
    }

    impl ScriptedSummarizer {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedSummarizer {
        fn name(&self) -> &str {
            "scripted-summarizer"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // System message tells us map vs reduce.
            let sys = messages
                .iter()
                .find(|m| matches!(m.role, litgraph_core::Role::System))
                .map(|m| m.text_content())
                .unwrap_or_default();
            let user = messages
                .iter()
                .find(|m| matches!(m.role, litgraph_core::Role::User))
                .map(|m| m.text_content())
                .unwrap_or_default();
            let out = if sys.contains("Combine") {
                // Reduce branch — count "---" separators to verify input shape.
                let n_inputs = user.matches("---").count() + 1;
                format!("R[{n_inputs}]")
            } else {
                // Map branch — echo first 8 chars of input as a marker.
                let marker: String = user.chars().take(8).collect();
                format!("M[{marker}]")
            };
            Ok(ChatResponse {
                message: Message::assistant(out),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt: 10,
                    completion: 5,
                    total: 15,
                    cache_creation: 0,
                    cache_read: 0,
                },
                model: "scripted-summarizer".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn empty_chunks_returns_empty() {
        let s = MapReduceSummarizer::new(ScriptedSummarizer::new());
        let r = s.summarize_chunks(&[]).await.unwrap();
        assert_eq!(r, "");
    }

    #[tokio::test]
    async fn single_chunk_skips_map_phase() {
        let inner = ScriptedSummarizer::new();
        let s = MapReduceSummarizer::new(inner.clone());
        let r = s
            .summarize_chunks(&["only-chunk content here".to_string()])
            .await
            .unwrap();
        // Reduce-only path. One LLM call total. Reduce sees 1 input.
        assert_eq!(inner.calls(), 1);
        assert_eq!(r, "R[1]");
    }

    #[tokio::test]
    async fn three_chunks_map_then_reduce() {
        let inner = ScriptedSummarizer::new();
        let s = MapReduceSummarizer::new(inner.clone());
        let chunks = vec![
            "AAAAAAAA chunk one".to_string(),
            "BBBBBBBB chunk two".to_string(),
            "CCCCCCCC chunk three".to_string(),
        ];
        let r = s.summarize_chunks(&chunks).await.unwrap();
        // 3 map calls + 1 reduce = 4.
        assert_eq!(inner.calls(), 4);
        // Reduce saw 3 inputs joined by `---`.
        assert_eq!(r, "R[3]");
    }

    #[tokio::test]
    async fn recursive_reduce_for_long_combined_output() {
        // 6 chunks. Recurse threshold set very low (10 chars) so the
        // combined map output (which is "M[...]" * 6 separated by ---)
        // exceeds threshold and forces recursion.
        let inner = ScriptedSummarizer::new();
        let s = MapReduceSummarizer::new(inner.clone())
            .with_recurse_threshold_chars(10)
            .with_recursive_reduce(true);
        let chunks: Vec<String> = (0..6)
            .map(|i| format!("chunk-content-{i}-aaaaaaaa"))
            .collect();
        let _r = s.summarize_chunks(&chunks).await.unwrap();
        // 6 map + group reduces + final reduce.
        // Group size = ceil(6/2) = 3; 2 groups → 2 reduces → tail-recurse on 2 → 1 final reduce.
        // Total: 6 + 2 + 1 = 9.
        assert_eq!(inner.calls(), 9);
    }

    #[tokio::test]
    async fn recursive_reduce_disabled_single_pass() {
        // Same fixture as recursive_reduce_for_long_combined_output but
        // recursion off → one big reduce regardless of size.
        let inner = ScriptedSummarizer::new();
        let s = MapReduceSummarizer::new(inner.clone())
            .with_recurse_threshold_chars(10)
            .with_recursive_reduce(false);
        let chunks: Vec<String> = (0..6)
            .map(|i| format!("chunk-content-{i}-aaaaaaaa"))
            .collect();
        let _r = s.summarize_chunks(&chunks).await.unwrap();
        // 6 map + 1 reduce.
        assert_eq!(inner.calls(), 7);
    }

    #[tokio::test]
    async fn max_concurrent_clamps_to_one_minimum() {
        let s = MapReduceSummarizer::new(ScriptedSummarizer::new()).with_max_concurrent(0);
        assert_eq!(s.max_concurrent, 1);
    }

    #[tokio::test]
    async fn custom_prompts_used() {
        // Use a map prompt that DOES contain "Combine" so the
        // ScriptedSummarizer's branch detector goes to the reduce
        // branch even on map calls — which lets us verify the
        // custom prompt is actually wired through.
        let inner = ScriptedSummarizer::new();
        let s = MapReduceSummarizer::new(inner.clone())
            .with_map_prompt("Combine and summarize this:")
            .with_reduce_prompt("Combine these:");
        let chunks = vec!["a".to_string(), "b".to_string()];
        let _r = s.summarize_chunks(&chunks).await.unwrap();
        // Both map AND reduce calls now hit the "Combine" branch →
        // 3 total calls (2 map + 1 reduce), all returning R[...].
        assert_eq!(inner.calls(), 3);
    }

    #[tokio::test]
    async fn map_phase_runs_concurrently() {
        // 8 chunks, max_concurrent=4. With a 50ms sleep per call:
        // - serial would take 8 * 50 = 400ms+
        // - concurrent (4-wide) takes ~2 * 50 = 100ms
        // We allow a generous upper bound to accommodate CI noise.
        struct SlowModel {
            calls: AtomicUsize,
        }
        #[async_trait]
        impl ChatModel for SlowModel {
            fn name(&self) -> &str {
                "slow"
            }
            async fn invoke(
                &self,
                _messages: Vec<Message>,
                _opts: &ChatOptions,
            ) -> Result<ChatResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
                Ok(ChatResponse {
                    message: Message::assistant("ok"),
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage {
                        prompt: 10,
                        completion: 5,
                        total: 15,
                        cache_creation: 0,
                        cache_read: 0,
                    },
                    model: "slow".into(),
                })
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
                unimplemented!()
            }
        }
        let inner = Arc::new(SlowModel {
            calls: AtomicUsize::new(0),
        });
        let s = MapReduceSummarizer::new(inner.clone() as Arc<dyn ChatModel>)
            .with_max_concurrent(4);
        let chunks: Vec<String> = (0..8).map(|i| format!("c{i}")).collect();
        let start = std::time::Instant::now();
        let _r = s.summarize_chunks(&chunks).await.unwrap();
        let elapsed = start.elapsed();
        // 8 map calls in waves of 4 = ~100ms; reduce adds another ~50ms; total ~150ms.
        // Allow up to 350ms to absorb CI scheduling jitter.
        assert!(
            elapsed < std::time::Duration::from_millis(350),
            "elapsed={elapsed:?}"
        );
        // Verify all 8 + 1 reduce ran.
        assert_eq!(inner.calls.load(Ordering::SeqCst), 9);
    }
}

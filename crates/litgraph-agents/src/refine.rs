//! `RefineSummarizer` — sequential running-summary chain for distilling
//! long documents while preserving cross-chunk context.
//!
//! # The pattern
//!
//! 1. **Initial**: run an `initial_prompt` over the first chunk to seed
//!    a running summary.
//! 2. **Refine** (for chunks 2..N): pass the running summary alongside
//!    each subsequent chunk, asking the model to *update* the summary
//!    given the new section.
//! 3. Return the final running summary.
//!
//! # Refine vs map-reduce (iter 300)
//!
//! Both summarize long inputs. They differ in tradeoff axis:
//!
//! - **Map-reduce** parallelizes the expensive map phase. N chunks →
//!   roughly `chunks/concurrency` wall-clock for map + 1 reduce.
//!   But each map call sees only its own chunk, so cross-chunk
//!   references are lost when the model summarizes them in
//!   isolation.
//! - **Refine** is sequential by design. N chunks → N model calls
//!   in series. Slower, but each call sees the running summary, so
//!   cross-chunk context is preserved (event in chunk 3 referenced
//!   in chunk 7 will likely survive into the final summary).
//!
//! Pick refine when fidelity-of-narrative matters and you can afford
//! the latency. Pick map-reduce when the chunk count is high and the
//! goal is bullet-point-style fact extraction (where cross-chunk
//! ordering is less load-bearing).
//!
//! # Why not stream the running summary
//!
//! Stream-output of intermediate states is tempting (the user could
//! watch the summary evolve). But `Result<String>` is the simpler
//! contract for batch summarization, and intermediate states are
//! often misleading — chunks 1..k can produce a summary that's
//! contradicted by chunk k+1, which the user-watching-streaming
//! would see and confuse for the final answer. Callers who genuinely
//! want progress observability should compose with iter-199 `Progress`
//! externally.

use std::sync::Arc;

use litgraph_core::model::ChatOptions;
use litgraph_core::{ChatModel, Message, Result};

const DEFAULT_INITIAL_PROMPT: &str =
    "You are a concise summarizer. Read the following text and produce a summary that \
     preserves the most important facts, names, numbers, and conclusions. Emit only the \
     summary — no commentary, no preamble.";

const DEFAULT_REFINE_PROMPT: &str =
    "You are a concise summarizer. Below is an existing summary, followed by an additional \
     section of the same source document. Produce a refined summary that integrates the new \
     section's important facts into the existing summary. Preserve key facts already in the \
     summary; merge or update where the new section contradicts or extends them; remove \
     trivia. Emit only the refined summary — no commentary.";

/// Sequential running-summary chain over arbitrary chunk lists.
///
/// Construct via `RefineSummarizer::new(chat_model)`; tune via the
/// fluent `with_*` builders. Default prompts cover the typical
/// "general document" use case; override for domain-specific tone.
pub struct RefineSummarizer {
    pub chat_model: Arc<dyn ChatModel>,
    pub initial_prompt: String,
    pub refine_prompt: String,
}

impl RefineSummarizer {
    pub fn new(chat_model: Arc<dyn ChatModel>) -> Self {
        Self {
            chat_model,
            initial_prompt: DEFAULT_INITIAL_PROMPT.to_string(),
            refine_prompt: DEFAULT_REFINE_PROMPT.to_string(),
        }
    }

    pub fn with_initial_prompt(mut self, p: impl Into<String>) -> Self {
        self.initial_prompt = p.into();
        self
    }

    pub fn with_refine_prompt(mut self, p: impl Into<String>) -> Self {
        self.refine_prompt = p.into();
        self
    }

    /// Run refine over a list of pre-chunked text segments.
    ///
    /// - `chunks.is_empty()` → returns the empty string.
    /// - `chunks.len() == 1` → one initial call; no refine.
    /// - `chunks.len() == N` → N model calls total: one initial,
    ///   `N − 1` refines.
    pub async fn summarize_chunks(&self, chunks: &[String]) -> Result<String> {
        if chunks.is_empty() {
            return Ok(String::new());
        }
        // Initial pass over the first chunk.
        let mut running = self
            .invoke(&self.initial_prompt, &chunks[0])
            .await?;
        // Refine through the remaining chunks, sequentially.
        for chunk in &chunks[1..] {
            let user_text = format!(
                "Existing summary:\n{running}\n\n--- New section ---\n{chunk}"
            );
            running = self.invoke(&self.refine_prompt, &user_text).await?;
        }
        Ok(running)
    }

    async fn invoke(&self, system_prompt: &str, user_text: &str) -> Result<String> {
        let messages = vec![
            Message::system(system_prompt),
            Message::user(user_text),
        ];
        let opts = ChatOptions::default();
        let resp = self.chat_model.invoke(messages, &opts).await?;
        Ok(resp.message.text_content())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Error};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Records call sequence: each call appends `(system_prefix,
    /// user_prefix)` so tests can assert on the order.
    struct RecordingModel {
        calls: AtomicUsize,
        log: Mutex<Vec<(String, String)>>,
        // Each call returns "S<n>" where n is the call index.
        // Lets the next refine see "Existing summary:\nS<prev>".
    }

    impl RecordingModel {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                log: Mutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn log(&self) -> Vec<(String, String)> {
            self.log.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatModel for RecordingModel {
        fn name(&self) -> &str {
            "recording"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
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
            self.log.lock().unwrap().push((
                sys.chars().take(40).collect::<String>(),
                user.chars().take(120).collect::<String>(),
            ));
            Ok(ChatResponse {
                message: Message::assistant(format!("S{n}")),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt: 10,
                    completion: 5,
                    total: 15,
                    cache_creation: 0,
                    cache_read: 0,
                },
                model: "recording".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Errors on the second call. Used to verify error propagation.
    struct FailOnSecondModel {
        calls: AtomicUsize,
    }

    impl FailOnSecondModel {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ChatModel for FailOnSecondModel {
        fn name(&self) -> &str {
            "fail-on-second"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n == 1 {
                return Err(Error::Provider("simulated 2nd-call failure".into()));
            }
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
                model: "fail-on-second".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn empty_chunks_returns_empty() {
        let s = RefineSummarizer::new(RecordingModel::new());
        let r = s.summarize_chunks(&[]).await.unwrap();
        assert_eq!(r, "");
    }

    #[tokio::test]
    async fn single_chunk_makes_one_call() {
        let inner = RecordingModel::new();
        let s = RefineSummarizer::new(inner.clone());
        let r = s.summarize_chunks(&["only-chunk".to_string()]).await.unwrap();
        assert_eq!(inner.calls(), 1);
        assert_eq!(r, "S0");
        // Single-call branch uses the initial prompt, NOT refine.
        let log = inner.log();
        assert!(log[0].0.starts_with("You are a concise summarizer"));
        assert!(!log[0].1.contains("Existing summary"));
    }

    #[tokio::test]
    async fn three_chunks_one_initial_two_refines() {
        let inner = RecordingModel::new();
        let s = RefineSummarizer::new(inner.clone());
        let chunks = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let r = s.summarize_chunks(&chunks).await.unwrap();
        assert_eq!(inner.calls(), 3);
        // Final result is the last call's output: S2.
        assert_eq!(r, "S2");
        let log = inner.log();
        // Call 0: initial prompt, user = "A".
        assert_eq!(log[0].1, "A");
        // Call 1: refine prompt, user = "Existing summary:\nS0\n\n--- New section ---\nB"
        assert!(log[1].1.contains("Existing summary:"));
        assert!(log[1].1.contains("S0"));
        assert!(log[1].1.contains("B"));
        // Call 2: refine prompt, user = "Existing summary:\nS1\n\n--- New section ---\nC"
        assert!(log[2].1.contains("S1"));
        assert!(log[2].1.contains("C"));
    }

    #[tokio::test]
    async fn running_summary_carried_forward() {
        // Verify each refine call sees the PRIOR call's output as
        // the running summary — not the original chunk.
        let inner = RecordingModel::new();
        let s = RefineSummarizer::new(inner.clone());
        let chunks: Vec<String> = (0..5).map(|i| format!("chunk-{i}")).collect();
        let _ = s.summarize_chunks(&chunks).await.unwrap();
        let log = inner.log();
        // The 4th call (index 3) is the 3rd refine — it should see "S2"
        // (the prior call's output) as the running summary.
        let user_at_3 = &log[3].1;
        assert!(user_at_3.contains("Existing summary:\nS2"));
        // And the new section should be chunk-3.
        assert!(user_at_3.contains("chunk-3"));
        // The last call (4) sees "S3" + "chunk-4".
        assert!(log[4].1.contains("Existing summary:\nS3"));
        assert!(log[4].1.contains("chunk-4"));
    }

    #[tokio::test]
    async fn error_in_refine_propagates_immediately() {
        let inner = FailOnSecondModel::new();
        let s = RefineSummarizer::new(inner.clone());
        let chunks = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let r = s.summarize_chunks(&chunks).await;
        assert!(r.is_err());
        // Initial succeeded, second (the first refine) failed → 2 calls total.
        // The 3rd chunk never got processed.
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn custom_prompts_wired_through() {
        let inner = RecordingModel::new();
        let s = RefineSummarizer::new(inner.clone())
            .with_initial_prompt("CUSTOM-INIT")
            .with_refine_prompt("CUSTOM-REFINE");
        let chunks = vec!["x".to_string(), "y".to_string()];
        let _ = s.summarize_chunks(&chunks).await.unwrap();
        let log = inner.log();
        assert_eq!(log[0].0, "CUSTOM-INIT");
        assert_eq!(log[1].0, "CUSTOM-REFINE");
    }

    #[tokio::test]
    async fn calls_are_sequential_not_parallel() {
        // 5 chunks at 30ms per call → expect ~150ms total wall-clock,
        // NOT < 60ms (which would mean parallel). The point of refine
        // is sequential, so this test pins that contract.
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
                _m: Vec<Message>,
                _o: &ChatOptions,
            ) -> Result<ChatResponse> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                tokio::time::sleep(std::time::Duration::from_millis(30)).await;
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
        let s = RefineSummarizer::new(inner.clone() as Arc<dyn ChatModel>);
        let chunks: Vec<String> = (0..5).map(|i| format!("c{i}")).collect();
        let start = std::time::Instant::now();
        let _ = s.summarize_chunks(&chunks).await.unwrap();
        let elapsed = start.elapsed();
        // Sequential lower bound: 5 * 30 = 150ms. Allow some scheduling slop.
        assert!(
            elapsed >= std::time::Duration::from_millis(140),
            "elapsed={elapsed:?} — refine should be sequential"
        );
        assert_eq!(inner.calls.load(Ordering::SeqCst), 5);
    }
}

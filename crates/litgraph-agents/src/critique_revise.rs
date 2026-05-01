//! `CritiqueReviseAgent` — generate a draft, then iteratively
//! critique-and-revise it through N rounds for higher-quality output.
//!
//! # The pattern
//!
//! 1. **Draft**: produce an initial response for the input.
//! 2. For each iteration:
//!    - **Critique**: ask the model to find weaknesses, errors, or
//!      missed angles in the current draft.
//!    - **Revise**: ask the model to produce a new draft that
//!      addresses the critique while preserving what's working.
//! 3. Return the final draft (plus the full iteration trace for
//!    debugging / observability).
//!
//! # Why a self-critique loop
//!
//! Single-pass LLM outputs frequently exhibit "first-draft" issues:
//! missed constraints, surface-level coverage of complex topics,
//! style/tone mismatches, factual sloppiness. The model itself
//! often *can* see these issues when prompted to look for them —
//! it just doesn't naturally double-check its own work in a single
//! invocation. The critique step makes the self-check explicit.
//!
//! Critique-revise has been shown to materially improve output
//! quality on tasks like long-form writing, code review, and
//! complex reasoning where the cost of an extra round-trip is
//! lower than the cost of shipping a flawed answer.
//!
//! # Distinct from neighboring patterns
//!
//! - **`ReactAgent`** (tool-calling loop): the model invokes
//!   external tools each step. Critique-revise has no tools — it's
//!   purely about output quality refinement.
//! - **`PlanAndExecuteAgent`** (plan-then-execute): two-phase
//!   different-purpose, plan first then run steps. Critique-revise
//!   is single-purpose (improve the same output) with a homogeneous
//!   loop body.
//! - **`MapReduceSummarizer`** / **`RefineSummarizer`** (iters 300,
//!   301): chunk-input chains. Critique-revise has a single
//!   atomic input.
//!
//! # When this is a win
//!
//! - Long-form writing where the first draft has structural issues.
//! - Code review where the first pass misses edge cases.
//! - Complex reasoning where the model "knows" but doesn't
//!   surface its concerns until prompted.
//!
//! # When this is NOT a win
//!
//! - Short factual answers (one round-trip is enough; critique adds
//!   noise).
//! - Tasks with crisp success criteria the LLM can self-evaluate
//!   in one pass (math, code with tests).
//! - Latency-sensitive paths (every iteration adds 2 round-trips).

use std::sync::Arc;

use litgraph_core::model::ChatOptions;
use litgraph_core::{ChatModel, Message, Result};

const DEFAULT_DRAFT_PROMPT: &str =
    "You are a careful, thorough writer. Produce a high-quality response to the user's request. \
     Take the request seriously; do not hedge or add unnecessary caveats.";

const DEFAULT_CRITIQUE_PROMPT: &str =
    "You are a critical reviewer. Below is a user request followed by a draft response. \
     Identify the most important weaknesses in the draft: factual errors, missed angles, \
     unclear sections, structural issues, or constraints from the request that the draft \
     fails to address. Be specific. Do not rewrite the draft; just enumerate the issues. \
     If the draft is already excellent, say so explicitly with one sentence.";

const DEFAULT_REVISE_PROMPT: &str =
    "You are a careful writer revising a draft based on reviewer feedback. Below is a user \
     request, the current draft, and a critique. Produce a revised draft that addresses the \
     critique while preserving the parts of the draft that were working. Emit only the \
     revised draft — no commentary, no preamble, no diff markers.";

/// One iteration's diagnostic record — the draft that entered the
/// critique step and the critique the model produced for it. Useful
/// for debugging "why did the agent's output change between iter N
/// and N+1?" without re-running.
#[derive(Debug, Clone)]
pub struct CritiqueReviseIteration {
    pub draft_before: String,
    pub critique: String,
}

/// Result of a `CritiqueReviseAgent::run` call.
#[derive(Debug, Clone)]
pub struct CritiqueReviseResult {
    pub initial_draft: String,
    pub final_draft: String,
    pub iterations: Vec<CritiqueReviseIteration>,
}

/// Single-input draft → critique → revise loop.
///
/// Construct via `CritiqueReviseAgent::new(chat_model)`; tune via
/// fluent `with_*_prompt` and `with_iterations` builders. Default
/// `iterations = 1` (one critique + one revise after the initial
/// draft → 3 model calls). Setting `iterations = 0` returns the
/// initial draft directly with no critique work — useful as an
/// A/B baseline.
pub struct CritiqueReviseAgent {
    pub chat_model: Arc<dyn ChatModel>,
    pub draft_prompt: String,
    pub critique_prompt: String,
    pub revise_prompt: String,
    pub iterations: usize,
}

impl CritiqueReviseAgent {
    pub fn new(chat_model: Arc<dyn ChatModel>) -> Self {
        Self {
            chat_model,
            draft_prompt: DEFAULT_DRAFT_PROMPT.to_string(),
            critique_prompt: DEFAULT_CRITIQUE_PROMPT.to_string(),
            revise_prompt: DEFAULT_REVISE_PROMPT.to_string(),
            iterations: 1,
        }
    }

    pub fn with_draft_prompt(mut self, p: impl Into<String>) -> Self {
        self.draft_prompt = p.into();
        self
    }

    pub fn with_critique_prompt(mut self, p: impl Into<String>) -> Self {
        self.critique_prompt = p.into();
        self
    }

    pub fn with_revise_prompt(mut self, p: impl Into<String>) -> Self {
        self.revise_prompt = p.into();
        self
    }

    pub fn with_iterations(mut self, n: usize) -> Self {
        self.iterations = n;
        self
    }

    /// Run the loop on `input`. Returns the final draft + the full
    /// iteration trace for debugging.
    ///
    /// Total LLM calls = `1 + 2 * iterations` (1 draft, then 2 per
    /// iteration for critique + revise). With `iterations = 0`,
    /// this is one call that returns the initial draft.
    pub async fn run(&self, input: &str) -> Result<CritiqueReviseResult> {
        let initial_draft = self.draft(input).await?;
        let mut current = initial_draft.clone();
        let mut trace = Vec::with_capacity(self.iterations);
        for _ in 0..self.iterations {
            let critique = self.critique(input, &current).await?;
            let revised = self.revise(input, &current, &critique).await?;
            trace.push(CritiqueReviseIteration {
                draft_before: current,
                critique,
            });
            current = revised;
        }
        Ok(CritiqueReviseResult {
            initial_draft,
            final_draft: current,
            iterations: trace,
        })
    }

    async fn draft(&self, input: &str) -> Result<String> {
        invoke_one(&self.chat_model, &self.draft_prompt, input).await
    }

    async fn critique(&self, input: &str, draft: &str) -> Result<String> {
        let user = format!("User request:\n{input}\n\n--- Draft ---\n{draft}");
        invoke_one(&self.chat_model, &self.critique_prompt, &user).await
    }

    async fn revise(&self, input: &str, draft: &str, critique: &str) -> Result<String> {
        let user = format!(
            "User request:\n{input}\n\n--- Current draft ---\n{draft}\n\n--- Critique ---\n{critique}"
        );
        invoke_one(&self.chat_model, &self.revise_prompt, &user).await
    }
}

async fn invoke_one(
    chat: &Arc<dyn ChatModel>,
    system_prompt: &str,
    user_text: &str,
) -> Result<String> {
    let messages = vec![
        Message::system(system_prompt),
        Message::user(user_text),
    ];
    let resp = chat.invoke(messages, &ChatOptions::default()).await?;
    Ok(resp.message.text_content())
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, Error};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;

    /// Returns "step{n}" where n is the call index. Records the
    /// system-prompt prefix and user-text length for each call so
    /// tests can probe the per-step plumbing.
    struct CountingModel {
        calls: AtomicUsize,
        log: Mutex<Vec<String>>, // first 50 chars of system prompt per call
    }

    impl CountingModel {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
                log: Mutex::new(Vec::new()),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
        fn log(&self) -> Vec<String> {
            self.log.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ChatModel for CountingModel {
        fn name(&self) -> &str {
            "counting"
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
            self.log
                .lock()
                .unwrap()
                .push(sys.chars().take(50).collect());
            Ok(ChatResponse {
                message: Message::assistant(format!("step{n}")),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt: 10,
                    completion: 5,
                    total: 15,
                    cache_creation: 0,
                    cache_read: 0,
                },
                model: "counting".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    /// Errors on the second call.
    struct FailOnSecond {
        calls: AtomicUsize,
    }

    impl FailOnSecond {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                calls: AtomicUsize::new(0),
            })
        }
    }

    #[async_trait]
    impl ChatModel for FailOnSecond {
        fn name(&self) -> &str {
            "fail2"
        }
        async fn invoke(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
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
                model: "fail2".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn iterations_zero_returns_draft_only() {
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone()).with_iterations(0);
        let res = agent.run("write a haiku").await.unwrap();
        assert_eq!(inner.calls(), 1);
        assert_eq!(res.initial_draft, "step0");
        assert_eq!(res.final_draft, "step0");
        assert!(res.iterations.is_empty());
    }

    #[tokio::test]
    async fn one_iteration_makes_three_calls() {
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone()).with_iterations(1);
        let res = agent.run("write a haiku").await.unwrap();
        // 1 draft + 1 critique + 1 revise = 3.
        assert_eq!(inner.calls(), 3);
        assert_eq!(res.initial_draft, "step0");
        assert_eq!(res.final_draft, "step2");
        assert_eq!(res.iterations.len(), 1);
        assert_eq!(res.iterations[0].draft_before, "step0");
        assert_eq!(res.iterations[0].critique, "step1");
    }

    #[tokio::test]
    async fn n_iterations_makes_1_plus_2n_calls() {
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone()).with_iterations(3);
        let res = agent.run("input").await.unwrap();
        // 1 + 2*3 = 7.
        assert_eq!(inner.calls(), 7);
        assert_eq!(res.iterations.len(), 3);
        // After 3 critique-revise cycles, final_draft = step6 (last revise call).
        assert_eq!(res.final_draft, "step6");
    }

    #[tokio::test]
    async fn iteration_trace_records_pre_revise_drafts() {
        // Iteration 0: draft_before = initial_draft (step0), critique = step1
        // Iteration 1: draft_before = revised-draft-from-iter-0 (step2), critique = step3
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone()).with_iterations(2);
        let res = agent.run("input").await.unwrap();
        assert_eq!(res.iterations.len(), 2);
        assert_eq!(res.iterations[0].draft_before, "step0");
        assert_eq!(res.iterations[0].critique, "step1");
        assert_eq!(res.iterations[1].draft_before, "step2");
        assert_eq!(res.iterations[1].critique, "step3");
        assert_eq!(res.final_draft, "step4");
    }

    #[tokio::test]
    async fn each_step_uses_its_own_prompt() {
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone())
            .with_draft_prompt("DRAFT-PROMPT-X")
            .with_critique_prompt("CRITIQUE-PROMPT-X")
            .with_revise_prompt("REVISE-PROMPT-X")
            .with_iterations(1);
        let _ = agent.run("input").await.unwrap();
        let log = inner.log();
        assert_eq!(log.len(), 3);
        assert!(log[0].starts_with("DRAFT-PROMPT-X"), "draft={}", log[0]);
        assert!(log[1].starts_with("CRITIQUE-PROMPT-X"), "crit={}", log[1]);
        assert!(log[2].starts_with("REVISE-PROMPT-X"), "rev={}", log[2]);
    }

    #[tokio::test]
    async fn error_in_critique_propagates() {
        // FailOnSecond errors on call index 1 = the first critique call.
        // Initial draft (call 0) succeeds; chain then stops.
        let inner = FailOnSecond::new();
        let agent = CritiqueReviseAgent::new(inner.clone()).with_iterations(2);
        let r = agent.run("input").await;
        assert!(r.is_err());
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn default_iterations_is_one() {
        let inner = CountingModel::new();
        let agent = CritiqueReviseAgent::new(inner.clone());
        assert_eq!(agent.iterations, 1);
        let _ = agent.run("input").await.unwrap();
        assert_eq!(inner.calls(), 3);
    }
}

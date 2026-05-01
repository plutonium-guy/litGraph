//! `DebateAgent` — multi-agent debate where N debaters argue across
//! rounds, then a judge synthesizes the best answer.
//!
//! # The pattern
//!
//! 1. **Round 0 (parallel)**: each debater independently produces an
//!    initial answer to the question.
//! 2. **Rounds 1..N (parallel within each round)**: each debater
//!    sees the other debaters' previous-round answers and refines
//!    its own — confirming or correcting based on what others said.
//! 3. **Judgment**: a separate judge model sees all debaters' final
//!    answers and synthesizes the best response.
//!
//! Within each round all debaters run in parallel via `JoinSet`;
//! rounds are sequential because each round depends on the prior
//! round's outputs. Total LLM calls: `n_debaters · (rounds + 1) + 1`
//! (the `+1` is the judge).
//!
//! # Why this works
//!
//! Du et al. (2023) "Improving Factuality and Reasoning in Language
//! Models through Multiagent Debate" — multiple models cross-checking
//! each other catches a class of errors single-model self-critique
//! misses: when a single model is *consistently* wrong about something,
//! self-critique can't surface the error because the same bias
//! generates both the draft and the critique. Different models (or
//! the same model with different sampling temperatures) often have
//! *different* bias patterns; debate exploits that diversity.
//!
//! # Distinct from `CritiqueReviseAgent` (iter 306)
//!
//! Both improve output quality through multi-call structure, but:
//!
//! - **CritiqueRevise** is *single-model self-improvement*. The same
//!   model drafts, critiques, and revises. Cheap and effective when
//!   the model can see its own draft's weaknesses.
//! - **Debate** is *multi-model cross-checking*. Different models
//!   (or different sampling configs) argue, exposing biases that
//!   single-model loops can't see. More expensive (N · (rounds+1) + 1
//!   calls vs 1 + 2·iterations) but catches a different class of
//!   errors.
//!
//! Pick based on bias profile: tasks where one model has a known
//! consistent failure mode (specific factual gaps, reasoning
//! hallucinations) benefit from debate; tasks where the model is
//! generally capable but inconsistent benefit more from
//! critique-revise.
//!
//! # When to use the SAME model for all debaters
//!
//! Setting all `debaters` to the same Arc<dyn ChatModel> with
//! different temperatures upstream gives sampling-based diversity
//! without paying for multiple provider relationships. Du et al's
//! original results came from same-model debate where the diversity
//! was purely sampling-driven; cross-provider debate is even
//! stronger but adds operational complexity.

use std::sync::Arc;

use litgraph_core::model::ChatOptions;
use litgraph_core::{ChatModel, Error, Message, Result};
use tokio::task::JoinSet;
use tracing::debug;

const DEFAULT_INITIAL_PROMPT: &str =
    "You are a careful, thorough reasoner. Provide a detailed, well-reasoned answer to the \
     user's question. Be specific. Cite your reasoning step-by-step. Do not hedge unnecessarily.";

const DEFAULT_DEBATE_PROMPT: &str =
    "You are a careful reasoner participating in a multi-agent debate. Below is the user's \
     question, then the other debaters' previous answers. Based on their responses, refine \
     your answer: confirm what you got right, correct anything you got wrong, address gaps \
     or contradictions you see in the other debaters' work. If your previous answer was \
     correct, restate it more clearly. Do not just agree with the majority — disagree if \
     you have good reason.";

const DEFAULT_JUDGE_PROMPT: &str =
    "You are an impartial judge synthesizing a final answer from multiple debaters' responses. \
     Below is the user's question followed by each debater's final answer. Produce the single \
     best response: synthesize where the debaters agree, resolve where they disagree by \
     evaluating their reasoning, and emit a clean final answer. Do not credit individual \
     debaters; just produce the answer.";

/// One round's debater outputs — used both for the iteration trace
/// (`DebateResult.rounds`) and for the per-round input to the next
/// round.
#[derive(Debug, Clone)]
pub struct DebateRound {
    /// One entry per debater, in the same order as `DebateAgent.debaters`.
    pub answers: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DebateResult {
    /// The judge's synthesized final answer.
    pub final_answer: String,
    /// Per-round trace, including round 0 (initial answers).
    /// `rounds.len() == debate.rounds + 1`.
    pub rounds: Vec<DebateRound>,
}

/// Multi-agent debate primitive.
///
/// Construct via `DebateAgent::new(debaters, judge)`. `debaters`
/// is a `Vec<Arc<dyn ChatModel>>` — pass the same Arc multiple
/// times for same-model debate, or distinct models for cross-
/// provider debate. `rounds` defaults to 1 (so the structure is:
/// initial round + 1 refinement round + judge = 2N+1 calls).
pub struct DebateAgent {
    pub debaters: Vec<Arc<dyn ChatModel>>,
    pub judge: Arc<dyn ChatModel>,
    pub initial_prompt: String,
    pub debate_prompt: String,
    pub judge_prompt: String,
    pub rounds: usize,
}

impl DebateAgent {
    pub fn new(debaters: Vec<Arc<dyn ChatModel>>, judge: Arc<dyn ChatModel>) -> Self {
        Self {
            debaters,
            judge,
            initial_prompt: DEFAULT_INITIAL_PROMPT.to_string(),
            debate_prompt: DEFAULT_DEBATE_PROMPT.to_string(),
            judge_prompt: DEFAULT_JUDGE_PROMPT.to_string(),
            rounds: 1,
        }
    }

    pub fn with_initial_prompt(mut self, p: impl Into<String>) -> Self {
        self.initial_prompt = p.into();
        self
    }

    pub fn with_debate_prompt(mut self, p: impl Into<String>) -> Self {
        self.debate_prompt = p.into();
        self
    }

    pub fn with_judge_prompt(mut self, p: impl Into<String>) -> Self {
        self.judge_prompt = p.into();
        self
    }

    pub fn with_rounds(mut self, n: usize) -> Self {
        self.rounds = n;
        self
    }

    pub async fn run(&self, question: &str) -> Result<DebateResult> {
        if self.debaters.is_empty() {
            return Err(Error::invalid("debate: at least one debater is required"));
        }
        // Round 0: each debater answers independently, in parallel.
        let initial = self.parallel_initial(question).await?;
        let mut trace = Vec::with_capacity(self.rounds + 1);
        trace.push(DebateRound {
            answers: initial.clone(),
        });
        let mut current = initial;
        // Refinement rounds.
        for round_idx in 0..self.rounds {
            let next = self.parallel_round(question, &current).await?;
            debug!(
                target: "litgraph_agents::debate",
                round = round_idx + 1,
                "debate round complete"
            );
            trace.push(DebateRound {
                answers: next.clone(),
            });
            current = next;
        }
        // Judge.
        let final_answer = self.judge(question, &current).await?;
        Ok(DebateResult {
            final_answer,
            rounds: trace,
        })
    }

    async fn parallel_initial(&self, question: &str) -> Result<Vec<String>> {
        let mut set: JoinSet<(usize, Result<String>)> = JoinSet::new();
        for (i, debater) in self.debaters.iter().enumerate() {
            let chat = Arc::clone(debater);
            let prompt = self.initial_prompt.clone();
            let q = question.to_string();
            set.spawn(async move { (i, invoke_one(&chat, &prompt, &q).await) });
        }
        collect_ordered(set, self.debaters.len()).await
    }

    async fn parallel_round(
        &self,
        question: &str,
        prev_answers: &[String],
    ) -> Result<Vec<String>> {
        let mut set: JoinSet<(usize, Result<String>)> = JoinSet::new();
        for (i, debater) in self.debaters.iter().enumerate() {
            let chat = Arc::clone(debater);
            let prompt = self.debate_prompt.clone();
            let user = format_round_input(question, i, prev_answers);
            set.spawn(async move { (i, invoke_one(&chat, &prompt, &user).await) });
        }
        collect_ordered(set, self.debaters.len()).await
    }

    async fn judge(&self, question: &str, final_answers: &[String]) -> Result<String> {
        let user = format_judge_input(question, final_answers);
        invoke_one(&self.judge, &self.judge_prompt, &user).await
    }
}

fn format_round_input(question: &str, my_idx: usize, prev_answers: &[String]) -> String {
    let mut s = format!("Question:\n{question}\n\n");
    s.push_str("Your previous answer:\n");
    s.push_str(&prev_answers[my_idx]);
    s.push_str("\n\n");
    s.push_str("Other debaters' previous answers:\n");
    for (i, ans) in prev_answers.iter().enumerate() {
        if i == my_idx {
            continue;
        }
        s.push_str(&format!("--- Debater {} ---\n{ans}\n\n", i + 1));
    }
    s
}

fn format_judge_input(question: &str, final_answers: &[String]) -> String {
    let mut s = format!("Question:\n{question}\n\n");
    for (i, ans) in final_answers.iter().enumerate() {
        s.push_str(&format!("--- Debater {} ---\n{ans}\n\n", i + 1));
    }
    s
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

/// Collect a `JoinSet<(idx, Result<String>)>` into a `Vec<String>`
/// preserving the order of debaters. Any failure short-circuits the
/// rest of the round.
async fn collect_ordered(
    mut set: JoinSet<(usize, Result<String>)>,
    expected_len: usize,
) -> Result<Vec<String>> {
    let mut results: Vec<Option<String>> = vec![None; expected_len];
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, Ok(s))) => results[idx] = Some(s),
            Ok((_, Err(e))) => return Err(e),
            Err(e) => return Err(Error::Other(format!("debate: task panicked: {e}"))),
        }
    }
    results
        .into_iter()
        .collect::<Option<Vec<String>>>()
        .ok_or_else(|| Error::Other("debate: not all debaters returned".into()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::ChatResponse;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A model that returns `"<tag>-<call_idx>"` so tests can verify
    /// which debater produced what.
    struct TaggedModel {
        tag: &'static str,
        calls: AtomicUsize,
    }

    impl TaggedModel {
        fn new(tag: &'static str) -> Arc<Self> {
            Arc::new(Self {
                tag,
                calls: AtomicUsize::new(0),
            })
        }
        fn calls(&self) -> usize {
            self.calls.load(Ordering::SeqCst)
        }
    }

    #[async_trait]
    impl ChatModel for TaggedModel {
        fn name(&self) -> &str {
            "tagged"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            Ok(ChatResponse {
                message: Message::assistant(format!("{}-{}", self.tag, n)),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt: 10,
                    completion: 5,
                    total: 15,
                    cache_creation: 0,
                    cache_read: 0,
                },
                model: "tagged".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn empty_debaters_errors() {
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(vec![], judge as Arc<dyn ChatModel>);
        let r = agent.run("question").await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn two_debaters_one_round_total_calls() {
        // 2 debaters × (1 initial + 1 round) = 4 debater calls + 1 judge = 5.
        let d1 = TaggedModel::new("A");
        let d2 = TaggedModel::new("B");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d1.clone() as Arc<dyn ChatModel>, d2.clone() as Arc<dyn ChatModel>],
            judge.clone() as Arc<dyn ChatModel>,
        )
        .with_rounds(1);
        let res = agent.run("q").await.unwrap();
        assert_eq!(d1.calls(), 2);
        assert_eq!(d2.calls(), 2);
        assert_eq!(judge.calls(), 1);
        assert_eq!(res.rounds.len(), 2); // round 0 + round 1
        assert_eq!(res.rounds[0].answers.len(), 2);
        assert_eq!(res.rounds[1].answers.len(), 2);
        // Judge output is the final_answer.
        assert!(res.final_answer.starts_with("J-"));
    }

    #[tokio::test]
    async fn three_debaters_two_rounds_total_calls() {
        // 3 × (1 + 2) = 9 + 1 judge = 10.
        let d1 = TaggedModel::new("A");
        let d2 = TaggedModel::new("B");
        let d3 = TaggedModel::new("C");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![
                d1.clone() as Arc<dyn ChatModel>,
                d2.clone() as Arc<dyn ChatModel>,
                d3.clone() as Arc<dyn ChatModel>,
            ],
            judge.clone() as Arc<dyn ChatModel>,
        )
        .with_rounds(2);
        let res = agent.run("q").await.unwrap();
        assert_eq!(d1.calls(), 3);
        assert_eq!(d2.calls(), 3);
        assert_eq!(d3.calls(), 3);
        assert_eq!(judge.calls(), 1);
        assert_eq!(res.rounds.len(), 3); // round 0, 1, 2
    }

    #[tokio::test]
    async fn rounds_zero_skips_refinement() {
        // rounds=0 → just initial answers + judge. 2 debaters × 1 = 2 + 1 = 3.
        let d1 = TaggedModel::new("A");
        let d2 = TaggedModel::new("B");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d1.clone() as Arc<dyn ChatModel>, d2.clone() as Arc<dyn ChatModel>],
            judge.clone() as Arc<dyn ChatModel>,
        )
        .with_rounds(0);
        let res = agent.run("q").await.unwrap();
        assert_eq!(d1.calls(), 1);
        assert_eq!(d2.calls(), 1);
        assert_eq!(judge.calls(), 1);
        assert_eq!(res.rounds.len(), 1); // just round 0
    }

    #[tokio::test]
    async fn single_debater_works() {
        // Degenerate but allowed: 1 debater → no peers to compare against.
        let d = TaggedModel::new("A");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d.clone() as Arc<dyn ChatModel>],
            judge.clone() as Arc<dyn ChatModel>,
        )
        .with_rounds(1);
        let res = agent.run("q").await.unwrap();
        assert_eq!(d.calls(), 2); // initial + 1 round
        assert_eq!(judge.calls(), 1);
        assert_eq!(res.rounds[0].answers.len(), 1);
    }

    #[tokio::test]
    async fn round_runs_in_parallel() {
        // 4 debaters with 50ms-per-call models. Sequential would be
        // 4*50 = 200ms per round; parallel should be ~50ms per round.
        // 1 initial + 1 round = 2 round-blocks of ~50ms each + 50ms judge = ~150ms total.
        // Sequential lower bound would be ~450ms.
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
        let make = || {
            Arc::new(SlowModel {
                calls: AtomicUsize::new(0),
            }) as Arc<dyn ChatModel>
        };
        let agent = DebateAgent::new(
            vec![make(), make(), make(), make()],
            make(),
        )
        .with_rounds(1);
        let start = std::time::Instant::now();
        let _ = agent.run("q").await.unwrap();
        let elapsed = start.elapsed();
        // Expect ~150ms (3 × 50ms blocks: initial-parallel, round-parallel, judge).
        // Allow up to 350ms for CI scheduling jitter.
        assert!(
            elapsed < std::time::Duration::from_millis(350),
            "elapsed={elapsed:?} — debate should run debaters in parallel within each round"
        );
    }

    #[tokio::test]
    async fn round_trace_records_per_round_outputs() {
        let d1 = TaggedModel::new("A");
        let d2 = TaggedModel::new("B");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d1.clone() as Arc<dyn ChatModel>, d2.clone() as Arc<dyn ChatModel>],
            judge as Arc<dyn ChatModel>,
        )
        .with_rounds(1);
        let res = agent.run("q").await.unwrap();
        // Round 0 (initial): A-0, B-0.
        assert_eq!(res.rounds[0].answers, vec!["A-0", "B-0"]);
        // Round 1 (refinement): A-1, B-1.
        assert_eq!(res.rounds[1].answers, vec!["A-1", "B-1"]);
    }

    #[tokio::test]
    async fn debater_failure_propagates() {
        // One debater errors on call 0; the chain should fail.
        struct FailModel;
        #[async_trait]
        impl ChatModel for FailModel {
            fn name(&self) -> &str {
                "fail"
            }
            async fn invoke(
                &self,
                _m: Vec<Message>,
                _o: &ChatOptions,
            ) -> Result<ChatResponse> {
                Err(Error::Provider("debater unavailable".into()))
            }
            async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
                unimplemented!()
            }
        }
        let d1 = TaggedModel::new("A");
        let d2 = Arc::new(FailModel) as Arc<dyn ChatModel>;
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d1 as Arc<dyn ChatModel>, d2],
            judge as Arc<dyn ChatModel>,
        );
        let r = agent.run("q").await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn default_rounds_is_one() {
        let d = TaggedModel::new("A");
        let judge = TaggedModel::new("J");
        let agent = DebateAgent::new(
            vec![d.clone() as Arc<dyn ChatModel>],
            judge as Arc<dyn ChatModel>,
        );
        assert_eq!(agent.rounds, 1);
    }
}

//! LLM-judge generation eval — faithfulness, answer-relevance, optional
//! correctness. Pairs with iter-65's deterministic `evaluate_retrieval` to
//! cover the full ragas-style RAG eval surface.
//!
//! Why an LLM judge: there's no programmatic way to ask "does this answer
//! follow from these contexts" or "does this answer the query." Both are
//! semantic-equivalence checks. Token overlap (BLEU/ROUGE) misses paraphrase;
//! exact-match misses anything but trivial Q&A.
//!
//! Tradeoffs to be honest about:
//!   - Variance: same case can grade 0.8 → 0.9 → 0.7 across runs. Pin a
//!     seed-friendly model (we use temperature=0.0).
//!   - Cost: each metric is one extra LLM call per case. A 1000-case suite
//!     with 3 metrics = 3000 calls. Use a cheap model like gpt-4o-mini.
//!   - Self-grading bias: don't use the same model under test as judge.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use litgraph_core::{ChatModel, ChatOptions, Error, Message, Result};

/// One labeled generation case. `answer` is what the system under test
/// produced; `contexts` are the retrieved passages it had access to;
/// `reference_answer` (optional) enables the `correctness` metric.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationCase {
    pub query: String,
    pub answer: String,
    pub contexts: Vec<String>,
    #[serde(default)]
    pub reference_answer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerCaseGenMetrics {
    pub query: String,
    pub faithfulness: f64,
    pub answer_relevance: f64,
    /// `None` when `reference_answer` was missing on the case.
    pub correctness: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenReport {
    pub n_cases: usize,
    pub faithfulness_macro: f64,
    pub answer_relevance_macro: f64,
    /// `None` when no case had a `reference_answer`.
    pub correctness_macro: Option<f64>,
    pub per_case: Vec<PerCaseGenMetrics>,
}

#[derive(Debug, Clone, Copy)]
pub struct GenEvalConfig {
    pub max_concurrency: usize,
    /// Skip correctness scoring even when `reference_answer` is present —
    /// useful when you only care about faithfulness/relevance and want to
    /// halve the eval cost.
    pub skip_correctness: bool,
}

impl Default for GenEvalConfig {
    fn default() -> Self {
        Self { max_concurrency: 8, skip_correctness: false }
    }
}

/// Run all three (or two, depending on config + dataset) judge metrics over
/// `cases`. Each case dispatches its metric calls concurrently up to
/// `max_concurrency`. Failed judge calls (network, parse error) score the
/// case at 0.0 for that metric — safer to under-credit than to silently
/// skip cases and inflate the macro average.
pub async fn evaluate_generation(
    judge: Arc<dyn ChatModel>,
    cases: &[GenerationCase],
    cfg: GenEvalConfig,
) -> Result<GenReport> {
    if cases.is_empty() {
        return Err(Error::other("evaluate_generation: empty cases"));
    }
    let sem = Arc::new(Semaphore::new(cfg.max_concurrency.max(1)));
    let mut tasks: JoinSet<PerCaseGenMetrics> = JoinSet::new();

    for case in cases {
        let judge = judge.clone();
        let sem = sem.clone();
        let case = case.clone();
        let do_correctness = !cfg.skip_correctness && case.reference_answer.is_some();

        tasks.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                // If the semaphore is closed, fail the case open (score 0).
                Err(_) => return PerCaseGenMetrics {
                    query: case.query, faithfulness: 0.0,
                    answer_relevance: 0.0, correctness: None,
                },
            };

            let faithfulness = judge_score(
                &*judge,
                &faithfulness_prompt(&case.contexts, &case.answer),
            ).await;

            let answer_relevance = judge_score(
                &*judge,
                &relevance_prompt(&case.query, &case.answer),
            ).await;

            let correctness = if do_correctness {
                let r = case.reference_answer.as_ref().unwrap();
                Some(judge_score(
                    &*judge,
                    &correctness_prompt(&case.query, &case.answer, r),
                ).await)
            } else { None };

            PerCaseGenMetrics {
                query: case.query,
                faithfulness, answer_relevance, correctness,
            }
        });
    }

    let mut per_case = Vec::with_capacity(cases.len());
    while let Some(r) = tasks.join_next().await {
        let m = r.map_err(|e| Error::other(format!("eval task join: {e}")))?;
        per_case.push(m);
    }
    // Restore input order.
    let order: std::collections::HashMap<&str, usize> = cases
        .iter()
        .enumerate()
        .map(|(i, c)| (c.query.as_str(), i))
        .collect();
    per_case.sort_by_key(|m| *order.get(m.query.as_str()).unwrap_or(&usize::MAX));

    let n = per_case.len() as f64;
    let faithfulness_macro: f64 = per_case.iter().map(|m| m.faithfulness).sum::<f64>() / n;
    let answer_relevance_macro: f64 = per_case.iter().map(|m| m.answer_relevance).sum::<f64>() / n;
    let correctness_scores: Vec<f64> = per_case.iter()
        .filter_map(|m| m.correctness).collect();
    let correctness_macro = if correctness_scores.is_empty() {
        None
    } else {
        Some(correctness_scores.iter().sum::<f64>() / correctness_scores.len() as f64)
    };

    Ok(GenReport {
        n_cases: per_case.len(),
        faithfulness_macro,
        answer_relevance_macro,
        correctness_macro,
        per_case,
    })
}

/// Run the judge with `prompt` as user message; parse a 0/1 (or yes/no)
/// from the start of the response. Errors → 0.0 (under-credit on failure).
async fn judge_score(judge: &dyn ChatModel, prompt: &str) -> f64 {
    let msgs = vec![
        Message::system(JUDGE_SYSTEM),
        Message::user(prompt.to_string()),
    ];
    let opts = ChatOptions {
        temperature: Some(0.0),
        max_tokens: Some(8),  // We just need "1" or "0"; longer responses get truncated.
        ..Default::default()
    };
    match judge.invoke(msgs, &opts).await {
        Ok(r) => parse_binary_score(&r.message.text_content()),
        Err(_) => 0.0,
    }
}

const JUDGE_SYSTEM: &str = "You are a strict evaluator. Reply with exactly one digit: 1 if the criterion is met, 0 if it is not. No explanation. No punctuation.";

fn faithfulness_prompt(contexts: &[String], answer: &str) -> String {
    let ctx = contexts.iter().enumerate()
        .map(|(i, c)| format!("[{}] {}", i + 1, c))
        .collect::<Vec<_>>()
        .join("\n\n");
    format!(
        "CONTEXTS:\n{ctx}\n\nANSWER:\n{answer}\n\n\
         Criterion: every factual claim in the ANSWER is supported by the CONTEXTS. \
         Reply 1 if supported, 0 if any claim is unsupported or contradicted."
    )
}

fn relevance_prompt(query: &str, answer: &str) -> String {
    format!(
        "QUERY:\n{query}\n\nANSWER:\n{answer}\n\n\
         Criterion: the ANSWER directly addresses the QUERY (not off-topic, not evasive). \
         Reply 1 if it does, 0 otherwise."
    )
}

fn correctness_prompt(query: &str, answer: &str, reference: &str) -> String {
    format!(
        "QUERY:\n{query}\n\nREFERENCE ANSWER:\n{reference}\n\nCANDIDATE ANSWER:\n{answer}\n\n\
         Criterion: the CANDIDATE conveys the same essential information as the REFERENCE \
         (paraphrase + extra detail OK; missing or wrong information NOT ok). \
         Reply 1 if equivalent, 0 otherwise."
    )
}

/// Parse the judge's reply. Accepts: leading `1` or `0`, or words `yes`/`no`
/// (case-insensitive). Anything else → 0.0 (be strict — judges that ramble
/// instead of complying with the contract shouldn't score full credit).
fn parse_binary_score(text: &str) -> f64 {
    let t = text.trim();
    if t.is_empty() { return 0.0; }
    let lower = t.to_ascii_lowercase();
    if let Some(c) = lower.chars().next() {
        match c {
            '1' => return 1.0,
            '0' => return 0.0,
            _ => {}
        }
    }
    if lower.starts_with("yes") { 1.0 } else { 0.0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
    use litgraph_core::{ChatResponse, ContentPart, Role};

    /// Judge that returns canned responses round-robin from the queue.
    struct ScriptedJudge {
        replies: parking_lot::Mutex<std::collections::VecDeque<String>>,
    }
    impl ScriptedJudge {
        fn new(replies: &[&str]) -> Self {
            Self {
                replies: parking_lot::Mutex::new(
                    replies.iter().map(|s| s.to_string()).collect()
                ),
            }
        }
    }
    #[async_trait]
    impl ChatModel for ScriptedJudge {
        fn name(&self) -> &str { "scripted-judge" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            let reply = self.replies.lock().pop_front()
                .unwrap_or_else(|| "0".into());
            Ok(ChatResponse {
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: reply }],
                    tool_calls: vec![], tool_call_id: None, name: None, cache: false,
                },
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted-judge".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[test]
    fn parse_binary_handles_canonical_replies() {
        assert_eq!(parse_binary_score("1"), 1.0);
        assert_eq!(parse_binary_score("0"), 0.0);
        assert_eq!(parse_binary_score("yes"), 1.0);
        assert_eq!(parse_binary_score("YES"), 1.0);
        assert_eq!(parse_binary_score("no"), 0.0);
        assert_eq!(parse_binary_score(""), 0.0);
    }

    #[test]
    fn parse_binary_strict_on_rambling_judges() {
        // A judge that explains itself instead of complying gets 0.
        // We don't try to extract semantics from natural language.
        assert_eq!(parse_binary_score("The answer is correct because..."), 0.0);
        assert_eq!(parse_binary_score("Sure, that's right."), 0.0);
    }

    #[tokio::test]
    async fn evaluate_generation_aggregates_macro_average() {
        // 2 cases × 2 metrics each = 4 judge calls (no correctness yet).
        // Order matters — JoinSet runs concurrent; the script is shared across
        // both cases, so each case sees pairs of replies in arrival order.
        // To keep the test deterministic, force max_concurrency=1.
        let judge: Arc<dyn ChatModel> = Arc::new(ScriptedJudge::new(&[
            "1", "1",     // case 1: faithfulness, relevance both pass
            "0", "1",     // case 2: faithfulness fails, relevance passes
        ]));
        let cases = vec![
            GenerationCase {
                query: "q1".into(), answer: "a1".into(),
                contexts: vec!["c1".into()], reference_answer: None,
            },
            GenerationCase {
                query: "q2".into(), answer: "a2".into(),
                contexts: vec!["c2".into()], reference_answer: None,
            },
        ];
        let report = evaluate_generation(
            judge, &cases,
            GenEvalConfig { max_concurrency: 1, skip_correctness: false },
        ).await.unwrap();
        assert_eq!(report.n_cases, 2);
        // Faithfulness: 1.0 + 0.0 / 2 = 0.5
        assert!((report.faithfulness_macro - 0.5).abs() < 1e-9);
        // Relevance: 1.0 + 1.0 / 2 = 1.0
        assert!((report.answer_relevance_macro - 1.0).abs() < 1e-9);
        // No reference_answer in either case → no correctness aggregate.
        assert!(report.correctness_macro.is_none());
        // Per-case order matches input order.
        assert_eq!(report.per_case[0].query, "q1");
        assert_eq!(report.per_case[1].query, "q2");
        assert_eq!(report.per_case[0].faithfulness, 1.0);
        assert_eq!(report.per_case[1].faithfulness, 0.0);
    }

    #[tokio::test]
    async fn correctness_scored_only_when_reference_present() {
        // Case 1 has reference; case 2 doesn't.
        // Case 1: faithfulness, relevance, correctness = "1", "1", "1"
        // Case 2: faithfulness, relevance only = "0", "1"
        let judge: Arc<dyn ChatModel> = Arc::new(ScriptedJudge::new(&[
            "1", "1", "1",  // case 1 — three calls
            "0", "1",       // case 2 — two calls
        ]));
        let cases = vec![
            GenerationCase {
                query: "q1".into(), answer: "a1".into(),
                contexts: vec![], reference_answer: Some("ref1".into()),
            },
            GenerationCase {
                query: "q2".into(), answer: "a2".into(),
                contexts: vec![], reference_answer: None,
            },
        ];
        let report = evaluate_generation(
            judge, &cases,
            GenEvalConfig { max_concurrency: 1, skip_correctness: false },
        ).await.unwrap();
        // Correctness aggregate is over the ONE case that had a reference.
        assert_eq!(report.correctness_macro, Some(1.0));
        assert_eq!(report.per_case[0].correctness, Some(1.0));
        assert_eq!(report.per_case[1].correctness, None);
    }

    #[tokio::test]
    async fn skip_correctness_overrides_dataset() {
        // Even with reference_answer present, skip_correctness=true must skip.
        let judge: Arc<dyn ChatModel> = Arc::new(ScriptedJudge::new(&["1", "1"]));
        let cases = vec![GenerationCase {
            query: "q".into(), answer: "a".into(),
            contexts: vec![], reference_answer: Some("ref".into()),
        }];
        let report = evaluate_generation(
            judge, &cases,
            GenEvalConfig { max_concurrency: 1, skip_correctness: true },
        ).await.unwrap();
        assert!(report.correctness_macro.is_none());
        assert!(report.per_case[0].correctness.is_none());
    }

    #[tokio::test]
    async fn empty_cases_errors() {
        let judge: Arc<dyn ChatModel> = Arc::new(ScriptedJudge::new(&[]));
        let err = evaluate_generation(
            judge, &[], GenEvalConfig::default()
        ).await.unwrap_err();
        assert!(format!("{err}").contains("empty cases"));
    }
}

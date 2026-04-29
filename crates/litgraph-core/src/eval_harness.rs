//! Golden-dataset eval harness — run N test cases against a target callable
//! with bounded concurrency, score each output, return per-case + aggregate
//! report.
//!
//! # Use cases
//!
//! - **Prompt regression**: did rewording the system prompt break anything?
//!   Run the same dataset against old/new prompt, compare aggregate scores.
//! - **Model A/B**: same dataset against two ChatModels, see which scores higher.
//! - **CI gate**: PR can't merge unless aggregate score ≥ baseline (set in repo).
//!
//! # vs the built-in `evaluators` module
//!
//! `evaluators` provides per-text scoring functions (`exact_match`,
//! `jaccard_similarity`, etc) — building blocks. This harness drives them
//! over a dataset with parallelism, an async target callable, and report
//! aggregation.
//!
//! # Concurrency
//!
//! Caller picks `max_parallel`. Default 4 — bounded so a 1000-case eval
//! against an LLM doesn't fan-out 1000 in-flight requests + trip rate
//! limits. Use `max_parallel = 1` for sequential / debuggable runs.

use std::sync::Arc;

use async_trait::async_trait;
use futures::stream::{FuturesUnordered, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{ChatModel, LlmJudge, Result};

/// One row in an eval dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCase {
    pub input: String,
    /// Optional gold answer. Some scorers (`ExactMatchScorer`,
    /// `JaccardScorer`) require it; others (`LengthScorer`,
    /// `RegexScorer`) don't.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
    /// Free-form per-case context (case ID, tags, difficulty level, …).
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub metadata: Value,
}

impl EvalCase {
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            expected: None,
            metadata: Value::Null,
        }
    }
    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }
    pub fn with_metadata(mut self, metadata: Value) -> Self {
        self.metadata = metadata;
        self
    }
}

#[derive(Debug, Clone, Default)]
pub struct EvalDataset {
    pub cases: Vec<EvalCase>,
}

impl EvalDataset {
    pub fn new(cases: Vec<EvalCase>) -> Self {
        Self { cases }
    }

    pub fn from_pairs<I, A, B>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (A, B)>,
        A: Into<String>,
        B: Into<String>,
    {
        Self {
            cases: pairs
                .into_iter()
                .map(|(i, e)| EvalCase::new(i).with_expected(e))
                .collect(),
        }
    }

    /// Parse JSONL — one `{"input":..., "expected":..., "metadata":...}`
    /// per line. Empty + whitespace-only lines skipped.
    pub fn from_jsonl(text: &str) -> Result<Self> {
        let mut cases = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let case: EvalCase = serde_json::from_str(trimmed).map_err(|e| {
                crate::Error::other(format!("eval dataset: line {}: {e}", i + 1))
            })?;
            cases.push(case);
        }
        Ok(Self { cases })
    }

    pub fn len(&self) -> usize { self.cases.len() }
    pub fn is_empty(&self) -> bool { self.cases.is_empty() }
}

/// Single-output scorer. Returns a score in `[0.0, 1.0]` (1 = perfect),
/// optional explanation. The `expected` is `Option` so scorers that don't
/// need a gold answer (length, format) work too.
#[async_trait]
pub trait Scorer: Send + Sync {
    fn name(&self) -> &str;
    async fn score(
        &self,
        input: &str,
        output: &str,
        expected: Option<&str>,
    ) -> Result<ScoreResult>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoreResult {
    pub score: f64,
    /// Optional human-readable detail. Surfaced in per-case report rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub explanation: Option<String>,
}

impl ScoreResult {
    pub fn pass() -> Self { Self { score: 1.0, explanation: None } }
    pub fn fail() -> Self { Self { score: 0.0, explanation: None } }
    pub fn new(score: f64) -> Self {
        Self { score: score.clamp(0.0, 1.0), explanation: None }
    }
    pub fn explained(score: f64, explanation: impl Into<String>) -> Self {
        Self {
            score: score.clamp(0.0, 1.0),
            explanation: Some(explanation.into()),
        }
    }
}

// --- Built-in scorers ---

pub struct ExactMatchScorer;

#[async_trait]
impl Scorer for ExactMatchScorer {
    fn name(&self) -> &str { "exact_match" }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let exp = expected.ok_or_else(|| {
            crate::Error::invalid("ExactMatchScorer requires `expected`")
        })?;
        Ok(if crate::evaluators::exact_match(output, exp) {
            ScoreResult::pass()
        } else {
            ScoreResult::fail()
        })
    }
}

pub struct JaccardScorer;

#[async_trait]
impl Scorer for JaccardScorer {
    fn name(&self) -> &str { "jaccard" }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let exp = expected.ok_or_else(|| {
            crate::Error::invalid("JaccardScorer requires `expected`")
        })?;
        Ok(ScoreResult::new(
            crate::evaluators::jaccard_similarity(output, exp) as f64,
        ))
    }
}

pub struct LevenshteinScorer;

#[async_trait]
impl Scorer for LevenshteinScorer {
    fn name(&self) -> &str { "levenshtein" }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let exp = expected.ok_or_else(|| {
            crate::Error::invalid("LevenshteinScorer requires `expected`")
        })?;
        Ok(ScoreResult::new(
            crate::evaluators::levenshtein_ratio(output, exp) as f64,
        ))
    }
}

/// Pass IFF the output contains every required substring (case-sensitive).
pub struct ContainsAllScorer {
    pub required: Vec<String>,
}

#[async_trait]
impl Scorer for ContainsAllScorer {
    fn name(&self) -> &str { "contains_all" }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        _expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let refs: Vec<&str> = self.required.iter().map(|s| s.as_str()).collect();
        Ok(if crate::evaluators::contains_all(output, &refs) {
            ScoreResult::pass()
        } else {
            ScoreResult::fail()
        })
    }
}

/// Pass IFF the output matches the regex.
pub struct RegexScorer {
    pub pattern: String,
}

#[async_trait]
impl Scorer for RegexScorer {
    fn name(&self) -> &str { "regex" }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        _expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let matched = crate::evaluators::regex_match(output, &self.pattern)
            .map_err(|e| crate::Error::invalid(format!("RegexScorer pattern: {e}")))?;
        Ok(if matched { ScoreResult::pass() } else { ScoreResult::fail() })
    }
}

/// LLM-as-judge scorer — wraps an `LlmJudge` (iter 128) as a `Scorer`.
/// Each case calls the judge model with `(prediction=output, reference=expected)`;
/// score is the judge's `[0, 1]` rating, explanation is the judge's
/// reasoning. Requires `expected`.
///
/// # Cost note
///
/// Adds one LLM call per case. For a 100-case eval at $0.01/call that's
/// $1 of judge spend on top of the target's spend. Use a cheap judge
/// model (gpt-4o-mini, claude-haiku) — judging is well within smaller-model
/// capability for most rubrics.
///
/// # Bias note
///
/// LLM judges have known biases (length, position, sycophancy). Stack with
/// at least one deterministic scorer (ExactMatch, ContainsAll) when using
/// LlmJudgeScorer for high-stakes decisions. The judge's `reasoning`
/// field surfaces in the per-case report's `explanation` so you can
/// audit individual scores.
pub struct LlmJudgeScorer {
    pub name: String,
    pub judge: Arc<LlmJudge>,
}

impl LlmJudgeScorer {
    /// Default name "llm_judge". Use `with_name` if multiple LLM judges
    /// are stacked (e.g. one strict, one lenient — different aggregate
    /// columns in the report).
    pub fn new(model: Arc<dyn ChatModel>, criteria: Option<String>) -> Self {
        Self {
            name: "llm_judge".into(),
            judge: Arc::new(LlmJudge::new(model, criteria)),
        }
    }

    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

#[async_trait]
impl Scorer for LlmJudgeScorer {
    fn name(&self) -> &str { &self.name }
    async fn score(
        &self,
        _input: &str,
        output: &str,
        expected: Option<&str>,
    ) -> Result<ScoreResult> {
        let exp = expected.ok_or_else(|| {
            crate::Error::invalid("LlmJudgeScorer requires `expected`")
        })?;
        let judged = self.judge.judge(output, exp).await?;
        Ok(ScoreResult::explained(judged.score as f64, judged.reasoning))
    }
}

// --- Per-case report ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalCaseResult {
    pub input: String,
    pub expected: Option<String>,
    pub output: Option<String>,
    /// Per-scorer score keyed by scorer name.
    pub scores: serde_json::Map<String, Value>,
    /// Set if the target callable errored. `output` will be None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregateScores {
    pub n_cases: usize,
    pub n_errors: usize,
    /// Per-scorer mean across non-errored cases. Empty if all cases errored.
    pub means: serde_json::Map<String, Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalReport {
    pub per_case: Vec<EvalCaseResult>,
    pub aggregate: AggregateScores,
}

/// Run `target` against every case in `dataset`, score each output via
/// every scorer in `scorers`, return a per-case + aggregate report.
///
/// `max_parallel` bounds in-flight target invocations. Default 4 (use
/// `0` to mean "1" — no panics).
///
/// `target` is `Fn(&str) -> Future<Output=Result<String>>` — async + can
/// fail per-case (errors surface in the report's `error` field, don't
/// abort the whole run).
pub async fn run_eval<F, Fut>(
    dataset: &EvalDataset,
    scorers: &[Arc<dyn Scorer>],
    target: F,
    max_parallel: usize,
) -> Result<EvalReport>
where
    F: Fn(String) -> Fut + Send + Sync + Clone + 'static,
    Fut: std::future::Future<Output = Result<String>> + Send + 'static,
{
    let max_parallel = max_parallel.max(1);
    let mut in_flight: FuturesUnordered<_> = FuturesUnordered::new();
    let mut iter = dataset.cases.iter().enumerate();
    // Per-case results buffered by index for stable output ordering.
    let mut buf: Vec<Option<EvalCaseResult>> = (0..dataset.len()).map(|_| None).collect();

    // Seed the first batch.
    for _ in 0..max_parallel {
        if let Some((idx, case)) = iter.next() {
            let target = target.clone();
            let scorers = scorers.to_vec();
            let case_owned = case.clone();
            in_flight.push(run_one(idx, case_owned, scorers, target));
        }
    }

    while let Some((idx, result)) = in_flight.next().await {
        buf[idx] = Some(result);
        if let Some((idx2, case)) = iter.next() {
            let target = target.clone();
            let scorers = scorers.to_vec();
            in_flight.push(run_one(idx2, case.clone(), scorers, target));
        }
    }

    let per_case: Vec<EvalCaseResult> = buf.into_iter().map(|opt| opt.expect("populated")).collect();
    let aggregate = aggregate_scores(&per_case);
    Ok(EvalReport { per_case, aggregate })
}

async fn run_one<F, Fut>(
    idx: usize,
    case: EvalCase,
    scorers: Vec<Arc<dyn Scorer>>,
    target: F,
) -> (usize, EvalCaseResult)
where
    F: Fn(String) -> Fut + Send + Sync,
    Fut: std::future::Future<Output = Result<String>> + Send,
{
    let input = case.input.clone();
    let target_result = target(input).await;
    let mut row = EvalCaseResult {
        input: case.input,
        expected: case.expected.clone(),
        output: None,
        scores: serde_json::Map::new(),
        error: None,
        metadata: case.metadata,
    };
    match target_result {
        Ok(output) => {
            for scorer in &scorers {
                let s = scorer
                    .score(&row.input, &output, row.expected.as_deref())
                    .await;
                let entry = match s {
                    Ok(sr) => serde_json::json!({"score": sr.score, "explanation": sr.explanation}),
                    Err(e) => serde_json::json!({"score": 0.0, "explanation": format!("scorer error: {e}")}),
                };
                row.scores.insert(scorer.name().to_string(), entry);
            }
            row.output = Some(output);
        }
        Err(e) => {
            row.error = Some(e.to_string());
        }
    }
    (idx, row)
}

fn aggregate_scores(per_case: &[EvalCaseResult]) -> AggregateScores {
    let n_errors = per_case.iter().filter(|c| c.error.is_some()).count();
    let mut sums: std::collections::HashMap<String, (f64, usize)> = std::collections::HashMap::new();
    for c in per_case {
        if c.error.is_some() {
            continue;
        }
        for (scorer_name, entry) in &c.scores {
            let s = entry.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
            let e = sums.entry(scorer_name.clone()).or_insert((0.0, 0));
            e.0 += s;
            e.1 += 1;
        }
    }
    let mut means = serde_json::Map::new();
    for (name, (sum, n)) in sums {
        let mean = if n == 0 { 0.0 } else { sum / n as f64 };
        means.insert(name, serde_json::json!(mean));
    }
    AggregateScores {
        n_cases: per_case.len(),
        n_errors,
        means,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset() -> EvalDataset {
        EvalDataset::from_pairs([
            ("what is 2+2?", "4"),
            ("capital of france?", "Paris"),
            ("largest planet?", "Jupiter"),
        ])
    }

    #[tokio::test]
    async fn perfect_target_scores_one_dot_oh() {
        // Echo the expected answer per case.
        let lookup: std::collections::HashMap<String, String> = vec![
            ("what is 2+2?".into(), "4".into()),
            ("capital of france?".into(), "Paris".into()),
            ("largest planet?".into(), "Jupiter".into()),
        ].into_iter().collect();
        let target = move |q: String| {
            let lookup = lookup.clone();
            async move { Ok(lookup.get(&q).cloned().unwrap_or_default()) }
        };
        let report = run_eval(
            &dataset(),
            &[Arc::new(ExactMatchScorer) as Arc<dyn Scorer>],
            target,
            4,
        ).await.unwrap();
        assert_eq!(report.per_case.len(), 3);
        assert_eq!(report.aggregate.n_errors, 0);
        let mean = report.aggregate.means["exact_match"].as_f64().unwrap();
        assert!((mean - 1.0).abs() < 1e-9);
    }

    #[tokio::test]
    async fn wrong_target_scores_zero() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("wrong".to_string()) };
        let report = run_eval(
            &dataset(),
            &[Arc::new(ExactMatchScorer) as Arc<dyn Scorer>],
            target,
            4,
        ).await.unwrap();
        let mean = report.aggregate.means["exact_match"].as_f64().unwrap();
        assert_eq!(mean, 0.0);
    }

    #[tokio::test]
    async fn errored_case_appears_in_report_does_not_abort() {
        let target = |q: String| async move {
            if q.contains("planet") {
                Err::<String, _>(crate::Error::other("upstream timeout"))
            } else {
                Ok("ok".to_string())
            }
        };
        let report = run_eval(
            &dataset(),
            &[Arc::new(JaccardScorer) as Arc<dyn Scorer>],
            target,
            4,
        ).await.unwrap();
        assert_eq!(report.per_case.len(), 3);
        assert_eq!(report.aggregate.n_errors, 1);
        let errored = report.per_case.iter().find(|c| c.error.is_some()).unwrap();
        assert!(errored.error.as_ref().unwrap().contains("upstream timeout"));
        assert!(errored.output.is_none());
    }

    #[tokio::test]
    async fn ordering_preserved_under_concurrency() {
        // Even with parallelism, per_case must be in dataset order.
        let target = |q: String| async move {
            // Random-ish delay per case to scramble completion order.
            let delay = if q.contains("planet") { 5 } else { 1 };
            tokio::time::sleep(std::time::Duration::from_millis(delay)).await;
            Ok::<_, crate::Error>(format!("answer-for-{q}"))
        };
        let report = run_eval(
            &dataset(),
            &[Arc::new(JaccardScorer) as Arc<dyn Scorer>],
            target,
            8,
        ).await.unwrap();
        let inputs: Vec<&str> = report.per_case.iter().map(|c| c.input.as_str()).collect();
        assert_eq!(inputs, vec!["what is 2+2?", "capital of france?", "largest planet?"]);
    }

    #[tokio::test]
    async fn multiple_scorers_aggregate_separately() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("Paris".to_string()) };
        let report = run_eval(
            &dataset(),
            &[
                Arc::new(ExactMatchScorer) as Arc<dyn Scorer>,
                Arc::new(JaccardScorer) as Arc<dyn Scorer>,
            ],
            target,
            4,
        ).await.unwrap();
        assert!(report.aggregate.means.contains_key("exact_match"));
        assert!(report.aggregate.means.contains_key("jaccard"));
        // Only "Paris" matches → exact_match mean = 1/3.
        let em = report.aggregate.means["exact_match"].as_f64().unwrap();
        assert!((em - 1.0/3.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn from_jsonl_parses_one_per_line() {
        let jsonl = "{\"input\":\"q1\",\"expected\":\"a1\"}\n\n{\"input\":\"q2\",\"expected\":\"a2\",\"metadata\":{\"tag\":\"easy\"}}";
        let ds = EvalDataset::from_jsonl(jsonl).unwrap();
        assert_eq!(ds.len(), 2);
        assert_eq!(ds.cases[0].input, "q1");
        assert_eq!(ds.cases[1].metadata["tag"], "easy");
    }

    #[tokio::test]
    async fn from_jsonl_bad_line_errors_with_line_number() {
        let jsonl = "{\"input\":\"ok\",\"expected\":\"a\"}\nnot valid json";
        let err = EvalDataset::from_jsonl(jsonl).unwrap_err();
        assert!(err.to_string().contains("line 2"));
    }

    #[tokio::test]
    async fn empty_dataset_returns_empty_report_no_panic() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("x".into()) };
        let report = run_eval(
            &EvalDataset::default(),
            &[Arc::new(ExactMatchScorer) as Arc<dyn Scorer>],
            target,
            4,
        ).await.unwrap();
        assert_eq!(report.per_case.len(), 0);
        assert_eq!(report.aggregate.n_cases, 0);
        assert!(report.aggregate.means.is_empty());
    }

    #[tokio::test]
    async fn max_parallel_zero_is_clamped_to_one() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("x".into()) };
        let report = run_eval(
            &dataset(),
            &[Arc::new(ContainsAllScorer { required: vec!["x".into()] }) as Arc<dyn Scorer>],
            target,
            0,  // would panic with naive implementation
        ).await.unwrap();
        assert_eq!(report.per_case.len(), 3);
    }

    #[tokio::test]
    async fn regex_scorer_passes_when_pattern_matches() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("the answer is 42".into()) };
        let ds = EvalDataset::new(vec![EvalCase::new("q")]);
        let report = run_eval(
            &ds,
            &[Arc::new(RegexScorer { pattern: r"\d+".into() }) as Arc<dyn Scorer>],
            target,
            1,
        ).await.unwrap();
        assert_eq!(report.aggregate.means["regex"].as_f64().unwrap(), 1.0);
    }

    #[tokio::test]
    async fn levenshtein_scorer_emits_continuous_score() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("Paris, France".into()) };
        let ds = EvalDataset::new(vec![EvalCase::new("q").with_expected("Paris")]);
        let report = run_eval(
            &ds,
            &[Arc::new(LevenshteinScorer) as Arc<dyn Scorer>],
            target,
            1,
        ).await.unwrap();
        let s = report.aggregate.means["levenshtein"].as_f64().unwrap();
        assert!(s > 0.0 && s < 1.0, "expected partial match, got {s}");
    }

    #[tokio::test]
    async fn scorer_requiring_expected_errors_when_missing() {
        let target = |_q: String| async move { Ok::<_, crate::Error>("hi".into()) };
        let ds = EvalDataset::new(vec![EvalCase::new("q")]);  // no expected
        let report = run_eval(
            &ds,
            &[Arc::new(ExactMatchScorer) as Arc<dyn Scorer>],
            target,
            1,
        ).await.unwrap();
        // ExactMatch without expected → scorer-error per case → score=0 + explanation.
        let explain = report.per_case[0].scores["exact_match"]["explanation"].as_str().unwrap();
        assert!(explain.contains("requires"));
    }

    // --- LlmJudgeScorer tests ---
    use crate::model::{ChatStream, FinishReason, TokenUsage};
    use crate::{ChatModel, ChatOptions, ChatResponse, Message};
    use async_trait::async_trait;

    /// Returns a fixed JSON payload as the assistant message for every
    /// invoke. StructuredChatModel parses it via the schema enforcement.
    struct CannedJudge {
        payload: String,
    }

    #[async_trait]
    impl ChatModel for CannedJudge {
        fn name(&self) -> &str { "canned-judge" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(self.payload.clone()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "canned".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn llm_judge_scorer_returns_judge_score_and_reasoning() {
        let model = Arc::new(CannedJudge {
            payload: r#"{"score": 0.8, "reasoning": "close but missed nuance"}"#.into(),
        });
        let scorer = LlmJudgeScorer::new(model, None);
        let result = scorer.score("input", "Paris is the capital", Some("Paris"))
            .await
            .unwrap();
        assert!((result.score - 0.8).abs() < 1e-6);
        assert_eq!(result.explanation.as_deref(), Some("close but missed nuance"));
    }

    #[tokio::test]
    async fn llm_judge_scorer_runs_through_eval_harness() {
        let model = Arc::new(CannedJudge {
            payload: r#"{"score": 1.0, "reasoning": "exact match"}"#.into(),
        });
        let scorer = Arc::new(LlmJudgeScorer::new(model, None)) as Arc<dyn Scorer>;
        let ds = EvalDataset::from_pairs([("q1", "a1"), ("q2", "a2")]);
        let target = |_q: String| async move { Ok::<_, crate::Error>("anything".into()) };
        let report = run_eval(&ds, &[scorer], target, 2).await.unwrap();
        assert_eq!(report.per_case.len(), 2);
        let mean = report.aggregate.means["llm_judge"].as_f64().unwrap();
        assert!((mean - 1.0).abs() < 1e-9);
        // Per-case rows carry the judge's reasoning in `explanation`.
        let explain = report.per_case[0].scores["llm_judge"]["explanation"]
            .as_str()
            .unwrap();
        assert_eq!(explain, "exact match");
    }

    #[tokio::test]
    async fn llm_judge_scorer_errors_when_expected_missing() {
        let model = Arc::new(CannedJudge {
            payload: r#"{"score": 1.0, "reasoning": "x"}"#.into(),
        });
        let scorer = LlmJudgeScorer::new(model, None);
        let err = scorer.score("input", "output", None).await.unwrap_err();
        assert!(err.to_string().contains("requires"));
    }

    #[tokio::test]
    async fn llm_judge_scorer_with_custom_name_uses_it_in_report() {
        let model = Arc::new(CannedJudge {
            payload: r#"{"score": 0.5, "reasoning": "ok"}"#.into(),
        });
        let scorer = Arc::new(
            LlmJudgeScorer::new(model, None).with_name("strict_judge"),
        ) as Arc<dyn Scorer>;
        let ds = EvalDataset::from_pairs([("q", "a")]);
        let target = |_q: String| async move { Ok::<_, crate::Error>("x".into()) };
        let report = run_eval(&ds, &[scorer], target, 1).await.unwrap();
        assert!(report.aggregate.means.contains_key("strict_judge"));
        assert!(!report.aggregate.means.contains_key("llm_judge"));
    }

    #[tokio::test]
    async fn llm_judge_scorer_propagates_judge_invalid_score_error() {
        // Judge returns score outside [0, 1] — LlmJudge raises Error::Parse.
        let model = Arc::new(CannedJudge {
            payload: r#"{"score": 1.5, "reasoning": "out of range"}"#.into(),
        });
        let scorer = Arc::new(LlmJudgeScorer::new(model, None)) as Arc<dyn Scorer>;
        let ds = EvalDataset::from_pairs([("q", "a")]);
        let target = |_q: String| async move { Ok::<_, crate::Error>("x".into()) };
        let report = run_eval(&ds, &[scorer], target, 1).await.unwrap();
        // Judge error → per-case scorer error → score=0 + explanation surfaces error.
        let entry = &report.per_case[0].scores["llm_judge"];
        assert_eq!(entry["score"].as_f64().unwrap(), 0.0);
        let explain = entry["explanation"].as_str().unwrap();
        assert!(explain.contains("scorer error"));
    }
}

//! Synthetic eval-set generation. Given a few seed cases, ask an LLM to
//! produce N more in the same shape and difficulty so a small hand-written
//! seed expands into a usable evaluation dataset.
//!
//! Why this is its own module rather than living next to `eval_harness`: the
//! generator depends on a chat model + structured output, while the harness
//! is pure functional scoring. Keeping them separate means callers who only
//! score don't pay for the structured-output dep graph.
//!
//! # Quality notes
//!
//! - Always provide ≥3 seeds. Less than that and the LLM has too little
//!   shape to copy.
//! - Set `criteria` to spell out invariants (length range, output format,
//!   forbidden phrases). The model otherwise drifts.
//! - For factual-recall datasets, audit the generated cases — LLMs
//!   hallucinate plausible-but-wrong gold answers.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{
    eval_harness::EvalCase, ChatModel, ChatOptions, Error, Message, Result, StructuredChatModel,
};

const SYNTH_SYSTEM: &str = "You expand a small set of evaluation seed cases into more cases. \
    The expansion must preserve the seeds' shape (input format, kind of expected answer, \
    difficulty) while introducing variety. Return ONLY the JSON object specified by the \
    schema. Do not duplicate any seed. Do not produce empty inputs.";

/// One generated case. Mirrors `EvalCase`'s wire shape but stays JSON-friendly
/// for the structured-output schema.
#[derive(Debug, Clone, Deserialize, Serialize)]
struct SynthCase {
    input: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    expected: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct SynthBatch {
    #[serde(default)]
    cases: Vec<SynthCase>,
}

/// Synthesise additional eval cases by varying the seeds. Returns *only* the
/// generated cases (not the seeds). Caller's responsibility to concatenate
/// if they want a combined dataset.
///
/// `target_count` is the model's target — we ask for that many but accept
/// fewer if the LLM trims; we never silently pad. The actual length is
/// `min(target_count, len(returned))` after dedup against seeds.
pub async fn synthesize_eval_cases(
    model: Arc<dyn ChatModel>,
    seeds: &[EvalCase],
    target_count: usize,
    criteria: Option<&str>,
) -> Result<Vec<EvalCase>> {
    if seeds.is_empty() {
        return Err(Error::invalid(
            "synthesize_eval_cases: need at least one seed",
        ));
    }
    if target_count == 0 {
        return Ok(Vec::new());
    }

    let schema = json!({
        "type": "object",
        "properties": {
            "cases": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "input": { "type": "string", "minLength": 1 },
                        "expected": { "type": ["string", "null"] }
                    },
                    "required": ["input"]
                }
            }
        },
        "required": ["cases"]
    });
    let structured = StructuredChatModel::new(model, schema, "EvalSynthBatch").with_strict(true);

    let seeds_payload = serde_json::to_string_pretty(
        &seeds
            .iter()
            .map(|c| SynthCase {
                input: c.input.clone(),
                expected: c.expected.clone(),
            })
            .collect::<Vec<_>>(),
    )
    .map_err(|e| Error::other(format!("synth seeds encode: {e}")))?;

    let criteria_block = match criteria {
        Some(c) if !c.trim().is_empty() => format!("Constraints / criteria:\n{c}\n\n"),
        _ => String::new(),
    };

    let user = format!(
        "{criteria_block}Seeds (do NOT duplicate any of these in your output):\n{seeds_payload}\n\n\
         Generate {target_count} new cases following the same shape. Output JSON object \
         {{\"cases\": [...]}} with at most {target_count} items.",
    );

    let messages = vec![Message::system(SYNTH_SYSTEM), Message::user(user)];

    let raw_value = structured
        .invoke_structured(messages, &ChatOptions::default())
        .await?;
    let batch: SynthBatch = serde_json::from_value(raw_value)
        .map_err(|e| Error::other(format!("synth: bad batch json: {e}")))?;

    let seen: std::collections::HashSet<String> =
        seeds.iter().map(|c| c.input.trim().to_lowercase()).collect();
    let mut out: Vec<EvalCase> = Vec::with_capacity(batch.cases.len().min(target_count));
    for c in batch.cases {
        let key = c.input.trim().to_lowercase();
        if c.input.trim().is_empty() || seen.contains(&key) {
            continue;
        }
        let mut case = EvalCase::new(c.input);
        if let Some(e) = c.expected {
            if !e.trim().is_empty() {
                case = case.with_expected(e);
            }
        }
        out.push(case);
        if out.len() >= target_count {
            break;
        }
    }
    Ok(out)
}

/// Convenience: passes through a fixed JSON `Value` as if the LLM produced
/// it. Useful for tests and for callers who already have generated cases
/// from another source and just want them parsed into `EvalCase`s.
pub fn parse_synth_response(raw: Value, seeds: &[EvalCase], cap: usize) -> Result<Vec<EvalCase>> {
    let batch: SynthBatch = serde_json::from_value(raw)
        .map_err(|e| Error::other(format!("synth parse: {e}")))?;
    let seen: std::collections::HashSet<String> =
        seeds.iter().map(|c| c.input.trim().to_lowercase()).collect();
    let mut out = Vec::with_capacity(batch.cases.len().min(cap));
    for c in batch.cases {
        let key = c.input.trim().to_lowercase();
        if c.input.trim().is_empty() || seen.contains(&key) {
            continue;
        }
        let mut case = EvalCase::new(c.input);
        if let Some(e) = c.expected {
            if !e.trim().is_empty() {
                case = case.with_expected(e);
            }
        }
        out.push(case);
        if out.len() >= cap {
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};
    use crate::model::{
        ChatOptions, ChatResponse, ChatStream, FinishReason, TokenUsage,
    };
    use async_trait::async_trait;

    /// Returns a fixed JSON blob whenever invoked — fine for verifying the
    /// dedup + cap + empty-input-trim behaviour without an actual LLM.
    struct CannedModel {
        canned: String,
    }

    #[async_trait]
    impl ChatModel for CannedModel {
        fn name(&self) -> &str {
            "canned"
        }
        async fn invoke(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant(self.canned.clone()),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "canned".into(),
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

    fn seeds() -> Vec<EvalCase> {
        vec![
            EvalCase::new("What is 2+2?").with_expected("4"),
            EvalCase::new("What is 5*3?").with_expected("15"),
        ]
    }

    #[tokio::test]
    async fn empty_seeds_rejected() {
        let model = Arc::new(CannedModel {
            canned: "{}".into(),
        }) as Arc<dyn ChatModel>;
        let err = synthesize_eval_cases(model, &[], 5, None).await.unwrap_err();
        assert!(format!("{err}").contains("seed"));
    }

    #[tokio::test]
    async fn target_zero_returns_empty() {
        let model = Arc::new(CannedModel {
            canned: "{}".into(),
        }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 0, None)
            .await
            .unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn parses_returned_cases_and_drops_duplicates_against_seeds() {
        let canned = json!({
            "cases": [
                {"input": "What is 7+1?", "expected": "8"},
                // dup of seed (case-insensitive trim) — must drop.
                {"input": "  WHAT is 2+2?  ", "expected": "4"},
                {"input": "What is 9-3?", "expected": "6"},
            ]
        })
        .to_string();
        let model = Arc::new(CannedModel { canned }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 10, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|c| c.input.contains("7+1")));
        assert!(out.iter().any(|c| c.input.contains("9-3")));
    }

    #[tokio::test]
    async fn caps_at_target_count() {
        let canned = json!({
            "cases": [
                {"input": "q1", "expected": "a1"},
                {"input": "q2", "expected": "a2"},
                {"input": "q3", "expected": "a3"},
                {"input": "q4", "expected": "a4"},
            ]
        })
        .to_string();
        let model = Arc::new(CannedModel { canned }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 2, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
    }

    #[tokio::test]
    async fn empty_input_dropped() {
        let canned = json!({
            "cases": [
                {"input": "   ", "expected": "x"},
                {"input": "real q", "expected": "y"},
            ]
        })
        .to_string();
        let model = Arc::new(CannedModel { canned }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 5, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].input, "real q");
    }

    #[tokio::test]
    async fn missing_expected_field_kept_as_none() {
        let canned = json!({
            "cases": [
                {"input": "open question with no gold"},
            ]
        })
        .to_string();
        let model = Arc::new(CannedModel { canned }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 5, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].expected.is_none());
    }

    #[tokio::test]
    async fn empty_string_expected_treated_as_none() {
        let canned = json!({
            "cases": [
                {"input": "q", "expected": "   "},
            ]
        })
        .to_string();
        let model = Arc::new(CannedModel { canned }) as Arc<dyn ChatModel>;
        let out = synthesize_eval_cases(model, &seeds(), 5, None)
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].expected.is_none());
    }

    #[test]
    fn parse_synth_response_dedups_and_caps() {
        let raw = json!({
            "cases": [
                {"input": "q1", "expected": "a1"},
                {"input": "What is 2+2?", "expected": "x"}, // dup
                {"input": "q2", "expected": "a2"},
            ]
        });
        let out = parse_synth_response(raw, &seeds(), 5).unwrap();
        assert_eq!(out.len(), 2);
    }
}

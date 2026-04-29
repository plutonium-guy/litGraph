//! LLM-as-judge evaluator. Pairs with iter-111 string evaluators for
//! cases where exact-match / Levenshtein / Jaccard are too brittle:
//! paraphrases, partial answers, multi-criteria scoring, semantic
//! equivalence.
//!
//! # When to use vs string evaluators
//!
//! - **String evaluators (iter 111)** — cheap, deterministic, no API
//!   cost. Use for: "did the model output this exact token?", regex,
//!   list-membership, JSON-validity.
//! - **LlmJudge (this file)** — flexible, nuanced, costs 1 LLM call per
//!   sample. Use for: "is this answer semantically correct?", "does
//!   this summary capture the key points?", "is the tone appropriate?".
//!
//! # Cost hygiene
//!
//! Default prompt asks for a SHORT reasoning (≤2 sentences). Output
//! schema is strict JSON → small completion tokens. Use a cheap model
//! (gpt-4o-mini, claude-haiku). Per-sample cost typically <$0.001.
//!
//! # Score scale
//!
//! `0.0` (completely wrong / off-topic) → `1.0` (matches reference
//! exactly in intent + factual content). Caller picks a threshold for
//! pass/fail (0.8 is a common default for "acceptable").

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{ChatModel, ChatOptions, Error, Message, Result, StructuredChatModel};

const DEFAULT_CRITERIA: &str =
    "Does the prediction match the reference answer in meaning and factual content? \
     Paraphrases count as correct. Missing key facts, factual errors, or off-topic \
     content should lower the score.";

const JUDGE_SYSTEM: &str = "You are a strict evaluator. Score how well a prediction \
    matches a reference answer on a 0.0–1.0 scale based on the given criteria. Return \
    ONLY the JSON object specified by the schema. Keep reasoning short (≤2 sentences).";

/// Parsed judge result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JudgeScore {
    /// 0.0 (completely wrong) → 1.0 (fully correct).
    pub score: f32,
    /// Short justification from the judge model. ≤2 sentences in practice.
    pub reasoning: String,
}

/// LLM-as-judge evaluator. Holds a wrapped `StructuredChatModel` that
/// constrains the judge's output to the `{score, reasoning}` schema.
pub struct LlmJudge {
    inner: Arc<StructuredChatModel>,
    criteria: String,
}

impl LlmJudge {
    /// Wrap a ChatModel as a judge. `criteria` is the scoring rubric
    /// passed to the judge on every call; `None` uses the default
    /// ("match in meaning and factual content, paraphrases allowed").
    pub fn new(model: Arc<dyn ChatModel>, criteria: Option<String>) -> Self {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "score": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "description": "0.0 = wrong, 1.0 = fully correct"
                },
                "reasoning": {
                    "type": "string",
                    "description": "≤2 sentences explaining the score"
                }
            },
            "required": ["score", "reasoning"]
        });
        let structured =
            StructuredChatModel::new(model, schema, "JudgeScore").with_strict(true);
        Self {
            inner: Arc::new(structured),
            criteria: criteria.unwrap_or_else(|| DEFAULT_CRITERIA.to_string()),
        }
    }

    /// Score a single (prediction, reference) pair against the judge's
    /// criteria. Returns `JudgeScore {score, reasoning}` or an error if
    /// the judge model fails / returns invalid JSON / returns a score
    /// outside `[0, 1]`.
    pub async fn judge(&self, prediction: &str, reference: &str) -> Result<JudgeScore> {
        let user_prompt = format!(
            "Criteria:\n{}\n\n---\n\nReference answer:\n{}\n\n---\n\nPrediction:\n{}\n\n\
             Score the prediction.",
            self.criteria, reference, prediction
        );
        let messages = vec![
            Message::system(JUDGE_SYSTEM),
            Message::user(user_prompt),
        ];
        let value = self
            .inner
            .invoke_structured(messages, &ChatOptions::default())
            .await?;
        let parsed: JudgeScore = serde_json::from_value(value)
            .map_err(|e| Error::parse(format!("LlmJudge: malformed JudgeScore: {e}")))?;
        if !parsed.score.is_finite() || !(0.0..=1.0).contains(&parsed.score) {
            return Err(Error::parse(format!(
                "LlmJudge: score {} out of [0.0, 1.0]",
                parsed.score
            )));
        }
        Ok(parsed)
    }

    /// Score a batch of (prediction, reference) pairs. Serial — one
    /// judge call per pair. For parallelism, wrap with your own
    /// `futures::stream::FuturesUnordered` at the call site.
    pub async fn judge_batch(
        &self,
        pairs: Vec<(String, String)>,
    ) -> Result<Vec<JudgeScore>> {
        let mut out = Vec::with_capacity(pairs.len());
        for (pred, refn) in pairs {
            out.push(self.judge(&pred, &refn).await?);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ChatResponse, ChatStream, FinishReason, TokenUsage};
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Canned-JSON chat fake. Returns the same JSON payload for every
    /// `invoke` call. StructuredChatModel will parse it via its
    /// tool-call / response_format dispatch; we fake the tool-call path
    /// by returning a ChatResponse whose message.tool_calls contains
    /// the scripted JSON as the arguments of a single JudgeScore call.
    struct ScriptedJudge {
        payloads: Mutex<Vec<String>>,
        seen: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedJudge {
        fn new(payloads: Vec<&str>) -> Arc<Self> {
            Arc::new(Self {
                payloads: Mutex::new(payloads.into_iter().map(str::to_string).rev().collect()),
                seen: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedJudge {
        fn name(&self) -> &str {
            "scripted-judge"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen.lock().unwrap().push(messages);
            let content = self
                .payloads
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| r#"{"score":0.0,"reasoning":"no more"}"#.into());
            // StructuredChatModel parses the JSON out of message text
            // content (it prefers response_format=json_schema / text
            // over tool-call dispatch). So return the JSON as the
            // assistant's text.
            Ok(ChatResponse {
                message: Message::assistant(content),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted".into(),
            })
        }
        async fn stream(
            &self,
            _m: Vec<Message>,
            _o: &ChatOptions,
        ) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn judges_matching_answer_high_score() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.95,"reasoning":"Prediction matches reference in both facts and intent."}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let score = judge
            .judge(
                "Paris is the capital of France.",
                "The capital of France is Paris.",
            )
            .await
            .unwrap();
        assert!(score.score >= 0.9);
        assert!(!score.reasoning.is_empty());
    }

    #[tokio::test]
    async fn judges_wrong_answer_low_score() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.0,"reasoning":"Prediction states a wrong capital."}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let score = judge
            .judge("London is the capital of France.", "Paris")
            .await
            .unwrap();
        assert!(score.score <= 0.1);
    }

    #[tokio::test]
    async fn score_out_of_range_errors() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":1.5,"reasoning":"bug"}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let err = judge.judge("p", "r").await.unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("out of"));
    }

    #[tokio::test]
    async fn score_nan_errors() {
        let chat = ScriptedJudge::new(vec![
            // Emit a valid-but-NaN-ish score. JSON spec says NaN is
            // invalid JSON so we can't literally return NaN; instead use
            // an out-of-range value that triggers the same validation.
            r#"{"score":-0.5,"reasoning":"underflow"}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        assert!(judge.judge("p", "r").await.is_err());
    }

    #[tokio::test]
    async fn custom_criteria_appears_in_prompt() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.8,"reasoning":"ok"}"#,
        ]);
        let criteria = "Rate ONLY on factual accuracy, ignore tone and phrasing.";
        let judge =
            LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, Some(criteria.into()));
        let _ = judge.judge("p", "r").await.unwrap();
        let seen = chat.seen.lock().unwrap();
        let user_msg = seen[0]
            .iter()
            .filter(|m| matches!(m.role, crate::Role::User))
            .map(|m| m.text_content())
            .collect::<String>();
        assert!(user_msg.contains("Rate ONLY on factual accuracy"));
    }

    #[tokio::test]
    async fn default_criteria_mentions_meaning_and_facts() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.5,"reasoning":"ok"}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let _ = judge.judge("p", "r").await.unwrap();
        let seen = chat.seen.lock().unwrap();
        let user_msg = seen[0]
            .iter()
            .filter(|m| matches!(m.role, crate::Role::User))
            .map(|m| m.text_content())
            .collect::<String>();
        assert!(user_msg.to_lowercase().contains("meaning"));
        assert!(user_msg.to_lowercase().contains("factual"));
    }

    #[tokio::test]
    async fn judge_batch_preserves_input_order() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.1,"reasoning":"first"}"#,
            r#"{"score":0.5,"reasoning":"second"}"#,
            r#"{"score":0.9,"reasoning":"third"}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let pairs = vec![
            ("a".into(), "b".into()),
            ("c".into(), "d".into()),
            ("e".into(), "f".into()),
        ];
        let scores = judge.judge_batch(pairs).await.unwrap();
        assert_eq!(scores.len(), 3);
        // Order preserved, and scores monotone (inputs match payloads in order).
        assert!(scores[0].score < scores[1].score);
        assert!(scores[1].score < scores[2].score);
    }

    #[tokio::test]
    async fn prompt_carries_reference_and_prediction_in_that_order() {
        let chat = ScriptedJudge::new(vec![
            r#"{"score":0.5,"reasoning":"ok"}"#,
        ]);
        let judge = LlmJudge::new(chat.clone() as Arc<dyn ChatModel>, None);
        let _ = judge
            .judge("THE_PREDICTION_TEXT", "THE_REFERENCE_TEXT")
            .await
            .unwrap();
        let seen = chat.seen.lock().unwrap();
        let user = seen[0]
            .iter()
            .filter(|m| matches!(m.role, crate::Role::User))
            .map(|m| m.text_content())
            .collect::<String>();
        let ref_pos = user.find("THE_REFERENCE_TEXT").unwrap();
        let pred_pos = user.find("THE_PREDICTION_TEXT").unwrap();
        assert!(ref_pos < pred_pos, "reference should appear before prediction");
    }
}

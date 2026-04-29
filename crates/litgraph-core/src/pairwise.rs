//! Pairwise evaluation — given the same input plus two candidate outputs,
//! ask an LLM judge which is better. Pairs with `LlmJudge` (single-output
//! grading) for A/B testing prompts, models, or fine-tunes.
//!
//! # When to use vs `LlmJudge`
//!
//! - **`LlmJudge`** — absolute scoring on a 0–1 scale against a reference.
//!   Right when you have a gold answer and want a per-sample number.
//! - **`PairwiseEvaluator` (this file)** — relative ranking when there's no
//!   gold answer or when you care about head-to-head wins between two
//!   versions of a prompt / model. Output is `{winner, confidence, reason}`.
//!
//! # Position-bias mitigation
//!
//! LLM judges famously prefer whichever candidate appears first. We expose a
//! `randomize_order: bool` flag (default true): the judge sees a randomly
//! ordered (A, B) pair and we map the answer back to the caller's
//! (left, right) labels. For a single comparison this halves the bias; for
//! batch comparisons, average over many cases drives it toward zero.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{ChatModel, ChatOptions, Error, Message, Result, StructuredChatModel};

const DEFAULT_PAIRWISE_CRITERIA: &str =
    "Pick the response that better answers the user's input — accuracy, \
     completeness, helpfulness, and absence of hallucinations matter. \
     Treat tone differences as ties unless the criteria say otherwise.";

const PAIRWISE_SYSTEM: &str = "You are a strict pairwise evaluator. Two \
    candidate responses (A and B) are given for the same input. Return ONLY \
    the JSON object specified by the schema. Keep `reason` short (≤2 \
    sentences).";

/// Which side won the comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Winner {
    Left,
    Right,
    Tie,
}

/// Result of a single A/B judgement, *unmapped* — refers to the candidates
/// the caller originally passed as `left` and `right`. Order randomization
/// is undone before this struct is returned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseResult {
    pub winner: Winner,
    /// `0.0` (coin flip) → `1.0` (clear). Reflects the judge's stated
    /// confidence; use with skepticism but it's a useful sort key.
    pub confidence: f32,
    pub reason: String,
}

/// Internal — what the judge LLM emits before we map A/B → left/right.
#[derive(Debug, Clone, Deserialize)]
struct RawJudgement {
    winner: String,
    #[serde(default)]
    confidence: f32,
    #[serde(default)]
    reason: String,
}

pub struct PairwiseEvaluator {
    inner: Arc<StructuredChatModel>,
    criteria: String,
}

impl PairwiseEvaluator {
    pub fn new(model: Arc<dyn ChatModel>, criteria: Option<String>) -> Self {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "winner": {
                    "type": "string",
                    "enum": ["A", "B", "tie"],
                    "description": "Which candidate is better, or 'tie'."
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0
                },
                "reason": {
                    "type": "string",
                    "description": "≤2 sentences."
                }
            },
            "required": ["winner", "confidence", "reason"]
        });
        let structured =
            StructuredChatModel::new(model, schema, "PairwiseJudgement").with_strict(true);
        Self {
            inner: Arc::new(structured),
            criteria: criteria.unwrap_or_else(|| DEFAULT_PAIRWISE_CRITERIA.to_string()),
        }
    }

    /// Compare `left` vs `right` for the same `input`. If `randomize_order`
    /// is true (default), the labels A/B shown to the judge are randomly
    /// shuffled and the result is mapped back. Pass `false` for
    /// reproducibility in tests.
    pub async fn compare(
        &self,
        input: &str,
        left: &str,
        right: &str,
        randomize_order: bool,
    ) -> Result<PairwiseResult> {
        // Randomization seed: cheap, deterministic-ish per input — avoids a
        // dep on `rand` in core. The seed is just (left, right, input)
        // hashes XOR'd. Good enough to scramble without being cryptographic.
        let swap = if randomize_order {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            input.hash(&mut h);
            left.hash(&mut h);
            right.hash(&mut h);
            (h.finish() & 1) == 0
        } else {
            false
        };
        let (a, b) = if swap { (right, left) } else { (left, right) };

        let prompt = format!(
            "Criteria:\n{}\n\nInput:\n{}\n\nResponse A:\n{}\n\nResponse B:\n{}",
            self.criteria.trim(),
            input,
            a,
            b
        );
        let messages = vec![
            Message::system(PAIRWISE_SYSTEM),
            Message::user(prompt),
        ];

        let raw_value = self
            .inner
            .invoke_structured(messages, &ChatOptions::default())
            .await?;
        let raw: RawJudgement = serde_json::from_value(raw_value)
            .map_err(|e| Error::other(format!("pairwise: bad judgement json: {e}")))?;

        let confidence = raw.confidence.clamp(0.0, 1.0);
        let raw_winner = match raw.winner.to_ascii_lowercase().as_str() {
            "a" => Winner::Left,
            "b" => Winner::Right,
            "tie" | "draw" | "neither" | "equal" => Winner::Tie,
            other => {
                return Err(Error::other(format!(
                    "pairwise: judge returned unknown winner `{other}`"
                )));
            }
        };
        let winner = if swap {
            match raw_winner {
                Winner::Left => Winner::Right,
                Winner::Right => Winner::Left,
                Winner::Tie => Winner::Tie,
            }
        } else {
            raw_winner
        };
        Ok(PairwiseResult {
            winner,
            confidence,
            reason: raw.reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{Message, Role};
    use crate::model::{
        ChatOptions, ChatResponse, ChatStream, FinishReason, TokenUsage,
    };
    use async_trait::async_trait;
    use std::sync::Mutex;

    /// Stub model that always returns the canned JSON the test wants.
    struct CannedJudge {
        canned: String,
        last_user: Arc<Mutex<Option<String>>>,
    }

    #[async_trait]
    impl ChatModel for CannedJudge {
        fn name(&self) -> &str {
            "canned"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            if let Some(m) = messages.iter().rev().find(|m| matches!(m.role, Role::User))
            {
                *self.last_user.lock().unwrap() = Some(m.text_content());
            }
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

    fn build(canned_json: &str) -> (PairwiseEvaluator, Arc<Mutex<Option<String>>>) {
        let last = Arc::new(Mutex::new(None));
        let model = Arc::new(CannedJudge {
            canned: canned_json.to_string(),
            last_user: last.clone(),
        }) as Arc<dyn ChatModel>;
        (PairwiseEvaluator::new(model, None), last)
    }

    #[tokio::test]
    async fn winner_a_maps_to_left_when_no_swap() {
        let (e, _) = build(r#"{"winner":"A","confidence":0.9,"reason":"yes"}"#);
        let r = e.compare("q", "left answer", "right answer", false).await.unwrap();
        assert_eq!(r.winner, Winner::Left);
        assert!((r.confidence - 0.9).abs() < 1e-6);
    }

    #[tokio::test]
    async fn winner_b_maps_to_right_when_no_swap() {
        let (e, _) = build(r#"{"winner":"B","confidence":0.5,"reason":"meh"}"#);
        let r = e.compare("q", "L", "R", false).await.unwrap();
        assert_eq!(r.winner, Winner::Right);
    }

    #[tokio::test]
    async fn tie_passes_through() {
        let (e, _) = build(r#"{"winner":"tie","confidence":0.1,"reason":"close"}"#);
        let r = e.compare("q", "L", "R", false).await.unwrap();
        assert_eq!(r.winner, Winner::Tie);
    }

    #[tokio::test]
    async fn confidence_clamped_to_unit_interval() {
        let (e, _) = build(r#"{"winner":"A","confidence":2.5,"reason":""}"#);
        let r = e.compare("q", "L", "R", false).await.unwrap();
        assert!((r.confidence - 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn unknown_winner_token_errors() {
        let (e, _) = build(r#"{"winner":"squirrel","confidence":0.5,"reason":"x"}"#);
        let err = e.compare("q", "L", "R", false).await.unwrap_err();
        assert!(format!("{err}").to_lowercase().contains("squirrel"));
    }

    #[tokio::test]
    async fn no_swap_when_disabled() {
        let (e, last) = build(r#"{"winner":"A","confidence":1.0,"reason":""}"#);
        e.compare("question", "LEFT", "RIGHT", false).await.unwrap();
        let user_msg = last.lock().unwrap().clone().unwrap();
        // A appears before B; A should be the original LEFT.
        let a_pos = user_msg.find("Response A:\nLEFT").unwrap();
        let b_pos = user_msg.find("Response B:\nRIGHT").unwrap();
        assert!(a_pos < b_pos);
    }

    #[tokio::test]
    async fn randomized_swap_remaps_winner() {
        // Construct the input so the deterministic seed swaps. We rely on
        // hashing being stable within a process, so we look at the actual
        // user message and verify the mapping is consistent — i.e. if the
        // judge sees LEFT as B and picks B, the result is `Left`.
        let canned = r#"{"winner":"B","confidence":1.0,"reason":""}"#;
        let (e, last) = build(canned);
        let r = e.compare("input", "LEFT", "RIGHT", true).await.unwrap();
        let msg = last.lock().unwrap().clone().unwrap();
        let left_seen_as_a = msg.find("Response A:\nLEFT").is_some();
        match r.winner {
            Winner::Left => assert!(!left_seen_as_a, "left won despite being labelled A and B winning"),
            Winner::Right => assert!(left_seen_as_a, "right won despite being labelled B and B winning"),
            Winner::Tie => panic!("unexpected tie"),
        }
    }
}

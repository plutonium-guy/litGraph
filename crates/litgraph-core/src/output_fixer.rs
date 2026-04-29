//! LLM-based output repair. When a parser fails on LLM output, send the
//! raw output + error + format-instructions back to a fixer ChatModel
//! and ask it to correct the formatting. Direct LangChain
//! `OutputFixingParser` parity (FEATURES.md line 78).
//!
//! # When this helps
//!
//! - **Malformed JSON** from a model that doesn't support strict
//!   structured-output mode (most local models, older checkpoints, some
//!   open-weight fine-tunes). Trailing commas, smart quotes, missing
//!   keys → fixer rewrites and you keep moving.
//! - **Wrong XML tag** — model emitted `<reasoning>` when prompt asked
//!   for `<thinking>`. Fixer renames.
//! - **Wrong format** — prompt asked for a numbered list, model emitted
//!   a comma-separated one. Fixer reformats.
//!
//! # When NOT to use
//!
//! - The error is semantic, not formatting. ("LLM said the wrong city" —
//!   no amount of repair-prompting fixes a wrong fact.)
//! - Cost-sensitive paths. Repair = extra LLM call ≈ 2x cost in the
//!   worst case. Use a cheaper fixer model (gpt-4o-mini, Claude Haiku)
//!   to amortize.
//! - You can use provider-native structured output (OpenAI's
//!   `response_format=json_schema`, Anthropic's tool-call mode). Native
//!   structured output never produces malformed JSON in the first place
//!   — `StructuredChatModel` (iter 89) is the right tool there.
//!
//! # Two entry points
//!
//! - [`fix_with_llm`] — pure function. Send raw + error + instructions to
//!   a fixer model, return its corrected text. No parse loop.
//! - [`parse_with_retry`] — generic loop. Try a parser; on fail, call
//!   `fix_with_llm`, re-parse, repeat until success or `max_retries`
//!   exhausted. Caller supplies the parse closure (works with any
//!   parser type that returns `Result<T>`).

use std::sync::Arc;

use crate::model::ChatModel;
use crate::{ChatOptions, Error, Message, Result};

/// Repair prompt template. Gets a `<format_instructions>` only if the
/// caller passed one; otherwise that block is omitted (some parsers
/// have no natural format prompt — boolean / regex match).
const REPAIR_TEMPLATE: &str = "The previous output failed to parse:

<error>
{error}
</error>

The previous output was:

<previous_output>
{raw}
</previous_output>
{format_block}
Return ONLY the corrected output. No explanation, no prose, no markdown code fences.";

/// Send raw + error + format-instructions to a fixer LLM. Returns the
/// model's correction as text. Single call — no parse loop. Use
/// [`parse_with_retry`] for the full retry loop.
///
/// `format_instructions` may be empty (we just omit that prompt block).
///
/// `model` SHOULD be a cheap one (gpt-4o-mini, claude-haiku) — repair
/// usually doesn't need a flagship model. Caller picks.
pub async fn fix_with_llm(
    raw: &str,
    error: &str,
    format_instructions: &str,
    model: Arc<dyn ChatModel>,
    chat_options: &ChatOptions,
) -> Result<String> {
    let format_block = if format_instructions.is_empty() {
        String::new()
    } else {
        format!(
            "\nThe correct format is:\n\n<format>\n{}\n</format>\n",
            format_instructions
        )
    };
    let prompt = REPAIR_TEMPLATE
        .replace("{error}", error)
        .replace("{raw}", raw)
        .replace("{format_block}", &format_block);
    let messages = vec![
        Message::system(
            "You are a strict output formatter. Given a previous output and the parse \
             error it produced, return ONLY a corrected version. Do not explain, do not \
             apologize, do not wrap in markdown.",
        ),
        Message::user(prompt),
    ];
    let resp = model.invoke(messages, chat_options).await?;
    Ok(resp.message.text_content())
}

/// Parse-with-retry loop. Call `parse_fn(text)`; on fail, ask the fixer
/// model to repair, re-parse, repeat. Returns the parsed value or the
/// final error after `max_retries`.
///
/// `max_retries=0` means: try once, no fixes. `max_retries=1` means:
/// initial try + one repair attempt. Default LangChain shape is
/// `max_retries=1` — repair is a one-shot rescue, not a debug loop.
///
/// `parse_fn` is generic over any closure that returns `Result<T>`.
/// Compose with iter-105 `parse_xml_tags`, iter-106 list parsers,
/// iter-107 `parse_react_step`, your own `serde_json::from_str::<MyT>`,
/// etc.
pub async fn parse_with_retry<T, F>(
    initial_raw: String,
    parse_fn: F,
    model: Arc<dyn ChatModel>,
    chat_options: &ChatOptions,
    format_instructions: &str,
    max_retries: usize,
) -> Result<T>
where
    F: Fn(&str) -> Result<T>,
{
    let mut current = initial_raw;
    let mut last_error: Option<Error> = None;

    for attempt in 0..=max_retries {
        match parse_fn(&current) {
            Ok(v) => return Ok(v),
            Err(e) if attempt == max_retries => {
                // Out of retries — surface the final parse error.
                return Err(e);
            }
            Err(e) => {
                let err_text = e.to_string();
                last_error = Some(e);
                current = fix_with_llm(
                    &current,
                    &err_text,
                    format_instructions,
                    model.clone(),
                    chat_options,
                )
                .await?;
            }
        }
    }
    // Loop body always returns or assigns to `current`. Fallback: surface
    // the last parse error. Should be unreachable given the bounds above.
    Err(last_error.unwrap_or_else(|| Error::other("parse_with_retry: no result")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use crate::model::{ChatResponse, ChatStream, FinishReason, TokenUsage};
    use std::sync::Mutex;

    /// Returns canned strings in order.
    struct ScriptedChat {
        responses: Mutex<Vec<String>>,
        seen_messages: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedChat {
        fn new(responses: Vec<&str>) -> Arc<Self> {
            Arc::new(Self {
                responses: Mutex::new(responses.into_iter().map(str::to_string).rev().collect()),
                seen_messages: Mutex::new(Vec::new()),
            })
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedChat {
        fn name(&self) -> &str {
            "scripted"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            self.seen_messages.lock().unwrap().push(messages);
            let content = self
                .responses
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| "(out of scripted responses)".into());
            Ok(ChatResponse {
                message: Message::assistant(content),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted".into(),
            })
        }
        async fn stream(&self, _: Vec<Message>, _: &ChatOptions) -> Result<ChatStream> {
            unimplemented!()
        }
    }

    #[tokio::test]
    async fn fix_with_llm_returns_model_correction() {
        let model = ScriptedChat::new(vec![r#"{"x": 1}"#]);
        let out = fix_with_llm(
            "{\"x\": 1,}",
            "trailing comma at line 1",
            "valid JSON object",
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
        )
        .await
        .unwrap();
        assert_eq!(out, r#"{"x": 1}"#);
    }

    #[tokio::test]
    async fn fix_with_llm_includes_error_and_raw_in_user_prompt() {
        let model = ScriptedChat::new(vec!["fixed"]);
        let _ = fix_with_llm(
            "garbage",
            "missing closing brace",
            "valid JSON",
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
        )
        .await
        .unwrap();
        let seen = model.seen_messages.lock().unwrap();
        let user_msg = seen[0].last().unwrap().text_content();
        assert!(user_msg.contains("garbage"));
        assert!(user_msg.contains("missing closing brace"));
        assert!(user_msg.contains("valid JSON"));
    }

    #[tokio::test]
    async fn fix_with_llm_omits_format_block_when_empty() {
        let model = ScriptedChat::new(vec!["fixed"]);
        let _ = fix_with_llm(
            "raw",
            "err",
            "", // no instructions
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
        )
        .await
        .unwrap();
        let seen = model.seen_messages.lock().unwrap();
        let user_msg = seen[0].last().unwrap().text_content();
        // The "<format>" block must NOT appear when no instructions given.
        assert!(!user_msg.contains("<format>"));
    }

    #[tokio::test]
    async fn parse_with_retry_succeeds_on_first_try_no_llm_call() {
        let model = ScriptedChat::new(vec![]);
        let parse = |s: &str| -> Result<i64> {
            s.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let out = parse_with_retry(
            "42".into(),
            parse,
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
            "an integer",
            3,
        )
        .await
        .unwrap();
        assert_eq!(out, 42);
        // No model invocation.
        assert_eq!(model.seen_messages.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn parse_with_retry_recovers_after_one_repair() {
        // Initial input bad; fixer returns a parseable correction.
        let model = ScriptedChat::new(vec!["42"]);
        let parse = |s: &str| -> Result<i64> {
            s.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let out = parse_with_retry(
            "forty two".into(),
            parse,
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
            "an integer",
            1,
        )
        .await
        .unwrap();
        assert_eq!(out, 42);
        assert_eq!(model.seen_messages.lock().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn parse_with_retry_exhausts_retries_returns_final_parse_error() {
        // Fixer returns more garbage; we should give up after `max_retries` tries.
        let model = ScriptedChat::new(vec!["still bad", "still bad", "still bad"]);
        let parse = |s: &str| -> Result<i64> {
            s.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let err = parse_with_retry(
            "garbage".into(),
            parse,
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
            "an integer",
            2,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Parse(_)));
        // 2 retries → 1 initial + 2 repair calls = 2 model invocations
        // (initial parse doesn't call model).
        assert_eq!(model.seen_messages.lock().unwrap().len(), 2);
    }

    #[tokio::test]
    async fn parse_with_retry_max_retries_zero_no_repair() {
        let model = ScriptedChat::new(vec![]);
        let parse = |s: &str| -> Result<i64> {
            s.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let err = parse_with_retry(
            "garbage".into(),
            parse,
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
            "an integer",
            0,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, Error::Parse(_)));
        // No repair calls.
        assert_eq!(model.seen_messages.lock().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn parse_with_retry_composes_with_existing_parsers() {
        // Realistic: use parse_xml_tags from iter 105 as the parse_fn.
        // First raw response missing the answer tag → fixer adds it.
        use crate::parse_xml_tags;
        let model = ScriptedChat::new(vec!["<answer>42</answer>"]);
        let parse = |s: &str| -> Result<i64> {
            let m = parse_xml_tags(s, &["answer"]);
            let v = m
                .get("answer")
                .ok_or_else(|| Error::parse("missing <answer> tag"))?;
            v.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let out = parse_with_retry(
            "the answer is 42 (no tags)".into(),
            parse,
            model.clone() as Arc<dyn ChatModel>,
            &ChatOptions::default(),
            "<answer>...</answer>",
            1,
        )
        .await
        .unwrap();
        assert_eq!(out, 42);
    }

    #[tokio::test]
    async fn fixer_model_failure_propagates_immediately() {
        // A fixer model that errors hard (rate limit) should NOT be
        // retried by the parse_with_retry loop — surface immediately.
        struct AlwaysFails;
        #[async_trait]
        impl ChatModel for AlwaysFails {
            fn name(&self) -> &str {
                "fails"
            }
            async fn invoke(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> Result<ChatResponse> {
                Err(Error::RateLimited {
                    retry_after_ms: None,
                })
            }
            async fn stream(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> Result<ChatStream> {
                unimplemented!()
            }
        }
        let model: Arc<dyn ChatModel> = Arc::new(AlwaysFails);
        let parse = |s: &str| -> Result<i64> {
            s.trim().parse::<i64>().map_err(|e| Error::parse(e.to_string()))
        };
        let err = parse_with_retry(
            "garbage".into(),
            parse,
            model,
            &ChatOptions::default(),
            "an integer",
            5,
        )
        .await
        .unwrap_err();
        // Surfaces fixer's error, NOT the parse error.
        assert!(matches!(err, Error::RateLimited { .. }));
    }
}

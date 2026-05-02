//! ReAct text-mode output parser — parse `Thought:` / `Action:` /
//! `Action Input:` / `Final Answer:` prose from LLMs that don't support
//! native tool-calling (Ollama local models, older Gemini checkpoints,
//! base completion models). Direct LangChain `ReActOutputParser` parity.
//!
//! # Why this exists
//!
//! iter-13's `ReactAgent` uses the provider's native tool-calling API
//! (OpenAI `tools`, Anthropic `tool_use`, Bedrock Converse `toolUse`).
//! That path covers 95% of real usage but NOT:
//!
//! - Local models via Ollama / vLLM / llama.cpp that return prose
//! - Older base-completion models (text-davinci-*, text-bison-*)
//! - Fine-tuned open-weight models that learned the ReAct format
//!   during training
//!
//! For those, the standard is ReAct prose format:
//!
//! ```text
//! Thought: I need to know today's weather in Paris.
//! Action: get_weather
//! Action Input: {"city": "Paris"}
//! ```
//!
//! ...then the runner feeds `Observation: <tool result>` back and the
//! model continues until it emits:
//!
//! ```text
//! Thought: I have the answer.
//! Final Answer: It's 15°C and raining.
//! ```
//!
//! This parser is the bridge from LLM prose → structured decision.
//!
//! # What it handles
//!
//! - Multi-line `Thought:` blocks (LLM reasoning is verbose)
//! - `Action Input:` as either JSON (`{"city": "Paris"}`) or raw string
//!   (`Paris` — some models skip JSON)
//! - Trailing prose after `Final Answer:` (LLMs add sign-off text)
//! - Missing `Thought:` (LLMs sometimes skip straight to Action)
//! - Case-insensitive labels (`action:` matches `Action:`)
//! - `Final Answer` takes precedence over `Action` when both appear
//!   (LLM sometimes hedges: "Action: get_weather\nFinal Answer: rain" —
//!   we take the final answer as authoritative)

use serde_json::Value;

use crate::{Error, Result};

/// A single ReAct step: either invoke a tool or emit a final answer.
#[derive(Debug, Clone, PartialEq)]
pub enum ReactStep {
    Action {
        thought: Option<String>,
        tool: String,
        input: Value,
    },
    Final {
        thought: Option<String>,
        answer: String,
    },
}

/// Parse a ReAct-format LLM response into one structured step.
///
/// Returns `Err` only if neither an `Action:` + `Action Input:` pair
/// NOR a `Final Answer:` is found. LLMs occasionally emit gibberish;
/// the runner decides retry vs fail.
///
/// ```ignore
/// let text = "Thought: check the weather\n\
///             Action: get_weather\n\
///             Action Input: {\"city\": \"Paris\"}";
/// let step = parse_react_step(text).unwrap();
/// match step {
///     ReactStep::Action { tool, input, .. } => {
///         assert_eq!(tool, "get_weather");
///         assert_eq!(input["city"], "Paris");
///     }
///     _ => panic!("expected Action"),
/// }
/// ```
pub fn parse_react_step(text: &str) -> Result<ReactStep> {
    let thought = extract_block(text, "Thought");

    // Final Answer wins over Action when both appear.
    if let Some(answer) = extract_block(text, "Final Answer") {
        return Ok(ReactStep::Final {
            thought,
            answer: answer.trim().to_string(),
        });
    }

    let tool = extract_block(text, "Action");
    let raw_input = extract_block(text, "Action Input");

    match (tool, raw_input) {
        (Some(tool), Some(raw_input)) => {
            let tool = tool.trim().to_string();
            if tool.is_empty() {
                return Err(Error::parse(
                    "ReAct parse: empty `Action:` value".to_string(),
                ));
            }
            let input = parse_action_input(raw_input.trim());
            Ok(ReactStep::Action {
                thought,
                tool,
                input,
            })
        }
        (Some(_), None) => Err(Error::parse(
            "ReAct parse: found `Action:` but no `Action Input:`".to_string(),
        )),
        (None, Some(_)) => Err(Error::parse(
            "ReAct parse: found `Action Input:` but no `Action:`".to_string(),
        )),
        (None, None) => Err(Error::parse(format!(
            "ReAct parse: no `Action:`+`Action Input:` pair and no `Final Answer:` in {:?}",
            snippet(text)
        ))),
    }
}

/// Extract the content of a labeled block — everything between `<label>:`
/// and the next recognized label (or EOF). Case-insensitive on the label.
/// Returns None if the label isn't found.
///
/// Also handles the common variant where the LLM wraps the label in
/// `**bold**` or surrounds with backticks.
fn extract_block(text: &str, label: &str) -> Option<String> {
    // Find the label (case-insensitive) followed by `:`. Allow optional
    // `**` markdown bold around the label.
    let start_byte = find_label(text, label)?;
    // Skip past `<label>:` — we know this is a simple ASCII match by
    // construction. Walk forward until we pass the `:`.
    let rest = &text[start_byte..];
    let colon = rest.find(':')?;
    let after_colon = &rest[colon + 1..];
    // When the LLM uses `**Label:**` markdown-bold, the `**` close marker
    // sits right after the colon — strip it so it doesn't bleed into the
    // block content. Also strip trailing bold markers at block END below.
    let after_colon = after_colon.trim_start_matches('*');

    // Now find where this block ends — the next label OR EOF.
    // Candidate labels are any of: Thought, Action, Action Input,
    // Observation, Final Answer.
    let labels = [
        "Thought:",
        "Action Input:",
        "Action:",
        "Observation:",
        "Final Answer:",
    ];
    let mut end = after_colon.len();
    for next_label in &labels {
        // Look for a newline-preceded label; a label mid-sentence is not
        // a section boundary (e.g. "Action should be careful" is prose,
        // not a new section).
        if let Some(idx) = find_label_at_line_start(after_colon, next_label) {
            if idx < end {
                end = idx;
            }
        }
    }

    let block = after_colon[..end].trim();
    if block.is_empty() {
        None
    } else {
        Some(block.to_string())
    }
}

/// Find the byte offset where `<label>:` starts in `text`. Label match
/// is case-insensitive and must be preceded by start-of-string,
/// whitespace, or a newline.
fn find_label(text: &str, label: &str) -> Option<usize> {
    let label_lower = label.to_ascii_lowercase();
    let text_lower = text.to_ascii_lowercase();
    let mut search_from = 0usize;
    while let Some(pos) = text_lower[search_from..].find(&label_lower) {
        let abs = search_from + pos;
        // Check prev char is line-start or whitespace.
        let prev_ok = abs == 0
            || matches!(text.as_bytes()[abs - 1], b'\n' | b'\r' | b' ' | b'\t' | b'*' | b'#');
        // Strip any trailing `**` (markdown bold close) before the colon.
        let after_label = abs + label.len();
        let mut cursor = after_label;
        let bytes = text.as_bytes();
        while cursor < bytes.len() && matches!(bytes[cursor], b'*' | b' ' | b'\t') {
            cursor += 1;
        }
        let has_colon = cursor < bytes.len() && bytes[cursor] == b':';
        if prev_ok && has_colon {
            return Some(abs);
        }
        search_from = abs + 1;
    }
    None
}

/// Find the byte offset of `label` in `text`, constrained to positions
/// that are line-starts (first non-whitespace on their line). Used for
/// section-end detection. Returns the offset of the newline/`*` prefix
/// — NOT the label itself — so the prior block's content excludes any
/// leading markdown-bold or whitespace belonging to the next label.
fn find_label_at_line_start(text: &str, label: &str) -> Option<usize> {
    let label_lower = label.to_ascii_lowercase();
    let text_lower = text.to_ascii_lowercase();
    let mut search_from = 0usize;
    while let Some(pos) = text_lower[search_from..].find(&label_lower) {
        let abs = search_from + pos;
        // Walk backward to the preceding newline (or start); record the
        // earliest whitespace/`*` char in that run so the returned offset
        // excludes the next label's leading decoration.
        let mut i = abs;
        let bytes = text.as_bytes();
        let mut is_line_start = true;
        let mut boundary = abs;
        while i > 0 {
            i -= 1;
            let b = bytes[i];
            if b == b'\n' || b == b'\r' {
                // Include the newline in the prior block's end offset
                // (actually exclude it — point to the newline itself).
                boundary = i;
                break;
            }
            if !matches!(b, b' ' | b'\t' | b'*' | b'#') {
                is_line_start = false;
                break;
            }
            boundary = i;
        }
        if is_line_start {
            return Some(boundary);
        }
        search_from = abs + 1;
    }
    None
}

/// Parse the `Action Input:` body — try JSON first, fall back to a raw
/// string value. LLMs trained on ReAct sometimes emit `{"city": "Paris"}`
/// and sometimes just `Paris`.
///
/// Also strips surrounding triple-backticks (`\`\`\`json ... \`\`\``) that
/// instruction-tuned models love.
fn parse_action_input(raw: &str) -> Value {
    let cleaned = strip_code_fence(raw);
    let cleaned = cleaned.trim();
    if cleaned.is_empty() {
        return Value::String(String::new());
    }
    if let Ok(v) = serde_json::from_str::<Value>(cleaned) {
        return v;
    }
    Value::String(cleaned.to_string())
}

fn strip_code_fence(s: &str) -> String {
    let s = s.trim();
    if let Some(inner) = s.strip_prefix("```") {
        // Drop optional language tag on first line.
        let inner = match inner.find('\n') {
            Some(idx) => &inner[idx + 1..],
            None => inner,
        };
        if let Some(stripped) = inner.strip_suffix("```") {
            return stripped.trim().to_string();
        }
        // Maybe there's trailing ``` somewhere — find last occurrence.
        if let Some(idx) = inner.rfind("```") {
            return inner[..idx].trim().to_string();
        }
    }
    s.to_string()
}

fn snippet(s: &str) -> String {
    if s.len() > 100 {
        format!("{}...", &s[..100])
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn action_with_json_input() {
        let text = "Thought: I need to check the weather.\n\
                    Action: get_weather\n\
                    Action Input: {\"city\": \"Paris\"}";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action {
                thought,
                tool,
                input,
            } => {
                assert_eq!(thought.as_deref(), Some("I need to check the weather."));
                assert_eq!(tool, "get_weather");
                assert_eq!(input["city"], "Paris");
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn action_with_string_input_falls_back_to_string_value() {
        let text = "Action: search\nAction Input: Paris weather today";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action { input, .. } => {
                assert_eq!(input, Value::String("Paris weather today".to_string()));
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn action_input_strips_code_fence() {
        let text = "Action: get_weather\n\
                    Action Input: ```json\n\
                    {\"city\": \"Paris\"}\n\
                    ```";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action { input, .. } => {
                assert_eq!(input["city"], "Paris");
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn final_answer_returned_as_final_variant() {
        let text = "Thought: I have what I need.\n\
                    Final Answer: It's 15°C and raining in Paris.";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Final { thought, answer } => {
                assert_eq!(thought.as_deref(), Some("I have what I need."));
                assert_eq!(answer, "It's 15°C and raining in Paris.");
            }
            _ => panic!("expected Final"),
        }
    }

    #[test]
    fn final_answer_wins_over_action_when_both_present() {
        // LLM hedges; take the final answer as authoritative.
        let text = "Thought: maybe search.\n\
                    Action: search\n\
                    Action Input: foo\n\
                    Final Answer: Actually I already know — 42.";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Final { answer, .. } => {
                assert_eq!(answer, "Actually I already know — 42.");
            }
            _ => panic!("expected Final"),
        }
    }

    #[test]
    fn multiline_thought_preserved() {
        let text = "Thought: Let me think.\n\
                    First, check the weather.\n\
                    Then, decide what to wear.\n\
                    Action: get_weather\n\
                    Action Input: {}";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action { thought, .. } => {
                let t = thought.unwrap();
                assert!(t.contains("First, check"));
                assert!(t.contains("Then, decide"));
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn missing_thought_still_parses_action() {
        let text = "Action: search\nAction Input: foo";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action {
                thought,
                tool,
                input,
            } => {
                assert!(thought.is_none());
                assert_eq!(tool, "search");
                assert_eq!(input, Value::String("foo".to_string()));
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn case_insensitive_labels() {
        let text = "thought: hmm\naction: search\naction input: foo";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action { tool, .. } => assert_eq!(tool, "search"),
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn markdown_bold_labels_accepted() {
        // Some LLMs emit **Action:** instead of Action:
        let text = "**Thought:** check weather\n**Action:** get_weather\n**Action Input:** {\"city\": \"Paris\"}";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Action { tool, input, .. } => {
                assert_eq!(tool, "get_weather");
                assert_eq!(input["city"], "Paris");
            }
            _ => panic!("expected Action"),
        }
    }

    #[test]
    fn no_action_and_no_final_errors() {
        let text = "I'm thinking about this...";
        assert!(parse_react_step(text).is_err());
    }

    #[test]
    fn action_without_input_errors() {
        let text = "Thought: hmm\nAction: search\n(no input)";
        let err = parse_react_step(text).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("Action Input"));
    }

    #[test]
    fn empty_action_value_errors() {
        let text = "Action:\nAction Input: foo";
        assert!(parse_react_step(text).is_err());
    }

    #[test]
    fn final_answer_multiline_with_trailing_prose_preserved() {
        let text = "Thought: done.\n\
                    Final Answer: The answer is 42.\n\
                    This is my best guess based on the data.";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Final { answer, .. } => {
                assert!(answer.starts_with("The answer is 42."));
                assert!(answer.contains("best guess"));
            }
            _ => panic!("expected Final"),
        }
    }

    #[test]
    fn observation_terminates_section_when_present() {
        // A re-fed trace with Observation between Action and next Thought.
        let text = "Thought: first\n\
                    Action: search\n\
                    Action Input: foo\n\
                    Observation: results\n\
                    Thought: now I know\n\
                    Final Answer: yes";
        let step = parse_react_step(text).unwrap();
        match step {
            ReactStep::Final { thought, answer } => {
                // Latest Thought before Final Answer wins (the FIRST
                // Thought:-labeled block; we take the first occurrence).
                assert_eq!(thought.as_deref(), Some("first"));
                assert_eq!(answer, "yes");
            }
            _ => panic!("expected Final"),
        }
    }
}

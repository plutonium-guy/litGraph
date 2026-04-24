//! Text-mode ReAct agent for LLMs WITHOUT native tool-calling.
//!
//! # When to use this vs `react::ReactAgent`
//!
//! - `ReactAgent` (iter 13) — uses the provider's native tool-calling API
//!   (OpenAI `tools`, Anthropic `tool_use`, Bedrock Converse `toolUse`).
//!   Faster, more reliable, supports parallel tool calls. **Use this for
//!   GPT-4, Claude, Gemini, Mistral Large, Cohere R+, DeepSeek, etc.**
//!
//! - `TextReActAgent` (this file) — parses prose in the classic
//!   Thought/Action/Action Input/Final Answer format. **Use this for:**
//!   - Local models via Ollama / vLLM / llama.cpp without tool-call fine-tunes
//!   - Base-completion checkpoints (older text-davinci, text-bison-*)
//!   - Fine-tunes trained on the ReAct format (some Llama-2 derivatives)
//!   - When the provider's tool-call API is broken or rate-limited and
//!     you need a graceful fallback
//!
//! Loop shape:
//!
//! ```text
//!   user input + system-prompt-with-tool-catalog
//!         │
//!         ▼
//!   LLM → prose ("Thought: ... Action: foo\nAction Input: ...")
//!         │
//!         ▼
//!   parse_react_step(prose)  (iter 107)
//!         │
//!     ┌───┴──────────────────┐
//!     │                      │
//!   Final → return answer   Action { tool, input }
//!                              │
//!                              ▼
//!                           run tool  →  Observation
//!                              │
//!                              ▼
//!                           append "Observation: ..." to prompt
//!                              │
//!                              ▼
//!                           (loop back to LLM)
//! ```
//!
//! Serial-only by design: text-mode models emit ONE action per turn, and
//! the Observation must be fed back before the next step.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use async_stream::try_stream;
use futures::Stream;
use litgraph_core::react_format_instructions;
use litgraph_core::react_parser::{parse_react_step, ReactStep};
use litgraph_core::tool::Tool;
use litgraph_core::{ChatModel, ChatOptions, Message, Role};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;

/// One turn of a text-ReAct trace. Exposed for debugging / observability
/// so callers can inspect the loop without parsing the full transcript.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TextReactTurn {
    Action {
        raw_response: String,
        thought: Option<String>,
        tool: String,
        input: serde_json::Value,
        observation: String,
        is_error: bool,
    },
    Final {
        raw_response: String,
        thought: Option<String>,
        answer: String,
    },
    ParseError {
        raw_response: String,
        error: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextReactResult {
    pub final_answer: Option<String>,
    pub trace: Vec<TextReactTurn>,
    pub iterations: u32,
    pub stopped_reason: StoppedReason,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StoppedReason {
    FinalAnswer,
    MaxIterations,
    ParseError,
    ToolNotFound,
}

/// Per-turn event surfaced by `TextReActAgent::stream()`. Variants are
/// stable (tagged with `type` in JSON). Fires in the order the agent
/// processes each turn:
///
/// `IterationStart` → `LlmResponse` → either `ParseError` (terminal) or
/// `ParsedAction` + `ToolStart` + `ToolResult` → loop, or `ParsedFinal`
/// → `Final` (terminal). `MaxIterations` is the alt terminal.
///
/// Unlike `react::AgentEvent` (parallel tool calls), text-mode is serial
/// — exactly one ToolStart/ToolResult pair per Action turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TextReactEvent {
    /// New model turn beginning. 1-based.
    IterationStart { iteration: u32 },
    /// Raw assistant prose returned by the LLM, before parsing. Useful
    /// for UIs that show the model's reasoning verbatim.
    LlmResponse { iteration: u32, text: String },
    /// The response parsed as a tool-call step. Fires before `ToolStart`.
    ParsedAction {
        iteration: u32,
        thought: Option<String>,
        tool: String,
        input: Value,
    },
    /// The response parsed as a final-answer step. Terminal-adjacent
    /// (followed by `Final`).
    ParsedFinal {
        iteration: u32,
        thought: Option<String>,
        answer: String,
    },
    /// Parser couldn't extract Action/Action Input/Final Answer. Terminal.
    ParseError { iteration: u32, error: String, raw_response: String },
    /// Tool invocation kicked off.
    ToolStart {
        iteration: u32,
        tool: String,
        input: Value,
    },
    /// Tool invocation finished. `is_error=true` if the tool raised — the
    /// agent will continue and let the LLM react to the failure.
    ToolResult {
        iteration: u32,
        tool: String,
        observation: String,
        is_error: bool,
        duration_ms: u64,
    },
    /// Terminal — agent reached Final Answer.
    Final {
        answer: String,
        iterations: u32,
    },
    /// Terminal — `max_iterations` reached without Final Answer.
    MaxIterations { iterations: u32 },
    /// Terminal — `tool_not_found` (LLM named an unknown tool). Fires
    /// AFTER the corresponding `ParsedAction` so subscribers see the
    /// invalid call before the stop.
    ToolNotFound {
        iteration: u32,
        tool: String,
        available: Vec<String>,
    },
}

pub type TextReactEventStream =
    Pin<Box<dyn Stream<Item = litgraph_core::Result<TextReactEvent>> + Send>>;

#[derive(Clone)]
pub struct TextReactAgentConfig {
    pub max_iterations: u32,
    pub system_prompt: Option<String>,
    pub chat_options: ChatOptions,
    /// If false, the caller provides the full system prompt including
    /// tool catalog. If true (default), a default ReAct instruction is
    /// prepended derived from tool schemas.
    pub auto_format_instructions: bool,
}

impl Default for TextReactAgentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            system_prompt: None,
            chat_options: ChatOptions::default(),
            auto_format_instructions: true,
        }
    }
}

pub struct TextReActAgent {
    model: Arc<dyn ChatModel>,
    tools: HashMap<String, Arc<dyn Tool>>,
    config: TextReactAgentConfig,
}

impl TextReActAgent {
    pub fn new(
        model: Arc<dyn ChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        config: TextReactAgentConfig,
    ) -> Self {
        let tools_map: HashMap<String, Arc<dyn Tool>> =
            tools.iter().map(|t| (t.schema().name, t.clone())).collect();
        Self {
            model,
            tools: tools_map,
            config,
        }
    }

    /// Run the agent on a user input. Blocks until `Final Answer`, max
    /// iterations, or an unrecoverable parse error.
    pub async fn invoke(&self, user_input: impl Into<String>) -> litgraph_core::Result<TextReactResult> {
        let user_input = user_input.into();
        let mut trace: Vec<TextReactTurn> = Vec::new();
        let mut messages: Vec<Message> = Vec::new();

        // System prompt: caller-provided + auto-generated tool catalog.
        if let Some(sp) = self.config.system_prompt.as_ref() {
            messages.push(Message::system(sp.clone()));
        }
        if self.config.auto_format_instructions {
            let tool_lines: Vec<String> = self
                .tools
                .values()
                .map(|t| {
                    let schema = t.schema();
                    format!("{}: {}", schema.name, schema.description)
                })
                .collect();
            let tool_refs: Vec<&str> = tool_lines.iter().map(|s| s.as_str()).collect();
            let instr = react_format_instructions(&tool_refs);
            // If the caller also set a system_prompt, append ours as a
            // second system message (providers all accept multiple
            // system messages; some merge them server-side).
            messages.push(Message::system(instr));
        }
        messages.push(Message::user(user_input));

        let max_iters = self.config.max_iterations;
        for iter in 0..max_iters {
            debug!("text_react: iter={}", iter + 1);
            let resp = self
                .model
                .invoke(messages.clone(), &self.config.chat_options)
                .await?;
            let raw = resp.message.text_content();

            // Echo the assistant's raw prose back into the history so the
            // model can see its own prior steps in the next turn.
            messages.push(Message::assistant(raw.clone()));

            let step = match parse_react_step(&raw) {
                Ok(step) => step,
                Err(e) => {
                    trace.push(TextReactTurn::ParseError {
                        raw_response: raw,
                        error: e.to_string(),
                    });
                    return Ok(TextReactResult {
                        final_answer: None,
                        trace,
                        iterations: iter + 1,
                        stopped_reason: StoppedReason::ParseError,
                    });
                }
            };

            match step {
                ReactStep::Final { thought, answer } => {
                    trace.push(TextReactTurn::Final {
                        raw_response: raw,
                        thought,
                        answer: answer.clone(),
                    });
                    return Ok(TextReactResult {
                        final_answer: Some(answer),
                        trace,
                        iterations: iter + 1,
                        stopped_reason: StoppedReason::FinalAnswer,
                    });
                }
                ReactStep::Action {
                    thought,
                    tool,
                    input,
                } => {
                    let tool_ref = self.tools.get(&tool);
                    let (observation, is_error) = match tool_ref {
                        Some(t) => match t.run(input.clone()).await {
                            Ok(v) => {
                                let s = match &v {
                                    serde_json::Value::String(s) => s.clone(),
                                    _ => v.to_string(),
                                };
                                (s, false)
                            }
                            Err(e) => (format!("tool error: {e}"), true),
                        },
                        None => {
                            let err = format!(
                                "tool `{}` not found. Available: {}",
                                tool,
                                self.tools
                                    .keys()
                                    .cloned()
                                    .collect::<Vec<_>>()
                                    .join(", ")
                            );
                            trace.push(TextReactTurn::Action {
                                raw_response: raw,
                                thought,
                                tool,
                                input,
                                observation: err,
                                is_error: true,
                            });
                            return Ok(TextReactResult {
                                final_answer: None,
                                trace,
                                iterations: iter + 1,
                                stopped_reason: StoppedReason::ToolNotFound,
                            });
                        }
                    };

                    // Feed observation back to the model — as a USER
                    // message prefixed with "Observation:" (matches the
                    // grammar in the system prompt).
                    messages.push(Message::user(format!("Observation: {observation}")));

                    trace.push(TextReactTurn::Action {
                        raw_response: raw,
                        thought,
                        tool,
                        input,
                        observation,
                        is_error,
                    });
                }
            }
        }

        Ok(TextReactResult {
            final_answer: None,
            trace,
            iterations: max_iters,
            stopped_reason: StoppedReason::MaxIterations,
        })
    }

    /// Stream per-turn `TextReactEvent`s for progressive UIs.
    ///
    /// Same loop as `invoke()` but yields each milestone as it happens:
    /// `IterationStart` → `LlmResponse` → `ParsedAction` → `ToolStart` →
    /// `ToolResult` → ... → `ParsedFinal` → `Final`. Stops with one of
    /// `Final`, `MaxIterations`, `ParseError`, or `ToolNotFound`.
    ///
    /// Caller drains the stream to completion; the agent does NOT cancel
    /// on drop mid-turn (the tool invocation runs to completion). For
    /// cancellation, use `tokio::time::timeout` around the consumer.
    pub fn stream(self: Arc<Self>, user_input: impl Into<String>) -> TextReactEventStream {
        let user_input = user_input.into();
        let agent = self;
        Box::pin(try_stream! {
            let mut messages: Vec<Message> = Vec::new();
            if let Some(sp) = agent.config.system_prompt.as_ref() {
                messages.push(Message::system(sp.clone()));
            }
            if agent.config.auto_format_instructions {
                let tool_lines: Vec<String> = agent
                    .tools
                    .values()
                    .map(|t| {
                        let schema = t.schema();
                        format!("{}: {}", schema.name, schema.description)
                    })
                    .collect();
                let tool_refs: Vec<&str> = tool_lines.iter().map(|s| s.as_str()).collect();
                let instr = react_format_instructions(&tool_refs);
                messages.push(Message::system(instr));
            }
            messages.push(Message::user(user_input));

            let max_iters = agent.config.max_iterations;
            for iter_idx in 0..max_iters {
                let iteration = iter_idx + 1;
                yield TextReactEvent::IterationStart { iteration };

                let resp = agent
                    .model
                    .invoke(messages.clone(), &agent.config.chat_options)
                    .await?;
                let raw = resp.message.text_content();
                yield TextReactEvent::LlmResponse {
                    iteration,
                    text: raw.clone(),
                };
                messages.push(Message::assistant(raw.clone()));

                let step = match parse_react_step(&raw) {
                    Ok(s) => s,
                    Err(e) => {
                        yield TextReactEvent::ParseError {
                            iteration,
                            error: e.to_string(),
                            raw_response: raw,
                        };
                        return;
                    }
                };

                match step {
                    ReactStep::Final { thought, answer } => {
                        yield TextReactEvent::ParsedFinal {
                            iteration,
                            thought,
                            answer: answer.clone(),
                        };
                        yield TextReactEvent::Final {
                            answer,
                            iterations: iteration,
                        };
                        return;
                    }
                    ReactStep::Action { thought, tool, input } => {
                        yield TextReactEvent::ParsedAction {
                            iteration,
                            thought,
                            tool: tool.clone(),
                            input: input.clone(),
                        };
                        let tool_ref = agent.tools.get(&tool);
                        let Some(t) = tool_ref else {
                            yield TextReactEvent::ToolNotFound {
                                iteration,
                                tool: tool.clone(),
                                available: agent.tools.keys().cloned().collect(),
                            };
                            return;
                        };
                        yield TextReactEvent::ToolStart {
                            iteration,
                            tool: tool.clone(),
                            input: input.clone(),
                        };
                        let started = std::time::Instant::now();
                        let (observation, is_error) = match t.run(input).await {
                            Ok(v) => {
                                let s = match &v {
                                    Value::String(s) => s.clone(),
                                    _ => v.to_string(),
                                };
                                (s, false)
                            }
                            Err(e) => (format!("tool error: {e}"), true),
                        };
                        let duration_ms = started.elapsed().as_millis() as u64;
                        yield TextReactEvent::ToolResult {
                            iteration,
                            tool: tool.clone(),
                            observation: observation.clone(),
                            is_error,
                            duration_ms,
                        };
                        messages.push(Message::user(format!("Observation: {observation}")));
                    }
                }
            }
            yield TextReactEvent::MaxIterations { iterations: max_iters };
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::model::{ChatOptions, ChatResponse, ChatStream, FinishReason, TokenUsage};
    use litgraph_core::tool::ToolSchema;
    use std::sync::Mutex;

    /// A fake chat model that returns a canned sequence of responses.
    struct ScriptedChat {
        responses: Mutex<Vec<String>>,
        seen_messages: Mutex<Vec<Vec<Message>>>,
    }

    impl ScriptedChat {
        fn new(responses: Vec<&str>) -> Self {
            Self {
                responses: Mutex::new(responses.into_iter().map(str::to_string).rev().collect()),
                seen_messages: Mutex::new(Vec::new()),
            }
        }
    }

    #[async_trait]
    impl ChatModel for ScriptedChat {
        fn name(&self) -> &str {
            "scripted-chat"
        }

        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> litgraph_core::Result<ChatResponse> {
            self.seen_messages.lock().unwrap().push(messages);
            let content = self
                .responses
                .lock()
                .unwrap()
                .pop()
                .unwrap_or_else(|| "Final Answer: (no more scripted responses)".to_string());
            Ok(ChatResponse {
                message: Message::assistant(content),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "scripted-chat".to_string(),
            })
        }

        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> litgraph_core::Result<ChatStream> {
            unimplemented!("scripted chat doesn't stream")
        }
    }

    /// Tool that returns a canned string for any input.
    struct EchoTool {
        name: String,
        response: String,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.clone(),
                description: "echoes back a canned response".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }

        async fn run(
            &self,
            _args: serde_json::Value,
        ) -> litgraph_core::Result<serde_json::Value> {
            Ok(serde_json::Value::String(self.response.clone()))
        }
    }

    fn echo(name: &str, resp: &str) -> Arc<dyn Tool> {
        Arc::new(EchoTool {
            name: name.to_string(),
            response: resp.to_string(),
        })
    }

    #[tokio::test]
    async fn final_answer_on_first_turn_short_circuits() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Thought: I know this\nFinal Answer: 42",
        ]));
        let agent = TextReActAgent::new(chat, vec![], TextReactAgentConfig::default());
        let result = agent.invoke("what's the answer").await.unwrap();
        assert_eq!(result.final_answer.as_deref(), Some("42"));
        assert_eq!(result.stopped_reason, StoppedReason::FinalAnswer);
        assert_eq!(result.iterations, 1);
    }

    #[tokio::test]
    async fn action_then_final_answer_roundtrips() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Thought: I need weather\nAction: get_weather\nAction Input: {\"city\": \"Paris\"}",
            "Thought: got it\nFinal Answer: 15°C in Paris",
        ]));
        let tools = vec![echo("get_weather", "15°C and raining")];
        let agent = TextReActAgent::new(chat.clone(), tools, TextReactAgentConfig::default());
        let result = agent.invoke("weather in Paris?").await.unwrap();
        assert_eq!(result.final_answer.as_deref(), Some("15°C in Paris"));
        assert_eq!(result.iterations, 2);
        assert_eq!(result.trace.len(), 2);
        // Turn 1 was an Action; turn 2 was Final.
        match &result.trace[0] {
            TextReactTurn::Action { tool, observation, .. } => {
                assert_eq!(tool, "get_weather");
                assert_eq!(observation, "15°C and raining");
            }
            _ => panic!("expected action turn first"),
        }
    }

    #[tokio::test]
    async fn observation_fed_back_as_user_message() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: get_weather\nAction Input: {}",
            "Final Answer: done",
        ]));
        let tools = vec![echo("get_weather", "sunny")];
        let agent = TextReActAgent::new(chat.clone(), tools, TextReactAgentConfig::default());
        let _ = agent.invoke("q").await.unwrap();

        // The second invocation to the model should have seen an
        // "Observation: sunny" user message.
        let seen = chat.seen_messages.lock().unwrap();
        let second_turn = &seen[1];
        let has_obs = second_turn
            .iter()
            .any(|m| matches!(m.role, Role::User) && m.text_content().contains("Observation: sunny"));
        assert!(has_obs, "expected Observation in 2nd turn messages");
    }

    #[tokio::test]
    async fn unknown_tool_stops_with_tool_not_found() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: nope\nAction Input: {}",
        ]));
        let agent = TextReActAgent::new(chat, vec![], TextReactAgentConfig::default());
        let result = agent.invoke("q").await.unwrap();
        assert_eq!(result.stopped_reason, StoppedReason::ToolNotFound);
        assert!(result.final_answer.is_none());
    }

    #[tokio::test]
    async fn parse_error_stops_with_parse_error_reason() {
        let chat = Arc::new(ScriptedChat::new(vec!["just some prose no labels"]));
        let agent = TextReActAgent::new(chat, vec![], TextReactAgentConfig::default());
        let result = agent.invoke("q").await.unwrap();
        assert_eq!(result.stopped_reason, StoppedReason::ParseError);
    }

    #[tokio::test]
    async fn max_iterations_cap_stops_the_loop() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: spin\nAction Input: {}",
            "Action: spin\nAction Input: {}",
            "Action: spin\nAction Input: {}",
            "Action: spin\nAction Input: {}",
        ]));
        let cfg = TextReactAgentConfig {
            max_iterations: 3,
            ..Default::default()
        };
        let agent = TextReActAgent::new(chat, vec![echo("spin", "spun")], cfg);
        let result = agent.invoke("q").await.unwrap();
        assert_eq!(result.stopped_reason, StoppedReason::MaxIterations);
        assert_eq!(result.iterations, 3);
        assert!(result.final_answer.is_none());
    }

    #[tokio::test]
    async fn auto_format_instructions_injects_tool_catalog_into_system() {
        let chat = Arc::new(ScriptedChat::new(vec!["Final Answer: ok"]));
        let tools = vec![echo("get_weather", "sunny"), echo("web_search", "results")];
        let agent = TextReActAgent::new(chat.clone(), tools, TextReactAgentConfig::default());
        let _ = agent.invoke("q").await.unwrap();
        let seen = chat.seen_messages.lock().unwrap();
        let sys = seen[0]
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.text_content().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(sys.contains("get_weather"));
        assert!(sys.contains("web_search"));
        assert!(sys.contains("Action Input:"));
    }

    #[tokio::test]
    async fn disabling_auto_format_instructions_omits_the_catalog() {
        let chat = Arc::new(ScriptedChat::new(vec!["Final Answer: ok"]));
        let cfg = TextReactAgentConfig {
            auto_format_instructions: false,
            system_prompt: Some("minimal".to_string()),
            ..Default::default()
        };
        let agent = TextReActAgent::new(chat.clone(), vec![echo("foo", "bar")], cfg);
        let _ = agent.invoke("q").await.unwrap();
        let seen = chat.seen_messages.lock().unwrap();
        let sys = seen[0]
            .iter()
            .filter(|m| matches!(m.role, Role::System))
            .map(|m| m.text_content().to_string())
            .collect::<Vec<_>>()
            .join("\n");
        // "foo" tool listing should NOT have been injected.
        assert!(!sys.contains("Action Input:"));
        assert!(sys.contains("minimal"));
    }

    #[tokio::test]
    async fn tool_error_captured_as_observation_not_fatal() {
        struct FailingTool;
        #[async_trait]
        impl Tool for FailingTool {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "always_fails".to_string(),
                    description: "raises an error".to_string(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn run(
                &self,
                _args: serde_json::Value,
            ) -> litgraph_core::Result<serde_json::Value> {
                Err(litgraph_core::Error::parse("kaboom"))
            }
        }
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: always_fails\nAction Input: {}",
            "Final Answer: gave up",
        ]));
        let tools: Vec<Arc<dyn Tool>> = vec![Arc::new(FailingTool)];
        let agent = TextReActAgent::new(chat, tools, TextReactAgentConfig::default());
        let result = agent.invoke("q").await.unwrap();
        // Agent continued past the tool error (agent decided "give up").
        assert_eq!(result.stopped_reason, StoppedReason::FinalAnswer);
        match &result.trace[0] {
            TextReactTurn::Action { is_error, observation, .. } => {
                assert!(*is_error);
                assert!(observation.contains("kaboom"));
            }
            _ => panic!("expected action turn with error"),
        }
    }

    // -------- streaming tests ---------------------------------------------

    use futures::StreamExt;

    async fn collect_stream(agent: Arc<TextReActAgent>, input: &str) -> Vec<TextReactEvent> {
        let mut s = agent.stream(input.to_string());
        let mut out = Vec::new();
        while let Some(ev) = s.next().await {
            out.push(ev.unwrap());
        }
        out
    }

    #[tokio::test]
    async fn stream_emits_iteration_then_response_then_final() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Thought: I know\nFinal Answer: 42",
        ]));
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![],
            TextReactAgentConfig::default(),
        ));
        let events = collect_stream(agent, "q").await;
        // Expect: IterationStart → LlmResponse → ParsedFinal → Final.
        assert!(matches!(events[0], TextReactEvent::IterationStart { iteration: 1 }));
        assert!(matches!(events[1], TextReactEvent::LlmResponse { iteration: 1, .. }));
        assert!(matches!(events[2], TextReactEvent::ParsedFinal { iteration: 1, .. }));
        match &events[3] {
            TextReactEvent::Final { answer, iterations } => {
                assert_eq!(answer, "42");
                assert_eq!(*iterations, 1);
            }
            other => panic!("expected Final, got {:?}", other),
        }
        assert_eq!(events.len(), 4);
    }

    #[tokio::test]
    async fn stream_emits_action_then_tool_start_then_tool_result() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: spin\nAction Input: {\"x\": 1}",
            "Final Answer: ok",
        ]));
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![echo("spin", "spun")],
            TextReactAgentConfig::default(),
        ));
        let events = collect_stream(agent, "q").await;
        // Iter 1: IterationStart, LlmResponse, ParsedAction, ToolStart, ToolResult.
        // Iter 2: IterationStart, LlmResponse, ParsedFinal, Final.
        let kinds: Vec<&str> = events
            .iter()
            .map(|e| match e {
                TextReactEvent::IterationStart { .. } => "iteration_start",
                TextReactEvent::LlmResponse { .. } => "llm_response",
                TextReactEvent::ParsedAction { .. } => "parsed_action",
                TextReactEvent::ParsedFinal { .. } => "parsed_final",
                TextReactEvent::ParseError { .. } => "parse_error",
                TextReactEvent::ToolStart { .. } => "tool_start",
                TextReactEvent::ToolResult { .. } => "tool_result",
                TextReactEvent::Final { .. } => "final",
                TextReactEvent::MaxIterations { .. } => "max_iterations",
                TextReactEvent::ToolNotFound { .. } => "tool_not_found",
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                "iteration_start",
                "llm_response",
                "parsed_action",
                "tool_start",
                "tool_result",
                "iteration_start",
                "llm_response",
                "parsed_final",
                "final",
            ]
        );
        match &events[3] {
            TextReactEvent::ToolStart { tool, input, .. } => {
                assert_eq!(tool, "spin");
                assert_eq!(input["x"], 1);
            }
            _ => panic!("wrong event"),
        }
        match &events[4] {
            TextReactEvent::ToolResult { observation, is_error, .. } => {
                assert_eq!(observation, "spun");
                assert!(!is_error);
            }
            _ => panic!("wrong event"),
        }
    }

    #[tokio::test]
    async fn stream_terminates_on_parse_error_without_final_event() {
        let chat = Arc::new(ScriptedChat::new(vec!["just prose no labels"]));
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![],
            TextReactAgentConfig::default(),
        ));
        let events = collect_stream(agent, "q").await;
        let last = events.last().unwrap();
        assert!(matches!(last, TextReactEvent::ParseError { .. }));
        // No Final/MaxIterations event after a ParseError.
        assert!(!events.iter().any(|e| matches!(e, TextReactEvent::Final { .. })));
    }

    #[tokio::test]
    async fn stream_emits_tool_not_found_when_unknown_tool_named() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: nope\nAction Input: {}",
        ]));
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![echo("known", "k")],
            TextReactAgentConfig::default(),
        ));
        let events = collect_stream(agent, "q").await;
        let last = events.last().unwrap();
        match last {
            TextReactEvent::ToolNotFound { tool, available, .. } => {
                assert_eq!(tool, "nope");
                assert_eq!(available, &vec!["known".to_string()]);
            }
            other => panic!("expected ToolNotFound, got {:?}", other),
        }
        // ParsedAction was emitted BEFORE ToolNotFound (subscriber sees the
        // bad call before the stop).
        assert!(matches!(
            events[events.len() - 2],
            TextReactEvent::ParsedAction { .. }
        ));
    }

    #[tokio::test]
    async fn stream_emits_max_iterations_when_loop_exhausts() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: spin\nAction Input: {}",
            "Action: spin\nAction Input: {}",
            "Action: spin\nAction Input: {}",
        ]));
        let cfg = TextReactAgentConfig {
            max_iterations: 2,
            ..Default::default()
        };
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![echo("spin", "spun")],
            cfg,
        ));
        let events = collect_stream(agent, "q").await;
        let last = events.last().unwrap();
        assert!(matches!(last, TextReactEvent::MaxIterations { iterations: 2 }));
    }

    #[tokio::test]
    async fn stream_tool_result_carries_duration_ms() {
        let chat = Arc::new(ScriptedChat::new(vec![
            "Action: spin\nAction Input: {}",
            "Final Answer: ok",
        ]));
        let agent = Arc::new(TextReActAgent::new(
            chat,
            vec![echo("spin", "spun")],
            TextReactAgentConfig::default(),
        ));
        let events = collect_stream(agent, "q").await;
        let tr = events
            .iter()
            .find(|e| matches!(e, TextReactEvent::ToolResult { .. }))
            .unwrap();
        match tr {
            TextReactEvent::ToolResult { duration_ms, .. } => {
                // Just check the field is present (echo returns instantly so 0 is OK).
                let _ = *duration_ms;
            }
            _ => unreachable!(),
        }
    }
}

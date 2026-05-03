//! Tool-calling ReAct agent built as a StateGraph.
//!
//! Layout (two nodes, a conditional edge, back-edge):
//!
//! ```text
//!         ┌──────────┐
//!   ──▶ model ───▶ should_continue?
//!         │             │
//!         │ tool_calls? ├── yes ──▶ tools ──▶ model  (loop)
//!         │             │
//!         └── no ───────┴── END
//! ```
//!
//! - model: calls the LLM with current messages + the tool catalog
//! - tools: runs all emitted tool calls in parallel (`JoinSet`), appends results
//!   as `Role::Tool` messages
//! - should_continue: conditional edge — if last message has tool calls, loop; else END
//!
//! # Streaming events
//!
//! `ReactAgent::stream(input)` returns a `Stream<Item = Result<AgentEvent>>`
//! that emits per-turn events: `IterationStart`, `LlmMessage`,
//! `ToolCallStart` (one per parallel call, emitted eagerly), `ToolCallResult`
//! (emitted as each call completes, in completion order — NOT submission
//! order), and finally `Final` / `MaxIterationsReached`. This drives
//! progress UIs that show "the agent is now calling `web_search` ..." in
//! real time, without waiting for the full loop to finish.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use litgraph_core::model::{ChatStreamEvent, FinishReason};
use litgraph_core::tool::{Tool, ToolCall};
use litgraph_core::{ChatModel, ChatOptions, Message, Role};
use litgraph_graph::{CompiledGraph, END, NodeOutput, START, StateGraph};
use serde::{Deserialize, Serialize};
use tokio::task::JoinSet;
use tracing::{debug, warn};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AgentState {
    #[serde(default)]
    pub messages: Vec<Message>,
    #[serde(default)]
    pub iterations: u32,
}

#[derive(Clone)]
pub struct ReactAgentConfig {
    pub system_prompt: Option<String>,
    pub max_iterations: u32,
    pub chat_options: ChatOptions,
    pub max_parallel_tools: usize,
    /// Optional chain of [`crate::middleware::ToolMiddleware`] hooks
    /// run around every tool dispatch. Default empty (no middleware
    /// — same fast path as before this field was added). Pass to
    /// `ReactAgent::with_config(...)` to attach.
    pub tool_middleware: crate::middleware::ToolMiddlewareChain,
}

impl Default for ReactAgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: 10,
            chat_options: ChatOptions::default(),
            max_parallel_tools: 16,
            tool_middleware: crate::middleware::ToolMiddlewareChain::new(),
        }
    }
}

/// High-level event surfaced by `ReactAgent::stream()`. Variants are stable
/// (tagged with `type` in JSON) so downstream consumers — Python iterators,
/// WebSocket bridges, log aggregators — can match on them.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AgentEvent {
    /// A new model-turn is about to start. `iteration` is 1-based.
    IterationStart { iteration: u32 },
    /// Partial assistant text from the streaming model — emitted only by
    /// `stream_tokens()`, never by `stream()`. Accumulate client-side for
    /// a progressive chat-UI render; the final full text also arrives in
    /// the `LlmMessage` event at the end of the turn.
    TokenDelta { text: String },
    /// The LLM replied. May carry tool_calls (intermediate turn) or plain
    /// text content (final turn). Emitted BEFORE any ToolCallStart events
    /// so consumers see the reasoning → tool-call ordering.
    LlmMessage { message: Message },
    /// A tool invocation has been submitted (kicked off concurrently with
    /// its siblings). Emitted once per call, in the order the model listed
    /// them, immediately after the corresponding `LlmMessage`.
    ToolCallStart {
        call_id: String,
        name: String,
        arguments: serde_json::Value,
    },
    /// A tool invocation finished. Emitted in **completion order** (not
    /// submission order) so slow tools don't block fast tools from
    /// surfacing. `duration_ms` measures just the tool execution (not the
    /// semaphore wait).
    ToolCallResult {
        call_id: String,
        name: String,
        result: String,
        is_error: bool,
        duration_ms: u64,
    },
    /// Terminal event — the agent returned a final assistant message with
    /// no tool calls. Carries the full message history for downstream
    /// callers that don't want to buffer every `LlmMessage`.
    Final { messages: Vec<Message>, iterations: u32 },
    /// Terminal event — `max_iterations` reached before the agent returned
    /// a final message. The last message may still contain tool calls.
    MaxIterationsReached {
        messages: Vec<Message>,
        iterations: u32,
    },
}

pub type AgentEventStream =
    Pin<Box<dyn Stream<Item = litgraph_graph::Result<AgentEvent>> + Send>>;

pub struct ReactAgent {
    compiled: CompiledGraph<AgentState>,
    // Retained refs for the streaming path — duplicates what's captured in
    // the graph closures, but the graph closures aren't reachable from the
    // outside, so we keep these here.
    model: Arc<dyn ChatModel>,
    tools_by_name: HashMap<String, Arc<dyn Tool>>,
    opts: ChatOptions,
    system_prompt: Option<String>,
    max_iterations: u32,
    max_parallel_tools: usize,
    tool_middleware: crate::middleware::ToolMiddlewareChain,
}

impl ReactAgent {
    pub fn new(
        model: Arc<dyn ChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        cfg: ReactAgentConfig,
    ) -> litgraph_graph::Result<Self> {
        let tools_by_name: HashMap<String, Arc<dyn Tool>> =
            tools.iter().map(|t| (t.schema().name, t.clone())).collect();
        let schemas: Vec<_> = tools.iter().map(|t| t.schema()).collect();

        let mut opts = cfg.chat_options.clone();
        // Inject tool schemas into every model call.
        opts.tools = schemas;

        let model_for_node = model.clone();
        let tools_for_node = tools_by_name.clone();
        let system_prompt = cfg.system_prompt.clone();
        let max_iters = cfg.max_iterations;
        let sem_size = cfg.max_parallel_tools;
        let middleware_for_node = cfg.tool_middleware.clone();
        let middleware_retained = cfg.tool_middleware.clone();

        let mut g = StateGraph::<AgentState>::new();

        // ----- MODEL NODE --------------------------------------------------
        let opts_clone = opts.clone();
        let system_prompt_for_node = system_prompt.clone();
        g.add_fallible_node("model", move |mut state: AgentState| {
            let m = model_for_node.clone();
            let opts = opts_clone.clone();
            let system_prompt = system_prompt_for_node.clone();
            Box::pin(async move {
                if state.iterations >= max_iters {
                    debug!("agent hit max_iterations={}", max_iters);
                    return Ok(NodeOutput::empty().goto(END));
                }
                state.iterations += 1;

                // Prepend system prompt if configured AND not already present.
                let mut msgs = state.messages.clone();
                if let Some(sp) = system_prompt.as_ref() {
                    if !msgs.first().map(|m| matches!(m.role, Role::System)).unwrap_or(false) {
                        msgs.insert(0, Message::system(sp.clone()));
                    }
                }

                let resp = m.invoke(msgs, &opts).await
                    .map_err(|e| litgraph_graph::GraphError::Other(format!("model error: {e}")))?;
                let assistant = resp.message.clone();
                let has_calls = !assistant.tool_calls.is_empty();
                let finish_toolcalls = matches!(resp.finish_reason, FinishReason::ToolCalls);

                let mut out = NodeOutput::update(AgentStateDelta {
                    messages: vec![assistant],
                    iterations: None,
                });
                // The iterations update we did in-place above is lost because the reducer
                // merges deltas, not full state. Re-emit it explicitly:
                out.update["iterations"] = serde_json::json!(state.iterations);

                if has_calls || finish_toolcalls {
                    out = out.goto("tools");
                } else {
                    out = out.goto(END);
                }
                Ok(out)
            })
        });

        // ----- TOOLS NODE --------------------------------------------------
        g.add_fallible_node("tools", move |state: AgentState| {
            let tools_by_name = tools_for_node.clone();
            let middleware = middleware_for_node.clone();
            Box::pin(async move {
                let last = state.messages.last().cloned();
                let Some(last) = last else {
                    return Ok(NodeOutput::empty().goto(END));
                };
                let calls = last.tool_calls.clone();
                if calls.is_empty() {
                    return Ok(NodeOutput::empty().goto(END));
                }

                // Concurrent tool execution — classic Rust advantage over LangChain Python.
                let sem = Arc::new(tokio::sync::Semaphore::new(sem_size.max(1)));
                let mut set: JoinSet<Message> = JoinSet::new();
                for tc in calls {
                    let tools_by_name = tools_by_name.clone();
                    let sem = sem.clone();
                    let mw = middleware.clone();
                    set.spawn(async move {
                        let _permit = match sem.acquire_owned().await {
                            Ok(p) => p,
                            Err(_) => return tool_error_message(&tc, "semaphore closed"),
                        };
                        run_one_tool(&tools_by_name, &tc, &mw).await
                    });
                }

                let mut out_msgs: Vec<Message> = Vec::new();
                while let Some(joined) = set.join_next().await {
                    match joined {
                        Ok(m) => out_msgs.push(m),
                        Err(e) => {
                            warn!("tool join failed: {e}");
                        }
                    }
                }

                let delta = AgentStateDelta { messages: out_msgs, iterations: None };
                Ok(NodeOutput::update(delta).goto("model"))
            })
        });

        g.add_edge(START, "model");
        // Dynamic routing via NodeOutput::goto — no static edges needed beyond START.

        let compiled = g.with_max_parallel(1).compile()?; // supersteps are linear here
        Ok(Self {
            compiled,
            model,
            tools_by_name,
            opts,
            system_prompt,
            max_iterations: max_iters,
            max_parallel_tools: sem_size,
            tool_middleware: middleware_retained,
        })
    }

    pub async fn invoke(&self, user: impl Into<String>) -> litgraph_graph::Result<AgentState> {
        let state = AgentState {
            messages: vec![Message::user(user)],
            iterations: 0,
        };
        self.compiled.invoke(state, None).await
    }

    pub async fn invoke_messages(&self, msgs: Vec<Message>) -> litgraph_graph::Result<AgentState> {
        self.compiled.invoke(AgentState { messages: msgs, iterations: 0 }, None).await
    }

    pub fn compiled(&self) -> &CompiledGraph<AgentState> { &self.compiled }

    /// Stream per-turn events from an initial user message. See `AgentEvent`
    /// for the event shape. The stream terminates after `Final` or
    /// `MaxIterationsReached` (or surfaces an error and ends).
    pub fn stream(&self, user: impl Into<String>) -> AgentEventStream {
        self.stream_messages(vec![Message::user(user)])
    }

    /// Token-level variant of `stream()`. Calls `model.stream()` on every
    /// turn (instead of `model.invoke()`), forwarding `ChatStreamEvent::Delta`
    /// as `AgentEvent::TokenDelta { text }` so chat UIs can render the LLM's
    /// reply character-by-character as it's generated. All other events
    /// (`IterationStart`, `LlmMessage`, `ToolCallStart/Result`, `Final`/
    /// `MaxIterationsReached`) match `stream()`'s shape and timing.
    ///
    /// The `LlmMessage` event still carries the full assembled message
    /// (from the stream's `Done` event) for consumers that want both the
    /// streamed tokens AND the final object.
    pub fn stream_tokens(&self, user: impl Into<String>) -> AgentEventStream {
        self.stream_messages_tokens(vec![Message::user(user)])
    }

    /// Stream variant that seeds the agent with a pre-built message history
    /// (e.g. prior conversation turns). Same event surface as `stream()`.
    pub fn stream_messages(&self, initial: Vec<Message>) -> AgentEventStream {
        let model = self.model.clone();
        let tools_by_name = self.tools_by_name.clone();
        let opts = self.opts.clone();
        let system_prompt = self.system_prompt.clone();
        let max_iters = self.max_iterations;
        let sem_size = self.max_parallel_tools;
        let middleware = self.tool_middleware.clone();

        Box::pin(async_stream::try_stream! {
            let mut msgs = initial;
            let mut iterations: u32 = 0;
            let sem = Arc::new(tokio::sync::Semaphore::new(sem_size.max(1)));

            loop {
                if iterations >= max_iters {
                    yield AgentEvent::MaxIterationsReached {
                        messages: msgs.clone(),
                        iterations,
                    };
                    return;
                }
                iterations += 1;
                yield AgentEvent::IterationStart { iteration: iterations };

                // ----- model turn -----
                let mut api_msgs = msgs.clone();
                if let Some(sp) = system_prompt.as_ref() {
                    if !api_msgs
                        .first()
                        .map(|m| matches!(m.role, Role::System))
                        .unwrap_or(false)
                    {
                        api_msgs.insert(0, Message::system(sp.clone()));
                    }
                }
                let resp = model
                    .invoke(api_msgs, &opts)
                    .await
                    .map_err(|e| litgraph_graph::GraphError::Other(format!("model error: {e}")))?;
                let assistant = resp.message.clone();
                msgs.push(assistant.clone());
                yield AgentEvent::LlmMessage { message: assistant.clone() };

                let has_calls = !assistant.tool_calls.is_empty();
                let finish_toolcalls = matches!(resp.finish_reason, FinishReason::ToolCalls);
                if !(has_calls || finish_toolcalls) {
                    yield AgentEvent::Final { messages: msgs.clone(), iterations };
                    return;
                }

                // ----- tools turn: emit Starts up front, then Results in
                // completion order. ToolCallStart's order reflects the
                // LLM's tool_calls order (stable). Results arrive as they
                // finish — that's the "show me fast tools' results first"
                // behavior consumers want.
                let calls = assistant.tool_calls.clone();
                for tc in &calls {
                    yield AgentEvent::ToolCallStart {
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    };
                }

                type ToolResult = (ToolCall, std::result::Result<serde_json::Value, litgraph_core::Error>, Duration);
                let mut set: JoinSet<ToolResult> = JoinSet::new();
                for tc in calls {
                    let tools = tools_by_name.clone();
                    let sem = sem.clone();
                    let mw = middleware.clone();
                    set.spawn(async move {
                        let _permit = match sem.acquire_owned().await {
                            Ok(p) => p,
                            Err(_) => {
                                return (
                                    tc.clone(),
                                    Err(litgraph_core::Error::other("semaphore closed")),
                                    Duration::ZERO,
                                );
                            }
                        };
                        let start = Instant::now();
                        // before-hooks: may mutate args or short-circuit.
                        let args = match mw.dispatch_before(&tc.name, &tc.arguments) {
                            Ok(a) => a,
                            Err(e) => {
                                return (
                                    tc.clone(),
                                    Err(litgraph_core::Error::other(e.0)),
                                    start.elapsed(),
                                );
                            }
                        };
                        let raw = match tools.get(&tc.name) {
                            Some(tool) => tool.run(args.clone()).await,
                            None => Err(litgraph_core::Error::other(format!(
                                "unknown tool `{}`",
                                tc.name
                            ))),
                        };
                        // after-hooks: may rewrite result.
                        let result = match raw {
                            Ok(v) => mw
                                .dispatch_after(&tc.name, &args, &v)
                                .map_err(|e| litgraph_core::Error::other(e.0)),
                            Err(e) => Err(e),
                        };
                        (tc, result, start.elapsed())
                    });
                }

                while let Some(joined) = set.join_next().await {
                    let (tc, result, duration) = joined.map_err(|e| {
                        litgraph_graph::GraphError::Other(format!("tool join: {e}"))
                    })?;
                    let (text, is_error) = match result {
                        Ok(v) => (
                            match v {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            },
                            false,
                        ),
                        Err(e) => (format!("error: {e}"), true),
                    };
                    msgs.push(Message::tool_response(&tc.id, text.clone()));
                    yield AgentEvent::ToolCallResult {
                        call_id: tc.id,
                        name: tc.name,
                        result: text,
                        is_error,
                        duration_ms: duration.as_millis() as u64,
                    };
                }
            }
        })
    }

    /// Token-streaming variant of `stream_messages`. Same loop + same event
    /// shape, but uses `model.stream()` per turn and forwards each
    /// `ChatStreamEvent::Delta { text }` as `AgentEvent::TokenDelta`. The
    /// `Done` event of each turn carries the assembled `ChatResponse` —
    /// from there the loop is identical to `stream_messages`.
    pub fn stream_messages_tokens(&self, initial: Vec<Message>) -> AgentEventStream {
        let model = self.model.clone();
        let tools_by_name = self.tools_by_name.clone();
        let opts = self.opts.clone();
        let system_prompt = self.system_prompt.clone();
        let max_iters = self.max_iterations;
        let sem_size = self.max_parallel_tools;
        let middleware = self.tool_middleware.clone();

        Box::pin(async_stream::try_stream! {
            let mut msgs = initial;
            let mut iterations: u32 = 0;
            let sem = Arc::new(tokio::sync::Semaphore::new(sem_size.max(1)));

            loop {
                if iterations >= max_iters {
                    yield AgentEvent::MaxIterationsReached {
                        messages: msgs.clone(),
                        iterations,
                    };
                    return;
                }
                iterations += 1;
                yield AgentEvent::IterationStart { iteration: iterations };

                // ----- model turn (streaming) -----
                let mut api_msgs = msgs.clone();
                if let Some(sp) = system_prompt.as_ref() {
                    if !api_msgs
                        .first()
                        .map(|m| matches!(m.role, Role::System))
                        .unwrap_or(false)
                    {
                        api_msgs.insert(0, Message::system(sp.clone()));
                    }
                }
                let mut chat_stream = model
                    .stream(api_msgs, &opts)
                    .await
                    .map_err(|e| litgraph_graph::GraphError::Other(format!("model stream: {e}")))?;

                let mut final_response: Option<litgraph_core::ChatResponse> = None;
                while let Some(item) = chat_stream.next().await {
                    let ev = item
                        .map_err(|e| litgraph_graph::GraphError::Other(format!("stream chunk: {e}")))?;
                    match ev {
                        ChatStreamEvent::Delta { text } => {
                            if !text.is_empty() {
                                yield AgentEvent::TokenDelta { text };
                            }
                        }
                        ChatStreamEvent::ToolCallDelta { .. } => {
                            // Provider already aggregates this into the final
                            // ChatResponse on `Done` — we don't surface it as
                            // a partial token (no human-readable use yet).
                        }
                        ChatStreamEvent::Done { response } => {
                            final_response = Some(response);
                            break;
                        }
                    }
                }
                let resp = final_response.ok_or_else(|| {
                    litgraph_graph::GraphError::Other(
                        "model stream ended without Done event".to_string(),
                    )
                })?;
                let assistant = resp.message.clone();
                msgs.push(assistant.clone());
                yield AgentEvent::LlmMessage { message: assistant.clone() };

                let has_calls = !assistant.tool_calls.is_empty();
                let finish_toolcalls = matches!(resp.finish_reason, FinishReason::ToolCalls);
                if !(has_calls || finish_toolcalls) {
                    yield AgentEvent::Final { messages: msgs.clone(), iterations };
                    return;
                }

                // ----- tools turn — identical to non-token stream path -----
                let calls = assistant.tool_calls.clone();
                for tc in &calls {
                    yield AgentEvent::ToolCallStart {
                        call_id: tc.id.clone(),
                        name: tc.name.clone(),
                        arguments: tc.arguments.clone(),
                    };
                }

                type ToolResult = (
                    ToolCall,
                    std::result::Result<serde_json::Value, litgraph_core::Error>,
                    Duration,
                );
                let mut set: tokio::task::JoinSet<ToolResult> = tokio::task::JoinSet::new();
                for tc in calls {
                    let tools = tools_by_name.clone();
                    let sem = sem.clone();
                    let mw = middleware.clone();
                    set.spawn(async move {
                        let _permit = match sem.acquire_owned().await {
                            Ok(p) => p,
                            Err(_) => {
                                return (
                                    tc.clone(),
                                    Err(litgraph_core::Error::other("semaphore closed")),
                                    Duration::ZERO,
                                );
                            }
                        };
                        let start = Instant::now();
                        let args = match mw.dispatch_before(&tc.name, &tc.arguments) {
                            Ok(a) => a,
                            Err(e) => {
                                return (
                                    tc.clone(),
                                    Err(litgraph_core::Error::other(e.0)),
                                    start.elapsed(),
                                );
                            }
                        };
                        let raw = match tools.get(&tc.name) {
                            Some(tool) => tool.run(args.clone()).await,
                            None => Err(litgraph_core::Error::other(format!(
                                "unknown tool `{}`",
                                tc.name
                            ))),
                        };
                        let result = match raw {
                            Ok(v) => mw
                                .dispatch_after(&tc.name, &args, &v)
                                .map_err(|e| litgraph_core::Error::other(e.0)),
                            Err(e) => Err(e),
                        };
                        (tc, result, start.elapsed())
                    });
                }

                while let Some(joined) = set.join_next().await {
                    let (tc, result, duration) = joined.map_err(|e| {
                        litgraph_graph::GraphError::Other(format!("tool join: {e}"))
                    })?;
                    let (text, is_error) = match result {
                        Ok(v) => (
                            match v {
                                serde_json::Value::String(s) => s,
                                other => other.to_string(),
                            },
                            false,
                        ),
                        Err(e) => (format!("error: {e}"), true),
                    };
                    msgs.push(Message::tool_response(&tc.id, text.clone()));
                    yield AgentEvent::ToolCallResult {
                        call_id: tc.id,
                        name: tc.name,
                        result: text,
                        is_error,
                        duration_ms: duration.as_millis() as u64,
                    };
                }
            }
        })
    }
}

async fn run_one_tool(
    tools: &HashMap<String, Arc<dyn Tool>>,
    tc: &ToolCall,
    middleware: &crate::middleware::ToolMiddlewareChain,
) -> Message {
    let Some(tool) = tools.get(&tc.name) else {
        return tool_error_message(tc, &format!("unknown tool `{}`", tc.name));
    };
    // before-hooks: may mutate args or short-circuit.
    let args = match middleware.dispatch_before(&tc.name, &tc.arguments) {
        Ok(a) => a,
        Err(e) => return tool_error_message(tc, &e.0),
    };
    let result = match tool.run(args.clone()).await {
        Ok(v) => v,
        Err(e) => return tool_error_message(tc, &e.to_string()),
    };
    // after-hooks: may rewrite result (PII scrub, redaction, …).
    let final_value = match middleware.dispatch_after(&tc.name, &args, &result) {
        Ok(v) => v,
        Err(e) => return tool_error_message(tc, &e.0),
    };
    let text = match final_value {
        serde_json::Value::String(s) => s,
        other => other.to_string(),
    };
    Message::tool_response(&tc.id, text)
}

fn tool_error_message(tc: &ToolCall, err: &str) -> Message {
    Message::tool_response(&tc.id, format!("error: {err}"))
}

/// Serde helper for partial-update deltas — the agent reducer merges `messages`
/// as an append and `iterations` as replace.
#[derive(Serialize)]
struct AgentStateDelta {
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    iterations: Option<u32>,
}

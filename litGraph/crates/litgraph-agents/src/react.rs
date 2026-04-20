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

use std::collections::HashMap;
use std::sync::Arc;

use litgraph_core::model::FinishReason;
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
}

impl Default for ReactAgentConfig {
    fn default() -> Self {
        Self {
            system_prompt: None,
            max_iterations: 10,
            chat_options: ChatOptions::default(),
            max_parallel_tools: 16,
        }
    }
}

pub struct ReactAgent {
    compiled: CompiledGraph<AgentState>,
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
        let system_prompt = cfg.system_prompt.clone();
        let max_iters = cfg.max_iterations;
        let sem_size = cfg.max_parallel_tools;

        let mut g = StateGraph::<AgentState>::new();

        // ----- MODEL NODE --------------------------------------------------
        let opts_clone = opts.clone();
        g.add_fallible_node("model", move |mut state: AgentState| {
            let m = model_for_node.clone();
            let opts = opts_clone.clone();
            let system_prompt = system_prompt.clone();
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
            let tools_by_name = tools_by_name.clone();
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
                    set.spawn(async move {
                        let _permit = match sem.acquire_owned().await {
                            Ok(p) => p,
                            Err(_) => return tool_error_message(&tc, "semaphore closed"),
                        };
                        run_one_tool(&tools_by_name, &tc).await
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
        Ok(Self { compiled })
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
}

async fn run_one_tool(tools: &HashMap<String, Arc<dyn Tool>>, tc: &ToolCall) -> Message {
    let Some(tool) = tools.get(&tc.name) else {
        return tool_error_message(tc, &format!("unknown tool `{}`", tc.name));
    };
    match tool.run(tc.arguments.clone()).await {
        Ok(v) => {
            let text = match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            };
            Message::tool_response(&tc.id, text)
        }
        Err(e) => tool_error_message(tc, &e.to_string()),
    }
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

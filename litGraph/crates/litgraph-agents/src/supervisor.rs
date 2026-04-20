//! Supervisor multi-agent pattern.
//!
//! A supervisor LLM sees the conversation + the catalog of available worker
//! agents, and picks one to delegate to. The chosen worker runs its own
//! `ReactAgent` (tools and all), returns its final assistant message, and the
//! supervisor either hands off again or decides the task is done.
//!
//! This is a pragmatic topology — one LLM call per hop, one `ReactAgent` per
//! worker. No magic. Users who need fancier orchestration write a StateGraph
//! directly.

use std::collections::HashMap;
use std::sync::Arc;

use litgraph_core::tool::{FnTool, Tool};
use litgraph_core::{ChatModel, Message, Result as LgResult};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::react::{AgentState, ReactAgent, ReactAgentConfig};

#[derive(Debug, Clone)]
pub struct SupervisorConfig {
    pub system_prompt: Option<String>,
    pub max_hops: u32,
}

impl Default for SupervisorConfig {
    fn default() -> Self {
        Self {
            system_prompt: Some(
                "You are a supervisor coordinating a team of specialists. Read the \
                 conversation; call `handoff` with the worker name most qualified to \
                 answer next. Call `finish` with the final answer when the user's \
                 question is fully resolved."
                    .into(),
            ),
            max_hops: 6,
        }
    }
}

pub struct SupervisorAgent {
    supervisor: ReactAgent,
    workers: HashMap<String, Arc<ReactAgent>>,
}

#[derive(Deserialize)]
struct HandoffArgs {
    worker: String,
    message: String,
}

#[derive(Serialize)]
struct HandoffOut {
    worker: String,
    reply: String,
}

#[derive(Deserialize)]
struct FinishArgs {
    answer: String,
}

#[derive(Serialize)]
struct FinishOut {
    answer: String,
}

impl SupervisorAgent {
    /// `workers` is a name→ReactAgent map. The supervisor LLM picks by name.
    pub fn new(
        supervisor_model: Arc<dyn ChatModel>,
        workers: HashMap<String, Arc<ReactAgent>>,
        cfg: SupervisorConfig,
    ) -> litgraph_graph::Result<Self> {
        let worker_list: Vec<String> = workers.keys().cloned().collect();
        let workers_arc = Arc::new(workers);

        // handoff tool: supervisor → worker
        let workers_for_handoff = workers_arc.clone();
        let handoff_tool: Arc<dyn Tool> = Arc::new(FnTool::new(
            "handoff",
            format!(
                "Delegate to a worker. Known workers: {}. Provide `worker` (name) and \
                 `message` (the instruction to forward).",
                worker_list.join(", ")
            ),
            json!({
                "type": "object",
                "properties": {
                    "worker":  { "type": "string" },
                    "message": { "type": "string" }
                },
                "required": ["worker", "message"]
            }),
            move |args: HandoffArgs| {
                let workers = workers_for_handoff.clone();
                Box::pin(async move {
                    let Some(w) = workers.get(&args.worker) else {
                        return Err(litgraph_core::Error::invalid(format!(
                            "unknown worker `{}`", args.worker
                        )));
                    };
                    let state: AgentState = w.invoke(args.message.clone()).await
                        .map_err(|e| litgraph_core::Error::other(format!("worker `{}`: {e}", args.worker)))?;
                    let reply = state
                        .messages
                        .iter()
                        .rev()
                        .find(|m| matches!(m.role, litgraph_core::Role::Assistant))
                        .map(|m| m.text_content())
                        .unwrap_or_default();
                    Ok(HandoffOut { worker: args.worker, reply })
                })
            },
        ));

        let finish_tool: Arc<dyn Tool> = Arc::new(FnTool::new(
            "finish",
            "Emit the final answer when the user's question is fully resolved. The \
             agent loop stops after this tool is called.",
            json!({
                "type": "object",
                "properties": { "answer": { "type": "string" } },
                "required": ["answer"]
            }),
            |args: FinishArgs| Box::pin(async move { Ok(FinishOut { answer: args.answer }) }),
        ));

        let supervisor = ReactAgent::new(
            supervisor_model,
            vec![handoff_tool, finish_tool],
            ReactAgentConfig {
                system_prompt: cfg.system_prompt,
                max_iterations: cfg.max_hops,
                ..Default::default()
            },
        )?;

        Ok(Self { supervisor, workers: (*workers_arc).clone() })
    }

    pub async fn invoke(&self, user: impl Into<String>) -> litgraph_graph::Result<AgentState> {
        self.supervisor.invoke(user).await
    }

    pub async fn invoke_messages(&self, msgs: Vec<Message>) -> litgraph_graph::Result<AgentState> {
        self.supervisor.invoke_messages(msgs).await
    }

    /// Internal accessor for tests / introspection.
    pub fn worker_names(&self) -> Vec<String> {
        self.workers.keys().cloned().collect()
    }
}

// Internal wrapper for the returned Result type.
type _LgResult<T> = LgResult<T>;

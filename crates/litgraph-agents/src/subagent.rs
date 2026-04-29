//! `Subagent` — a ReactAgent wrapped as a `Tool` so a parent agent can spawn
//! it dynamically. Mirrors the `task` tool from LangChain `deepagents`: each
//! invocation runs the inner agent in an isolated context (no shared message
//! history with the parent), returning a single string answer plus
//! bookkeeping.
//!
//! Two scoping wins:
//! * Context quarantine — the parent's tool-call loop never sees the
//!   subagent's intermediate steps; only the final answer comes back, which
//!   keeps the parent's prompt budget under control.
//! * Tool isolation — the subagent has its own tool list, so the parent can
//!   delegate noisy / dangerous work (e.g. shell exec) to a constrained child
//!   without exposing those tools to itself.
//!
//! Concurrency: the subagent's `invoke` is async, and the parent's
//! `JoinSet` already runs tools in parallel — calling `Subagent` once per
//! parent tool-call slot fan-outs naturally.

use std::sync::Arc;

use async_trait::async_trait;
use serde_json::{json, Value};

use litgraph_core::message::Role;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{ChatModel, Error, Result};

use crate::react::{ReactAgent, ReactAgentConfig};

/// Subagent tool. Holds an inner `ReactAgent` and exposes it through the
/// `Tool` trait. Each `run()` call spawns a fresh agent invocation; state
/// from previous calls does not leak.
pub struct SubagentTool {
    name: String,
    description: String,
    inner: Arc<ReactAgent>,
}

impl SubagentTool {
    /// Build a subagent. `name` shows up in the parent's tool schema, so use
    /// something that telegraphs the subagent's specialty (e.g.
    /// `"research_agent"` or `"shell_executor"`). `description` is what the
    /// parent LLM reads to decide *when* to delegate.
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        model: Arc<dyn ChatModel>,
        tools: Vec<Arc<dyn Tool>>,
        cfg: ReactAgentConfig,
    ) -> Result<Self> {
        let agent = ReactAgent::new(model, tools, cfg).map_err(|e| {
            Error::invalid(format!("subagent: failed to build inner ReactAgent: {e}"))
        })?;
        Ok(Self {
            name: name.into(),
            description: description.into(),
            inner: Arc::new(agent),
        })
    }

    /// Wrap an already-built `ReactAgent`. Convenient when the agent was
    /// built with custom resilience wrappers around the inner model.
    pub fn from_agent(
        name: impl Into<String>,
        description: impl Into<String>,
        agent: Arc<ReactAgent>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            inner: agent,
        }
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: self.name.clone(),
            description: format!(
                "{}\n\nDelegate a self-contained task to a subagent that will run its own \
                 tool-calling loop and return a single answer string. The parent does NOT see \
                 the subagent's intermediate steps. Use this to keep the parent's context \
                 small or to scope sub-tasks to a constrained tool set.",
                self.description
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Self-contained task description for the subagent. \
                                        State all required context — the subagent has none of \
                                        the parent's prior messages."
                    }
                },
                "required": ["task"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let task = args
            .get("task")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("subagent: missing `task`"))?;
        if task.trim().is_empty() {
            return Err(Error::invalid("subagent: `task` is empty"));
        }
        let final_state = self
            .inner
            .invoke(task.to_string())
            .await
            .map_err(|e| Error::invalid(format!("subagent: inner invocation failed: {e}")))?;

        // Final assistant message — the last `Role::Assistant` with no
        // pending tool_calls. Fall back to "no answer" if the loop bailed
        // out via max_iterations with a tool-call message at the tail.
        let answer = final_state
            .messages
            .iter()
            .rev()
            .find(|m| matches!(m.role, Role::Assistant) && m.tool_calls.is_empty())
            .map(|m| m.text_content())
            .unwrap_or_default();

        Ok(json!({
            "answer": answer,
            "iterations": final_state.iterations,
            "message_count": final_state.messages.len(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use litgraph_core::message::Message;
    use litgraph_core::model::{
        ChatOptions, ChatResponse, ChatStream, FinishReason, TokenUsage,
    };
    use std::sync::Mutex;

    /// Trivial chat model that always returns the user's last message as the
    /// assistant reply. Enough to drive a one-shot ReactAgent loop.
    struct EchoModel {
        calls: Arc<Mutex<u32>>,
    }

    #[async_trait]
    impl ChatModel for EchoModel {
        fn name(&self) -> &str {
            "echo"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> Result<ChatResponse> {
            *self.calls.lock().unwrap() += 1;
            let last_user = messages
                .iter()
                .rev()
                .find(|m| matches!(m.role, Role::User))
                .map(|m| m.text_content())
                .unwrap_or_default();
            Ok(ChatResponse {
                message: Message::assistant(format!("echo: {last_user}")),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "echo".into(),
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

    fn build(name: &str) -> SubagentTool {
        let model: Arc<dyn ChatModel> = Arc::new(EchoModel {
            calls: Arc::new(Mutex::new(0)),
        });
        SubagentTool::new(
            name,
            "Test subagent.",
            model,
            Vec::new(),
            ReactAgentConfig {
                max_iterations: 2,
                ..Default::default()
            },
        )
        .unwrap()
    }

    #[tokio::test]
    async fn schema_uses_provided_name_and_extends_description() {
        let s = build("research_agent");
        let sch = s.schema();
        assert_eq!(sch.name, "research_agent");
        assert!(sch.description.contains("Test subagent."));
        assert!(sch.description.contains("subagent"));
        assert_eq!(sch.parameters["required"], json!(["task"]));
    }

    #[tokio::test]
    async fn run_returns_answer_iterations_and_count() {
        let s = build("a");
        let out = s.run(json!({"task": "hello"})).await.unwrap();
        assert_eq!(out["answer"], "echo: hello");
        // System prompt absent + 1 user + 1 assistant final = 2; iterations 1.
        assert_eq!(out["iterations"], 1);
        assert_eq!(out["message_count"], 2);
    }

    #[tokio::test]
    async fn run_rejects_missing_task() {
        let s = build("a");
        let err = s.run(json!({})).await.unwrap_err();
        assert!(format!("{err}").contains("task"));
    }

    #[tokio::test]
    async fn run_rejects_empty_task() {
        let s = build("a");
        let err = s.run(json!({"task": "  "})).await.unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[tokio::test]
    async fn each_invocation_is_isolated() {
        // Two calls; second must not see the first's messages.
        let s = build("a");
        let r1 = s.run(json!({"task": "first"})).await.unwrap();
        let r2 = s.run(json!({"task": "second"})).await.unwrap();
        assert_eq!(r1["message_count"], 2);
        assert_eq!(r2["message_count"], 2);
        assert_eq!(r1["answer"], "echo: first");
        assert_eq!(r2["answer"], "echo: second");
    }

    #[tokio::test]
    async fn from_agent_wraps_existing_react_agent() {
        let model: Arc<dyn ChatModel> = Arc::new(EchoModel {
            calls: Arc::new(Mutex::new(0)),
        });
        let agent = Arc::new(
            ReactAgent::new(model, Vec::new(), ReactAgentConfig::default()).unwrap(),
        );
        let s = SubagentTool::from_agent("wrapped", "wrapped agent", agent);
        let out = s.run(json!({"task": "ping"})).await.unwrap();
        assert_eq!(out["answer"], "echo: ping");
    }
}

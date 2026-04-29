//! Supervisor orchestrates 2 worker ReactAgents through scripted models.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use litgraph_agents::{ReactAgent, ReactAgentConfig, SupervisorAgent, SupervisorConfig};
use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
use litgraph_core::tool::ToolCall;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Message, Result, Role,
};
use serde_json::json;

struct ScriptedModel {
    name: String,
    script: Mutex<Vec<ChatResponse>>,
}

impl ScriptedModel {
    fn new(name: &str, script: Vec<ChatResponse>) -> Self {
        Self { name: name.into(), script: Mutex::new(script) }
    }
}

#[async_trait]
impl ChatModel for ScriptedModel {
    fn name(&self) -> &str { &self.name }
    async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
        let mut s = self.script.lock().unwrap();
        Ok(s.remove(0))
    }
    async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
        unimplemented!()
    }
}

fn tool_call(id: &str, name: &str, args: serde_json::Value) -> ChatResponse {
    ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: vec![ContentPart::Text { text: String::new() }],
            tool_calls: vec![ToolCall { id: id.into(), name: name.into(), arguments: args }],
            tool_call_id: None, name: None, cache: false,
        },
        finish_reason: FinishReason::ToolCalls,
        usage: TokenUsage::default(),
        model: "s".into(),
    }
}

fn plain(text: &str) -> ChatResponse {
    ChatResponse {
        message: Message::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: TokenUsage::default(),
        model: "s".into(),
    }
}

#[tokio::test]
async fn supervisor_routes_to_worker_then_finishes() {
    // Math worker: produces "42" on any input.
    let math_model = Arc::new(ScriptedModel::new("math", vec![plain("42")]));
    let math_worker = Arc::new(
        ReactAgent::new(math_model, vec![], ReactAgentConfig::default()).unwrap(),
    );

    // Chitchat worker: unused in this test.
    let chit_model = Arc::new(ScriptedModel::new("chit", vec![plain("hello")]));
    let chit_worker = Arc::new(
        ReactAgent::new(chit_model, vec![], ReactAgentConfig::default()).unwrap(),
    );

    let mut workers = HashMap::new();
    workers.insert("math".to_string(), math_worker);
    workers.insert("chit".to_string(), chit_worker);

    // Supervisor plan:
    //   1) handoff to math ("what is 6*7?")
    //   2) finish with the math answer
    let supervisor_model = Arc::new(ScriptedModel::new("sup", vec![
        tool_call("h1", "handoff", json!({"worker": "math", "message": "what is 6*7?"})),
        tool_call("f1", "finish", json!({"answer": "The answer is 42."})),
        plain("done"),
    ]));

    let sup = SupervisorAgent::new(supervisor_model, workers, SupervisorConfig::default()).unwrap();
    let state = sup.invoke("compute 6*7 and tell me").await.unwrap();

    // At least the user message + two assistant turns + two tool responses should be present.
    assert!(state.messages.len() >= 5);
    let worker_names = sup.worker_names();
    assert!(worker_names.contains(&"math".to_string()));
    assert!(worker_names.contains(&"chit".to_string()));

    // The finish tool's serialized output should appear as a Tool message.
    let tool_msgs: Vec<_> = state
        .messages
        .iter()
        .filter(|m| matches!(m.role, Role::Tool))
        .collect();
    assert!(tool_msgs.iter().any(|m| m.text_content().contains("42")),
        "expected a tool response carrying the math answer");
}

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use litgraph_agents::{ReactAgent, ReactAgentConfig};
use litgraph_core::model::{ChatStream, ChatStreamEvent, FinishReason};
use litgraph_core::tool::{FnTool, Tool, ToolCall};
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ContentPart, Message, Role, TokenUsage, Result,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

struct ScriptedModel {
    /// Each call returns the next response in order.
    script: Mutex<Vec<ChatResponse>>,
    call_count: std::sync::atomic::AtomicU32,
}

impl ScriptedModel {
    fn new(script: Vec<ChatResponse>) -> Self {
        Self {
            script: Mutex::new(script),
            call_count: std::sync::atomic::AtomicU32::new(0),
        }
    }
    fn calls(&self) -> u32 { self.call_count.load(std::sync::atomic::Ordering::SeqCst) }
}

#[async_trait]
impl ChatModel for ScriptedModel {
    fn name(&self) -> &str { "scripted" }
    async fn invoke(&self, _messages: Vec<Message>, _opts: &ChatOptions) -> Result<ChatResponse> {
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let mut s = self.script.lock().unwrap();
        Ok(s.remove(0))
    }
    async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
        unimplemented!("not used here")
    }
}

fn with_tool_call(id: &str, name: &str, args: serde_json::Value) -> ChatResponse {
    ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: vec![ContentPart::Text { text: String::new() }],
            tool_calls: vec![ToolCall { id: id.into(), name: name.into(), arguments: args }],
            tool_call_id: None,
            name: None,
            cache: false,
        },
        finish_reason: FinishReason::ToolCalls,
        usage: TokenUsage::default(),
        model: "scripted".into(),
    }
}

fn plain_text(text: &str) -> ChatResponse {
    ChatResponse {
        message: Message::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: TokenUsage::default(),
        model: "scripted".into(),
    }
}

fn add_tool() -> Arc<dyn Tool> {
    #[derive(Deserialize)]
    struct Args { a: i64, b: i64 }
    #[derive(Serialize)]
    struct Out { sum: i64 }
    Arc::new(FnTool::new(
        "add",
        "Add two integers.",
        json!({
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"}
            },
            "required": ["a", "b"]
        }),
        |args: Args| Box::pin(async move { Ok(Out { sum: args.a + args.b }) }),
    )) as Arc<dyn Tool>
}

#[tokio::test]
async fn agent_runs_tool_then_responds() {
    let model = Arc::new(ScriptedModel::new(vec![
        with_tool_call("c1", "add", json!({"a": 2, "b": 3})),
        plain_text("Result: 5"),
    ]));

    let agent = ReactAgent::new(model.clone(), vec![add_tool()], ReactAgentConfig::default())
        .unwrap();
    let state = agent.invoke("what is 2+3").await.unwrap();

    assert_eq!(model.calls(), 2);
    let last = state.messages.last().unwrap();
    assert_eq!(last.role, Role::Assistant);
    assert_eq!(last.text_content(), "Result: 5");

    // Check tool message exists with sum=5 content.
    let tool_msg = state.messages.iter().find(|m| matches!(m.role, Role::Tool)).unwrap();
    assert!(tool_msg.text_content().contains("\"sum\":5"));
}

#[tokio::test]
async fn agent_stops_at_max_iterations() {
    // Model always returns a tool call → infinite loop unless max_iterations caps it.
    let mut script = Vec::new();
    for i in 0..5 {
        script.push(with_tool_call(&format!("c{i}"), "add", json!({"a": 1, "b": 1})));
    }
    let model = Arc::new(ScriptedModel::new(script));

    let cfg = ReactAgentConfig { max_iterations: 3, ..Default::default() };
    let agent = ReactAgent::new(model.clone(), vec![add_tool()], cfg).unwrap();
    let _state = agent.invoke("loop please").await.unwrap();
    assert!(model.calls() <= 4);
}

#[tokio::test]
async fn agent_runs_parallel_tool_calls() {
    // One assistant turn emits THREE tool calls — they should all run concurrently.
    let mut script = Vec::new();
    script.push(ChatResponse {
        message: Message {
            role: Role::Assistant,
            content: vec![],
            tool_calls: vec![
                ToolCall { id: "a".into(), name: "add".into(), arguments: json!({"a":1,"b":1}) },
                ToolCall { id: "b".into(), name: "add".into(), arguments: json!({"a":2,"b":2}) },
                ToolCall { id: "c".into(), name: "add".into(), arguments: json!({"a":3,"b":3}) },
            ],
            tool_call_id: None,
            name: None,
            cache: false,
        },
        finish_reason: FinishReason::ToolCalls,
        usage: TokenUsage::default(),
        model: "scripted".into(),
    });
    script.push(plain_text("done"));

    let model = Arc::new(ScriptedModel::new(script));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let state = agent.invoke("triple sum").await.unwrap();

    let tool_responses: Vec<_> = state.messages.iter().filter(|m| matches!(m.role, Role::Tool)).collect();
    assert_eq!(tool_responses.len(), 3);
}

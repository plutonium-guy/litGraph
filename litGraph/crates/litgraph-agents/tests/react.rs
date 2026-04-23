use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use litgraph_agents::{AgentEvent, ReactAgent, ReactAgentConfig};
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
        // Real stream impl for token-streaming tests: pop the next scripted
        // ChatResponse, emit one Delta per character of the assistant text,
        // then a Done with the full response.
        self.call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let resp = { let mut s = self.script.lock().unwrap(); s.remove(0) };
        let text = resp.message.text_content();
        let chunks: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        let resp_clone = resp.clone();
        let s = async_stream::try_stream! {
            for c in chunks {
                yield ChatStreamEvent::Delta { text: c };
            }
            yield ChatStreamEvent::Done { response: resp_clone };
        };
        Ok(Box::pin(s) as ChatStream)
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

// ---------- streaming (iter 76) ----------

async fn collect_events(
    mut stream: litgraph_agents::AgentEventStream,
) -> Vec<AgentEvent> {
    let mut out = Vec::new();
    while let Some(ev) = stream.next().await {
        out.push(ev.unwrap());
    }
    out
}

#[tokio::test]
async fn stream_emits_iteration_llm_tool_final_in_order() {
    let model = Arc::new(ScriptedModel::new(vec![
        with_tool_call("c1", "add", json!({"a": 4, "b": 5})),
        plain_text("Answer: 9"),
    ]));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let events = collect_events(agent.stream("what is 4+5")).await;

    // Expected shape — exact:
    //   IterationStart(1), LlmMessage(tool_call), ToolCallStart, ToolCallResult,
    //   IterationStart(2), LlmMessage("Answer: 9"), Final
    assert!(
        matches!(events[0], AgentEvent::IterationStart { iteration: 1 }),
        "first event must be IterationStart(1), got {:?}", events[0]
    );
    assert!(matches!(events[1], AgentEvent::LlmMessage { .. }));
    match &events[2] {
        AgentEvent::ToolCallStart { call_id, name, .. } => {
            assert_eq!(call_id, "c1");
            assert_eq!(name, "add");
        }
        other => panic!("expected ToolCallStart, got {other:?}"),
    }
    match &events[3] {
        AgentEvent::ToolCallResult { call_id, name, result, is_error, .. } => {
            assert_eq!(call_id, "c1");
            assert_eq!(name, "add");
            assert!(result.contains("\"sum\":9"));
            assert!(!is_error);
        }
        other => panic!("expected ToolCallResult, got {other:?}"),
    }
    assert!(matches!(events[4], AgentEvent::IterationStart { iteration: 2 }));
    assert!(matches!(events[5], AgentEvent::LlmMessage { .. }));
    match events.last().unwrap() {
        AgentEvent::Final { iterations, messages } => {
            assert_eq!(*iterations, 2);
            assert_eq!(messages.last().unwrap().text_content(), "Answer: 9");
        }
        other => panic!("expected Final, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_parallel_tools_emit_all_starts_then_all_results() {
    // 3 parallel tool calls. All ToolCallStart must precede any ToolCallResult
    // (we emit Starts up front before spawning the JoinSet).
    let model = Arc::new(ScriptedModel::new(vec![
        ChatResponse {
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
        },
        plain_text("done"),
    ]));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let events = collect_events(agent.stream("triple")).await;

    let mut starts_seen = 0;
    let mut results_seen = 0;
    let mut result_before_all_starts = false;
    for ev in &events {
        match ev {
            AgentEvent::ToolCallStart { .. } => {
                if results_seen > 0 {
                    // Result came before a Start — violates the invariant.
                    panic!("ToolCallStart emitted AFTER a ToolCallResult — ordering wrong");
                }
                starts_seen += 1;
            }
            AgentEvent::ToolCallResult { .. } => {
                if starts_seen < 3 {
                    result_before_all_starts = true;
                }
                results_seen += 1;
            }
            _ => {}
        }
    }
    assert_eq!(starts_seen, 3);
    assert_eq!(results_seen, 3);
    assert!(
        !result_before_all_starts,
        "all ToolCallStarts must be emitted before any ToolCallResult"
    );
}

#[tokio::test]
async fn stream_tool_error_surfaces_as_is_error_event() {
    // Scripting a tool_call for an UNKNOWN tool → the stream must emit
    // ToolCallResult { is_error: true } rather than bailing out.
    let model = Arc::new(ScriptedModel::new(vec![
        with_tool_call("x1", "nope", json!({})),
        plain_text("recovered"),
    ]));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let events = collect_events(agent.stream("use nope")).await;

    let err_ev = events.iter().find_map(|e| match e {
        AgentEvent::ToolCallResult { is_error: true, result, name, .. } => Some((name.clone(), result.clone())),
        _ => None,
    }).expect("expected at least one is_error=true result");
    assert_eq!(err_ev.0, "nope");
    assert!(err_ev.1.contains("unknown tool"), "got: {}", err_ev.1);
}

#[tokio::test]
async fn stream_max_iterations_terminates_with_dedicated_event() {
    // Model always returns tool_calls → the agent caps at max_iterations and
    // we expect `MaxIterationsReached`, NOT `Final`.
    let mut script = Vec::new();
    for i in 0..5 {
        script.push(with_tool_call(&format!("c{i}"), "add", json!({"a":1,"b":1})));
    }
    let model = Arc::new(ScriptedModel::new(script));
    let cfg = ReactAgentConfig { max_iterations: 2, ..Default::default() };
    let agent = ReactAgent::new(model, vec![add_tool()], cfg).unwrap();
    let events = collect_events(agent.stream("loop")).await;

    let last = events.last().unwrap();
    match last {
        AgentEvent::MaxIterationsReached { iterations, .. } => {
            assert_eq!(*iterations, 2);
        }
        other => panic!("expected MaxIterationsReached, got {other:?}"),
    }
    // No Final event present.
    assert!(!events.iter().any(|e| matches!(e, AgentEvent::Final { .. })));
}

// ---------- Token streaming (iter 81) ----------

#[tokio::test]
async fn stream_tokens_emits_one_token_delta_per_character_then_llm_message() {
    // Final answer "Hi!" should emit 3 TokenDelta events ("H", "i", "!") then
    // an LlmMessage with the full text, then Final.
    let model = Arc::new(ScriptedModel::new(vec![plain_text("Hi!")]));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let events = collect_events(agent.stream_tokens("hi")).await;

    let token_texts: Vec<String> = events.iter().filter_map(|e| match e {
        AgentEvent::TokenDelta { text } => Some(text.clone()),
        _ => None,
    }).collect();
    assert_eq!(token_texts, vec!["H", "i", "!"]);

    // Concatenated tokens equal the LlmMessage text.
    let llm_msg_text = events.iter().find_map(|e| match e {
        AgentEvent::LlmMessage { message } => Some(message.text_content()),
        _ => None,
    }).expect("expected LlmMessage event");
    assert_eq!(token_texts.concat(), llm_msg_text);
    assert_eq!(llm_msg_text, "Hi!");

    // Stream terminates with Final.
    assert!(matches!(events.last().unwrap(), AgentEvent::Final { .. }));
}

#[tokio::test]
async fn stream_tokens_drives_full_react_loop_with_streaming_on_every_turn() {
    // Tool-call turn (empty text → 0 token deltas) → tools run → final-answer
    // turn streams "Result: 9". Verify the event sequence carries TokenDeltas
    // ONLY for the final turn, not the tool-call turn.
    let model = Arc::new(ScriptedModel::new(vec![
        with_tool_call("c1", "add", json!({"a": 4, "b": 5})),
        plain_text("Result: 9"),
    ]));
    let agent = ReactAgent::new(model, vec![add_tool()], ReactAgentConfig::default()).unwrap();
    let events = collect_events(agent.stream_tokens("what is 4+5")).await;

    // Tool-call turn has empty text content → 0 TokenDelta in iteration 1.
    // Final turn has "Result: 9" → 9 TokenDeltas in iteration 2.
    let mut iter_token_counts: Vec<(u32, usize)> = Vec::new();
    let mut current_iter: Option<u32> = None;
    let mut current_count = 0usize;
    for ev in &events {
        match ev {
            AgentEvent::IterationStart { iteration } => {
                if let Some(it) = current_iter.take() {
                    iter_token_counts.push((it, current_count));
                }
                current_iter = Some(*iteration);
                current_count = 0;
            }
            AgentEvent::TokenDelta { .. } => current_count += 1,
            _ => {}
        }
    }
    if let Some(it) = current_iter {
        iter_token_counts.push((it, current_count));
    }
    assert_eq!(iter_token_counts, vec![(1, 0), (2, "Result: 9".len())]);

    // Tool ran successfully.
    let tool_result = events.iter().find_map(|e| match e {
        AgentEvent::ToolCallResult { result, is_error: false, .. } => Some(result.clone()),
        _ => None,
    }).expect("expected successful tool result");
    assert!(tool_result.contains("\"sum\":9"));
}

#[tokio::test]
async fn stream_tokens_serializes_tagged_json() {
    let ev = AgentEvent::TokenDelta { text: "abc".into() };
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["type"], "token_delta");
    assert_eq!(v["text"], "abc");
}

#[tokio::test]
async fn stream_serializes_to_tagged_json() {
    // The Python binding relies on #[serde(tag = "type")] so downstream
    // consumers can match on a `type` field. Lock the wire format.
    let ev = AgentEvent::IterationStart { iteration: 1 };
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["type"], "iteration_start");
    assert_eq!(v["iteration"], 1);

    let ev = AgentEvent::ToolCallResult {
        call_id: "x".into(), name: "add".into(),
        result: "5".into(), is_error: false, duration_ms: 17,
    };
    let v = serde_json::to_value(&ev).unwrap();
    assert_eq!(v["type"], "tool_call_result");
    assert_eq!(v["duration_ms"], 17);
}

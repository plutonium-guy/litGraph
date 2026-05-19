//! Integration tests for `MistralRsChat` driven by `MockModelBackend`.

use std::sync::Arc;

use futures::StreamExt;
use litgraph_core::model::ChatStreamEvent;
use litgraph_core::{ChatModel, ChatOptions, FinishReason, Message};
use litgraph_providers_mistralrs::{
    GenOptions, MistralRsChat, MockModelBackend, flatten_messages,
};

fn chat(canned: &str) -> MistralRsChat {
    MistralRsChat::with_backend(Arc::new(MockModelBackend::new(canned)))
}

#[tokio::test]
async fn invoke_returns_canned_response_as_assistant_message() {
    let m = chat("the answer is 42");
    let resp = m
        .invoke(vec![Message::user("what?")], &ChatOptions::default())
        .await
        .unwrap();
    assert_eq!(resp.message.text_content(), "the answer is 42");
    assert!(matches!(resp.finish_reason, FinishReason::Stop));
    assert_eq!(resp.model, "mistralrs-mock");
    assert!(resp.usage.total >= resp.usage.completion);
}

#[tokio::test]
async fn invoke_reports_length_finish_when_max_tokens_hits_completion() {
    let m = chat("one two three four five");
    let opts = ChatOptions {
        max_tokens: Some(5),
        ..Default::default()
    };
    let resp = m.invoke(vec![Message::user("count")], &opts).await.unwrap();
    // MockModelBackend reports completion_tokens = word_count(canned) = 5.
    // With max_tokens=5, the adapter must label this as `Length`, not `Stop`.
    assert!(matches!(resp.finish_reason, FinishReason::Length));
}

#[tokio::test]
async fn invoke_honours_stop_strings_and_reports_stop_finish() {
    let m = chat("first STOP second");
    let opts = ChatOptions {
        stop: Some(vec!["STOP".into()]),
        ..Default::default()
    };
    let resp = m.invoke(vec![Message::user("hi")], &opts).await.unwrap();
    assert_eq!(resp.message.text_content(), "first ");
    assert!(matches!(resp.finish_reason, FinishReason::Stop));
}

#[tokio::test]
async fn stream_emits_delta_then_done() {
    let m = chat("streamed body");
    let mut s = m
        .stream(vec![Message::user("hi")], &ChatOptions::default())
        .await
        .unwrap();
    let mut got_delta = None;
    let mut got_done = None;
    while let Some(ev) = s.next().await {
        match ev.unwrap() {
            ChatStreamEvent::Delta { text } => got_delta = Some(text),
            ChatStreamEvent::Done { response } => got_done = Some(response),
            other => panic!("unexpected event: {other:?}"),
        }
    }
    assert_eq!(got_delta.as_deref(), Some("streamed body"));
    let done = got_done.unwrap();
    assert_eq!(done.message.text_content(), "streamed body");
}

#[tokio::test]
async fn name_passthrough_uses_backend_identifier() {
    let backend = MockModelBackend::new("x").with_identifier("dev-llama-2");
    let m = MistralRsChat::with_backend(Arc::new(backend));
    assert_eq!(m.name(), "dev-llama-2");
}

#[test]
fn flatten_messages_emits_role_tagged_transcript_with_trailing_open() {
    // Locks the transcript format. Future model-specific templates
    // (Llama 3, Mistral instruct) wrap on top of this baseline.
    let msgs = vec![
        Message::system("you are a duck"),
        Message::user("hi"),
        Message::assistant("hello"),
    ];
    let flat = flatten_messages(&msgs);
    assert!(flat.contains("<system>\nyou are a duck\n</system>"));
    assert!(flat.contains("<user>\nhi\n</user>"));
    assert!(flat.contains("<assistant>\nhello\n</assistant>"));
    assert!(flat.ends_with("<assistant>\n"), "must open response slot");
}

#[test]
fn gen_options_projection_picks_up_temperature_and_max_tokens() {
    let opts = ChatOptions {
        temperature: Some(0.7),
        max_tokens: Some(128),
        stop: Some(vec!["END".into()]),
        ..Default::default()
    };
    let g = GenOptions::from_chat_options(&opts);
    assert_eq!(g.temperature, Some(0.7));
    assert_eq!(g.max_tokens, Some(128));
    assert_eq!(g.stop, vec!["END".to_string()]);
}

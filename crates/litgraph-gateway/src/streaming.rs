//! OpenAI-compatible SSE relay for streaming completions.
//!
//! Failover ends when the upstream stream is established. A later failure
//! is emitted in-band because the HTTP status and any earlier tokens have
//! already reached the client.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use litgraph_core::{ChatStreamEvent, FinishReason, TokenUsage};
use serde_json::json;

pub type UsageMeter = Arc<dyn Fn(TokenUsage) + Send + Sync>;

pub fn sse_relay(
    upstream: litgraph_core::ChatStream,
    alias: String,
    completion_id: String,
    deployment_id: String,
    meter: UsageMeter,
) -> Response {
    let stream = async_stream::stream! {
        let mut upstream = upstream;
        let mut usage = TokenUsage::default();
        let mut relayed_completion_chars = 0usize;
        let mut terminal = false;
        let created = now_secs();

        while let Some(item) = upstream.next().await {
            match item {
                Ok(ChatStreamEvent::Delta { text }) => {
                    relayed_completion_chars += text.len();
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(
                        json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text},
                                "finish_reason": null,
                            }],
                        })
                        .to_string(),
                    ));
                }
                Ok(ChatStreamEvent::ToolCallDelta {
                    index,
                    id,
                    name,
                    arguments_delta,
                }) => {
                    yield Ok(Event::default().data(
                        json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [{
                                "index": 0,
                                "delta": {"tool_calls": [{
                                    "index": index,
                                    "id": id,
                                    "type": "function",
                                    "function": {
                                        "name": name,
                                        "arguments": arguments_delta,
                                    },
                                }]},
                                "finish_reason": null,
                            }],
                        })
                        .to_string(),
                    ));
                }
                Ok(ChatStreamEvent::Done { response }) => {
                    usage = response.usage;
                    yield Ok(Event::default().data(
                        json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": finish_str(response.finish_reason),
                            }],
                        })
                        .to_string(),
                    ));
                    yield Ok(Event::default().data(
                        json!({
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": alias,
                            "choices": [],
                            "usage": {
                                "prompt_tokens": usage.prompt,
                                "completion_tokens": usage.completion,
                                "total_tokens": usage.total,
                            },
                        })
                        .to_string(),
                    ));
                    terminal = true;
                    break;
                }
                Err(error) => {
                    tracing::warn!(deployment = %deployment_id, %error, "stream failed mid-flight");
                    yield Ok(Event::default().data(
                        json!({"error": {
                            "message": "The upstream stream ended unexpectedly.",
                            "type": "server_error",
                            "code": "upstream_stream_error",
                        }})
                        .to_string(),
                    ));
                    terminal = true;
                    break;
                }
            }
        }

        if usage.total == 0 && relayed_completion_chars > 0 {
            usage.completion = relayed_completion_chars.div_ceil(4).max(1) as u32;
            usage.total = usage.completion;
        }
        meter(usage);

        if !terminal {
            tracing::warn!(deployment = %deployment_id, "stream ended without a terminal event");
            yield Ok(Event::default().data(
                json!({"error": {
                    "message": "The upstream stream ended unexpectedly.",
                    "type": "server_error",
                    "code": "upstream_stream_incomplete",
                }})
                .to_string(),
            ));
        }
        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream).into_response()
}

fn finish_str(reason: FinishReason) -> &'static str {
    match reason {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::ContentFilter => "content_filter",
        FinishReason::Other => "stop",
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use litgraph_core::{ChatResponse, Error, Message};
    use std::sync::atomic::{AtomicU32, Ordering};

    async fn relay_for_test(items: Vec<Result<ChatStreamEvent, Error>>) -> (String, u32) {
        let upstream = Box::pin(futures::stream::iter(items));
        let metered = Arc::new(AtomicU32::new(0));
        let observed = metered.clone();
        let response = sse_relay(
            upstream,
            "ollama".into(),
            "chatcmpl-test".into(),
            "local".into(),
            Arc::new(move |usage| {
                observed.store(usage.total, Ordering::SeqCst);
            }),
        );
        let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        (
            String::from_utf8(body.to_vec()).unwrap(),
            metered.load(Ordering::SeqCst),
        )
    }

    fn done(prompt: u32, completion: u32) -> ChatStreamEvent {
        ChatStreamEvent::Done {
            response: ChatResponse {
                message: Message::assistant("Hello"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage {
                    prompt,
                    completion,
                    total: prompt + completion,
                    ..Default::default()
                },
                model: "upstream-model-name".into(),
            },
        }
    }

    #[tokio::test]
    async fn relay_emits_deltas_then_done_and_extracts_usage() {
        let (body, metered) = relay_for_test(vec![
            Ok(ChatStreamEvent::Delta { text: "Hel".into() }),
            Ok(ChatStreamEvent::Delta { text: "lo".into() }),
            Ok(done(10, 5)),
        ])
        .await;
        assert!(body.contains("\"content\":\"Hel\""));
        assert!(body.contains("\"content\":\"lo\""));
        assert!(body.contains("data: [DONE]"));
        assert_eq!(metered, 15);
    }

    #[tokio::test]
    async fn stream_that_dies_early_still_meters_relayed_tokens() {
        let (body, metered) = relay_for_test(vec![
            Ok(ChatStreamEvent::Delta {
                text: "partial".into(),
            }),
            Err(Error::provider("connection reset")),
        ])
        .await;
        assert!(body.contains("partial"));
        assert!(body.contains("\"error\""));
        assert!(metered > 0);
    }

    #[tokio::test]
    async fn chunks_echo_the_client_alias_not_the_upstream_model() {
        let (body, _) = relay_for_test(vec![
            Ok(ChatStreamEvent::Delta { text: "x".into() }),
            Ok(done(1, 1)),
        ])
        .await;
        assert!(body.contains("\"model\":\"ollama\""));
        assert!(!body.contains("upstream-model-name"));
    }
}

//! Drop-in HTTP server for litGraph. Wrap any [`ChatModel`] in a REST +
//! SSE endpoint with a single call:
//!
//! ```no_run
//! # use std::sync::Arc;
//! # use litgraph_core::ChatModel;
//! # async fn run(model: Arc<dyn ChatModel>) -> std::io::Result<()> {
//! litgraph_serve::serve_chat(model, "0.0.0.0:8080").await
//! # }
//! ```
//!
//! # Endpoints
//!
//! | Method | Path | Body | Returns |
//! |--------|------|------|---------|
//! | `POST` | `/invoke` | `{ messages: [...], options?: {...} }` | `ChatResponse` JSON |
//! | `POST` | `/stream` | same as `/invoke` | `text/event-stream` with `ChatStreamEvent` per `data:` line |
//! | `POST` | `/batch`  | `{ inputs: [[...], [...]], options?: {...} }` | array of `ChatResponse` |
//! | `GET`  | `/health` | — | `{ status: "ok" }` |
//! | `GET`  | `/info` | — | `{ name, endpoints: [...] }` |
//!
//! # Why axum
//!
//! - First-class `tokio` integration — same runtime as the rest of
//!   litGraph; no extra threadpool.
//! - Zero-config TLS off, JSON in/out via Serde.
//! - Tower middleware compatible — callers can stack `tower-http` for
//!   CORS, compression, tracing without us hard-coding it. Keeps this
//!   crate's dep tree minimal (no `tower-http` pulled in by default).
//!
//! # Auth
//!
//! Out of scope for v1. The recommended pattern is to wrap the
//! returned [`Router`] (via [`router_for`]) with a tower layer that
//! checks `Authorization: Bearer …`. We don't bake one in because
//! every deployment has its own auth model (JWT, mTLS, IP allow-list,
//! …) and forcing one is the LangServe complaint we're trying to
//! avoid.
//!
//! # Streaming format
//!
//! Server-Sent Events. Each [`ChatStreamEvent`] is JSON-encoded and
//! sent as a `data:` line. Stream ends with a `data: [DONE]` sentinel
//! after the final `Done` event so curl-style clients can break the
//! loop without keeping a JSON parser open.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    body::Body,
    extract::State,
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures::StreamExt;
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Message};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tracing::{debug, error};

#[cfg(feature = "studio")]
pub mod studio;

pub mod resume;

/// Wraps the model + any per-instance config the handlers need. Cheap
/// to clone (Arc inside).
#[derive(Clone)]
struct AppState {
    model: Arc<dyn ChatModel>,
}

#[derive(Debug, Deserialize)]
struct InvokeRequest {
    messages: Vec<Message>,
    #[serde(default)]
    options: ChatOptions,
}

#[derive(Debug, Deserialize)]
struct BatchRequest {
    inputs: Vec<Vec<Message>>,
    #[serde(default)]
    options: ChatOptions,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: String,
}

/// Build the [`axum::Router`] for a [`ChatModel`] without binding a
/// listener. Use this when you want to mount our routes alongside your
/// own application's, or wrap them in tower middleware (CORS, auth,
/// rate limiting).
pub fn router_for(model: Arc<dyn ChatModel>) -> Router {
    let state = AppState { model };
    Router::new()
        .route("/health", get(health))
        .route("/info", get(info))
        .route("/invoke", post(invoke))
        .route("/stream", post(stream))
        .route("/batch", post(batch))
        .with_state(state)
}

/// Serve `model` on `addr` until the process is killed. Convenience for
/// the common case; for graceful shutdown / signal handling use
/// [`router_for`] + `axum::serve` directly.
pub async fn serve_chat(model: Arc<dyn ChatModel>, addr: &str) -> std::io::Result<()> {
    let parsed: SocketAddr = addr
        .parse()
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, format!("bad addr {addr}: {e}")))?;
    let listener = tokio::net::TcpListener::bind(parsed).await?;
    debug!("litgraph-serve listening on {parsed} (model={})", model.name());
    axum::serve(listener, router_for(model)).await?;
    Ok(())
}

// ---- handlers --------------------------------------------------------------

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn info(State(s): State<AppState>) -> Json<Value> {
    Json(json!({
        "name": s.model.name(),
        "endpoints": ["/health", "/info", "/invoke", "/stream", "/batch"],
        "schema": {
            "invoke": {
                "request": { "messages": "Vec<Message>", "options": "ChatOptions?" },
                "response": "ChatResponse"
            },
            "stream": {
                "request": "same as /invoke",
                "response": "text/event-stream of ChatStreamEvent + [DONE] sentinel"
            },
            "batch": {
                "request": { "inputs": "Vec<Vec<Message>>", "options": "ChatOptions?" },
                "response": "Vec<ChatResponse>"
            }
        }
    }))
}

async fn invoke(
    State(s): State<AppState>,
    Json(req): Json<InvokeRequest>,
) -> Result<Json<ChatResponse>, ApiError> {
    let resp = s.model.invoke(req.messages, &req.options).await?;
    Ok(Json(resp))
}

async fn batch(
    State(s): State<AppState>,
    Json(req): Json<BatchRequest>,
) -> Result<Json<Vec<ChatResponse>>, ApiError> {
    let out = s.model.batch(req.inputs, &req.options).await?;
    Ok(Json(out))
}

async fn stream(
    State(s): State<AppState>,
    Json(req): Json<InvokeRequest>,
) -> Result<Response, ApiError> {
    let inner_stream = s.model.stream(req.messages, &req.options).await?;
    let sse_stream = inner_stream.map(|item| {
        let line = match item {
            Ok(ev) => match serde_json::to_string(&ev) {
                Ok(s) => format!("data: {s}\n\n"),
                Err(e) => format!("event: error\ndata: {}\n\n", json!({"error": e.to_string()})),
            },
            Err(e) => format!("event: error\ndata: {}\n\n", json!({"error": e.to_string()})),
        };
        Ok::<_, std::io::Error>(line)
    });
    // Append the [DONE] sentinel so curl-style clients can terminate.
    let done = futures::stream::once(async { Ok::<_, std::io::Error>("data: [DONE]\n\n".to_string()) });
    let combined = sse_stream.chain(done);
    let body = Body::from_stream(combined);

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        // Disable proxy buffering so deltas reach the client promptly
        // even behind nginx.
        .header("X-Accel-Buffering", "no")
        .body(body)
        .map_err(|e| ApiError::Internal(format!("response build: {e}")))
}

// ---- error type ------------------------------------------------------------

/// Internal error wrapper that converts any litgraph_core::Error into a
/// 500 response with a JSON body. Validation errors (bad JSON in the
/// request) are handled by axum directly and return 400.
enum ApiError {
    /// Anything that came out of the model.
    Model(litgraph_core::Error),
    /// Internal server problem (response builder, etc.).
    Internal(String),
}

impl From<litgraph_core::Error> for ApiError {
    fn from(e: litgraph_core::Error) -> Self {
        ApiError::Model(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, msg) = match self {
            ApiError::Model(e) => {
                error!("model error: {e}");
                (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
            }
            ApiError::Internal(m) => {
                error!("internal error: {m}");
                (StatusCode::INTERNAL_SERVER_ERROR, m)
            }
        };
        (status, Json(ErrorResponse { error: msg })).into_response()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use axum::body::to_bytes;
    use axum::http::Request;
    use litgraph_core::model::{ChatStream, ChatStreamEvent};
    use litgraph_core::{FinishReason, Message, Result as CoreResult, TokenUsage};
    use tower::ServiceExt;

    /// Stub model — returns canned content + canned stream chunks. Lets
    /// us test the HTTP plumbing without a real provider.
    struct StubModel;

    #[async_trait]
    impl ChatModel for StubModel {
        fn name(&self) -> &str {
            "stub"
        }
        async fn invoke(
            &self,
            messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> CoreResult<ChatResponse> {
            // Echo back: confirms we received the body intact.
            let last = messages
                .last()
                .map(|m| m.text_content())
                .unwrap_or_default();
            Ok(ChatResponse {
                message: Message::assistant(format!("you said: {last}")),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage::default(),
                model: "stub".into(),
            })
        }

        async fn stream(
            &self,
            _messages: Vec<Message>,
            _opts: &ChatOptions,
        ) -> CoreResult<ChatStream> {
            let events: Vec<CoreResult<ChatStreamEvent>> = vec![
                Ok(ChatStreamEvent::Delta {
                    text: "Hello".into(),
                }),
                Ok(ChatStreamEvent::Delta { text: " world".into() }),
                Ok(ChatStreamEvent::Done {
                    response: ChatResponse {
                        message: Message::assistant("Hello world"),
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage::default(),
                        model: "stub".into(),
                    },
                }),
            ];
            Ok(Box::pin(futures::stream::iter(events)))
        }
    }

    fn app() -> Router {
        router_for(Arc::new(StubModel))
    }

    async fn body_to_string(resp: Response) -> String {
        let bytes = to_bytes(resp.into_body(), 1024 * 1024).await.unwrap();
        String::from_utf8(bytes.to_vec()).unwrap()
    }

    async fn body_to_json(resp: Response) -> Value {
        let s = body_to_string(resp).await;
        serde_json::from_str(&s).unwrap_or_else(|e| panic!("not json: {s} ({e})"))
    }

    #[tokio::test]
    async fn health_returns_ok() {
        let resp = app()
            .oneshot(Request::get("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        assert_eq!(v["status"], "ok");
    }

    #[tokio::test]
    async fn info_lists_endpoints_and_model_name() {
        let resp = app()
            .oneshot(Request::get("/info").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let v = body_to_json(resp).await;
        assert_eq!(v["name"], "stub");
        let eps = v["endpoints"].as_array().unwrap();
        let names: Vec<&str> = eps.iter().filter_map(|x| x.as_str()).collect();
        assert!(names.contains(&"/invoke"));
        assert!(names.contains(&"/stream"));
        assert!(names.contains(&"/batch"));
    }

    #[tokio::test]
    async fn invoke_round_trips_message() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}]
        });
        let resp = app()
            .oneshot(
                Request::post("/invoke")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        // Stub echoes "you said: <last user content>".
        let text = v["message"]["content"][0]["text"].as_str().unwrap();
        assert!(text.starts_with("you said: ping"), "{text}");
    }

    #[tokio::test]
    async fn invoke_rejects_non_json() {
        let resp = app()
            .oneshot(
                Request::post("/invoke")
                    .header("content-type", "application/json")
                    .body(Body::from("not json"))
                    .unwrap(),
            )
            .await
            .unwrap();
        // axum returns 400 on bad JSON before the handler runs.
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stream_emits_sse_lines_and_done_sentinel() {
        let body = json!({
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        });
        let resp = app()
            .oneshot(
                Request::post("/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(
            resp.headers()
                .get("content-type")
                .map(|v| v.to_str().unwrap()),
            Some("text/event-stream")
        );
        let text = body_to_string(resp).await;
        // Three events from the stub + DONE = at least 4 `data:` lines.
        let n_data_lines = text.matches("data:").count();
        assert!(n_data_lines >= 4, "got {n_data_lines}\n{text}");
        assert!(text.ends_with("data: [DONE]\n\n"), "trailing: {text}");
        // Each event payload is JSON we can parse.
        for line in text.lines() {
            if let Some(payload) = line.strip_prefix("data: ") {
                if payload == "[DONE]" {
                    continue;
                }
                let _: Value = serde_json::from_str(payload).unwrap_or_else(|e| {
                    panic!("bad event payload {payload}: {e}")
                });
            }
        }
    }

    #[tokio::test]
    async fn batch_runs_through_default_impl() {
        let body = json!({
            "inputs": [
                [{"role": "user", "content": [{"type": "text", "text": "a"}]}],
                [{"role": "user", "content": [{"type": "text", "text": "b"}]}]
            ]
        });
        let resp = app()
            .oneshot(
                Request::post("/batch")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let v = body_to_json(resp).await;
        let arr = v.as_array().unwrap();
        assert_eq!(arr.len(), 2);
        assert!(arr[0]["message"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("you said: a"));
        assert!(arr[1]["message"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("you said: b"));
    }

    #[tokio::test]
    async fn unknown_route_404() {
        let resp = app()
            .oneshot(Request::get("/nope").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn invoke_with_empty_messages_array() {
        // Stub doesn't error on empty input — confirms we don't impose
        // server-level validation that would reject legitimate calls.
        let body = json!({ "messages": [] });
        let resp = app()
            .oneshot(
                Request::post("/invoke")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn model_error_returns_500_with_json_body() {
        struct FailingModel;
        #[async_trait]
        impl ChatModel for FailingModel {
            fn name(&self) -> &str {
                "fail"
            }
            async fn invoke(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> CoreResult<ChatResponse> {
                Err(litgraph_core::Error::other("boom"))
            }
            async fn stream(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> CoreResult<ChatStream> {
                Err(litgraph_core::Error::other("boom"))
            }
        }
        let app = router_for(Arc::new(FailingModel));
        let body = json!({ "messages": [] });
        let resp = app
            .oneshot(
                Request::post("/invoke")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
        let v = body_to_json(resp).await;
        assert!(v["error"].as_str().unwrap().contains("boom"));
    }

    #[tokio::test]
    async fn stream_emits_error_event_when_inner_stream_fails() {
        struct ErroringStream;
        #[async_trait]
        impl ChatModel for ErroringStream {
            fn name(&self) -> &str {
                "err-stream"
            }
            async fn invoke(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> CoreResult<ChatResponse> {
                unimplemented!()
            }
            async fn stream(
                &self,
                _: Vec<Message>,
                _: &ChatOptions,
            ) -> CoreResult<ChatStream> {
                let events: Vec<CoreResult<ChatStreamEvent>> = vec![
                    Ok(ChatStreamEvent::Delta { text: "ok".into() }),
                    Err(litgraph_core::Error::other("mid-stream boom")),
                ];
                Ok(Box::pin(futures::stream::iter(events)))
            }
        }
        let app = router_for(Arc::new(ErroringStream));
        let body = json!({ "messages": [] });
        let resp = app
            .oneshot(
                Request::post("/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let text = body_to_string(resp).await;
        assert!(text.contains("event: error"));
        assert!(text.contains("mid-stream boom"));
        assert!(text.ends_with("data: [DONE]\n\n"));
    }
}

//! The OpenAI-compatible HTTP surface.

use std::sync::Arc;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use litgraph_core::{ChatOptions, Message};
use litgraph_observability::cost::PriceSheet;
use rand::RngCore;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::error::GatewayError;
use crate::keys::KeyStore;
use crate::registry::{Registry, RoutingStrategy};
use crate::tenant::{PolicyDecision, TenantPolicy};

pub struct GatewayState {
    pub registry: Registry,
    pub keys: KeyStore,
    pub policy: TenantPolicy,
    pub strategy: Box<dyn RoutingStrategy>,
    pub prices: PriceSheet,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    #[serde(default)]
    pub messages: Vec<WireMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
pub struct WireMessage {
    pub role: String,
    #[serde(default)]
    pub content: String,
}

pub fn router(state: Arc<GatewayState>) -> Router {
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(state)
}

fn bearer(headers: &HeaderMap) -> Option<&str> {
    let raw = headers.get(axum::http::header::AUTHORIZATION)?.to_str().ok()?;
    let (scheme, value) = raw.split_once(' ')?;
    scheme.eq_ignore_ascii_case("bearer").then(|| value.trim())
}

async fn list_models(
    State(s): State<Arc<GatewayState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, GatewayError> {
    let token = bearer(&headers).ok_or(GatewayError::Unauthorized)?;
    let key = s.keys.authenticate(token).map_err(|_| GatewayError::Unauthorized)?;
    let data: Vec<Value> = s
        .registry
        .group_names()
        .into_iter()
        .filter(|g| key.allows_group(g))
        .map(|g| json!({"id": g, "object": "model", "owned_by": "litgraph"}))
        .collect();
    Ok(Json(json!({"object": "list", "data": data})))
}

async fn chat_completions(
    State(s): State<Arc<GatewayState>>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Response, GatewayError> {
    // Deserialize manually rather than via the `Json<T>` extractor: axum's
    // extractor rejects a malformed/mistyped body BEFORE this handler runs,
    // with a plain-text 400 that has no `error` object. An OpenAI SDK that
    // parses every error through `error.message/type/code` throws a parse
    // error instead of a normal `APIError`. Routing the failure through
    // `GatewayError::BadRequest` keeps every failure mode in the OpenAI
    // envelope. The serde detail is not client-facing (it can echo body
    // content); only the trace gets it.
    let req: ChatCompletionRequest = serde_json::from_slice(&body).map_err(|e| {
        tracing::warn!(error = %e, "malformed chat completion request body");
        GatewayError::BadRequest {
            message: "could not parse request body as JSON".into(),
        }
    })?;

    // 1. authenticate
    let token = bearer(&headers).ok_or(GatewayError::Unauthorized)?;
    let key = s.keys.authenticate(token).map_err(|_| GatewayError::Unauthorized)?;

    // 2. authorize the group. Distinguish "not yours" from "doesn't exist"
    //    only for groups that exist, so probing cannot enumerate config.
    let group = s
        .registry
        .group(&req.model)
        .ok_or_else(|| GatewayError::ModelNotFound { model: req.model.clone() })?;
    if !key.allows_group(&req.model) {
        return Err(GatewayError::GroupForbidden);
    }

    // 3. tenant gate
    match s.policy.check(&key) {
        PolicyDecision::Allow => {}
        PolicyDecision::RateLimited { retry_after_ms } => {
            return Err(GatewayError::RateLimited { retry_after_ms })
        }
        PolicyDecision::BudgetExhausted { spent_usd, cap_usd } => {
            return Err(GatewayError::BudgetExhausted { spent_usd, cap_usd })
        }
    }

    // 4-5. route and dispatch
    let messages: Vec<Message> = req.messages.iter().map(to_core_message).collect();
    let opts = ChatOptions {
        temperature: req.temperature,
        max_tokens: req.max_tokens,
        ..Default::default()
    };

    let completion_id = completion_id();
    if req.stream {
        let (upstream, used) =
            crate::dispatch::dispatch_stream(group, s.strategy.as_ref(), messages, &opts)
                .await
                .map_err(dispatch_error)?;
        let price = s.prices.lookup(&used.upstream_model);
        if price.is_none() {
            // Same silently-free hazard as the non-streaming path below.
            tracing::warn!(
                upstream_model = %used.upstream_model,
                "no price entry for this deployment; stream served unmetered"
            );
        }
        let state = s.clone();
        let key_id = key.id.clone();
        let meter = Arc::new(move |usage: litgraph_core::TokenUsage| {
            if let Some(price) = price {
                let usd = (usage.prompt as f64 / 1_000_000.0) * price.prompt_per_mtok
                    + (usage.completion as f64 / 1_000_000.0) * price.completion_per_mtok;
                state.policy.record_spend(&key_id, usd);
            }
        });
        return Ok(crate::streaming::sse_relay(
            upstream,
            req.model,
            completion_id,
            used.id.clone(),
            meter,
        ));
    }

    let (resp, used) =
        crate::dispatch::dispatch_invoke(group, s.strategy.as_ref(), messages, &opts)
            .await
            .map_err(dispatch_error)?;

    // 6. meter
    match s.prices.lookup(&used.upstream_model) {
        Some(price) => {
            let usd = (resp.usage.prompt as f64 / 1_000_000.0) * price.prompt_per_mtok
                + (resp.usage.completion as f64 / 1_000_000.0) * price.completion_per_mtok;
            s.policy.record_spend(&key.id, usd);
        }
        // An unpriced deployment serves traffic that costs real money upstream
        // while `spent_today` never moves, so `max_usd_per_day` never trips for
        // it. Nothing else would reveal a missing `PriceSheet` row, so warn:
        // silently-free traffic must at least be greppable. Server-side only —
        // the upstream model name is never surfaced to the client.
        None => tracing::warn!(
            upstream_model = %used.upstream_model,
            "no price entry for this deployment; request served unmetered"
        ),
    }

    // 7. respond, echoing the alias the client asked for
    Ok(Json(json!({
        "id": completion_id,
        "object": "chat.completion",
        "created": unix_seconds(),
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text_of(&resp.message)},
            "finish_reason": finish_str(resp.finish_reason),
        }],
        "usage": {
            "prompt_tokens": resp.usage.prompt,
            "completion_tokens": resp.usage.completion,
            "total_tokens": resp.usage.total,
        },
    }))
    .into_response())
}

fn dispatch_error(error: crate::dispatch::DispatchError) -> GatewayError {
    match error {
        crate::dispatch::DispatchError::Upstream { message } => {
            GatewayError::UpstreamRejected { message }
        }
        crate::dispatch::DispatchError::AllDeploymentsUnavailable => {
            GatewayError::NoDeploymentAvailable
        }
    }
}

fn completion_id() -> String {
    let mut bytes = [0u8; 16];
    rand::thread_rng().fill_bytes(&mut bytes);
    format!("chatcmpl-{}", crate::keys::hex(&bytes))
}

fn unix_seconds() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn to_core_message(m: &WireMessage) -> Message {
    match m.role.as_str() {
        "system" => Message::system(&m.content),
        "assistant" => Message::assistant(&m.content),
        _ => Message::user(&m.content),
    }
}

fn text_of(m: &Message) -> String {
    use litgraph_core::ContentPart;
    m.content
        .iter()
        .filter_map(|p| match p {
            ContentPart::Text { text } => Some(text.as_str()),
            _ => None,
        })
        .collect::<Vec<_>>()
        .join("")
}

fn finish_str(r: litgraph_core::FinishReason) -> &'static str {
    use litgraph_core::FinishReason as F;
    match r {
        F::Stop => "stop",
        F::Length => "length",
        F::ToolCalls => "tool_calls",
        F::ContentFilter => "content_filter",
        _ => "stop",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    use litgraph_core::{ChatModel, ChatOptions, ChatResponse, ChatStream, ChatStreamEvent,
                        FinishReason, Message, Result, TokenUsage};
    use litgraph_observability::cost::{ModelPrice, PriceSheet};
    use crate::config::KeyConfig;
    use crate::keys::{generate_key, KeyStore};
    use crate::registry::{Deployment, Registry, WeightedRandom};
    use crate::tenant::{MemorySpendStore, SpendStore, TenantPolicy, TestClock};
    use std::sync::OnceLock;

    /// Upstream that always succeeds with fixed text and non-zero usage.
    struct Echo;

    #[async_trait::async_trait]
    impl ChatModel for Echo {
        fn name(&self) -> &str { "upstream-model-name" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            Ok(ChatResponse {
                message: Message::assistant("hi"),
                finish_reason: FinishReason::Stop,
                usage: TokenUsage { prompt: 10, completion: 5, total: 15, ..Default::default() },
                model: "upstream-model-name".into(),
            })
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            Ok(Box::pin(futures::stream::iter(vec![
                Ok(ChatStreamEvent::Delta { text: "hi".into() }),
                Ok(ChatStreamEvent::Done {
                    response: ChatResponse {
                        message: Message::assistant("hi"),
                        finish_reason: FinishReason::Stop,
                        usage: TokenUsage {
                            prompt: 10,
                            completion: 5,
                            total: 15,
                            ..Default::default()
                        },
                        model: "upstream-model-name".into(),
                    },
                }),
            ])))
        }
    }

    /// One key, minted once so the plaintext and its stored hash agree.
    fn test_key() -> &'static (String, String, String) {
        static KEY: OnceLock<(String, String, String)> = OnceLock::new();
        KEY.get_or_init(generate_key)
    }

    fn test_plaintext_key() -> String {
        test_key().0.clone()
    }

    /// One deployment in group "gpt-4o", plus a second group the key is
    /// NOT allowed to use, so the 403-vs-404 distinction is testable.
    fn test_state() -> Arc<GatewayState> {
        test_state_and_store().0
    }

    /// Same fixture, but also handing back the spend store so a test can
    /// assert what was actually metered rather than only what was returned.
    fn test_state_and_store() -> (Arc<GatewayState>, Arc<MemorySpendStore>) {
        let (_, prefix, hash) = test_key();
        let keys = KeyStore::from_configs(&[KeyConfig {
            id: "team-a".into(),
            prefix: prefix.clone(),
            hash: hash.clone(),
            groups: vec!["gpt-4o".into()],
            rpm: None,
            max_usd_per_day: None,
        }])
        .expect("valid key config");

        let registry = Registry::for_test(vec![
            Arc::new(Deployment::for_test("gpt4o-openai-primary", "gpt-4o", 1, Arc::new(Echo))),
            Arc::new(Deployment::for_test(
                "sonnet-anthropic-primary",
                "claude-sonnet-4-5",
                1,
                Arc::new(Echo),
            )),
        ]);

        let mut prices = PriceSheet::new();
        prices.set(
            "test-model",
            ModelPrice { prompt_per_mtok: 1.0, completion_per_mtok: 2.0 },
        );

        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::new(clock.clone()));
        let state = Arc::new(GatewayState {
            registry,
            keys,
            policy: TenantPolicy::new(clock.clone(), store.clone()),
            strategy: Box::new(WeightedRandom::seeded(1)),
            prices,
        });
        (state, store)
    }

    #[tokio::test]
    async fn missing_bearer_is_401() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"gpt-4o","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn malformed_json_uses_the_openai_error_envelope() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", test_plaintext_key()))
                    .header("content-type", "application/json")
                    .body(Body::from("{"))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let value: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(value["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn key_cannot_use_a_group_it_does_not_allow() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", test_plaintext_key()))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"claude-sonnet-4-5","messages":[]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        // Configured but not allowed -> 403; not configured at all -> 404.
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);
    }

    #[tokio::test]
    async fn happy_path_returns_openai_shaped_completion() {
        let (state, store) = test_state_and_store();
        let app = router(state);
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", test_plaintext_key()))
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["object"], "chat.completion");
        // The client sees the alias it asked for, not the upstream model name.
        assert_eq!(v["model"], "gpt-4o");
        assert!(v["choices"][0]["message"]["content"].is_string());
        assert!(v["usage"]["total_tokens"].is_number());

        // The response id must be opaque. "gpt4o-openai-primary" is the
        // deployment that served this call, and it is shaped like this repo's
        // real config ids on purpose — echoing one here would tell every
        // tenant which upstream handled their request. Asserting shape alone
        // would not catch that, which is how it shipped the first time.
        //
        // The fixture id must also be something that cannot occur inside a
        // random hex string: a short id like "d1" appears by chance in a
        // 32-char hex id about 12% of the time and makes this flake.
        let id = v["id"].as_str().expect("id must be a string");
        assert!(id.starts_with("chatcmpl-"), "id must keep the OpenAI prefix: {id}");
        assert!(
            !id.contains("gpt4o-openai-primary"),
            "response id leaked the deployment id: {id}"
        );

        // Metering actually ran, with the right amount against the right key.
        // Echo reports {prompt: 10, completion: 5} and the fixture prices
        // "test-model" at $1/Mtok prompt and $2/Mtok completion, so:
        //   10/1e6 * 1.0  +  5/1e6 * 2.0  =  2e-5
        // Without this, a wrong formula, a wrong key id, or a no-op
        // record_spend would all still pass.
        let spent = store.spent_today("team-a");
        assert!(
            (spent - 2e-5).abs() < 1e-12,
            "expected 2e-5 USD metered to team-a, got {spent}"
        );
        assert_eq!(store.spent_today("team-b"), 0.0, "spend must not bleed across keys");
    }

    #[tokio::test]
    async fn completion_ids_are_opaque_and_unique_per_request() {
        // A constant id would satisfy "does not contain the deployment id",
        // so pin uniqueness separately: two calls must not collide.
        async fn one_id() -> String {
            let app = router(test_state());
            let resp = app
                .oneshot(
                    Request::post("/v1/chat/completions")
                        .header("authorization", format!("Bearer {}", test_plaintext_key()))
                        .header("content-type", "application/json")
                        .body(Body::from(
                            r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}"#,
                        ))
                        .unwrap(),
                )
                .await
                .unwrap();
            let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
            let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            v["id"].as_str().unwrap().to_string()
        }

        let (a, b) = (one_id().await, one_id().await);
        assert_ne!(a, b, "completion ids must not repeat across requests");
    }

    #[tokio::test]
    async fn streaming_path_returns_sse_chunks_and_done() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::post("/v1/chat/completions")
                    .header("authorization", format!("Bearer {}", test_plaintext_key()))
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r#"{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}],"stream":true}"#,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert!(resp.headers()["content-type"]
            .to_str()
            .unwrap()
            .starts_with("text/event-stream"));
        let body = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let body = String::from_utf8(body.to_vec()).unwrap();
        assert!(body.contains("chat.completion.chunk"));
        assert!(body.contains("data: [DONE]"));
    }

    #[tokio::test]
    async fn models_endpoint_lists_only_groups_the_key_allows() {
        let app = router(test_state());
        let resp = app
            .oneshot(
                Request::get("/v1/models")
                    .header("authorization", format!("Bearer {}", test_plaintext_key()))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let bytes = to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        let ids: Vec<&str> =
            v["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
        assert_eq!(ids, vec!["gpt-4o"]);
    }
}

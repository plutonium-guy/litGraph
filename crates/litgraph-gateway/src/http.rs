//! The OpenAI-compatible HTTP surface.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use litgraph_core::{ChatOptions, Message};
use litgraph_observability::cost::PriceSheet;
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
    Json(req): Json<ChatCompletionRequest>,
) -> Result<Json<Value>, GatewayError> {
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
    let (resp, used) =
        crate::dispatch::dispatch_invoke(group, s.strategy.as_ref(), messages, &opts)
            .await
            .map_err(|e| match e {
                crate::dispatch::DispatchError::Upstream { message } => {
                    GatewayError::UpstreamRejected { message }
                }
                crate::dispatch::DispatchError::AllDeploymentsUnavailable => {
                    GatewayError::NoDeploymentAvailable
                }
            })?;

    // 6. meter
    if let Some(price) = s.prices.lookup(&used.upstream_model) {
        let usd = (resp.usage.prompt as f64 / 1_000_000.0) * price.prompt_per_mtok
            + (resp.usage.completion as f64 / 1_000_000.0) * price.completion_per_mtok;
        s.policy.record_spend(&key.id, usd);
    }

    // 7. respond, echoing the alias the client asked for
    Ok(Json(json!({
        "id": format!("chatcmpl-{}", &used.id),
        "object": "chat.completion",
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
    })))
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

    use litgraph_core::{ChatModel, ChatOptions, ChatResponse, ChatStream,
                        FinishReason, Message, Result, TokenUsage};
    use litgraph_observability::cost::{ModelPrice, PriceSheet};
    use crate::config::KeyConfig;
    use crate::keys::{generate_key, KeyStore};
    use crate::registry::{Deployment, Registry, WeightedRandom};
    use crate::tenant::{MemorySpendStore, TenantPolicy, TestClock};
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
            unreachable!("Task 6 covers non-streaming only")
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
            Arc::new(Deployment::for_test("d1", "gpt-4o", 1, Arc::new(Echo))),
            Arc::new(Deployment::for_test("d2", "claude-sonnet-4-5", 1, Arc::new(Echo))),
        ]);

        let mut prices = PriceSheet::new();
        prices.set(
            "test-model",
            ModelPrice { prompt_per_mtok: 1.0, completion_per_mtok: 2.0 },
        );

        let clock = Arc::new(TestClock::new());
        Arc::new(GatewayState {
            registry,
            keys,
            policy: TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock.clone()))),
            strategy: Box::new(WeightedRandom::seeded(1)),
            prices,
        })
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
        let app = router(test_state());
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

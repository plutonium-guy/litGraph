//! Test and benchmark harness over a deterministic in-process model.

use std::sync::Arc;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use futures::stream;
use litgraph_core::{
    ChatModel, ChatOptions, ChatResponse, ChatStream, ChatStreamEvent, FinishReason, Message,
    Result, TokenUsage,
};
use litgraph_observability::cost::PriceSheet;
use tokio::sync::oneshot;
use tower::ServiceExt;

use crate::config::KeyConfig;
use crate::http::{router, GatewayState};
use crate::keys::{generate_key, KeyStore};
use crate::registry::{Deployment, Registry, WeightedRandom};
use crate::tenant::{MemorySpendStore, TenantPolicy, TestClock};

struct ScriptedModel {
    chunks: usize,
}

#[async_trait::async_trait]
impl ChatModel for ScriptedModel {
    fn name(&self) -> &str {
        "scripted"
    }

    async fn invoke(&self, _messages: Vec<Message>, _opts: &ChatOptions) -> Result<ChatResponse> {
        Ok(response("ok"))
    }

    async fn stream(&self, _messages: Vec<Message>, _opts: &ChatOptions) -> Result<ChatStream> {
        Ok(scripted_stream(self.chunks))
    }
}

pub struct BenchState {
    state: Arc<GatewayState>,
    plaintext_key: String,
}

pub async fn spawn_test_gateway() -> (std::net::SocketAddr, String, oneshot::Sender<()>) {
    let bench = bench_state();
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    tokio::spawn(async move {
        let server =
            axum::serve(listener, router(bench.state)).with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            });
        let _ = server.await;
    });
    (address, bench.plaintext_key, shutdown_tx)
}

pub fn bench_state() -> BenchState {
    let (plaintext_key, prefix, hash) = generate_key();
    let keys = KeyStore::from_configs(&[KeyConfig {
        id: "bench".into(),
        prefix,
        hash,
        groups: vec!["ollama".into()],
        rpm: None,
        max_usd_per_day: None,
    }])
    .unwrap();
    let deployment = Arc::new(Deployment::for_test(
        "local",
        "ollama",
        1,
        Arc::new(ScriptedModel { chunks: 2 }),
    ));
    let clock = Arc::new(TestClock::new());
    BenchState {
        state: Arc::new(GatewayState {
            registry: Registry::for_test(vec![deployment]),
            keys,
            policy: TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock))),
            strategy: Box::new(WeightedRandom::seeded(1)),
            prices: PriceSheet::new(),
        }),
        plaintext_key,
    }
}

pub async fn invoke_once(bench: Arc<BenchState>) -> StatusCode {
    router(bench.state.clone())
        .oneshot(
            Request::post("/v1/chat/completions")
                .header("authorization", format!("Bearer {}", bench.plaintext_key))
                .header("content-type", "application/json")
                .body(Body::from(
                    r#"{"model":"ollama","messages":[{"role":"user","content":"hi"}]}"#,
                ))
                .unwrap(),
        )
        .await
        .unwrap()
        .status()
}

pub async fn relay_n_chunks(chunks: usize) -> usize {
    let response = crate::streaming::sse_relay(
        scripted_stream(chunks),
        "ollama".into(),
        "chatcmpl-bench".into(),
        "local".into(),
        Arc::new(|_| {}),
    );
    to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap()
        .len()
}

fn scripted_stream(chunks: usize) -> ChatStream {
    let mut events = Vec::with_capacity(chunks + 1);
    for _ in 0..chunks {
        events.push(Ok(ChatStreamEvent::Delta { text: "x".into() }));
    }
    events.push(Ok(ChatStreamEvent::Done {
        response: response(&"x".repeat(chunks)),
    }));
    Box::pin(stream::iter(events))
}

fn response(text: &str) -> ChatResponse {
    ChatResponse {
        message: Message::assistant(text),
        finish_reason: FinishReason::Stop,
        usage: TokenUsage {
            prompt: 2,
            completion: 1,
            total: 3,
            ..Default::default()
        },
        model: "scripted".into(),
    }
}

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::model::{ChatStream, FinishReason, TokenUsage};
use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Message, Result};
use litgraph_observability::{
    CallbackBus, CostTracker, InstrumentedChatModel, ModelPrice, PriceSheet,
};

struct Fake { hits: Arc<AtomicU32> }

#[async_trait]
impl ChatModel for Fake {
    fn name(&self) -> &str { "gpt-5" }
    async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
        self.hits.fetch_add(1, Ordering::SeqCst);
        Ok(ChatResponse {
            message: Message::assistant("hi"),
            finish_reason: FinishReason::Stop,
            usage: TokenUsage { prompt: 500_000, completion: 250_000, total: 750_000 , cache_creation: 0, cache_read: 0 },
            model: "gpt-5".into(),
        })
    }
    async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
        unimplemented!()
    }
}

#[tokio::test]
async fn instrumented_model_flows_into_cost_tracker() {
    let mut prices = PriceSheet::new();
    prices.set("gpt-5", ModelPrice { prompt_per_mtok: 2.0, completion_per_mtok: 10.0 });
    let tracker = Arc::new(CostTracker::new(prices));

    let bus = CallbackBus::new().with_flush_interval(Duration::from_millis(5));
    bus.subscribe(tracker.clone());
    let (handle, _task) = bus.start();

    let fake = Arc::new(Fake { hits: Arc::new(AtomicU32::new(0)) });
    let inst = InstrumentedChatModel::new(fake, handle);

    for _ in 0..3 {
        let _ = inst.invoke(vec![Message::user("hi")], &ChatOptions::default()).await.unwrap();
    }

    // Give the bus a moment to flush.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let snap = tracker.snapshot();
    assert_eq!(snap.calls, 3);
    assert_eq!(snap.prompt_tokens, 1_500_000);
    assert_eq!(snap.completion_tokens, 750_000);
    // 3 calls * (0.5M * $2 + 0.25M * $10) = 3 * (1.0 + 2.5) = 10.5
    assert!((snap.usd - 10.5).abs() < 1e-6, "usd={}", snap.usd);
    let gpt5 = snap.per_model.get("gpt-5").unwrap();
    assert_eq!(gpt5.calls, 3);
}

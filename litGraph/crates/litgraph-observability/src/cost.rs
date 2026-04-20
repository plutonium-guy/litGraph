//! Token + cost accounting.
//!
//! `CostTracker` is a `Callback` — subscribe it to a `CallbackBus` and running
//! totals update live. Query with `.snapshot()` or `.usd()`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::callback::Callback;
use crate::event::Event;

/// Per-million-token prices in USD for a single model.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct ModelPrice {
    pub prompt_per_mtok: f64,
    pub completion_per_mtok: f64,
}

#[derive(Debug, Clone, Default)]
pub struct PriceSheet {
    inner: HashMap<String, ModelPrice>,
}

impl PriceSheet {
    pub fn new() -> Self { Self::default() }

    pub fn set(&mut self, model: impl Into<String>, price: ModelPrice) -> &mut Self {
        self.inner.insert(model.into(), price);
        self
    }

    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.inner.get(model).copied()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Snapshot {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub usd: f64,
    pub per_model: HashMap<String, PerModel>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerModel {
    pub calls: u64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub usd: f64,
}

struct Inner {
    prices: PriceSheet,
    snap: Snapshot,
}

pub struct CostTracker {
    inner: Arc<RwLock<Inner>>,
}

impl CostTracker {
    pub fn new(prices: PriceSheet) -> Self {
        Self { inner: Arc::new(RwLock::new(Inner { prices, snap: Snapshot::default() })) }
    }

    pub fn snapshot(&self) -> Snapshot { self.inner.read().snap.clone() }
    pub fn usd(&self) -> f64 { self.inner.read().snap.usd }

    pub fn reset(&self) {
        let mut g = self.inner.write();
        g.snap = Snapshot::default();
    }
}

#[async_trait]
impl Callback for CostTracker {
    async fn on_events(&self, events: &[Event]) {
        // Only aggregate terminal LLM events with usage — start events carry nothing.
        let mut g = self.inner.write();
        for e in events {
            if let Event::Llm { phase: crate::event::Phase::End, model, usage: Some(u), .. } = e {
                let price = g.prices.get(model).unwrap_or_default();
                let cost = (u.prompt as f64 / 1_000_000.0) * price.prompt_per_mtok
                    + (u.completion as f64 / 1_000_000.0) * price.completion_per_mtok;

                g.snap.calls += 1;
                g.snap.prompt_tokens += u.prompt as u64;
                g.snap.completion_tokens += u.completion as u64;
                g.snap.usd += cost;

                let pm = g.snap.per_model.entry(model.clone()).or_default();
                pm.calls += 1;
                pm.prompt_tokens += u.prompt as u64;
                pm.completion_tokens += u.completion as u64;
                pm.usd += cost;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::model::TokenUsage;

    #[tokio::test]
    async fn tracks_cost_from_events() {
        let mut prices = PriceSheet::new();
        prices.set("gpt-5", ModelPrice { prompt_per_mtok: 2.50, completion_per_mtok: 10.00 });
        let tracker = CostTracker::new(prices);
        let events = vec![
            Event::Llm {
                phase: crate::event::Phase::End,
                model: "gpt-5".into(),
                usage: Some(TokenUsage { prompt: 1_000_000, completion: 500_000, total: 1_500_000 , cache_creation: 0, cache_read: 0 }),
                error: None,
                ts_ms: 0,
            },
        ];
        tracker.on_events(&events).await;
        let snap = tracker.snapshot();
        assert_eq!(snap.calls, 1);
        assert_eq!(snap.prompt_tokens, 1_000_000);
        assert!((snap.usd - (2.50 + 5.00)).abs() < 1e-6);
    }
}

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

    /// Exact match only. Fast path for explicitly-set keys.
    pub fn get(&self, model: &str) -> Option<ModelPrice> {
        self.inner.get(model).copied()
    }

    /// Lookup with case-insensitive substring fallback. Tries exact match
    /// first, then iterates entries and returns the LONGEST key that is a
    /// substring of `model` (case-insensitive). Longest-key wins is critical
    /// — both "gpt-4" and "gpt-4o" appear in "gpt-4o-2024-11-20", but the
    /// caller wants the gpt-4o price, not the gpt-4 price.
    pub fn lookup(&self, model: &str) -> Option<ModelPrice> {
        if let Some(p) = self.inner.get(model) {
            return Some(*p);
        }
        let m = model.to_ascii_lowercase();
        let mut best: Option<(usize, ModelPrice)> = None;
        for (key, price) in &self.inner {
            let k = key.to_ascii_lowercase();
            if m.contains(&k) && best.is_none_or(|(len, _)| k.len() > len) {
                best = Some((k.len(), *price));
            }
        }
        best.map(|(_, p)| p)
    }

    /// Number of registered models (useful in tests + reporting).
    pub fn len(&self) -> usize { self.inner.len() }
    pub fn is_empty(&self) -> bool { self.inner.is_empty() }

    /// Iterate `(model_key, price)` pairs in arbitrary order — language
    /// bindings need this to expose the catalog as a dict.
    pub fn iter(&self) -> impl Iterator<Item = (&str, ModelPrice)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), *v))
    }
}

/// Per-million-token shorthand used in the price tables — easier to read than
/// `f64` literals at point of definition.
const fn p(prompt_mtok: f64, completion_mtok: f64) -> ModelPrice {
    ModelPrice { prompt_per_mtok: prompt_mtok, completion_per_mtok: completion_mtok }
}

/// Current public list-prices for the major hosted-LLM endpoints.
///
/// Sources: vendor pricing pages as of 2026-04. Prices change frequently —
/// override what this gets wrong via `sheet.set(...)`. Keys are short model
/// IDs that match against full versioned names via `PriceSheet::lookup()`'s
/// longest-substring rule (e.g. "gpt-4o" matches "gpt-4o-2024-11-20").
///
/// Per-million-token (USD). Bedrock Anthropic prices are roughly identical
/// to direct Anthropic so we don't duplicate; if you need separate Bedrock
/// pricing, set them explicitly.
pub fn default_prices() -> PriceSheet {
    let mut s = PriceSheet::new();
    let entries: &[(&str, ModelPrice)] = &[
        // ── OpenAI ──────────────────────────────────────────────────────
        ("gpt-4o-mini",            p(0.15, 0.60)),
        ("gpt-4o",                 p(2.50, 10.00)),
        ("gpt-4-turbo",            p(10.00, 30.00)),
        ("gpt-4",                  p(30.00, 60.00)),
        ("gpt-3.5-turbo",          p(0.50, 1.50)),
        ("o1-mini",                p(3.00, 12.00)),
        ("o1",                     p(15.00, 60.00)),
        ("o3-mini",                p(1.10, 4.40)),
        ("text-embedding-3-large", p(0.13, 0.0)),
        ("text-embedding-3-small", p(0.02, 0.0)),
        // ── Anthropic ───────────────────────────────────────────────────
        ("claude-haiku-4",         p(1.00, 5.00)),
        ("claude-sonnet-4",        p(3.00, 15.00)),
        ("claude-opus-4-7",        p(15.00, 75.00)),
        ("claude-opus-4",          p(15.00, 75.00)),
        ("claude-3-5-sonnet",      p(3.00, 15.00)),
        ("claude-3-5-haiku",       p(0.80, 4.00)),
        ("claude-3-opus",          p(15.00, 75.00)),
        ("claude-3-sonnet",        p(3.00, 15.00)),
        ("claude-3-haiku",         p(0.25, 1.25)),
        // ── Gemini ──────────────────────────────────────────────────────
        ("gemini-2.0-flash",       p(0.075, 0.30)),
        ("gemini-1.5-flash",       p(0.075, 0.30)),
        ("gemini-1.5-pro",         p(1.25, 5.00)),
        ("text-embedding-004",     p(0.025, 0.0)),
        // ── Cohere ──────────────────────────────────────────────────────
        ("command-r-plus",         p(2.50, 10.00)),
        ("command-r",              p(0.15, 0.60)),
        ("embed-english-v3",       p(0.10, 0.0)),
        ("embed-multilingual-v3",  p(0.10, 0.0)),
        // ── Voyage ──────────────────────────────────────────────────────
        ("voyage-3-large",         p(0.18, 0.0)),
        ("voyage-3",               p(0.06, 0.0)),
        ("rerank-2-lite",          p(0.02, 0.0)),
        ("rerank-2",               p(0.05, 0.0)),
        // ── Jina ────────────────────────────────────────────────────────
        ("jina-embeddings-v3",     p(0.018, 0.0)),
        ("jina-reranker-v2",       p(0.018, 0.0)),
        // ── Groq (LPU; Llama family) ────────────────────────────────────
        ("llama-3.3-70b",          p(0.59, 0.79)),
        ("llama-3.1-70b",          p(0.59, 0.79)),
        ("llama-3.1-8b",           p(0.05, 0.08)),
        // ── Together / Fireworks (rough OSS-model averages) ─────────────
        ("mixtral-8x7b",           p(0.50, 0.50)),
        ("mixtral-8x22b",          p(1.20, 1.20)),
        // ── Mistral La Plateforme ───────────────────────────────────────
        ("mistral-large",          p(2.00, 6.00)),
        ("mistral-small",          p(0.20, 0.60)),
        ("codestral",              p(0.30, 0.90)),
        ("pixtral-large",          p(2.00, 6.00)),
        // ── DeepSeek ────────────────────────────────────────────────────
        ("deepseek-chat",          p(0.27, 1.10)),
        ("deepseek-reasoner",      p(0.55, 2.19)),
        // ── xAI ─────────────────────────────────────────────────────────
        ("grok-2",                 p(2.00, 10.00)),
        // ── AWS Titan embed ─────────────────────────────────────────────
        ("titan-embed-text-v2",    p(0.02, 0.0)),
        ("titan-embed-text-v1",    p(0.10, 0.0)),
    ];
    for (k, v) in entries { s.set(*k, *v); }
    s
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
                // Use substring fallback so versioned model IDs
                // ("gpt-4o-2024-11-20") still match the registered "gpt-4o" key.
                let price = g.prices.lookup(model).unwrap_or_default();
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

    #[test]
    fn lookup_exact_match_takes_precedence() {
        let mut s = PriceSheet::new();
        s.set("gpt-4o", p(2.50, 10.00));
        s.set("gpt-4o-mini", p(0.15, 0.60));
        let exact = s.lookup("gpt-4o").unwrap();
        assert!((exact.prompt_per_mtok - 2.50).abs() < 1e-9);
    }

    #[test]
    fn lookup_substring_finds_longest_key_when_versioned() {
        // Both "gpt-4" and "gpt-4o" are substrings of the versioned name; the
        // longest match (gpt-4o) must win — otherwise we'd quote the agent
        // 12x the right price for using gpt-4o-2024-11-20.
        let mut s = PriceSheet::new();
        s.set("gpt-4", p(30.00, 60.00));
        s.set("gpt-4o", p(2.50, 10.00));
        let p_versioned = s.lookup("gpt-4o-2024-11-20").unwrap();
        assert!((p_versioned.prompt_per_mtok - 2.50).abs() < 1e-9,
            "expected gpt-4o pricing, got {}", p_versioned.prompt_per_mtok);
    }

    #[test]
    fn lookup_case_insensitive() {
        let mut s = PriceSheet::new();
        s.set("Claude-Opus-4-7", p(15.00, 75.00));
        let p1 = s.lookup("claude-OPUS-4-7-1m").unwrap();
        assert!((p1.completion_per_mtok - 75.00).abs() < 1e-9);
    }

    #[test]
    fn lookup_returns_none_when_no_match() {
        let mut s = PriceSheet::new();
        s.set("gpt-4o", p(2.50, 10.00));
        assert!(s.lookup("anthropic-claude-opus-4-7").is_none());
    }

    #[test]
    fn default_prices_covers_major_providers() {
        let s = default_prices();
        // Smoke: every documented family should have at least one entry that
        // matches via substring lookup with a real-world versioned name.
        for versioned in &[
            "gpt-4o-2024-11-20", "gpt-4o-mini-2024-07-18",
            "claude-opus-4-7-1m", "claude-sonnet-4-20251022",
            "gemini-2.0-flash-exp", "command-r-plus-08-2024",
            "voyage-3-large", "jina-embeddings-v3",
            "mistral-large-latest", "deepseek-chat",
            "amazon.titan-embed-text-v2:0",
        ] {
            assert!(s.lookup(versioned).is_some(),
                "default_prices missing entry for: {versioned}");
        }
        // Sanity: the catalog has a non-trivial number of models.
        assert!(s.len() >= 30, "got {} entries", s.len());
    }

    #[tokio::test]
    async fn tracker_with_default_prices_handles_versioned_model_id() {
        let tracker = CostTracker::new(default_prices());
        // Real-world: a chat response would carry the FULL versioned name.
        let events = vec![
            Event::Llm {
                phase: crate::event::Phase::End,
                model: "gpt-4o-2024-11-20".into(),
                usage: Some(TokenUsage {
                    prompt: 1_000_000, completion: 500_000, total: 1_500_000,
                    cache_creation: 0, cache_read: 0,
                }),
                error: None,
                ts_ms: 0,
            },
        ];
        tracker.on_events(&events).await;
        let snap = tracker.snapshot();
        // gpt-4o = $2.50 prompt + $10.00 completion per Mtok → $7.50 total.
        assert!((snap.usd - 7.50).abs() < 1e-6, "got ${}", snap.usd);
    }
}

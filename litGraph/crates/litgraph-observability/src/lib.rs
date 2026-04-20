//! Observability for litGraph.
//!
//! Three layers, all production-grade on day 1:
//!
//! 1. **Callback bus** — providers and graph nodes emit typed [`Event`]s into a bus.
//!    Subscribers implement [`Callback`] and receive batched slices to avoid
//!    per-token GIL thrash when a Python consumer is listening.
//! 2. **Cost accounting** — built-in callback that tallies [`TokenUsage`] per model
//!    and applies a user-supplied price sheet to report running USD cost.
//! 3. **Tracing** — `tracing` spans wrap every call site. With the `otel` feature,
//!    spans export to OTLP — vendor-neutral, works with Grafana/Honeycomb/Datadog/LS.
//!
//! No LangSmith lock-in. OTel-first.

pub mod callback;
pub mod cost;
pub mod event;
pub mod instrument;
pub mod otel;

pub use callback::{Callback, CallbackBus, CallbackHandle};
pub use cost::{CostTracker, PriceSheet, ModelPrice};
pub use event::{Event, Phase};
pub use instrument::InstrumentedChatModel;
pub use otel::{LangSmithConfig, langsmith_otlp_endpoint, langsmith_otlp_headers};

//! `litgraph-gateway` — an OpenAI-compatible LLM gateway.
//!
//! Virtual API keys, per-tenant rate limits and spend caps, and weighted
//! routing with failover across N deployments per model alias.
//!
//! A deployment is an `Arc<dyn ChatModel>`, so this crate is a thin edge
//! over a pool of them rather than a second execution engine.

pub mod config;
pub mod keys;
pub mod tenant;

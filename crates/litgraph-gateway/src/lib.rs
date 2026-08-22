//! `litgraph-gateway` — an OpenAI-compatible LLM gateway.
//!
//! Virtual API keys, per-tenant rate limits and spend caps, and weighted
//! routing with failover across N deployments per model alias.
//!
//! A deployment is an `Arc<dyn ChatModel>`, so this crate is a thin edge
//! over a pool of them rather than a second execution engine.

pub mod config;
pub mod dispatch;
pub mod error;
pub mod http;
pub mod keys;
pub mod registry;
pub mod streaming;
pub mod tenant;

#[doc(hidden)]
pub mod testing;

use std::sync::Arc;

use thiserror::Error;

use crate::config::{ConfigError, GatewayConfig};
use crate::http::GatewayState;
use crate::keys::{AuthError, KeyStore};
use crate::registry::{Registry, WeightedRandom};
use crate::tenant::{MemorySpendStore, SystemClock, TenantPolicy};

#[derive(Debug, Error)]
pub enum BuildError {
    #[error(transparent)]
    Config(#[from] ConfigError),
    #[error(transparent)]
    Keys(#[from] AuthError),
}

pub fn build_state(cfg: &GatewayConfig) -> Result<GatewayState, BuildError> {
    let clock = Arc::new(SystemClock);
    Ok(GatewayState {
        registry: Registry::from_config(cfg)?,
        keys: KeyStore::from_configs(&cfg.key)?,
        policy: TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::new(clock))),
        strategy: Box::new(WeightedRandom::new()),
        prices: litgraph_observability::cost::default_prices(),
    })
}

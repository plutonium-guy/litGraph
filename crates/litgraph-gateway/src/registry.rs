//! Deployments, groups, and the routing strategy.
//!
//! A deployment is an `Arc<dyn ChatModel>` plus its deployment-scoped
//! state. Deployment-scoped means shared across all tenants: every
//! tenant's failures are evidence about the same upstream, so one shared
//! breaker per deployment is the correct granularity.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use litgraph_core::circuit_breaker::{CircuitBreaker, CircuitState};
use litgraph_core::ChatModel;
use parking_lot::Mutex;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::config::{ConfigError, DeploymentConfig, GatewayConfig};

const BREAKER_THRESHOLD: usize = 5;
const BREAKER_COOLDOWN: Duration = Duration::from_secs(30);

pub struct Deployment {
    pub id: String,
    pub group: String,
    /// The name sent upstream, which may differ from `group`.
    pub upstream_model: String,
    pub weight: u32,
    pub model: Arc<dyn ChatModel>,
    breaker: Arc<CircuitBreaker>,
}

impl std::fmt::Debug for Deployment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Deployment")
            .field("id", &self.id)
            .field("group", &self.group)
            .field("weight", &self.weight)
            .finish_non_exhaustive()
    }
}

impl Deployment {
    /// Best-effort health snapshot for telemetry and logging.
    ///
    /// NOT an admission check: `CircuitBreaker::state` reports `Open` even
    /// after the cooldown has expired (the `Open -> HalfOpenProbing`
    /// transition happens inside `call`). Filtering on this would exclude a
    /// recovered deployment forever. Admission belongs to
    /// `CircuitBreaker::call` in the dispatch layer.
    pub fn is_available(&self) -> bool {
        !matches!(self.breaker.state(), CircuitState::Open)
    }

    /// The deployment-scoped breaker. Admission decisions MUST go through
    /// `CircuitBreaker::call`, which handles half-open probing and cooldown
    /// expiry; `is_available` above is telemetry only.
    pub fn breaker(&self) -> &Arc<CircuitBreaker> {
        &self.breaker
    }

    #[doc(hidden)]
    pub fn for_test(id: &str, group: &str, weight: u32, model: Arc<dyn ChatModel>) -> Self {
        Self {
            id: id.into(),
            group: group.into(),
            upstream_model: "test-model".into(),
            weight,
            model,
            breaker: Arc::new(CircuitBreaker::new(BREAKER_THRESHOLD, BREAKER_COOLDOWN)),
        }
    }

    #[doc(hidden)]
    pub fn trip_for_test(&self) {
        self.breaker.trip(BREAKER_COOLDOWN);
    }
}

pub struct ModelGroup {
    pub name: String,
    pub deployments: Vec<Arc<Deployment>>,
}

pub struct Registry {
    groups: HashMap<String, ModelGroup>,
}

impl Registry {
    pub fn group(&self, name: &str) -> Option<&ModelGroup> {
        self.groups.get(name)
    }

    pub fn group_names(&self) -> Vec<String> {
        let mut v: Vec<_> = self.groups.keys().cloned().collect();
        v.sort();
        v
    }

    /// Build from config, constructing one provider client per deployment.
    /// Clients are built once here and shared forever — constructing one
    /// per request would pay a TLS handshake every time.
    pub fn from_config(cfg: &GatewayConfig) -> Result<Self, ConfigError> {
        let mut groups: HashMap<String, ModelGroup> = HashMap::new();
        for d in &cfg.deployment {
            let model = build_model(d)?;
            let dep = Arc::new(Deployment {
                id: d.id.clone(),
                group: d.group.clone(),
                upstream_model: d.model.clone(),
                weight: d.weight.max(1),
                model,
                breaker: Arc::new(CircuitBreaker::new(BREAKER_THRESHOLD, BREAKER_COOLDOWN)),
            });
            groups
                .entry(d.group.clone())
                .or_insert_with(|| ModelGroup { name: d.group.clone(), deployments: Vec::new() })
                .deployments
                .push(dep);
        }
        Ok(Self { groups })
    }

    #[doc(hidden)]
    pub fn for_test(deployments: Vec<Arc<Deployment>>) -> Self {
        let mut groups: HashMap<String, ModelGroup> = HashMap::new();
        for d in deployments {
            groups
                .entry(d.group.clone())
                .or_insert_with(|| ModelGroup { name: d.group.clone(), deployments: Vec::new() })
                .deployments
                .push(d);
        }
        Self { groups }
    }
}

fn build_model(d: &DeploymentConfig) -> Result<Arc<dyn ChatModel>, ConfigError> {
    match d.provider.as_str() {
        // "openai" covers every OpenAI-compatible endpoint via base_url.
        "openai" => {
            let api_key = std::env::var(&d.api_key_env)
                .map_err(|_| ConfigError::MissingEnv(d.api_key_env.clone()))?;
            let cfg = litgraph_providers_openai::OpenAIConfig::new(api_key, d.model.clone())
                .with_base_url(d.base_url.clone());
            let chat = litgraph_providers_openai::OpenAIChat::new(cfg)
                .map_err(|e| ConfigError::ProviderBuild(d.id.clone(), e.to_string()))?;
            Ok(Arc::new(chat))
        }
        "ollama" => {
            let cfg = litgraph_providers_openai::OpenAIConfig::new("ollama", d.model.clone())
                .with_base_url(d.base_url.clone());
            let chat = litgraph_providers_openai::OpenAIChat::new(cfg)
                .map_err(|e| ConfigError::ProviderBuild(d.id.clone(), e.to_string()))?;
            Ok(Arc::new(chat))
        }
        other => Err(ConfigError::UnknownProvider {
            deployment_id: d.id.clone(),
            provider: other.to_string(),
        }),
    }
}

/// How a deployment is chosen from the available candidates in a group.
pub trait RoutingStrategy: Send + Sync {
    fn pick(&self, candidates: &[Arc<Deployment>]) -> Option<Arc<Deployment>>;
}

/// Weighted random. Chosen over least-latency because it needs no
/// per-deployment latency state and is deterministic under a seeded RNG,
/// which makes distribution assertable in tests.
pub struct WeightedRandom {
    rng: Mutex<StdRng>,
}

impl WeightedRandom {
    pub fn new() -> Self {
        Self { rng: Mutex::new(StdRng::from_entropy()) }
    }
    pub fn seeded(seed: u64) -> Self {
        Self { rng: Mutex::new(StdRng::seed_from_u64(seed)) }
    }
}

impl Default for WeightedRandom {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingStrategy for WeightedRandom {
    fn pick(&self, candidates: &[Arc<Deployment>]) -> Option<Arc<Deployment>> {
        if candidates.is_empty() {
            return None;
        }
        let total: u32 = candidates.iter().map(|d| d.weight).sum();
        if total == 0 {
            return candidates.first().cloned();
        }
        let mut roll = self.rng.lock().gen_range(0..total);
        for d in candidates {
            if roll < d.weight {
                return Some(d.clone());
            }
            roll -= d.weight;
        }
        candidates.last().cloned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use litgraph_core::{ChatModel, ChatOptions, ChatResponse, ChatStream, Message, Result};

    struct Dummy(&'static str);

    #[async_trait::async_trait]
    impl ChatModel for Dummy {
        fn name(&self) -> &str { self.0 }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            unreachable!("routing tests never dispatch")
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unreachable!("routing tests never dispatch")
        }
    }

    fn dep(id: &str, group: &str, weight: u32) -> Arc<Deployment> {
        Arc::new(Deployment::for_test(id, group, weight, Arc::new(Dummy("m"))))
    }

    #[test]
    fn weighted_pick_is_deterministic_under_a_seeded_rng() {
        let a = dep("a", "g", 3);
        let b = dep("b", "g", 1);
        let candidates = vec![a, b];

        let first: Vec<String> = {
            let strat = WeightedRandom::seeded(42);
            (0..20).map(|_| strat.pick(&candidates).unwrap().id.clone()).collect()
        };
        let second: Vec<String> = {
            let strat = WeightedRandom::seeded(42);
            (0..20).map(|_| strat.pick(&candidates).unwrap().id.clone()).collect()
        };
        assert_eq!(first, second, "same seed must produce the same sequence");
    }

    #[test]
    fn weighted_pick_respects_weights_over_many_draws() {
        let candidates = vec![dep("heavy", "g", 9), dep("light", "g", 1)];
        let strat = WeightedRandom::seeded(7);
        let mut heavy = 0;
        for _ in 0..1_000 {
            if strat.pick(&candidates).unwrap().id == "heavy" {
                heavy += 1;
            }
        }
        assert!((700..=980).contains(&heavy), "expected ~90% heavy, got {heavy}/1000");
    }

    #[test]
    fn unavailable_deployments_are_never_picked() {
        let open = dep("open", "g", 1);
        open.trip_for_test();
        let healthy = dep("healthy", "g", 1);
        let available: Vec<_> =
            vec![open, healthy].into_iter().filter(|d| d.is_available()).collect();

        assert_eq!(available.len(), 1);
        let strat = WeightedRandom::seeded(1);
        assert_eq!(strat.pick(&available).unwrap().id, "healthy");
    }

    #[test]
    fn pick_returns_none_when_no_candidates_remain() {
        let strat = WeightedRandom::seeded(1);
        assert!(strat.pick(&[]).is_none());
    }

    #[test]
    fn registry_groups_deployments_by_alias() {
        let reg = Registry::for_test(vec![
            dep("d1", "gpt-4o", 1),
            dep("d2", "gpt-4o", 1),
            dep("d3", "claude", 1),
        ]);
        assert_eq!(reg.group("gpt-4o").unwrap().deployments.len(), 2);
        assert_eq!(reg.group("claude").unwrap().deployments.len(), 1);
        assert!(reg.group("nope").is_none());
    }

    #[test]
    fn ollama_deployment_builds_without_an_api_key_environment_variable() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "local"
group = "local-chat"
provider = "ollama"
model = "qwen2.5:0.5b"
base_url = "http://127.0.0.1:11434/v1"
"#,
        )
        .expect("valid Ollama config");

        let registry = Registry::from_config(&cfg).expect("Ollama does not require a key env");
        let deployment = &registry.group("local-chat").unwrap().deployments[0];
        assert_eq!(deployment.upstream_model, "qwen2.5:0.5b");
    }
}

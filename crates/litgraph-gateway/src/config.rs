//! Gateway configuration — deployments (physical endpoints), groups
//! (the aliases clients request), and virtual keys (tenants).
//!
//! Validation is total: a config that parses is a config the registry can
//! build from. Cross-references (a key naming a group no deployment
//! serves) are rejected here rather than surfacing as a 404 at runtime.

use std::collections::HashSet;

use serde::Deserialize;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ConfigError {
    #[error("config is not valid TOML: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("duplicate deployment id {0:?}")]
    DuplicateDeploymentId(String),
    #[error("duplicate key id {0:?}")]
    DuplicateKeyId(String),
    #[error("duplicate key prefix {0:?}")]
    DuplicateKeyPrefix(String),
    #[error("key {key_id:?} allows group {group:?}, which no deployment serves")]
    UnknownGroup { key_id: String, group: String },
    #[error("config declares no deployments")]
    NoDeployments,
    #[error("deployment {0:?} names env var that is not set")]
    MissingEnv(String),
    #[error("deployment {deployment_id:?} uses unsupported provider {provider:?}")]
    UnknownProvider { deployment_id: String, provider: String },
    #[error("deployment {0:?} failed to build its provider client: {1}")]
    ProviderBuild(String, String),
}

fn default_weight() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeploymentConfig {
    /// Unique across the config. Appears in traces, never in client errors.
    pub id: String,
    /// The alias clients send as `"model"`. Many deployments may share one.
    pub group: String,
    /// Provider adapter to construct. v1 supports "openai" (which covers
    /// every OpenAI-compatible endpoint via `base_url`).
    pub provider: String,
    /// The model name sent upstream, which may differ from `group`.
    pub model: String,
    pub base_url: String,
    /// Env var holding the upstream credential. Never the secret itself.
    pub api_key_env: String,
    #[serde(default = "default_weight")]
    pub weight: u32,
    /// Deployment-scoped requests/minute — protects the upstream, shared
    /// across all tenants. Distinct from a key's tenant-scoped `rpm`.
    #[serde(default)]
    pub rpm: Option<u32>,
}

#[derive(Clone, Deserialize)]
pub struct KeyConfig {
    pub id: String,
    /// Lookup index for this key. Must match the prefix embedded in the
    /// plaintext token. Emitted by `keygen` alongside the hash.
    pub prefix: String,
    /// argon2id PHC string. Never a plaintext key.
    pub hash: String,
    /// Groups this key may invoke. Exact match; no globs in v1.
    pub groups: Vec<String>,
    /// Tenant-scoped requests/minute. `None` = unlimited.
    #[serde(default)]
    pub rpm: Option<u32>,
    /// Rolling daily spend ceiling in USD. `None` = uncapped.
    #[serde(default)]
    pub max_usd_per_day: Option<f64>,
}

/// Hand-written so `hash` — a live argon2id PHC string — never appears in
/// logs or trace output via `format!("{cfg:?}")` or `tracing::debug!`. A
/// derived `Debug` would print it verbatim; `#[derive(Debug)]` on
/// `GatewayConfig` also relies on this impl for the same reason.
impl std::fmt::Debug for KeyConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyConfig")
            .field("id", &self.id)
            .field("prefix", &self.prefix)
            .field("hash", &"<redacted>")
            .field("groups", &self.groups)
            .field("rpm", &self.rpm)
            .field("max_usd_per_day", &self.max_usd_per_day)
            .finish()
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct GatewayConfig {
    #[serde(default)]
    pub deployment: Vec<DeploymentConfig>,
    #[serde(default)]
    pub key: Vec<KeyConfig>,
}

impl GatewayConfig {
    pub fn from_toml_str(s: &str) -> Result<Self, ConfigError> {
        let cfg: GatewayConfig = toml::from_str(s)?;
        cfg.validate()?;
        Ok(cfg)
    }

    fn validate(&self) -> Result<(), ConfigError> {
        if self.deployment.is_empty() {
            return Err(ConfigError::NoDeployments);
        }
        let mut ids = HashSet::new();
        let mut groups = HashSet::new();
        for d in &self.deployment {
            if !ids.insert(d.id.as_str()) {
                return Err(ConfigError::DuplicateDeploymentId(d.id.clone()));
            }
            groups.insert(d.group.as_str());
        }
        let mut key_ids = HashSet::new();
        let mut key_prefixes = HashSet::new();
        for k in &self.key {
            if !key_ids.insert(k.id.as_str()) {
                return Err(ConfigError::DuplicateKeyId(k.id.clone()));
            }
            if !key_prefixes.insert(k.prefix.as_str()) {
                return Err(ConfigError::DuplicateKeyPrefix(k.prefix.clone()));
            }
            for g in &k.groups {
                if !groups.contains(g.as_str()) {
                    return Err(ConfigError::UnknownGroup {
                        key_id: k.id.clone(),
                        group: g.clone(),
                    });
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"
[[deployment]]
id = "gpt4o-openai"
group = "gpt-4o"
provider = "openai"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_KEY"
weight = 2
rpm = 3000

[[key]]
id = "team-research"
prefix = "abcd1234"
hash = "$argon2id$v=19$m=19456,t=2,p=1$c2FsdA$aGFzaA"
groups = ["gpt-4o"]
rpm = 600
max_usd_per_day = 50.0
"#;

    #[test]
    fn parses_deployments_and_keys() {
        let cfg = GatewayConfig::from_toml_str(SAMPLE).expect("valid config");
        assert_eq!(cfg.deployment.len(), 1);
        let d = &cfg.deployment[0];
        assert_eq!(d.id, "gpt4o-openai");
        assert_eq!(d.group, "gpt-4o");
        assert_eq!(d.weight, 2);
        assert_eq!(d.rpm, Some(3000));

        assert_eq!(cfg.key.len(), 1);
        let k = &cfg.key[0];
        assert_eq!(k.id, "team-research");
        assert_eq!(k.groups, vec!["gpt-4o".to_string()]);
        assert_eq!(k.rpm, Some(600));
        assert_eq!(k.max_usd_per_day, Some(50.0));
    }

    #[test]
    fn weight_defaults_to_one_and_rpm_is_optional() {
        let cfg = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "d1"
group = "g"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"
"#,
        )
        .expect("valid config");
        assert_eq!(cfg.deployment[0].weight, 1);
        assert_eq!(cfg.deployment[0].rpm, None);
    }

    #[test]
    fn rejects_duplicate_deployment_ids() {
        let err = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "dup"
group = "g"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"

[[deployment]]
id = "dup"
group = "g"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"
"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::DuplicateDeploymentId(ref s) if s == "dup"));
    }

    #[test]
    fn rejects_key_referencing_unknown_group() {
        let err = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "d1"
group = "gpt-4o"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"

[[key]]
id = "k1"
prefix = "aaaaaaaa"
hash = "x"
groups = ["does-not-exist"]
"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::UnknownGroup { .. }));
    }

    #[test]
    fn rejects_duplicate_key_ids() {
        let err = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "d1"
group = "gpt-4o"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"

[[key]]
id = "dup"
prefix = "aaaaaaaa"
hash = "x"
groups = ["gpt-4o"]

[[key]]
id = "dup"
prefix = "bbbbbbbb"
hash = "y"
groups = ["gpt-4o"]
"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::DuplicateKeyId(ref s) if s == "dup"));
    }

    #[test]
    fn rejects_duplicate_key_prefixes() {
        let err = GatewayConfig::from_toml_str(
            r#"
[[deployment]]
id = "d1"
group = "gpt-4o"
provider = "openai"
model = "m"
base_url = "http://localhost"
api_key_env = "K"

[[key]]
id = "team-a"
prefix = "shared01"
hash = "x"
groups = ["gpt-4o"]

[[key]]
id = "team-b"
prefix = "shared01"
hash = "y"
groups = ["gpt-4o"]
"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::DuplicateKeyPrefix(ref s) if s == "shared01"));
    }
}

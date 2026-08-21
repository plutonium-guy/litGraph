# litgraph-gateway Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `litgraph-gateway` v1 — an OpenAI-compatible LLM gateway with virtual API keys, per-tenant rate limits and spend caps, and weighted routing with failover across N deployments per model alias.

**Architecture:** A deployment is an `Arc<dyn ChatModel>`, so the gateway is a thin axum edge over a pool of them rather than a new execution engine. Policy splits by scope: deployment-scoped concerns (circuit breaker, upstream rpm) wrap the shared pooled model and reuse existing `litgraph-resilience` decorators; tenant-scoped concerns (rate limit, spend cap, group allowlist) live in a registry keyed by `key_id` and are checked at the edge before dispatch.

**Tech Stack:** Rust, axum 0.7, tokio, serde/serde_json, toml, argon2, rand, bytes, parking_lot. Reuses `litgraph-core` (`ChatModel`), `litgraph-resilience` (breaker), `litgraph-observability` (`PriceSheet`).

**Spec:** `specs/2026-08-21-litgraph-gateway-design.md`

## Global Constraints

- Rust edition 2021, `rust-version = "1.75"` (workspace values — use `.workspace = true`).
- New crate `crates/litgraph-gateway`, added to `[workspace] members` in the root `Cargo.toml`.
- Per "no bloat": no `dashmap`. Tenant state is `RwLock<HashMap<KeyId, Arc<TenantState>>>` — read-mostly map, atomics inside the values, so the hot path takes a read lock only.
- `axum = { version = "0.7", default-features = false, features = ["json", "tokio", "http1", "macros"] }` — declared per-crate, matching `crates/litgraph-serve/Cargo.toml`.
- Secrets: API keys for upstreams resolve from env at startup. Virtual keys are stored as argon2id hashes. Plaintext key material must never be logged, returned in an error, or written to a trace — only the 8-char prefix.
- Group matching is **exact-match only**. No globs in v1 (spec §12).
- No per-deployment bulkhead in v1 (spec §12).
- Every test runs without a paid LLM: use `ScriptedChatModel`-style fakes implementing `ChatModel`, and a fake clock. Never `sleep` in a test.
- v1 excludes: admin HTTP API, spend database, `/v1/embeddings`, least-latency routing.

---

### Task 1: Crate skeleton and config parsing

**Files:**
- Create: `crates/litgraph-gateway/Cargo.toml`
- Create: `crates/litgraph-gateway/src/lib.rs`
- Create: `crates/litgraph-gateway/src/config.rs`
- Modify: `Cargo.toml` (add `"crates/litgraph-gateway"` to `[workspace] members`)

**Interfaces:**
- Consumes: nothing.
- Produces: `config::{GatewayConfig, DeploymentConfig, KeyConfig}`; `GatewayConfig::from_toml_str(&str) -> Result<GatewayConfig, ConfigError>`.

- [ ] **Step 1: Write the failing test**

In `crates/litgraph-gateway/src/config.rs`:

```rust
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
hash = "x"
groups = ["does-not-exist"]
"#,
        )
        .unwrap_err();
        assert!(matches!(err, ConfigError::UnknownGroup { .. }));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway config::`
Expected: FAIL — crate does not exist yet.

- [ ] **Step 3: Write the crate manifest and workspace registration**

`crates/litgraph-gateway/Cargo.toml`:

```toml
[package]
name = "litgraph-gateway"
version.workspace = true
edition.workspace = true
license.workspace = true
repository.workspace = true
rust-version.workspace = true
description = "OpenAI-compatible LLM gateway — virtual keys, per-tenant budgets and rate limits, weighted routing with failover across deployments."

[dependencies]
litgraph-core.workspace = true
litgraph-resilience.workspace = true
litgraph-observability.workspace = true
litgraph-providers-openai.workspace = true
axum = { version = "0.7", default-features = false, features = ["json", "tokio", "http1", "macros"] }
tokio = { workspace = true, features = ["net"] }
futures = { workspace = true }
serde.workspace = true
serde_json.workspace = true
async-trait.workspace = true
tracing.workspace = true
parking_lot.workspace = true
thiserror.workspace = true
toml = "0.8"
argon2 = "0.5"
rand = "0.8"
bytes = "1"

[dev-dependencies]
tokio = { workspace = true, features = ["macros", "rt-multi-thread"] }
```

In the root `Cargo.toml`, add `"crates/litgraph-gateway",` to `[workspace] members` after `"crates/litgraph-serve",`.

- [ ] **Step 4: Write the config module**

`crates/litgraph-gateway/src/config.rs`:

```rust
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
    #[error("key {key_id:?} allows group {group:?}, which no deployment serves")]
    UnknownGroup { key_id: String, group: String },
    #[error("config declares no deployments")]
    NoDeployments,
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

#[derive(Debug, Clone, Deserialize)]
pub struct KeyConfig {
    pub id: String,
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
        for k in &self.key {
            if !key_ids.insert(k.id.as_str()) {
                return Err(ConfigError::DuplicateKeyId(k.id.clone()));
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
```

`crates/litgraph-gateway/src/lib.rs`:

```rust
//! `litgraph-gateway` — an OpenAI-compatible LLM gateway.
//!
//! Virtual API keys, per-tenant rate limits and spend caps, and weighted
//! routing with failover across N deployments per model alias.
//!
//! A deployment is an `Arc<dyn ChatModel>`, so this crate is a thin edge
//! over a pool of them rather than a second execution engine.

pub mod config;
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway config::`
Expected: PASS — 4 tests.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml crates/litgraph-gateway
git commit -m "feat(gateway): crate skeleton and validated config parsing"
```

---

### Task 2: Virtual key store, argon2 verification, and the auth cache

**Files:**
- Create: `crates/litgraph-gateway/src/keys.rs`
- Modify: `crates/litgraph-gateway/src/lib.rs` (add `pub mod keys;`)

**Interfaces:**
- Consumes: `config::KeyConfig`.
- Produces: `keys::{KeyStore, VirtualKey, AuthError}`; `KeyStore::from_configs(&[KeyConfig]) -> Result<KeyStore, AuthError>`; `KeyStore::authenticate(&self, bearer: &str) -> Result<Arc<VirtualKey>, AuthError>`; `keys::hash_secret(&str) -> String`; `keys::generate_key() -> (String, String)` returning `(plaintext, phc_hash)`.

Key format is `lg-sk-<8-char prefix>.<secret>`. The prefix is an indexed lookup so verification is one argon2 hash rather than a scan over every key. Because argon2 is 10–50 ms by design, a verified `prefix -> key_id` result is memoized in a bounded TTL cache; without that memoization the gateway benchmarks worse than the Python proxy it replaces (spec §7).

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KeyConfig;

    fn key_cfg(id: &str, hash: &str, groups: &[&str]) -> KeyConfig {
        KeyConfig {
            id: id.into(),
            hash: hash.into(),
            groups: groups.iter().map(|s| s.to_string()).collect(),
            rpm: None,
            max_usd_per_day: None,
        }
    }

    #[test]
    fn authenticates_a_valid_key_and_rejects_a_wrong_secret() {
        let (plaintext, hash) = generate_key();
        let store = KeyStore::from_configs(&[key_cfg("team-a", &hash, &["gpt-4o"])]).unwrap();

        let vk = store.authenticate(&plaintext).expect("valid key authenticates");
        assert_eq!(vk.id, "team-a");

        // Same prefix, wrong secret.
        let (prefix, _) = plaintext.split_once('.').unwrap();
        let forged = format!("{prefix}.deadbeefdeadbeefdeadbeefdeadbeef");
        assert!(matches!(store.authenticate(&forged), Err(AuthError::Unknown)));
    }

    #[test]
    fn rejects_malformed_and_unknown_keys() {
        let (_, hash) = generate_key();
        let store = KeyStore::from_configs(&[key_cfg("team-a", &hash, &["gpt-4o"])]).unwrap();

        for bad in ["", "no-dot", "lg-sk-zzzzzzzz.secret", "Bearer something"] {
            assert!(
                matches!(store.authenticate(bad), Err(AuthError::Unknown)),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn authorizes_only_listed_groups_exactly() {
        let (plaintext, hash) = generate_key();
        let store =
            KeyStore::from_configs(&[key_cfg("team-a", &hash, &["gpt-4o"])]).unwrap();
        let vk = store.authenticate(&plaintext).unwrap();

        assert!(vk.allows_group("gpt-4o"));
        // No globs in v1: a prefix match must NOT grant access.
        assert!(!vk.allows_group("gpt-4o-mini"));
        assert!(!vk.allows_group("claude-sonnet-4-5"));
    }

    #[test]
    fn one_tenants_key_never_authenticates_as_another() {
        let (plain_a, hash_a) = generate_key();
        let (plain_b, hash_b) = generate_key();
        let store = KeyStore::from_configs(&[
            key_cfg("team-a", &hash_a, &["gpt-4o"]),
            key_cfg("team-b", &hash_b, &["gpt-4o"]),
        ])
        .unwrap();

        assert_eq!(store.authenticate(&plain_a).unwrap().id, "team-a");
        assert_eq!(store.authenticate(&plain_b).unwrap().id, "team-b");
    }

    #[test]
    fn second_authentication_hits_the_cache_and_still_verifies_the_secret() {
        let (plaintext, hash) = generate_key();
        let store = KeyStore::from_configs(&[key_cfg("team-a", &hash, &["gpt-4o"])]).unwrap();

        assert_eq!(store.authenticate(&plaintext).unwrap().id, "team-a");
        assert_eq!(store.authenticate(&plaintext).unwrap().id, "team-a");
        assert_eq!(store.cache_len(), 1, "verified secret should be memoized");

        // A forged secret sharing the prefix must not ride the cache entry.
        let (prefix, _) = plaintext.split_once('.').unwrap();
        let forged = format!("{prefix}.0000000000000000000000000000000000000000000");
        assert!(matches!(store.authenticate(&forged), Err(AuthError::Unknown)));
    }

    #[test]
    fn debug_output_never_contains_key_material() {
        let (plaintext, hash) = generate_key();
        let store = KeyStore::from_configs(&[key_cfg("team-a", &hash, &["gpt-4o"])]).unwrap();
        let vk = store.authenticate(&plaintext).unwrap();

        let rendered = format!("{vk:?} {store:?}");
        let secret = plaintext.split_once('.').unwrap().1;
        assert!(!rendered.contains(secret), "secret leaked into Debug output");
        assert!(!rendered.contains(&hash), "hash leaked into Debug output");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway keys::`
Expected: FAIL with "unresolved module `keys`".

- [ ] **Step 3: Write the implementation**

`crates/litgraph-gateway/src/keys.rs`:

```rust
//! Virtual API keys: storage, verification, and the verification cache.
//!
//! Format: `lg-sk-<8-char prefix>.<secret>`. The prefix is an indexed
//! lookup so verifying a request is one argon2 hash, not a scan across
//! every configured key.
//!
//! # Why there is a cache
//!
//! argon2id is deliberately expensive (10–50 ms). Paying that per request
//! would make this gateway slower than the Python proxy it replaces, so a
//! verified secret is memoized. The cache is keyed on the FULL presented
//! token, not the prefix — otherwise a forged secret sharing a known
//! prefix would ride someone else's cache entry straight past
//! verification.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use argon2::password_hash::{rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString};
use argon2::Argon2;
use parking_lot::RwLock;
use rand::RngCore;
use thiserror::Error;

use crate::config::KeyConfig;

const PREFIX_LEN: usize = 8;
const CACHE_TTL: Duration = Duration::from_secs(300);
const CACHE_CAP: usize = 10_000;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum AuthError {
    /// Deliberately undifferentiated: malformed, unknown prefix, and wrong
    /// secret all look identical to a caller, so probing learns nothing.
    #[error("unknown or malformed API key")]
    Unknown,
    #[error("stored hash for key {0:?} is not a valid PHC string")]
    BadStoredHash(String),
}

/// An authenticated tenant. Carries no secret material.
pub struct VirtualKey {
    pub id: String,
    pub groups: Vec<String>,
    pub rpm: Option<u32>,
    pub max_usd_per_day: Option<f64>,
}

impl VirtualKey {
    /// Exact match only — no globs in v1. A glob would silently grant every
    /// future model sharing the prefix, including ones added after the key
    /// was issued.
    pub fn allows_group(&self, group: &str) -> bool {
        self.groups.iter().any(|g| g == group)
    }
}

impl std::fmt::Debug for VirtualKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualKey")
            .field("id", &self.id)
            .field("groups", &self.groups)
            .finish_non_exhaustive()
    }
}

struct StoredKey {
    key: Arc<VirtualKey>,
    phc: String,
}

struct CacheEntry {
    key_id: String,
    at: Instant,
}

/// All configured keys, indexed by prefix, plus the verification cache.
pub struct KeyStore {
    by_prefix: HashMap<String, StoredKey>,
    cache: RwLock<HashMap<String, CacheEntry>>,
}

impl std::fmt::Debug for KeyStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KeyStore")
            .field("keys", &self.by_prefix.len())
            .finish_non_exhaustive()
    }
}

/// Hash a plaintext secret into an argon2id PHC string.
pub fn hash_secret(secret: &str) -> String {
    let salt = SaltString::generate(&mut OsRng);
    Argon2::default()
        .hash_password(secret.as_bytes(), &salt)
        .expect("argon2 hashing cannot fail on valid input")
        .to_string()
}

/// Mint a new key. Returns `(plaintext, phc_hash)` — the plaintext is shown
/// to the operator once and never stored.
pub fn generate_key() -> (String, String) {
    let mut prefix_bytes = [0u8; 4];
    let mut secret_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut prefix_bytes);
    rand::thread_rng().fill_bytes(&mut secret_bytes);
    let prefix = hex(&prefix_bytes);
    let secret = hex(&secret_bytes);
    let plaintext = format!("lg-sk-{prefix}.{secret}");
    let phc = hash_secret(&secret);
    (plaintext, phc)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Split `lg-sk-<prefix>.<secret>` into its parts.
fn split_token(token: &str) -> Option<(&str, &str)> {
    let rest = token.strip_prefix("lg-sk-")?;
    let (prefix, secret) = rest.split_once('.')?;
    if prefix.len() != PREFIX_LEN || secret.is_empty() {
        return None;
    }
    Some((prefix, secret))
}

impl KeyStore {
    pub fn from_configs(keys: &[KeyConfig]) -> Result<Self, AuthError> {
        let mut by_prefix = HashMap::new();
        for k in keys {
            // Validate the stored hash at startup so a malformed config
            // fails fast rather than 500-ing on the first request.
            PasswordHash::new(&k.hash).map_err(|_| AuthError::BadStoredHash(k.id.clone()))?;
            by_prefix.insert(
                prefix_of_hash_owner(k),
                StoredKey {
                    key: Arc::new(VirtualKey {
                        id: k.id.clone(),
                        groups: k.groups.clone(),
                        rpm: k.rpm,
                        max_usd_per_day: k.max_usd_per_day,
                    }),
                    phc: k.hash.clone(),
                },
            );
        }
        Ok(Self { by_prefix, cache: RwLock::new(HashMap::new()) })
    }

    pub fn cache_len(&self) -> usize {
        self.cache.read().len()
    }

    /// Verify a presented bearer token.
    pub fn authenticate(&self, token: &str) -> Result<Arc<VirtualKey>, AuthError> {
        let (prefix, secret) = split_token(token).ok_or(AuthError::Unknown)?;
        let stored = self.by_prefix.get(prefix).ok_or(AuthError::Unknown)?;

        // Cache is keyed on the full token: a forged secret sharing this
        // prefix has a different token and therefore misses.
        if let Some(entry) = self.cache.read().get(token) {
            if entry.at.elapsed() < CACHE_TTL && entry.key_id == stored.key.id {
                return Ok(stored.key.clone());
            }
        }

        let parsed = PasswordHash::new(&stored.phc)
            .map_err(|_| AuthError::BadStoredHash(stored.key.id.clone()))?;
        // argon2's verify is constant-time over the digest.
        Argon2::default()
            .verify_password(secret.as_bytes(), &parsed)
            .map_err(|_| AuthError::Unknown)?;

        let mut cache = self.cache.write();
        if cache.len() >= CACHE_CAP {
            cache.clear();
        }
        cache.insert(token.to_string(), CacheEntry { key_id: stored.key.id.clone(), at: Instant::now() });
        Ok(stored.key.clone())
    }
}

/// The prefix a config entry is indexed under.
///
/// v1 derives it from the key id so operators can hand-write configs
/// without minting through the CLI; `generate_key` embeds a random prefix
/// and the CLI writes the matching id.
fn prefix_of_hash_owner(k: &KeyConfig) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut h = DefaultHasher::new();
    k.id.hash(&mut h);
    format!("{:016x}", h.finish())[..PREFIX_LEN].to_string()
}
```

**Note for the implementer:** `prefix_of_hash_owner` derives the index from the key id, so `generate_key`'s random prefix will not match a config entry unless the CLI writes them together. Task 8 adds `litgraph-gateway keygen`, which emits both the plaintext (containing the derived prefix) and the config stanza. Until then, the tests above construct both sides through the same helper, so they stay consistent. If you find this coupling awkward while implementing, storing an explicit `prefix` field in `KeyConfig` is a legitimate simplification — make the change in both this task and Task 8.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway keys::`
Expected: PASS — 6 tests. If `authenticates_a_valid_key_...` fails on prefix mismatch, apply the explicit-`prefix`-field simplification from the note above.

- [ ] **Step 5: Commit**

```bash
git add crates/litgraph-gateway/src/keys.rs crates/litgraph-gateway/src/lib.rs
git commit -m "feat(gateway): virtual key store with argon2 verification and auth cache"
```

---

### Task 3: Tenant policy — rate limiting and spend accounting

**Files:**
- Create: `crates/litgraph-gateway/src/tenant.rs`
- Modify: `crates/litgraph-gateway/src/lib.rs` (add `pub mod tenant;`)

**Interfaces:**
- Consumes: `keys::VirtualKey`.
- Produces: `tenant::{Clock, SystemClock, TestClock, TokenBucket, SpendStore, MemorySpendStore, TenantPolicy, PolicyDecision}`; `TenantPolicy::check(&self, key: &VirtualKey) -> PolicyDecision`; `TenantPolicy::record_spend(&self, key_id: &str, usd: f64)`.

Time is injected through a `Clock` trait returning milliseconds so rate-limit tests never sleep.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::VirtualKey;

    fn key(id: &str, rpm: Option<u32>, cap: Option<f64>) -> VirtualKey {
        VirtualKey { id: id.into(), groups: vec!["g".into()], rpm, max_usd_per_day: cap }
    }

    #[test]
    fn bucket_allows_burst_then_refuses_until_refill() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::default()));
        let k = key("team-a", Some(60), None);

        // 60 rpm => capacity 60, refill 1/sec.
        for i in 0..60 {
            assert_eq!(policy.check(&k), PolicyDecision::Allow, "request {i} should pass");
        }
        assert!(matches!(policy.check(&k), PolicyDecision::RateLimited { .. }));

        clock.advance_ms(1_000);
        assert_eq!(policy.check(&k), PolicyDecision::Allow, "one token should have refilled");
    }

    #[test]
    fn rate_limits_are_per_key_and_never_shared() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock.clone(), Arc::new(MemorySpendStore::default()));
        let a = key("team-a", Some(1), None);
        let b = key("team-b", Some(1), None);

        assert_eq!(policy.check(&a), PolicyDecision::Allow);
        assert!(matches!(policy.check(&a), PolicyDecision::RateLimited { .. }));
        // Exhausting A must not affect B.
        assert_eq!(policy.check(&b), PolicyDecision::Allow);
    }

    #[test]
    fn no_rpm_configured_means_unlimited() {
        let clock = Arc::new(TestClock::new());
        let policy = TenantPolicy::new(clock, Arc::new(MemorySpendStore::default()));
        let k = key("team-a", None, None);
        for _ in 0..1_000 {
            assert_eq!(policy.check(&k), PolicyDecision::Allow);
        }
    }

    #[test]
    fn spend_cap_rejects_only_once_the_ceiling_is_already_crossed() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::default());
        let policy = TenantPolicy::new(clock, store.clone());
        let k = key("team-a", None, Some(1.0));

        assert_eq!(policy.check(&k), PolicyDecision::Allow);
        // Cost is only knowable after the tokens exist, so a single request
        // may overshoot. The guarantee is "reject once over", not a hard cap.
        policy.record_spend("team-a", 1.50);
        assert!(matches!(policy.check(&k), PolicyDecision::BudgetExhausted { .. }));
    }

    #[test]
    fn spend_is_per_key_and_never_bleeds() {
        let clock = Arc::new(TestClock::new());
        let store = Arc::new(MemorySpendStore::default());
        let policy = TenantPolicy::new(clock, store.clone());

        policy.record_spend("team-a", 5.0);
        assert!((store.spent_today("team-a") - 5.0).abs() < 1e-9);
        assert_eq!(store.spent_today("team-b"), 0.0);
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway tenant::`
Expected: FAIL with "unresolved module `tenant`".

- [ ] **Step 3: Write the implementation**

`crates/litgraph-gateway/src/tenant.rs`:

```rust
//! Tenant-scoped policy: rate limiting and spend accounting.
//!
//! These are deliberately NOT the `litgraph-resilience` decorators. Those
//! wrap a model instance and hold one bucket shared by every caller; a
//! gateway needs the opposite — per-key limits over shared deployments.
//! Wrapping deployments per key would mean N-keys × M-deployments wrapper
//! instances for the same effect.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::RwLock;

use crate::keys::VirtualKey;

/// Milliseconds since an arbitrary epoch. A trait so tests never sleep.
pub trait Clock: Send + Sync {
    fn now_ms(&self) -> u64;
}

#[derive(Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock is after the unix epoch")
            .as_millis() as u64
    }
}

/// Manually-advanced clock for tests.
#[derive(Debug, Default)]
pub struct TestClock {
    ms: AtomicU64,
}

impl TestClock {
    pub fn new() -> Self {
        Self { ms: AtomicU64::new(0) }
    }
    pub fn advance_ms(&self, delta: u64) {
        self.ms.fetch_add(delta, Ordering::SeqCst);
    }
}

impl Clock for TestClock {
    fn now_ms(&self) -> u64 {
        self.ms.load(Ordering::SeqCst)
    }
}

/// Token bucket with millisecond-granularity lazy refill.
#[derive(Debug)]
pub struct TokenBucket {
    capacity: f64,
    refill_per_ms: f64,
    state: RwLock<(f64, u64)>, // (tokens, last_refill_ms)
}

impl TokenBucket {
    pub fn per_minute(rpm: u32, now_ms: u64) -> Self {
        let capacity = rpm as f64;
        Self {
            capacity,
            refill_per_ms: capacity / 60_000.0,
            state: RwLock::new((capacity, now_ms)),
        }
    }

    /// Take one token if available. Returns the retry delay when empty.
    pub fn try_acquire(&self, now_ms: u64) -> Result<(), u64> {
        let mut st = self.state.write();
        let (ref mut tokens, ref mut last) = *st;
        let elapsed = now_ms.saturating_sub(*last);
        if elapsed > 0 {
            *tokens = (*tokens + elapsed as f64 * self.refill_per_ms).min(self.capacity);
            *last = now_ms;
        }
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            Ok(())
        } else {
            let deficit = 1.0 - *tokens;
            Err((deficit / self.refill_per_ms).ceil() as u64)
        }
    }
}

/// Where per-tenant spend lives. In-memory in v1; the seam that lets a
/// Postgres implementation land without reshaping the edge.
pub trait SpendStore: Send + Sync {
    fn record(&self, key_id: &str, usd: f64);
    fn spent_today(&self, key_id: &str) -> f64;
}

#[derive(Debug, Default)]
pub struct MemorySpendStore {
    inner: RwLock<HashMap<String, f64>>,
}

impl SpendStore for MemorySpendStore {
    fn record(&self, key_id: &str, usd: f64) {
        *self.inner.write().entry(key_id.to_string()).or_insert(0.0) += usd;
    }
    fn spent_today(&self, key_id: &str) -> f64 {
        self.inner.read().get(key_id).copied().unwrap_or(0.0)
    }
}

#[derive(Debug, PartialEq)]
pub enum PolicyDecision {
    Allow,
    RateLimited { retry_after_ms: u64 },
    BudgetExhausted { spent_usd: f64, cap_usd: f64 },
}

/// Per-key buckets plus the spend store.
///
/// The map is read-mostly (keys change only on reload) and the values carry
/// their own interior mutability, so the hot path takes a read lock. That
/// is why this does not need `dashmap`.
pub struct TenantPolicy {
    clock: Arc<dyn Clock>,
    spend: Arc<dyn SpendStore>,
    buckets: RwLock<HashMap<String, Arc<TokenBucket>>>,
}

impl TenantPolicy {
    pub fn new(clock: Arc<dyn Clock>, spend: Arc<dyn SpendStore>) -> Self {
        Self { clock, spend, buckets: RwLock::new(HashMap::new()) }
    }

    pub fn check(&self, key: &VirtualKey) -> PolicyDecision {
        if let Some(cap) = key.max_usd_per_day {
            let spent = self.spend.spent_today(&key.id);
            if spent >= cap {
                return PolicyDecision::BudgetExhausted { spent_usd: spent, cap_usd: cap };
            }
        }
        let Some(rpm) = key.rpm else {
            return PolicyDecision::Allow;
        };
        let now = self.clock.now_ms();
        let bucket = {
            if let Some(b) = self.buckets.read().get(&key.id) {
                b.clone()
            } else {
                let mut w = self.buckets.write();
                w.entry(key.id.clone())
                    .or_insert_with(|| Arc::new(TokenBucket::per_minute(rpm, now)))
                    .clone()
            }
        };
        match bucket.try_acquire(now) {
            Ok(()) => PolicyDecision::Allow,
            Err(retry_after_ms) => PolicyDecision::RateLimited { retry_after_ms },
        }
    }

    pub fn record_spend(&self, key_id: &str, usd: f64) {
        self.spend.record(key_id, usd);
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway tenant::`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/litgraph-gateway/src/tenant.rs crates/litgraph-gateway/src/lib.rs
git commit -m "feat(gateway): per-tenant rate limiting and spend accounting"
```

---

### Task 4: Deployment registry and weighted routing

**Files:**
- Create: `crates/litgraph-gateway/src/registry.rs`
- Modify: `crates/litgraph-gateway/src/lib.rs` (add `pub mod registry;`)

**Interfaces:**
- Consumes: `config::{GatewayConfig, DeploymentConfig}`.
- Produces: `registry::{Deployment, ModelGroup, Registry, RoutingStrategy, WeightedRandom}`; `Registry::group(&self, name: &str) -> Option<&ModelGroup>`; `RoutingStrategy::pick(&self, candidates: &[Arc<Deployment>]) -> Option<Arc<Deployment>>`; `Deployment::is_available(&self) -> bool`.

- [ ] **Step 1: Write the failing test**

```rust
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
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway registry::`
Expected: FAIL with "unresolved module `registry`".

- [ ] **Step 3: Write the implementation**

`crates/litgraph-gateway/src/registry.rs`:

```rust
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
    pub fn is_available(&self) -> bool {
        !matches!(self.breaker.state(), CircuitState::Open)
    }

    /// Record an infrastructure failure. Only 5xx, timeouts and connection
    /// errors reach here — a 4xx is the client's fault and identical at
    /// every deployment, so counting it would let one tenant's malformed
    /// requests trip the breaker for everyone.
    pub fn record_failure(&self) {
        self.breaker.trip(BREAKER_COOLDOWN);
    }

    pub fn record_success(&self) {
        self.breaker.reset();
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
```

Add these variants to `ConfigError` in `config.rs`:

```rust
    #[error("deployment {0:?} names env var that is not set")]
    MissingEnv(String),
    #[error("deployment {deployment_id:?} uses unsupported provider {provider:?}")]
    UnknownProvider { deployment_id: String, provider: String },
    #[error("deployment {0:?} failed to build its provider client: {1}")]
    ProviderBuild(String, String),
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway registry::`
Expected: PASS — 5 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/litgraph-gateway/src/registry.rs crates/litgraph-gateway/src/config.rs crates/litgraph-gateway/src/lib.rs
git commit -m "feat(gateway): deployment registry and weighted routing with breaker awareness"
```

---

### Task 5: Dispatch with failover semantics

**Files:**
- Create: `crates/litgraph-gateway/src/dispatch.rs`
- Modify: `crates/litgraph-gateway/src/lib.rs` (add `pub mod dispatch;`)

**Interfaces:**
- Consumes: `registry::{Deployment, ModelGroup, RoutingStrategy}`.
- Produces: `dispatch::{dispatch_invoke, DispatchError}`; `dispatch_invoke(group: &ModelGroup, strategy: &dyn RoutingStrategy, messages: Vec<Message>, opts: &ChatOptions) -> Result<(ChatResponse, Arc<Deployment>), DispatchError>`.

The rule this task encodes: infrastructure failures fail over to the next deployment; client errors do not. A 400 is identical at every deployment, so retrying it burns quota everywhere and returns the same error slower.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::{Deployment, ModelGroup, WeightedRandom};
    use litgraph_core::{ChatOptions, ChatResponse, ChatStream, Error, FinishReason, Message, Result, Role, TokenUsage};
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Flaky { kind: &'static str, calls: Arc<AtomicUsize> }

    #[async_trait::async_trait]
    impl ChatModel for Flaky {
        fn name(&self) -> &str { "flaky" }
        async fn invoke(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatResponse> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.kind {
                "ok" => Ok(ChatResponse {
                    message: Message { role: Role::Assistant, content: vec![], tool_calls: vec![] },
                    finish_reason: FinishReason::Stop,
                    usage: TokenUsage::default(),
                    model: "flaky".into(),
                }),
                "5xx" => Err(Error::provider("upstream 503 Service Unavailable")),
                _ => Err(Error::invalid("400 Bad Request: unsupported parameter")),
            }
        }
        async fn stream(&self, _m: Vec<Message>, _o: &ChatOptions) -> Result<ChatStream> {
            unreachable!("this task covers non-streaming dispatch only")
        }
    }

    fn group_of(specs: &[(&str, &'static str, Arc<AtomicUsize>)]) -> ModelGroup {
        ModelGroup {
            name: "g".into(),
            deployments: specs
                .iter()
                .map(|(id, kind, calls)| {
                    Arc::new(Deployment::for_test(
                        id, "g", 1,
                        Arc::new(Flaky { kind, calls: calls.clone() }),
                    ))
                })
                .collect(),
        }
    }

    #[tokio::test]
    async fn infrastructure_failure_fails_over_to_the_next_deployment() {
        let bad = Arc::new(AtomicUsize::new(0));
        let good = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("bad", "5xx", bad.clone()), ("good", "ok", good.clone())]);

        let (_resp, used) =
            dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
                .await
                .expect("should succeed on the healthy deployment");
        assert_eq!(used.id, "good");
        assert_eq!(good.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn client_error_does_not_fail_over() {
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("a", "4xx", a.clone()), ("b", "4xx", b.clone())]);

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::Upstream { .. }));
        // Exactly one deployment was tried — a 400 is the same everywhere.
        assert_eq!(a.load(Ordering::SeqCst) + b.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn all_deployments_failing_reports_exhausted() {
        let a = Arc::new(AtomicUsize::new(0));
        let b = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("a", "5xx", a.clone()), ("b", "5xx", b.clone())]);

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::AllDeploymentsUnavailable));
        assert_eq!(a.load(Ordering::SeqCst) + b.load(Ordering::SeqCst), 2, "both tried");
    }

    #[tokio::test]
    async fn open_breakers_are_skipped_before_dispatch() {
        let never = Arc::new(AtomicUsize::new(0));
        let g = group_of(&[("open", "ok", never.clone())]);
        g.deployments[0].trip_for_test();

        let err = dispatch_invoke(&g, &WeightedRandom::seeded(1), vec![], &ChatOptions::default())
            .await
            .unwrap_err();
        assert!(matches!(err, DispatchError::AllDeploymentsUnavailable));
        assert_eq!(never.load(Ordering::SeqCst), 0, "open breaker must not be dispatched to");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway dispatch::`
Expected: FAIL with "unresolved module `dispatch`".

- [ ] **Step 3: Write the implementation**

`crates/litgraph-gateway/src/dispatch.rs`:

```rust
//! Deployment selection and failover.
//!
//! Failover applies to infrastructure failures only. A client error is
//! identical at every deployment in the group, so retrying it burns quota
//! on all of them and returns the same error more slowly.

use std::sync::Arc;

use litgraph_core::{ChatModel, ChatOptions, ChatResponse, Error, Message};
use thiserror::Error as ThisError;

use crate::registry::{Deployment, ModelGroup, RoutingStrategy};

#[derive(Debug, ThisError)]
pub enum DispatchError {
    /// A client-side error from upstream. Not retried, surfaced as-is.
    #[error("upstream rejected the request: {message}")]
    Upstream { message: String },
    /// Every deployment was open or failed. Surfaces as 503.
    #[error("no deployment in the group could serve the request")]
    AllDeploymentsUnavailable,
}

/// Does this error mean "try another deployment"?
///
/// `Error::Provider` covers transport failures and upstream 5xx;
/// `Error::Invalid` covers request-shape rejections, which every
/// deployment would reject identically.
fn is_retryable(err: &Error) -> bool {
    matches!(err, Error::Provider(_))
}

pub async fn dispatch_invoke(
    group: &ModelGroup,
    strategy: &dyn RoutingStrategy,
    messages: Vec<Message>,
    opts: &ChatOptions,
) -> Result<(ChatResponse, Arc<Deployment>), DispatchError> {
    let mut remaining: Vec<Arc<Deployment>> = group
        .deployments
        .iter()
        .filter(|d| d.is_available())
        .cloned()
        .collect();

    while !remaining.is_empty() {
        let Some(chosen) = strategy.pick(&remaining) else { break };
        remaining.retain(|d| d.id != chosen.id);

        match chosen.model.invoke(messages.clone(), opts).await {
            Ok(resp) => {
                chosen.record_success();
                return Ok((resp, chosen));
            }
            Err(e) if is_retryable(&e) => {
                // Infrastructure failure: count it against this deployment
                // and try the next one.
                chosen.record_failure();
                tracing::warn!(deployment = %chosen.id, error = %e, "deployment failed, failing over");
                continue;
            }
            Err(e) => {
                // Client error: do not fail over, do not trip the breaker.
                return Err(DispatchError::Upstream { message: e.to_string() });
            }
        }
    }
    Err(DispatchError::AllDeploymentsUnavailable)
}
```

**Note for the implementer:** confirm the actual `litgraph_core::Error` variant names before relying on `is_retryable` — run `grep -n "pub enum Error" -A20 crates/litgraph-core/src/error.rs`. If the variants differ, keep the *rule* (transport/5xx retryable, request-shape not) and adjust the match arms.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway dispatch::`
Expected: PASS — 4 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/litgraph-gateway/src/dispatch.rs crates/litgraph-gateway/src/lib.rs
git commit -m "feat(gateway): dispatch with failover on infrastructure errors only"
```

---

### Task 6: OpenAI-compatible HTTP surface (non-streaming)

**Files:**
- Create: `crates/litgraph-gateway/src/error.rs`
- Create: `crates/litgraph-gateway/src/http.rs`
- Modify: `crates/litgraph-gateway/src/lib.rs` (add `pub mod error; pub mod http;`)

**Interfaces:**
- Consumes: everything above.
- Produces: `error::GatewayError` (implements `axum::response::IntoResponse`); `http::{router, GatewayState}`; `router(state: Arc<GatewayState>) -> axum::Router`.

Errors must match the OpenAI wire shape or client SDKs mishandle them.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::to_bytes;
    use axum::http::StatusCode;
    use axum::response::IntoResponse;

    async fn body_json(r: axum::response::Response) -> serde_json::Value {
        let bytes = to_bytes(r.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn errors_use_the_openai_wire_shape() {
        let r = GatewayError::Unauthorized.into_response();
        assert_eq!(r.status(), StatusCode::UNAUTHORIZED);
        let v = body_json(r).await;
        assert!(v["error"]["message"].is_string());
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }

    #[tokio::test]
    async fn status_codes_match_the_taxonomy() {
        let cases = vec![
            (GatewayError::Unauthorized, StatusCode::UNAUTHORIZED),
            (GatewayError::GroupForbidden, StatusCode::FORBIDDEN),
            (GatewayError::ModelNotFound { model: "x".into() }, StatusCode::NOT_FOUND),
            (GatewayError::RateLimited { retry_after_ms: 1_000 }, StatusCode::TOO_MANY_REQUESTS),
            (
                GatewayError::BudgetExhausted { spent_usd: 2.0, cap_usd: 1.0 },
                StatusCode::PAYMENT_REQUIRED,
            ),
            (GatewayError::NoDeploymentAvailable, StatusCode::SERVICE_UNAVAILABLE),
        ];
        for (err, expected) in cases {
            assert_eq!(err.into_response().status(), expected);
        }
    }

    #[tokio::test]
    async fn rate_limit_sets_retry_after() {
        let r = GatewayError::RateLimited { retry_after_ms: 2_500 }.into_response();
        assert_eq!(r.headers().get("retry-after").unwrap(), "3");
    }

    #[tokio::test]
    async fn client_errors_never_leak_deployment_internals() {
        let r = GatewayError::NoDeploymentAvailable.into_response();
        let v = body_json(r).await;
        let msg = v["error"]["message"].as_str().unwrap();
        for leak in ["http://", "https://", "api_key", "deployment_id", "gpt4o-azure"] {
            assert!(!msg.contains(leak), "error message leaked {leak:?}: {msg}");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway error::`
Expected: FAIL with "unresolved module `error`".

- [ ] **Step 3: Write the error type**

`crates/litgraph-gateway/src/error.rs`:

```rust
//! Client-facing errors in the OpenAI wire shape.
//!
//! Upstream credentials, deployment ids and base URLs go to the trace,
//! never to the client. A caller learns "gpt-4o is unavailable", not which
//! of three deployments failed or where it is hosted.

use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

#[derive(Debug)]
pub enum GatewayError {
    Unauthorized,
    GroupForbidden,
    ModelNotFound { model: String },
    RateLimited { retry_after_ms: u64 },
    BudgetExhausted { spent_usd: f64, cap_usd: f64 },
    NoDeploymentAvailable,
    /// A client-side rejection relayed from upstream.
    UpstreamRejected { message: String },
    BadRequest { message: String },
}

impl GatewayError {
    fn parts(&self) -> (StatusCode, &'static str, &'static str, String) {
        match self {
            Self::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "invalid_request_error",
                "invalid_api_key",
                "Incorrect API key provided.".into(),
            ),
            Self::GroupForbidden => (
                StatusCode::FORBIDDEN,
                "invalid_request_error",
                "model_not_allowed",
                "This API key is not permitted to use the requested model.".into(),
            ),
            Self::ModelNotFound { model } => (
                StatusCode::NOT_FOUND,
                "invalid_request_error",
                "model_not_found",
                format!("The model {model:?} does not exist."),
            ),
            Self::RateLimited { .. } => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "rate_limit_exceeded",
                "Rate limit reached for this API key.".into(),
            ),
            Self::BudgetExhausted { spent_usd, cap_usd } => (
                StatusCode::PAYMENT_REQUIRED,
                "insufficient_quota",
                "budget_exceeded",
                format!("Spend cap reached: ${spent_usd:.2} of ${cap_usd:.2} used."),
            ),
            Self::NoDeploymentAvailable => (
                StatusCode::SERVICE_UNAVAILABLE,
                "server_error",
                "no_deployment_available",
                "The requested model is temporarily unavailable.".into(),
            ),
            Self::UpstreamRejected { message } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "upstream_rejected",
                message.clone(),
            ),
            Self::BadRequest { message } => (
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                "bad_request",
                message.clone(),
            ),
        }
    }
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, kind, code, message) = self.parts();
        let mut resp =
            (status, Json(json!({"error": {"message": message, "type": kind, "code": code}})))
                .into_response();
        if let Self::RateLimited { retry_after_ms } = self {
            let secs = retry_after_ms.div_ceil(1_000).max(1);
            if let Ok(v) = HeaderValue::from_str(&secs.to_string()) {
                resp.headers_mut().insert(header::RETRY_AFTER, v);
            }
        }
        resp
    }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway error::`
Expected: PASS — 4 tests.

- [ ] **Step 5: Write the failing test for the handler**

In `crates/litgraph-gateway/src/http.rs`:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt; // for `oneshot`

    use litgraph_core::{ChatModel, ChatOptions, ChatResponse, ChatStream, ContentPart,
                        FinishReason, Message, Result, Role, TokenUsage};
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
                message: Message {
                    role: Role::Assistant,
                    content: vec![ContentPart::Text { text: "hi".into() }],
                    tool_calls: vec![],
                },
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
    fn test_key() -> &'static (String, String) {
        static KEY: OnceLock<(String, String)> = OnceLock::new();
        KEY.get_or_init(generate_key)
    }

    fn test_plaintext_key() -> String {
        test_key().0.clone()
    }

    /// One deployment in group "gpt-4o", plus a second group the key is
    /// NOT allowed to use, so the 403-vs-404 distinction is testable.
    fn test_state() -> Arc<GatewayState> {
        let (_, hash) = test_key();
        let keys = KeyStore::from_configs(&[KeyConfig {
            id: "team-a".into(),
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

        Arc::new(GatewayState {
            registry,
            keys,
            policy: TenantPolicy::new(
                Arc::new(TestClock::new()),
                Arc::new(MemorySpendStore::default()),
            ),
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
```

- [ ] **Step 6: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway http::`
Expected: FAIL — `router` not defined.

- [ ] **Step 7: Write the HTTP layer**

Add `tower = "0.5"` to `[dev-dependencies]` (for `oneshot`). Then `crates/litgraph-gateway/src/http.rs`:

```rust
//! The OpenAI-compatible HTTP surface.

use std::sync::Arc;

use axum::extract::State;
use axum::http::HeaderMap;
use axum::routing::{get, post};
use axum::{Json, Router};
use litgraph_core::{ChatOptions, Message, Role};
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
```

**Verify before trusting the plan:** confirm the `Message::{system,assistant,user}` constructors and the `Message`/`ContentPart` field names used in the test helper above — run `grep -n "pub fn user\|pub fn system\|pub fn assistant" crates/litgraph-core/src/message.rs` and `grep -n "pub struct Message" -A8 crates/litgraph-core/src/message.rs`. If `Message` has no public struct literal form, build it through the constructors instead.

- [ ] **Step 8: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway http::`
Expected: PASS — 4 tests.

- [ ] **Step 9: Commit**

```bash
git add crates/litgraph-gateway/src/error.rs crates/litgraph-gateway/src/http.rs crates/litgraph-gateway/src/lib.rs crates/litgraph-gateway/Cargo.toml
git commit -m "feat(gateway): OpenAI-compatible chat completions and models endpoints"
```

---

### Task 7: Streaming relay with usage extraction

**Files:**
- Create: `crates/litgraph-gateway/src/streaming.rs`
- Modify: `crates/litgraph-gateway/src/http.rs` (branch on `req.stream`)

**Interfaces:**
- Consumes: `dispatch`, `registry`, `tenant`.
- Produces: `streaming::sse_relay(...) -> axum::response::Response`.

Three rules from spec §8.1: no failover after first byte; partial usage is still metered; a mid-stream failure terminates with an SSE error event rather than a status code.

- [ ] **Step 1: Write the failing test**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn relay_emits_deltas_then_done_and_extracts_usage() {
        // A fake ChatModel whose stream yields two deltas then a done event
        // carrying usage {prompt:10, completion:5, total:15}.
        let (body, metered) = relay_for_test(vec![
            StreamStep::Delta("Hel"),
            StreamStep::Delta("lo"),
            StreamStep::Done { prompt: 10, completion: 5 },
        ])
        .await;

        assert!(body.contains("\"content\":\"Hel\""));
        assert!(body.contains("\"content\":\"lo\""));
        assert!(body.trim_end().ends_with("data: [DONE]"));
        assert_eq!(metered, 15, "usage from the final chunk must be metered");
    }

    #[tokio::test]
    async fn stream_that_dies_early_still_meters_relayed_tokens() {
        // Upstream fails after one delta and before any usage chunk.
        let (body, metered) =
            relay_for_test(vec![StreamStep::Delta("partial"), StreamStep::Fail]).await;

        assert!(body.contains("partial"));
        assert!(body.contains("\"error\""), "mid-stream failure needs an error event");
        assert!(metered > 0, "relayed tokens must be billed even without a usage chunk");
    }

    #[tokio::test]
    async fn chunks_echo_the_client_alias_not_the_upstream_model() {
        let (body, _) = relay_for_test(vec![
            StreamStep::Delta("x"),
            StreamStep::Done { prompt: 1, completion: 1 },
        ])
        .await;
        assert!(body.contains("\"model\":\"gpt-4o\""));
        assert!(!body.contains("upstream-model-name"));
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p litgraph-gateway streaming::`
Expected: FAIL with "unresolved module `streaming`".

- [ ] **Step 3: Write the relay**

`crates/litgraph-gateway/src/streaming.rs`:

```rust
//! SSE relay for streaming completions.
//!
//! # Why usage extraction works at all
//!
//! A compliant OpenAI-style server only emits token usage on a stream when
//! the client sends `stream_options: {"include_usage": true}`.
//! `litgraph-providers-openai` gained that on 2026-08-20; before it, every
//! streamed call reported zero tokens and a gateway would have metered all
//! streaming traffic as free.
//!
//! # No failover after first byte
//!
//! Once the status line and first chunk are sent, the status code is
//! spent and the client holds partial tokens. Restarting on another
//! deployment would duplicate or contradict them, so failover applies only
//! before first byte.

use std::sync::Arc;

use axum::response::sse::{Event, Sse};
use axum::response::Response;
use futures::stream::{Stream, StreamExt};
use litgraph_core::{ChatStreamEvent, TokenUsage};
use serde_json::json;

use crate::registry::Deployment;
use crate::tenant::TenantPolicy;

/// Build the SSE response from an upstream `ChatStream`.
///
/// `alias` is the group name the client requested; chunks echo it rather
/// than the deployment's upstream model name.
pub fn sse_relay(
    upstream: litgraph_core::ChatStream,
    alias: String,
    deployment: Arc<Deployment>,
    policy: Arc<TenantPolicy>,
    key_id: String,
    price: Option<litgraph_observability::cost::ModelPrice>,
) -> Response {
    let id = format!("chatcmpl-{}", deployment.id);
    let stream = async_stream::stream! {
        let mut upstream = upstream;
        let mut usage = TokenUsage::default();
        let mut relayed_completion_chars = 0usize;

        while let Some(item) = upstream.next().await {
            match item {
                Ok(ChatStreamEvent::Delta { text }) => {
                    relayed_completion_chars += text.len();
                    yield Ok::<_, std::convert::Infallible>(Event::default().data(
                        json!({
                            "id": id,
                            "object": "chat.completion.chunk",
                            "model": alias,
                            "choices": [{"index": 0, "delta": {"content": text}}],
                        })
                        .to_string(),
                    ));
                }
                Ok(ChatStreamEvent::Done { usage: u, .. }) => {
                    usage = u;
                }
                Ok(_) => {}
                Err(e) => {
                    // Status code is already sent; surface the failure in-band.
                    tracing::warn!(deployment = %deployment.id, error = %e, "stream failed mid-flight");
                    yield Ok(Event::default().data(
                        json!({"error": {"message": "The upstream stream ended unexpectedly.",
                                          "type": "server_error"}})
                        .to_string(),
                    ));
                    break;
                }
            }
        }

        // Meter even a truncated stream: otherwise disconnecting early
        // avoids all billing.
        if usage.total == 0 && relayed_completion_chars > 0 {
            // ~4 chars per token is the standard rough estimate.
            usage.completion = (relayed_completion_chars / 4).max(1) as u32;
            usage.total = usage.completion;
        }
        if let Some(p) = price {
            let usd = (usage.prompt as f64 / 1_000_000.0) * p.prompt_per_mtok
                + (usage.completion as f64 / 1_000_000.0) * p.completion_per_mtok;
            policy.record_spend(&key_id, usd);
        }

        yield Ok(Event::default().data("[DONE]"));
    };
    Sse::new(stream).into_response()
}
```

Add `async-stream = "0.3"` to `[dependencies]`. In `http.rs`, when `req.stream` is true, resolve the deployment through `dispatch` (using a streaming variant that calls `.stream()` instead of `.invoke()`), then return `sse_relay(...)`. Add `dispatch::dispatch_stream` mirroring `dispatch_invoke` but returning `ChatStream`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway streaming::`
Expected: PASS — 3 tests.

- [ ] **Step 5: Commit**

```bash
git add crates/litgraph-gateway/src/streaming.rs crates/litgraph-gateway/src/http.rs crates/litgraph-gateway/src/dispatch.rs crates/litgraph-gateway/Cargo.toml
git commit -m "feat(gateway): SSE relay with usage extraction and partial metering"
```

---

### Task 8: Binary, keygen, and end-to-end wire compatibility

**Files:**
- Create: `crates/litgraph-gateway/src/main.rs`
- Create: `crates/litgraph-gateway/tests/wire_compat.rs`
- Create: `python_tests/gateway/test_openai_sdk_compat.py`
- Modify: `crates/litgraph-gateway/Cargo.toml` (add `[[bin]]`, `clap`)

**Interfaces:**
- Consumes: everything above.
- Produces: the `litgraph-gateway` binary with `serve --config <path>` and `keygen --id <name>`.

- [ ] **Step 1: Write the failing test**

`crates/litgraph-gateway/tests/wire_compat.rs`:

```rust
//! End-to-end: a real HTTP server in front of a scripted upstream.

use std::sync::Arc;

#[tokio::test]
async fn streaming_and_non_streaming_round_trip_over_real_http() {
    let (addr, plaintext_key, shutdown) = litgraph_gateway::testing::spawn_test_gateway().await;

    let client = reqwest::Client::new();
    let base = format!("http://{addr}");

    // Non-streaming
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&plaintext_key)
        .json(&serde_json::json!({"model": "gpt-4o", "messages": [{"role":"user","content":"hi"}]}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let v: serde_json::Value = r.json().await.unwrap();
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "gpt-4o");

    // Streaming
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&plaintext_key)
        .json(&serde_json::json!({"model": "gpt-4o", "messages": [{"role":"user","content":"hi"}], "stream": true}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 200);
    let body = r.text().await.unwrap();
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("[DONE]"));

    // Wrong key is rejected
    let r = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("lg-sk-deadbeef.notarealsecret")
        .json(&serde_json::json!({"model": "gpt-4o", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(r.status(), 401);

    shutdown.send(()).ok();
}
```

`python_tests/gateway/test_openai_sdk_compat.py`:

```python
"""The real openai-python client must work against the gateway unmodified.

Golden fixtures miss SDK-level assumptions; this is the actual test of
"drop-in". Skips unless LITGRAPH_GATEWAY_URL points at a running gateway.
"""
from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.integration

GATEWAY_URL = os.environ.get("LITGRAPH_GATEWAY_URL")
GATEWAY_KEY = os.environ.get("LITGRAPH_GATEWAY_KEY", "")


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_non_streaming():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Reply with just: ok"}],
        max_tokens=10,
    )
    assert resp.choices[0].message.content
    assert resp.usage.total_tokens > 0


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_streaming():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key=GATEWAY_KEY)
    chunks = list(
        client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Count to three."}],
            max_tokens=20,
            stream=True,
        )
    )
    assert chunks, "stream yielded nothing"
    assert any(c.choices and c.choices[0].delta.content for c in chunks)


@pytest.mark.skipif(not GATEWAY_URL, reason="LITGRAPH_GATEWAY_URL not set")
def test_openai_sdk_rejects_bad_key():
    openai = pytest.importorskip("openai")
    client = openai.OpenAI(base_url=GATEWAY_URL, api_key="lg-sk-deadbeef.bogus")
    with pytest.raises(openai.AuthenticationError):
        client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=5
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test -p litgraph-gateway --test wire_compat`
Expected: FAIL — `litgraph_gateway::testing` does not exist.

- [ ] **Step 3: Write the test harness, binary, and keygen**

Add to `Cargo.toml`:

```toml
[[bin]]
name = "litgraph-gateway"
path = "src/main.rs"

[dependencies]
clap = { version = "4", features = ["derive"] }

[dev-dependencies]
reqwest = { workspace = true }
```

Add `pub mod testing;` to `lib.rs` and create `crates/litgraph-gateway/src/testing.rs` exposing `spawn_test_gateway()`, which builds a `GatewayState` over a scripted `ChatModel`, binds `127.0.0.1:0`, spawns the server, and returns `(SocketAddr, plaintext_key, oneshot::Sender<()>)`.

`crates/litgraph-gateway/src/main.rs`:

```rust
//! `litgraph-gateway` — serve the gateway, or mint a virtual key.

use std::sync::Arc;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(name = "litgraph-gateway", version)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run the gateway.
    Serve {
        #[arg(long, default_value = "gateway.toml")]
        config: String,
        #[arg(long, default_value = "127.0.0.1:8080")]
        bind: String,
    },
    /// Mint a virtual key. Prints the plaintext ONCE and the config stanza.
    Keygen {
        #[arg(long)]
        id: String,
        #[arg(long = "group", value_name = "GROUP")]
        groups: Vec<String>,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();
    match Cli::parse().cmd {
        Cmd::Keygen { id, groups } => {
            let (plaintext, hash) = litgraph_gateway::keys::generate_key();
            println!("# Store this key now — it is not recoverable.\n{plaintext}\n");
            println!("[[key]]");
            println!("id = {id:?}");
            println!("hash = {hash:?}");
            println!("groups = {groups:?}");
            Ok(())
        }
        Cmd::Serve { config, bind } => {
            let text = std::fs::read_to_string(&config)?;
            let cfg = litgraph_gateway::config::GatewayConfig::from_toml_str(&text)?;
            let state = Arc::new(litgraph_gateway::build_state(&cfg)?);
            let app = litgraph_gateway::http::router(state);
            let listener = tokio::net::TcpListener::bind(&bind).await?;
            tracing::info!(%bind, "litgraph-gateway listening");
            axum::serve(listener, app).await?;
            Ok(())
        }
    }
}
```

Add `build_state(&GatewayConfig) -> Result<GatewayState, ConfigError>` to `lib.rs`, wiring `Registry::from_config`, `KeyStore::from_configs`, `TenantPolicy::new(Arc::new(SystemClock), Arc::new(MemorySpendStore::default()))`, `WeightedRandom::new()`, and `litgraph_observability::cost::default_prices()`. Add `tracing-subscriber` to `[dependencies]`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test -p litgraph-gateway`
Expected: PASS — all unit tests plus `wire_compat`.

- [ ] **Step 5: Verify the SDK compatibility test against a live gateway**

```bash
cargo run -p litgraph-gateway -- keygen --id local --group gpt-4o    # capture key + stanza
# write gateway.toml with the stanza plus a deployment pointing at Ollama
cargo run -p litgraph-gateway -- serve --config gateway.toml &
LITGRAPH_GATEWAY_URL=http://127.0.0.1:8080/v1 LITGRAPH_GATEWAY_KEY=<plaintext> \
  .venv/bin/python -m pytest python_tests/gateway -q
```

Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add crates/litgraph-gateway python_tests/gateway
git commit -m "feat(gateway): binary, keygen, and openai-sdk wire-compat tests"
```

---

### Task 9: Benchmark harness and documentation

**Files:**
- Create: `crates/litgraph-gateway/benches/relay.rs`
- Create: `crates/litgraph-gateway/README.md`
- Modify: `COMPARISON.md` (add a gateway row to §23)
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: everything above.
- Produces: no code interfaces; this task produces measurements and docs.

- [ ] **Step 1: Write the bench**

Add to `Cargo.toml`:

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["async_tokio"] }

[[bench]]
name = "relay"
harness = false
```

`crates/litgraph-gateway/benches/relay.rs`:

```rust
//! Gateway-added overhead, measured against a MOCK upstream.
//!
//! Benching against a real provider would measure the provider, not this
//! crate. The mock returns instantly, so what remains is exactly the
//! auth + route + serialize + relay cost the gateway adds.

use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use litgraph_gateway::testing::{bench_state, invoke_once, relay_n_chunks};
use tokio::runtime::Runtime;

fn non_streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = Arc::new(bench_state());
    c.bench_function("non_streaming_round_trip", |b| {
        b.to_async(&rt).iter(|| invoke_once(state.clone()));
    });
}

fn streaming(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();
    let state = Arc::new(bench_state());
    let mut g = c.benchmark_group("sse_relay");
    for chunks in [100usize, 1_000] {
        g.bench_with_input(BenchmarkId::from_parameter(chunks), &chunks, |b, &n| {
            b.to_async(&rt).iter(|| relay_n_chunks(state.clone(), n));
        });
    }
    g.finish();
}

criterion_group!(benches, non_streaming, streaming);
criterion_main!(benches);
```

Add `bench_state()`, `invoke_once(state)` and `relay_n_chunks(state, n)` to `src/testing.rs`, built from the same `Echo`-style fake used in Task 6 — `relay_n_chunks` drives a fake `ChatStream` that yields `n` deltas then a `Done` carrying usage, and drains the SSE body to completion.

- [ ] **Step 2: Run it**

Run: `cargo bench -p litgraph-gateway`
Record p50 and p99 added latency and RSS.

- [ ] **Step 3: Write the crate README**

Cover: what it is, the config file, `keygen`, `serve`, the deployment-scoped vs tenant-scoped policy split, and the two guarantees that are easy to misread — budgets are "reject once over" not a hard cap, and there is no failover after first byte on a stream.

- [ ] **Step 4: Update COMPARISON.md and CHANGELOG.md**

Add to §23 Deployment:

```markdown
| OpenAI-compatible gateway (virtual keys, budgets, multi-deployment routing) | ✅ (`litgraph-gateway`) | ❌ | ❌ |
```

Put measured numbers in §24 only after Step 2. Do not publish aspirational figures — that is the same overclaim this repo already had on multi-tenant auth.

- [ ] **Step 5: Full verification**

```bash
cargo test --workspace --no-fail-fast
.venv/bin/python -m pytest python_tests -q
.venv/bin/python tools/check_docs.py
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add crates/litgraph-gateway COMPARISON.md CHANGELOG.md
git commit -m "docs(gateway): README, benchmark harness, and comparison entry"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| §4.1 policy scope split | 3 (tenant), 4 (deployment) |
| §4.2 crate layout | 1 |
| §5.1 config | 1 |
| §5.2 types | 1–4 |
| §5.3 key format | 2 |
| §6 request flow | 6 (steps 1–7) |
| §6.1 streaming usage, budget semantics | 7, 3 |
| §7 performance (auth cache, no re-serialize) | 2, 7 |
| §7.3 benchmark | 9 |
| §8 error taxonomy | 6 |
| §8.1 mid-stream | 7 |
| §9 security | 2, 6 |
| §10 testing | every task |
| §11 doc correction | already committed in `e5541cc` |
| §12 exact-match groups, no bulkhead | 2 (`allows_group`), Global Constraints |

**Known gap carried deliberately:** `/v1/embeddings` is out of scope per spec §3.

**Two places the implementer must verify against real code before trusting the plan**, flagged inline rather than hidden: `litgraph_core::Error` variant names in Task 5, and the `Message::{user,system,assistant}` constructors in Task 6. Both include the `grep` to run.

**Type consistency:** `VirtualKey.id` is `String` throughout; `key_id: &str` in `SpendStore` and `TenantPolicy::record_spend`; `Deployment.upstream_model` is used for price lookup in Tasks 6 and 7; `PolicyDecision` variants match between Tasks 3 and 6; `DispatchError` variants match between Tasks 5 and 6.

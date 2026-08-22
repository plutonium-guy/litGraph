//! Virtual API keys: storage, verification, and the verification cache.
//!
//! Format: `lg-sk-<8-char prefix>.<secret>`. The prefix is an indexed
//! lookup so verifying a request is one argon2 hash, not a scan across
//! every configured key.
//!
//! # Why there is a cache
//!
//! argon2id is deliberately expensive (10-50 ms). Paying that per request
//! would make this gateway slower than the Python proxy it replaces, so a
//! verified secret is memoized. The cache is keyed on the FULL presented
//! token, not the prefix — otherwise a forged secret sharing a known
//! prefix would ride someone else's cache entry straight past
//! verification.
//!
//! # Prefix indexing (ruling R2)
//!
//! `KeyConfig::prefix` is an explicit field rather than something derived
//! from the key id, because the lookup index used by `KeyStore` and the
//! prefix embedded in a minted plaintext token must agree exactly.
//! `generate_key` returns the prefix alongside the plaintext and hash so
//! callers (and Task 8's `keygen` CLI) can write a config stanza that
//! actually verifies.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use argon2::password_hash::{
    rand_core::OsRng, PasswordHash, PasswordHasher, PasswordVerifier, SaltString,
};
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
    /// `GatewayConfig::validate` already rejects this at config-parse time,
    /// but `KeyStore::from_configs` takes a bare `&[KeyConfig]` — nothing
    /// in its signature guarantees the caller validated first (a future
    /// hot-reload path or a hand-built `Vec<KeyConfig>` could skip it). A
    /// silent overwrite here would mean the last key with a given prefix
    /// wins and every earlier tenant sharing it is locked out, so this
    /// file — the actual security boundary — enforces uniqueness itself
    /// too. The redundancy with config validation is deliberate defense
    /// in depth, not an oversight.
    #[error("duplicate key prefix {0:?}")]
    DuplicateKeyPrefix(String),
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

/// Mint a new key. Returns `(plaintext, prefix, phc_hash)` — the plaintext
/// is shown to the operator once and never stored. `prefix` is the lookup
/// index a `KeyConfig` must carry for the plaintext to verify.
pub fn generate_key() -> (String, String, String) {
    let mut prefix_bytes = [0u8; 4];
    let mut secret_bytes = [0u8; 32];
    rand::thread_rng().fill_bytes(&mut prefix_bytes);
    rand::thread_rng().fill_bytes(&mut secret_bytes);
    let prefix = hex(&prefix_bytes);
    let secret = hex(&secret_bytes);
    let plaintext = format!("lg-sk-{prefix}.{secret}");
    let phc = hash_secret(&secret);
    (plaintext, prefix, phc)
}

/// Encode bytes as lowercase hex. `pub(crate)` so other gateway modules
/// (e.g. `http`'s opaque response ids) can reuse it instead of growing a
/// second copy.
pub(crate) fn hex(bytes: &[u8]) -> String {
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
            // Enforced again here (see AuthError::DuplicateKeyPrefix doc):
            // a plain `HashMap::insert` would silently let a later key
            // overwrite an earlier one's index entry.
            if by_prefix.contains_key(&k.prefix) {
                return Err(AuthError::DuplicateKeyPrefix(k.prefix.clone()));
            }
            by_prefix.insert(
                k.prefix.clone(),
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
        Ok(Self {
            by_prefix,
            cache: RwLock::new(HashMap::new()),
        })
    }

    pub fn cache_len(&self) -> usize {
        self.cache.read().len()
    }

    /// Verify a presented bearer token.
    ///
    /// # Known, accepted timing side channel
    ///
    /// Malformed input and an unknown prefix return immediately, while a
    /// known prefix with the wrong secret pays a full argon2 verify
    /// (10-50 ms). That timing gap lets an attacker enumerate which 8-char
    /// prefixes are live by observing response latency across many guesses.
    ///
    /// This is a deliberate, reasoned trade-off, not an oversight — do not
    /// "fix" it by paying argon2 on every miss including garbage input.
    /// That would turn any unauthenticated request into a guaranteed
    /// 10-50 ms CPU burn, which is a far cheaper denial-of-service lever
    /// than the enumeration it prevents. What actually leaks is only which
    /// prefixes exist; a prefix without its paired 256-bit secret is
    /// worthless to an attacker. Do not change this control flow.
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
        cache.insert(
            token.to_string(),
            CacheEntry {
                key_id: stored.key.id.clone(),
                at: Instant::now(),
            },
        );
        Ok(stored.key.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::KeyConfig;

    fn key_cfg(id: &str, prefix: &str, hash: &str, groups: &[&str]) -> KeyConfig {
        KeyConfig {
            id: id.into(),
            prefix: prefix.into(),
            hash: hash.into(),
            groups: groups.iter().map(|s| s.to_string()).collect(),
            rpm: None,
            max_usd_per_day: None,
        }
    }

    #[test]
    fn authenticates_a_valid_key_and_rejects_a_wrong_secret() {
        let (plaintext, prefix, hash) = generate_key();
        let store =
            KeyStore::from_configs(&[key_cfg("team-a", &prefix, &hash, &["gpt-4o"])]).unwrap();

        let vk = store
            .authenticate(&plaintext)
            .expect("valid key authenticates");
        assert_eq!(vk.id, "team-a");

        // Same prefix, wrong secret.
        let (prefix_part, _) = plaintext.split_once('.').unwrap();
        let forged = format!("{prefix_part}.deadbeefdeadbeefdeadbeefdeadbeef");
        assert!(matches!(store.authenticate(&forged), Err(AuthError::Unknown)));
    }

    #[test]
    fn rejects_malformed_and_unknown_keys() {
        let (_, prefix, hash) = generate_key();
        let store =
            KeyStore::from_configs(&[key_cfg("team-a", &prefix, &hash, &["gpt-4o"])]).unwrap();

        for bad in ["", "no-dot", "lg-sk-zzzzzzzz.secret", "Bearer something"] {
            assert!(
                matches!(store.authenticate(bad), Err(AuthError::Unknown)),
                "expected rejection for {bad:?}"
            );
        }
    }

    #[test]
    fn authorizes_only_listed_groups_exactly() {
        let (plaintext, prefix, hash) = generate_key();
        let store =
            KeyStore::from_configs(&[key_cfg("team-a", &prefix, &hash, &["gpt-4o"])]).unwrap();
        let vk = store.authenticate(&plaintext).unwrap();

        assert!(vk.allows_group("gpt-4o"));
        // No globs in v1: a prefix match must NOT grant access.
        assert!(!vk.allows_group("gpt-4o-mini"));
        assert!(!vk.allows_group("claude-sonnet-4-5"));
    }

    #[test]
    fn one_tenants_key_never_authenticates_as_another() {
        let (plain_a, prefix_a, hash_a) = generate_key();
        let (plain_b, prefix_b, hash_b) = generate_key();
        let store = KeyStore::from_configs(&[
            key_cfg("team-a", &prefix_a, &hash_a, &["gpt-4o"]),
            key_cfg("team-b", &prefix_b, &hash_b, &["gpt-4o"]),
        ])
        .unwrap();

        assert_eq!(store.authenticate(&plain_a).unwrap().id, "team-a");
        assert_eq!(store.authenticate(&plain_b).unwrap().id, "team-b");
    }

    #[test]
    fn second_authentication_hits_the_cache_and_still_verifies_the_secret() {
        let (plaintext, prefix, hash) = generate_key();
        let store =
            KeyStore::from_configs(&[key_cfg("team-a", &prefix, &hash, &["gpt-4o"])]).unwrap();

        assert_eq!(store.authenticate(&plaintext).unwrap().id, "team-a");
        assert_eq!(store.authenticate(&plaintext).unwrap().id, "team-a");
        assert_eq!(store.cache_len(), 1, "verified secret should be memoized");

        // A forged secret sharing the prefix must not ride the cache entry.
        let (prefix_part, _) = plaintext.split_once('.').unwrap();
        let forged = format!("{prefix_part}.0000000000000000000000000000000000000000000");
        assert!(matches!(store.authenticate(&forged), Err(AuthError::Unknown)));
    }

    #[test]
    fn debug_output_never_contains_key_material() {
        let (plaintext, prefix, hash) = generate_key();
        let cfg = key_cfg("team-a", &prefix, &hash, &["gpt-4o"]);
        let store = KeyStore::from_configs(&[cfg.clone()]).unwrap();
        let vk = store.authenticate(&plaintext).unwrap();

        let gateway_cfg = crate::config::GatewayConfig {
            deployment: Vec::new(),
            key: vec![cfg.clone()],
        };

        let rendered = format!("{vk:?} {store:?} {cfg:?} {gateway_cfg:?}");
        let secret = plaintext.split_once('.').unwrap().1;
        assert!(!rendered.contains(secret), "secret leaked into Debug output");
        assert!(!rendered.contains(&hash), "hash leaked into Debug output");
    }

    #[test]
    fn from_configs_rejects_colliding_prefixes_even_without_config_validation() {
        // Bypasses GatewayConfig::validate entirely — constructs the
        // KeyStore directly from a hand-built slice, as a future hot-reload
        // path or unvalidated caller might.
        let (_, _, hash_a) = generate_key();
        let (_, _, hash_b) = generate_key();
        let err = KeyStore::from_configs(&[
            key_cfg("team-a", "shared01", &hash_a, &["gpt-4o"]),
            key_cfg("team-b", "shared01", &hash_b, &["gpt-4o"]),
        ])
        .unwrap_err();
        assert!(matches!(err, AuthError::DuplicateKeyPrefix(ref s) if s == "shared01"));
    }
}

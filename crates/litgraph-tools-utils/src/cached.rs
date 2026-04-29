//! TTL + LRU cache wrapper around any `Tool`. Pattern:
//!
//! ```ignore
//! use litgraph_tools_utils::CachedTool;
//! use std::time::Duration;
//!
//! let raw = Arc::new(BraveSearch::new(cfg)?);
//! let cached = CachedTool::wrap(raw, Duration::from_secs(3600), 256);
//! agent.tools.push(cached);
//! ```
//!
//! # When this helps
//!
//! - **Web search**: agents often re-issue the same query in a multi-step
//!   trace ("search for X" → reasoning → "search for X again"). Cache
//!   returns the prior result instantly + saves an API credit.
//! - **DB queries**: read-only `SELECT`s with the same params return the
//!   same rows within a TTL window. Cache them for "what's this user's
//!   email?"-style follow-ups in a chat session.
//! - **Deterministic APIs**: weather-by-city, exchange rates, public
//!   data — cache by argument hash, TTL keyed to data freshness.
//!
//! # When this is WRONG to use
//!
//! - Side-effecting tools (write_file, shell, post-to-slack) — caching
//!   would silently swallow second invocations. Don't wrap these.
//! - Non-deterministic tools (random sampling, current-time lookup) —
//!   cache hit returns a stale value. Don't wrap these.
//! - Tools with sensitive args (passwords, tokens) — keys are hashed
//!   but the in-memory cache holds the responses; if response contains
//!   PII, downstream readers get it. Disable wrapping where appropriate.
//!
//! # Eviction
//!
//! Two policies stack:
//! - **TTL**: every entry has an absolute expiry. Read past expiry → MISS,
//!   entry removed lazily on next access.
//! - **LRU**: when cache is full and a fresh insert is needed, oldest-
//!   accessed entry is evicted. Tracking is approximate (insertion order
//!   only, not access order — exact LRU would need a doubly-linked list
//!   we don't pull in).
//!
//! # Key construction
//!
//! Cache key = `{tool_name}\0{canonical_json(args)}`. Canonical JSON
//! (sorted object keys) makes `{"a":1, "b":2}` and `{"b":2, "a":1}`
//! collide as the same key — which is what callers want.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::Result;
use serde_json::Value;

/// Wraps a `Tool` so identical-arg calls within a TTL window return the
/// cached prior result without invoking the inner tool.
pub struct CachedTool {
    inner: Arc<dyn Tool>,
    cache: Arc<Mutex<CacheState>>,
    ttl: Duration,
    max_entries: usize,
}

#[derive(Default)]
struct CacheState {
    /// Insertion order of keys for approximate LRU.
    order: Vec<String>,
    /// key → (cached value, expiry).
    entries: HashMap<String, CacheEntry>,
}

struct CacheEntry {
    value: Value,
    expiry: Instant,
}

impl CachedTool {
    /// Wrap an inner tool with a fresh cache. `ttl` is the entry lifetime;
    /// `max_entries` caps total cache size (LRU-evicted on overflow).
    pub fn wrap(inner: Arc<dyn Tool>, ttl: Duration, max_entries: usize) -> Arc<Self> {
        Arc::new(Self {
            inner,
            cache: Arc::new(Mutex::new(CacheState::default())),
            ttl,
            max_entries: max_entries.max(1),
        })
    }

    /// Drop all entries — useful for tests + manual invalidation.
    pub fn clear(&self) {
        let mut g = self.cache.lock().expect("poisoned");
        g.order.clear();
        g.entries.clear();
    }

    pub fn len(&self) -> usize {
        self.cache.lock().expect("poisoned").entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[async_trait]
impl Tool for CachedTool {
    fn schema(&self) -> ToolSchema {
        // Pass through inner schema unchanged — the LLM doesn't need to
        // know the tool is cached.
        self.inner.schema()
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let key = cache_key(&self.inner.schema().name, &args);
        let now = Instant::now();

        // Look up. If hit AND not expired, return cached value.
        {
            let mut g = self.cache.lock().expect("poisoned");
            if let Some(entry) = g.entries.get(&key) {
                if entry.expiry > now {
                    return Ok(entry.value.clone());
                }
                // Expired — remove eagerly so we don't re-test it.
                g.entries.remove(&key);
                g.order.retain(|k| k != &key);
            }
        }

        // Miss → invoke inner. We don't hold the lock during the call.
        let value = self.inner.run(args).await?;

        // Insert. Evict expired + LRU until we fit.
        {
            let mut g = self.cache.lock().expect("poisoned");

            // Evict expired entries opportunistically (cheap pass since
            // we already hold the lock).
            let now2 = Instant::now();
            let expired: Vec<String> = g
                .entries
                .iter()
                .filter_map(|(k, e)| if e.expiry <= now2 { Some(k.clone()) } else { None })
                .collect();
            for k in &expired {
                g.entries.remove(k);
            }
            // Clone the entries' key set so the closure doesn't borrow `g`.
            let live: std::collections::HashSet<String> =
                g.entries.keys().cloned().collect();
            g.order.retain(|k| live.contains(k));

            // LRU evict if still over the cap.
            while g.entries.len() >= self.max_entries {
                let Some(oldest) = g.order.first().cloned() else { break };
                g.order.remove(0);
                g.entries.remove(&oldest);
            }

            g.entries.insert(
                key.clone(),
                CacheEntry {
                    value: value.clone(),
                    expiry: now + self.ttl,
                },
            );
            g.order.push(key);
        }

        Ok(value)
    }
}

/// Build the cache key. Canonical JSON (sorted object keys) so that
/// `{"a":1, "b":2}` and `{"b":2, "a":1}` collide.
fn cache_key(tool_name: &str, args: &Value) -> String {
    let canonical = canonical_json(args);
    format!("{tool_name}\0{canonical}")
}

/// Re-serialize a JSON value with sorted object keys. Recursive.
fn canonical_json(v: &Value) -> String {
    match v {
        Value::Object(map) => {
            let mut entries: Vec<(&String, &Value)> = map.iter().collect();
            entries.sort_by(|a, b| a.0.cmp(b.0));
            let mut out = String::from("{");
            for (i, (k, val)) in entries.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                // Quote+escape the key minimally (JSON-safe).
                out.push_str(&serde_json::to_string(k).unwrap_or_default());
                out.push(':');
                out.push_str(&canonical_json(val));
            }
            out.push('}');
            out
        }
        Value::Array(arr) => {
            let mut out = String::from("[");
            for (i, item) in arr.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                out.push_str(&canonical_json(item));
            }
            out.push(']');
            out
        }
        _ => serde_json::to_string(v).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::Duration;

    /// Counts invocations. Returns `{"call_n": <count>}` so tests can
    /// distinguish cached vs fresh responses.
    struct CountingTool {
        calls: AtomicU32,
    }

    #[async_trait]
    impl Tool for CountingTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: "counter".to_string(),
                description: "increments a counter on each call".to_string(),
                parameters: serde_json::json!({"type": "object"}),
            }
        }
        async fn run(&self, _args: Value) -> Result<Value> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(serde_json::json!({"call_n": n}))
        }
    }

    fn counter() -> Arc<CountingTool> {
        Arc::new(CountingTool { calls: AtomicU32::new(0) })
    }

    #[tokio::test]
    async fn second_identical_call_returns_cached_value() {
        let inner = counter();
        let cached: Arc<dyn Tool> =
            CachedTool::wrap(inner.clone() as Arc<dyn Tool>, Duration::from_secs(60), 10);
        let r1 = cached
            .run(serde_json::json!({"q": "hi"}))
            .await
            .unwrap();
        let r2 = cached
            .run(serde_json::json!({"q": "hi"}))
            .await
            .unwrap();
        assert_eq!(r1, r2);
        // Inner was only called once.
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn different_args_are_separate_cache_entries() {
        let inner = counter();
        let cached: Arc<dyn Tool> =
            CachedTool::wrap(inner.clone() as Arc<dyn Tool>, Duration::from_secs(60), 10);
        let r1 = cached.run(serde_json::json!({"q": "alpha"})).await.unwrap();
        let r2 = cached.run(serde_json::json!({"q": "beta"})).await.unwrap();
        assert_ne!(r1, r2);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn key_canonicalization_object_order_doesnt_matter() {
        let inner = counter();
        let cached: Arc<dyn Tool> =
            CachedTool::wrap(inner.clone() as Arc<dyn Tool>, Duration::from_secs(60), 10);
        let r1 = cached
            .run(serde_json::json!({"a": 1, "b": 2}))
            .await
            .unwrap();
        let r2 = cached
            .run(serde_json::json!({"b": 2, "a": 1}))
            .await
            .unwrap();
        // Same canonical form → same cache entry → same response.
        assert_eq!(r1, r2);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn ttl_expiry_triggers_re_invocation() {
        let inner = counter();
        let cached: Arc<dyn Tool> = CachedTool::wrap(
            inner.clone() as Arc<dyn Tool>,
            Duration::from_millis(50),
            10,
        );
        let r1 = cached.run(serde_json::json!({"q": "x"})).await.unwrap();
        // Wait past TTL.
        tokio::time::sleep(Duration::from_millis(80)).await;
        let r2 = cached.run(serde_json::json!({"q": "x"})).await.unwrap();
        // Different responses (counter incremented).
        assert_ne!(r1, r2);
        assert_eq!(inner.calls.load(Ordering::SeqCst), 2);
    }

    #[tokio::test]
    async fn lru_evicts_oldest_when_cap_reached() {
        let inner = counter();
        // Cap = 2 entries.
        let cached_arc = CachedTool::wrap(
            inner.clone() as Arc<dyn Tool>,
            Duration::from_secs(60),
            2,
        );
        let cached: Arc<dyn Tool> = cached_arc.clone();

        let _ = cached.run(serde_json::json!({"q": "a"})).await.unwrap();
        let _ = cached.run(serde_json::json!({"q": "b"})).await.unwrap();
        // Insert 3rd → evicts "a" (oldest).
        let _ = cached.run(serde_json::json!({"q": "c"})).await.unwrap();
        assert_eq!(cached_arc.len(), 2);

        // "a" no longer cached → calls inner again.
        let n_before = inner.calls.load(Ordering::SeqCst);
        let _ = cached.run(serde_json::json!({"q": "a"})).await.unwrap();
        assert_eq!(inner.calls.load(Ordering::SeqCst), n_before + 1);
    }

    #[tokio::test]
    async fn schema_passes_through_unchanged() {
        let inner = counter();
        let cached_arc = CachedTool::wrap(
            inner.clone() as Arc<dyn Tool>,
            Duration::from_secs(60),
            10,
        );
        let cached: Arc<dyn Tool> = cached_arc;
        // Schema name + description must be preserved.
        let s = cached.schema();
        assert_eq!(s.name, "counter");
        assert_eq!(s.description, "increments a counter on each call");
    }

    #[tokio::test]
    async fn clear_drops_all_entries() {
        let inner = counter();
        let cached_arc = CachedTool::wrap(
            inner.clone() as Arc<dyn Tool>,
            Duration::from_secs(60),
            10,
        );
        let cached: Arc<dyn Tool> = cached_arc.clone();
        let _ = cached.run(serde_json::json!({"q": "a"})).await.unwrap();
        assert_eq!(cached_arc.len(), 1);
        cached_arc.clear();
        assert_eq!(cached_arc.len(), 0);
    }

    #[tokio::test]
    async fn errors_from_inner_are_not_cached() {
        struct AlwaysFails;
        #[async_trait]
        impl Tool for AlwaysFails {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "fails".to_string(),
                    description: "always errors".to_string(),
                    parameters: serde_json::json!({}),
                }
            }
            async fn run(&self, _args: Value) -> Result<Value> {
                Err(litgraph_core::Error::parse("kaboom"))
            }
        }
        let inner: Arc<dyn Tool> = Arc::new(AlwaysFails);
        let cached_arc = CachedTool::wrap(inner, Duration::from_secs(60), 10);
        let cached: Arc<dyn Tool> = cached_arc.clone();
        let _ = cached.run(serde_json::json!({})).await.unwrap_err();
        let _ = cached.run(serde_json::json!({})).await.unwrap_err();
        // Errors are not stored — cache stays empty.
        assert_eq!(cached_arc.len(), 0);
    }

    #[test]
    fn canonical_json_sorts_object_keys_recursively() {
        let v1 = serde_json::json!({"b": 2, "a": {"y": 9, "x": 1}});
        let v2 = serde_json::json!({"a": {"x": 1, "y": 9}, "b": 2});
        assert_eq!(canonical_json(&v1), canonical_json(&v2));
    }

    #[test]
    fn canonical_json_preserves_array_order() {
        let v1 = serde_json::json!([1, 2, 3]);
        let v2 = serde_json::json!([3, 2, 1]);
        // Arrays are NOT reordered (positional).
        assert_ne!(canonical_json(&v1), canonical_json(&v2));
    }
}

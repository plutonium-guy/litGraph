//! Long-term memory store — namespace+key JSON document API.
//!
//! Mirrors LangGraph's `BaseStore`: cross-thread / cross-session memory keyed by
//! a namespace tuple plus a document key. Distinct from the `Checkpointer`
//! (thread-scoped) and `Memory` (conversation-scoped) traits.
//!
//! ```rust
//! use litgraph_core::store::{InMemoryStore, Store, StoreItem};
//! use serde_json::json;
//!
//! # async fn _ex() -> litgraph_core::Result<()> {
//! let store = InMemoryStore::new();
//! let ns = &["users".to_string(), "alice".to_string()];
//! store.put(ns, "pref:lang", &json!({"value": "rust"}), None).await?;
//! let got: Option<StoreItem> = store.get(ns, "pref:lang").await?;
//! assert_eq!(got.unwrap().value["value"], "rust");
//! # Ok(()) }
//! ```
//!
//! Concrete implementations: `InMemoryStore` (this module), Postgres / Redis
//! live in their own crates.

use crate::error::Result;
use async_trait::async_trait;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

/// Namespace tuple. `("users", "alice")` is a 2-segment namespace; segments
/// must be non-empty. Stored hierarchically so prefix-matching works for
/// `list_namespaces`.
pub type Namespace = [String];

/// One stored document.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoreItem {
    pub namespace: Vec<String>,
    pub key: String,
    pub value: Value,
    /// Unix-millisecond TTL deadline; `None` = never expires.
    pub expires_at_ms: Option<u64>,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

impl StoreItem {
    fn is_expired(&self, now_ms: u64) -> bool {
        matches!(self.expires_at_ms, Some(t) if t <= now_ms)
    }
}

/// Search filters. `match` does exact JSON-path equality on the value;
/// `query_text` is for stores that index value-text (LIKE / FTS).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchFilter {
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    pub query_text: Option<String>,
    /// (json-pointer-style path, expected JSON value) pairs. Item must match
    /// every pair to be returned. Path `/foo/bar` reads `value["foo"]["bar"]`.
    pub matches: Vec<(String, Value)>,
}

/// Long-term memory store. Implementations must be `Send + Sync` so a single
/// store instance can serve a parallel graph executor.
#[async_trait]
pub trait Store: Send + Sync {
    /// Insert or replace one item. `ttl_ms` is a relative-from-now TTL.
    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: &Value,
        ttl_ms: Option<u64>,
    ) -> Result<()>;

    /// Fetch one item; expired items return `None` and are evicted on read.
    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<StoreItem>>;

    /// Delete one item; returns `true` if it existed.
    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool>;

    /// List items in a namespace (or namespace prefix), filter by `SearchFilter`.
    async fn search(
        &self,
        namespace_prefix: &Namespace,
        filter: &SearchFilter,
    ) -> Result<Vec<StoreItem>>;

    /// List the leaf namespaces under a prefix. Useful for tenant discovery.
    async fn list_namespaces(
        &self,
        prefix: &Namespace,
        limit: Option<usize>,
    ) -> Result<Vec<Vec<String>>>;
}

/// In-process store. `Arc<RwLock<...>>` so clones share state. Production
/// deployments use the Postgres / Redis implementations.
#[derive(Debug, Default, Clone)]
pub struct InMemoryStore {
    inner: Arc<RwLock<BTreeMap<(Vec<String>, String), StoreItem>>>,
}

impl InMemoryStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.read().len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.read().is_empty()
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn json_pointer_get<'a>(value: &'a Value, path: &str) -> Option<&'a Value> {
    if path.is_empty() || path == "/" {
        return Some(value);
    }
    let normalized = if let Some(stripped) = path.strip_prefix('/') {
        stripped.to_string()
    } else {
        path.to_string()
    };
    let mut cur = value;
    for seg in normalized.split('/') {
        let unescaped = seg.replace("~1", "/").replace("~0", "~");
        cur = match cur {
            Value::Object(map) => map.get(&unescaped)?,
            Value::Array(arr) => {
                let idx: usize = unescaped.parse().ok()?;
                arr.get(idx)?
            }
            _ => return None,
        };
    }
    Some(cur)
}

fn matches_filter(item: &StoreItem, filter: &SearchFilter) -> bool {
    for (path, expected) in &filter.matches {
        match json_pointer_get(&item.value, path) {
            Some(v) if v == expected => continue,
            _ => return false,
        }
    }
    if let Some(q) = &filter.query_text {
        let body = item.value.to_string();
        if !body.to_lowercase().contains(&q.to_lowercase()) {
            return false;
        }
    }
    true
}

#[async_trait]
impl Store for InMemoryStore {
    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: &Value,
        ttl_ms: Option<u64>,
    ) -> Result<()> {
        let now = now_ms();
        let ns: Vec<String> = namespace.to_vec();
        let mut guard = self.inner.write();
        let existing_created = guard.get(&(ns.clone(), key.to_string())).map(|i| i.created_at_ms);
        let item = StoreItem {
            namespace: ns.clone(),
            key: key.to_string(),
            value: value.clone(),
            expires_at_ms: ttl_ms.map(|t| now + t),
            created_at_ms: existing_created.unwrap_or(now),
            updated_at_ms: now,
        };
        guard.insert((ns, key.to_string()), item);
        Ok(())
    }

    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<StoreItem>> {
        let now = now_ms();
        let ns: Vec<String> = namespace.to_vec();
        let mut guard = self.inner.write();
        let composite = (ns, key.to_string());
        let Some(item) = guard.get(&composite) else {
            return Ok(None);
        };
        if item.is_expired(now) {
            guard.remove(&composite);
            return Ok(None);
        }
        Ok(Some(item.clone()))
    }

    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool> {
        let mut guard = self.inner.write();
        Ok(guard.remove(&(namespace.to_vec(), key.to_string())).is_some())
    }

    async fn search(
        &self,
        namespace_prefix: &Namespace,
        filter: &SearchFilter,
    ) -> Result<Vec<StoreItem>> {
        let now = now_ms();
        let prefix: Vec<String> = namespace_prefix.to_vec();
        let mut to_evict: Vec<(Vec<String>, String)> = Vec::new();
        let mut hits: Vec<StoreItem> = {
            let guard = self.inner.read();
            guard
                .iter()
                .filter_map(|((ns, key), item)| {
                    if !ns.starts_with(&prefix) {
                        return None;
                    }
                    if item.is_expired(now) {
                        to_evict.push((ns.clone(), key.clone()));
                        return None;
                    }
                    if !matches_filter(item, filter) {
                        return None;
                    }
                    Some(item.clone())
                })
                .collect()
        };
        if !to_evict.is_empty() {
            let mut guard = self.inner.write();
            for k in &to_evict {
                guard.remove(k);
            }
        }
        hits.sort_by(|a, b| b.updated_at_ms.cmp(&a.updated_at_ms));
        let offset = filter.offset.unwrap_or(0);
        let limit = filter.limit.unwrap_or(usize::MAX);
        Ok(hits.into_iter().skip(offset).take(limit).collect())
    }

    async fn list_namespaces(
        &self,
        prefix: &Namespace,
        limit: Option<usize>,
    ) -> Result<Vec<Vec<String>>> {
        let p: Vec<String> = prefix.to_vec();
        let mut seen: BTreeMap<Vec<String>, ()> = BTreeMap::new();
        {
            let guard = self.inner.read();
            for ((ns, _), _) in guard.iter() {
                if ns.starts_with(&p) {
                    seen.insert(ns.clone(), ());
                }
            }
        }
        let cap = limit.unwrap_or(usize::MAX);
        Ok(seen.into_keys().take(cap).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn put_get_roundtrip() {
        let store = InMemoryStore::new();
        let ns = vec!["users".to_string(), "alice".to_string()];
        store
            .put(&ns, "pref", &json!({"lang": "rust"}), None)
            .await
            .unwrap();
        let got = store.get(&ns, "pref").await.unwrap().unwrap();
        assert_eq!(got.value["lang"], "rust");
        assert_eq!(got.namespace, ns);
        assert_eq!(got.key, "pref");
    }

    #[tokio::test]
    async fn delete_returns_true_only_when_existed() {
        let store = InMemoryStore::new();
        let ns = vec!["t".to_string()];
        assert!(!store.delete(&ns, "k").await.unwrap());
        store.put(&ns, "k", &json!(1), None).await.unwrap();
        assert!(store.delete(&ns, "k").await.unwrap());
        assert!(store.get(&ns, "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn search_filters_by_match_and_query_text() {
        let store = InMemoryStore::new();
        let ns = vec!["docs".to_string()];
        store
            .put(&ns, "a", &json!({"role": "admin", "name": "alice"}), None)
            .await
            .unwrap();
        store
            .put(&ns, "b", &json!({"role": "user", "name": "bob"}), None)
            .await
            .unwrap();
        let filter = SearchFilter {
            matches: vec![("/role".into(), json!("admin"))],
            ..Default::default()
        };
        let hits = store.search(&ns, &filter).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key, "a");

        let filter = SearchFilter {
            query_text: Some("bob".into()),
            ..Default::default()
        };
        let hits = store.search(&ns, &filter).await.unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].key, "b");
    }

    #[tokio::test]
    async fn search_respects_namespace_prefix() {
        let store = InMemoryStore::new();
        store
            .put(&["t".to_string(), "a".to_string()], "k", &json!(1), None)
            .await
            .unwrap();
        store
            .put(&["t".to_string(), "b".to_string()], "k", &json!(2), None)
            .await
            .unwrap();
        store
            .put(&["other".to_string()], "k", &json!(3), None)
            .await
            .unwrap();
        let hits = store
            .search(&["t".to_string()], &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[tokio::test]
    async fn ttl_evicts_on_read() {
        let store = InMemoryStore::new();
        let ns = vec!["t".to_string()];
        store.put(&ns, "k", &json!("v"), Some(0)).await.unwrap();
        // 0 ms TTL ⇒ already expired.
        assert!(store.get(&ns, "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn list_namespaces_returns_distinct_paths() {
        let store = InMemoryStore::new();
        store
            .put(&["t".to_string(), "a".to_string()], "k", &json!(1), None)
            .await
            .unwrap();
        store
            .put(&["t".to_string(), "a".to_string()], "k2", &json!(2), None)
            .await
            .unwrap();
        store
            .put(&["t".to_string(), "b".to_string()], "k", &json!(3), None)
            .await
            .unwrap();
        let nss = store
            .list_namespaces(&["t".to_string()], None)
            .await
            .unwrap();
        assert_eq!(nss.len(), 2);
        assert!(nss.contains(&vec!["t".to_string(), "a".to_string()]));
        assert!(nss.contains(&vec!["t".to_string(), "b".to_string()]));
    }

    #[tokio::test]
    async fn search_sorts_by_updated_desc_and_paginates() {
        let store = InMemoryStore::new();
        let ns = vec!["t".to_string()];
        for i in 0..5 {
            store
                .put(&ns, &format!("k{i}"), &json!(i), None)
                .await
                .unwrap();
        }
        let filter = SearchFilter {
            limit: Some(2),
            offset: Some(1),
            ..Default::default()
        };
        let hits = store.search(&ns, &filter).await.unwrap();
        assert_eq!(hits.len(), 2);
    }

    #[test]
    fn json_pointer_root_returns_value_itself() {
        let v = json!({"x": 1});
        assert_eq!(json_pointer_get(&v, "").unwrap(), &v);
        assert_eq!(json_pointer_get(&v, "/").unwrap(), &v);
        assert_eq!(json_pointer_get(&v, "/x").unwrap(), &json!(1));
    }
}

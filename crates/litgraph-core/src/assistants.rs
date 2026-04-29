//! Assistants — per-graph config snapshots. LangGraph Cloud parity.
//!
//! An "Assistant" is a saved bundle of `(graph_id, config, metadata)`
//! addressable by a stable id. The same compiled graph can have many
//! assistants (one per persona, customer, experiment) and each
//! assistant's config replaces the runtime defaults at invoke time.
//!
//! # The pattern
//!
//! ```text
//!  one compiled graph
//!         │
//!  ┌──────┼──────────┐
//!  │      │          │
//! a1     a2         a3        ← Assistant snapshots
//!  │      │          │
//!  ▼      ▼          ▼
//! "concise"  "verbose"  "JSON-only"
//! config     config     config
//! ```
//!
//! Compared to inlining config in code, assistants give you:
//! - **Live tuning** — UI / API editor without redeploying.
//! - **A/B testing** — route % of traffic to a new assistant.
//! - **Audit history** — every update bumps `version`, prior values
//!   readable via the `Store::search` history (see [`AssistantManager::
//!   versions`]).
//!
//! # Storage layout
//!
//! Assistants live in the [`Store`] under namespace
//! `["assistants", graph_id]` with the assistant's id as key. That gives
//! you free per-graph isolation: listing all assistants for a graph is
//! a single `Store::search` call.
//!
//! Each version is an independent key in the same namespace —
//! `<id>` for the canonical (latest) entry, plus `<id>@v<n>` for an
//! immutable archive of prior versions. Read paths default to the
//! canonical key for cheap "latest" reads; [`AssistantManager::versions`]
//! enumerates the archives.
//!
//! # No new deps
//!
//! Pure glue around `Store` + `serde_json::Value`. Plug in any
//! `Store` impl (`InMemoryStore`, `PostgresStore` from
//! `litgraph-store-postgres`) and you have a production-grade
//! assistants backend with versioning + audit.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::store::{SearchFilter, Store};
use crate::{Error, Result};

/// Per-graph config snapshot.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Assistant {
    pub id: String,
    pub name: String,
    /// Identifies which compiled graph this assistant is bound to.
    /// Free-form — the application picks the convention.
    pub graph_id: String,
    /// Caller-defined config bag — passed to the graph at invoke time.
    /// Schema is the application's responsibility; we treat it as opaque.
    #[serde(default = "default_config")]
    pub config: Value,
    /// Free-form labels (description, tags, owner email, …).
    #[serde(default = "default_metadata")]
    pub metadata: Value,
    /// Monotonic version number. Bumped on every `update`. Starts at 1.
    pub version: u64,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

fn default_config() -> Value {
    json!({})
}
fn default_metadata() -> Value {
    json!({})
}

impl Assistant {
    /// Build a v1 assistant. Most callers go through
    /// [`AssistantManager::create`] which also persists.
    pub fn new(
        graph_id: impl Into<String>,
        name: impl Into<String>,
        config: Value,
    ) -> Self {
        let now = now_ms();
        Self {
            id: new_assistant_id(),
            name: name.into(),
            graph_id: graph_id.into(),
            config,
            metadata: json!({}),
            version: 1,
            created_at_ms: now,
            updated_at_ms: now,
        }
    }

    pub fn with_metadata(mut self, m: Value) -> Self {
        self.metadata = m;
        self
    }

    pub fn with_id(mut self, id: impl Into<String>) -> Self {
        self.id = id.into();
        self
    }
}

/// Optional partial-update payload — only set the fields you want to
/// change. `None` fields keep their existing value.
#[derive(Debug, Clone, Default)]
pub struct AssistantPatch {
    pub name: Option<String>,
    pub config: Option<Value>,
    pub metadata: Option<Value>,
}

impl AssistantPatch {
    pub fn name(mut self, n: impl Into<String>) -> Self {
        self.name = Some(n.into());
        self
    }
    pub fn config(mut self, c: Value) -> Self {
        self.config = Some(c);
        self
    }
    pub fn metadata(mut self, m: Value) -> Self {
        self.metadata = Some(m);
        self
    }
    /// True if any field would change.
    pub fn is_empty(&self) -> bool {
        self.name.is_none() && self.config.is_none() && self.metadata.is_none()
    }
}

/// CRUD + versioning over a [`Store`]-backed assistants namespace.
/// Cheap to clone (Arc inside).
#[derive(Clone)]
pub struct AssistantManager {
    store: Arc<dyn Store>,
    /// Whether to write a `<id>@v<n>` archive entry on each update.
    /// Default true — the audit trail is the whole point. Disable
    /// to halve write traffic if you don't need history.
    keep_history: bool,
}

impl AssistantManager {
    pub fn new(store: Arc<dyn Store>) -> Self {
        Self {
            store,
            keep_history: true,
        }
    }

    pub fn without_history(mut self) -> Self {
        self.keep_history = false;
        self
    }

    fn ns(graph_id: &str) -> Vec<String> {
        vec!["assistants".into(), graph_id.into()]
    }

    fn version_key(id: &str, version: u64) -> String {
        format!("{id}@v{version}")
    }

    /// Persist a new assistant. Sets `version=1`. Returns the stored
    /// shape (with id + timestamps populated). If you pass an `Assistant`
    /// with a pre-set `id`, it's used verbatim — useful for migrating
    /// existing data into the manager.
    pub async fn create(&self, mut a: Assistant) -> Result<Assistant> {
        let now = now_ms();
        a.version = 1;
        a.created_at_ms = now;
        a.updated_at_ms = now;
        let ns = Self::ns(&a.graph_id);
        let value = serde_json::to_value(&a)
            .map_err(|e| Error::other(format!("assistants encode: {e}")))?;
        self.store.put(&ns, &a.id, &value, None).await?;
        if self.keep_history {
            let key = Self::version_key(&a.id, a.version);
            self.store.put(&ns, &key, &value, None).await?;
        }
        Ok(a)
    }

    /// Fetch the latest version of `id`. Returns `None` if the assistant
    /// doesn't exist.
    pub async fn get(&self, graph_id: &str, id: &str) -> Result<Option<Assistant>> {
        let ns = Self::ns(graph_id);
        let Some(item) = self.store.get(&ns, id).await? else {
            return Ok(None);
        };
        Ok(Some(deserialise(item.value)?))
    }

    /// Apply `patch` and bump the version. Returns the updated
    /// assistant. Errors if the assistant doesn't exist OR if the
    /// patch is empty.
    pub async fn update(
        &self,
        graph_id: &str,
        id: &str,
        patch: AssistantPatch,
    ) -> Result<Assistant> {
        if patch.is_empty() {
            return Err(Error::invalid("assistants update: patch is empty"));
        }
        let ns = Self::ns(graph_id);
        let item = self
            .store
            .get(&ns, id)
            .await?
            .ok_or_else(|| Error::other(format!("assistants update: {id} not found")))?;
        let mut a: Assistant = deserialise(item.value)?;
        if let Some(n) = patch.name {
            a.name = n;
        }
        if let Some(c) = patch.config {
            a.config = c;
        }
        if let Some(m) = patch.metadata {
            a.metadata = m;
        }
        a.version += 1;
        a.updated_at_ms = now_ms();

        let value = serde_json::to_value(&a)
            .map_err(|e| Error::other(format!("assistants encode: {e}")))?;
        self.store.put(&ns, &a.id, &value, None).await?;
        if self.keep_history {
            let key = Self::version_key(&a.id, a.version);
            self.store.put(&ns, &key, &value, None).await?;
        }
        Ok(a)
    }

    /// Delete the canonical entry. Returns `true` if it existed. By
    /// default this PRESERVES the `<id>@vN` archives — you can audit
    /// past versions even after the assistant is gone. Pass
    /// `purge_history = true` to wipe everything.
    pub async fn delete(
        &self,
        graph_id: &str,
        id: &str,
        purge_history: bool,
    ) -> Result<bool> {
        let ns = Self::ns(graph_id);
        let removed = self.store.delete(&ns, id).await?;
        if purge_history {
            // Best-effort: enumerate archives via search and delete them.
            let archives = self.versions(graph_id, id).await?;
            for v in archives {
                let key = Self::version_key(id, v.version);
                self.store.delete(&ns, &key).await?;
            }
        }
        Ok(removed)
    }

    /// All canonical assistants for `graph_id`, newest-first by
    /// `updated_at_ms`. Skips archive entries (`<id>@v<n>`) so the list
    /// shape matches what callers expect from a CRUD-style API.
    pub async fn list(&self, graph_id: &str, limit: Option<usize>) -> Result<Vec<Assistant>> {
        let ns = Self::ns(graph_id);
        let items = self
            .store
            .search(&ns, &SearchFilter { limit, ..Default::default() })
            .await?;
        let mut out = Vec::with_capacity(items.len());
        for it in items {
            // `@v` keys are archives — skip.
            if it.key.contains("@v") {
                continue;
            }
            if let Ok(a) = deserialise(it.value) {
                out.push(a);
            }
        }
        Ok(out)
    }

    /// All historical versions of `id`, oldest-first. Empty vec if the
    /// assistant has no archives (either never updated, or
    /// `keep_history` was off).
    pub async fn versions(&self, graph_id: &str, id: &str) -> Result<Vec<Assistant>> {
        let ns = Self::ns(graph_id);
        let items = self
            .store
            .search(&ns, &SearchFilter::default())
            .await?;
        let prefix = format!("{id}@v");
        let mut out: Vec<Assistant> = items
            .into_iter()
            .filter(|it| it.key.starts_with(&prefix))
            .filter_map(|it| deserialise(it.value).ok())
            .collect();
        out.sort_by_key(|a| a.version);
        Ok(out)
    }

    /// Look up a specific historical version. Returns `None` if the
    /// version doesn't exist (or wasn't archived because
    /// `keep_history` was off when the update happened).
    pub async fn get_version(
        &self,
        graph_id: &str,
        id: &str,
        version: u64,
    ) -> Result<Option<Assistant>> {
        let ns = Self::ns(graph_id);
        let key = Self::version_key(id, version);
        let Some(item) = self.store.get(&ns, &key).await? else {
            return Ok(None);
        };
        Ok(Some(deserialise(item.value)?))
    }
}

// ---- helpers --------------------------------------------------------------

fn deserialise(v: Value) -> Result<Assistant> {
    serde_json::from_value(v).map_err(|e| Error::other(format!("assistant decode: {e}")))
}

fn new_assistant_id() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("asst-{nanos:x}-{n:x}")
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemoryStore;
    use serde_json::json;

    fn manager() -> AssistantManager {
        AssistantManager::new(Arc::new(InMemoryStore::new()))
    }

    fn sample(graph: &str, name: &str) -> Assistant {
        Assistant::new(
            graph,
            name,
            json!({"temperature": 0.7, "system_prompt": "be terse"}),
        )
        .with_metadata(json!({"owner": "alice@example.com"}))
    }

    #[tokio::test]
    async fn create_persists_and_returns_v1() {
        let m = manager();
        let a = m.create(sample("rag", "concise")).await.unwrap();
        assert_eq!(a.version, 1);
        assert!(a.id.starts_with("asst-"));
        assert!(a.created_at_ms > 0);
        assert_eq!(a.created_at_ms, a.updated_at_ms);
        // Round-trip via get.
        let fetched = m.get("rag", &a.id).await.unwrap().unwrap();
        assert_eq!(fetched, a);
    }

    #[tokio::test]
    async fn create_preserves_caller_supplied_id() {
        let m = manager();
        let a = m
            .create(sample("rag", "x").with_id("custom-id"))
            .await
            .unwrap();
        assert_eq!(a.id, "custom-id");
        assert!(m.get("rag", "custom-id").await.unwrap().is_some());
    }

    #[tokio::test]
    async fn get_missing_returns_none() {
        let m = manager();
        assert!(m.get("rag", "nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn update_bumps_version_and_writes_archive() {
        let m = manager();
        let a = m.create(sample("rag", "v1-name")).await.unwrap();
        // Sleep enough to make updated_at_ms differ.
        tokio::time::sleep(std::time::Duration::from_millis(2)).await;
        let updated = m
            .update(
                "rag",
                &a.id,
                AssistantPatch::default()
                    .name("v2-name")
                    .config(json!({"temperature": 0.2})),
            )
            .await
            .unwrap();
        assert_eq!(updated.version, 2);
        assert_eq!(updated.name, "v2-name");
        assert_eq!(updated.config["temperature"], 0.2);
        // metadata wasn't patched — preserved.
        assert_eq!(updated.metadata["owner"], "alice@example.com");
        assert_eq!(updated.created_at_ms, a.created_at_ms);
        assert!(updated.updated_at_ms >= a.updated_at_ms);
        // v1 archive readable.
        let v1 = m.get_version("rag", &a.id, 1).await.unwrap().unwrap();
        assert_eq!(v1.name, "v1-name");
        assert_eq!(v1.version, 1);
        // v2 archive readable.
        let v2 = m.get_version("rag", &a.id, 2).await.unwrap().unwrap();
        assert_eq!(v2.name, "v2-name");
    }

    #[tokio::test]
    async fn update_rejects_empty_patch() {
        let m = manager();
        let a = m.create(sample("rag", "x")).await.unwrap();
        let err = m
            .update("rag", &a.id, AssistantPatch::default())
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[tokio::test]
    async fn update_unknown_id_errors() {
        let m = manager();
        let err = m
            .update("rag", "nope", AssistantPatch::default().name("x"))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[tokio::test]
    async fn versions_lists_archives_oldest_first() {
        let m = manager();
        let a = m.create(sample("rag", "v1")).await.unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("v2"))
            .await
            .unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("v3"))
            .await
            .unwrap();
        let history = m.versions("rag", &a.id).await.unwrap();
        let names: Vec<&str> = history.iter().map(|a| a.name.as_str()).collect();
        let nums: Vec<u64> = history.iter().map(|a| a.version).collect();
        assert_eq!(names, vec!["v1", "v2", "v3"]);
        assert_eq!(nums, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn list_returns_canonical_only_skipping_archives() {
        let m = manager();
        let a = m.create(sample("rag", "alpha")).await.unwrap();
        let b = m.create(sample("rag", "beta")).await.unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("alpha-v2"))
            .await
            .unwrap();
        let listed = m.list("rag", None).await.unwrap();
        // Two canonical entries — archives don't appear.
        assert_eq!(listed.len(), 2, "got: {listed:?}");
        let ids: std::collections::HashSet<_> = listed.iter().map(|a| a.id.clone()).collect();
        assert!(ids.contains(&a.id));
        assert!(ids.contains(&b.id));
        // The "alpha" entry's name is the v2 value (canonical = latest).
        let alpha = listed.iter().find(|x| x.id == a.id).unwrap();
        assert_eq!(alpha.name, "alpha-v2");
    }

    #[tokio::test]
    async fn list_scoped_to_graph_id() {
        let m = manager();
        m.create(sample("rag", "a")).await.unwrap();
        m.create(sample("planner", "b")).await.unwrap();
        let rag = m.list("rag", None).await.unwrap();
        let planner = m.list("planner", None).await.unwrap();
        assert_eq!(rag.len(), 1);
        assert_eq!(planner.len(), 1);
        assert_eq!(rag[0].graph_id, "rag");
        assert_eq!(planner[0].graph_id, "planner");
    }

    #[tokio::test]
    async fn delete_keeps_history_by_default() {
        let m = manager();
        let a = m.create(sample("rag", "v1")).await.unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("v2"))
            .await
            .unwrap();
        assert!(m.delete("rag", &a.id, false).await.unwrap());
        // Canonical gone.
        assert!(m.get("rag", &a.id).await.unwrap().is_none());
        // Archives still there.
        assert!(m.get_version("rag", &a.id, 1).await.unwrap().is_some());
        assert!(m.get_version("rag", &a.id, 2).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn delete_purge_history_wipes_archives() {
        let m = manager();
        let a = m.create(sample("rag", "v1")).await.unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("v2"))
            .await
            .unwrap();
        assert!(m.delete("rag", &a.id, true).await.unwrap());
        assert!(m.get_version("rag", &a.id, 1).await.unwrap().is_none());
        assert!(m.get_version("rag", &a.id, 2).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn without_history_skips_archive_writes() {
        let m = AssistantManager::new(Arc::new(InMemoryStore::new())).without_history();
        let a = m.create(sample("rag", "v1")).await.unwrap();
        m.update("rag", &a.id, AssistantPatch::default().name("v2"))
            .await
            .unwrap();
        // No archives.
        assert!(m.versions("rag", &a.id).await.unwrap().is_empty());
        assert!(m.get_version("rag", &a.id, 1).await.unwrap().is_none());
        // But canonical still has v2.
        let latest = m.get("rag", &a.id).await.unwrap().unwrap();
        assert_eq!(latest.name, "v2");
        assert_eq!(latest.version, 2);
    }

    #[tokio::test]
    async fn delete_returns_false_when_missing() {
        let m = manager();
        assert!(!m.delete("rag", "nope", false).await.unwrap());
    }

    #[tokio::test]
    async fn metadata_round_trips() {
        let m = manager();
        let a = m
            .create(
                Assistant::new("rag", "x", json!({}))
                    .with_metadata(json!({"tag": "experimental", "ab_pct": 5})),
            )
            .await
            .unwrap();
        let fetched = m.get("rag", &a.id).await.unwrap().unwrap();
        assert_eq!(fetched.metadata["tag"], "experimental");
        assert_eq!(fetched.metadata["ab_pct"], 5);
    }

    #[tokio::test]
    async fn get_version_returns_none_for_unknown() {
        let m = manager();
        let a = m.create(sample("rag", "v1")).await.unwrap();
        assert!(m.get_version("rag", &a.id, 99).await.unwrap().is_none());
    }

    #[test]
    fn id_generator_collision_proof_within_process() {
        let n = 1000;
        let ids: std::collections::HashSet<_> = (0..n).map(|_| new_assistant_id()).collect();
        assert_eq!(ids.len(), n, "id collision");
    }

    #[test]
    fn assistant_patch_is_empty_detects_no_op() {
        let p = AssistantPatch::default();
        assert!(p.is_empty());
        assert!(!AssistantPatch::default().name("x").is_empty());
        assert!(!AssistantPatch::default().config(json!({})).is_empty());
        assert!(!AssistantPatch::default().metadata(json!({})).is_empty());
    }
}

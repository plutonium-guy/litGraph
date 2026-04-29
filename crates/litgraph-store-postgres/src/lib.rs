//! Postgres-backed implementation of [`litgraph_core::Store`] — durable
//! distributed long-term memory for agents.
//!
//! Counterpart to the in-process `InMemoryStore` shipped in core. Use
//! Postgres when:
//! - multiple processes need to read/write the same memories
//!   (LangGraph 1.1 cross-thread Store semantics);
//! - data must outlive process restarts (preferences, learned facts);
//! - you want to use one store for both the checkpointer (already
//!   ships postgres) and long-term memory.
//!
//! # Storage layout
//!
//! ```sql
//! CREATE TABLE IF NOT EXISTS litgraph_kv_store (
//!     namespace      TEXT[] NOT NULL,
//!     key            TEXT   NOT NULL,
//!     value          JSONB  NOT NULL,
//!     expires_at_ms  BIGINT,
//!     created_at_ms  BIGINT NOT NULL,
//!     updated_at_ms  BIGINT NOT NULL,
//!     PRIMARY KEY (namespace, key)
//! );
//! CREATE INDEX IF NOT EXISTS idx_litgraph_kv_namespace
//!     ON litgraph_kv_store USING GIN (namespace);
//! CREATE INDEX IF NOT EXISTS idx_litgraph_kv_expires
//!     ON litgraph_kv_store (expires_at_ms)
//!     WHERE expires_at_ms IS NOT NULL;
//! ```
//!
//! Namespace is a real `TEXT[]` so prefix matching uses a single
//! parameterised condition (`namespace[1:N] = $prefix`) rather than
//! a string-concat hack.
//!
//! # TTL semantics
//!
//! Expired items return `None` from `get` AND get evicted lazily on
//! that read (mirrors `InMemoryStore`). Background sweep is the
//! caller's job — write a cron that runs:
//!
//! ```sql
//! DELETE FROM litgraph_kv_store
//! WHERE expires_at_ms IS NOT NULL AND expires_at_ms <= EXTRACT(EPOCH FROM now()) * 1000;
//! ```
//!
//! # Why not let the SQL filter sub-set on JSON paths
//!
//! Postgres can do `value #> '{a,b}' = $expected` natively, and we use
//! that. Each `(path, expected)` in the [`SearchFilter::matches`]
//! becomes one AND clause in the prepared statement. Up to 8 filter
//! clauses; beyond that we bail to client-side filter (predictable plan
//! costs, fewer surprises with weird JSON).

use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use litgraph_core::store::{Namespace, SearchFilter, Store, StoreItem};
use litgraph_core::{Error, Result};
use serde_json::Value;
use tokio_postgres::types::ToSql;
use tokio_postgres::NoTls;
use tracing::debug;

const DDL: &str = r#"
CREATE TABLE IF NOT EXISTS litgraph_kv_store (
    namespace      TEXT[] NOT NULL,
    key            TEXT   NOT NULL,
    value          JSONB  NOT NULL,
    expires_at_ms  BIGINT,
    created_at_ms  BIGINT NOT NULL,
    updated_at_ms  BIGINT NOT NULL,
    PRIMARY KEY (namespace, key)
);
CREATE INDEX IF NOT EXISTS idx_litgraph_kv_namespace
    ON litgraph_kv_store USING GIN (namespace);
CREATE INDEX IF NOT EXISTS idx_litgraph_kv_expires
    ON litgraph_kv_store (expires_at_ms)
    WHERE expires_at_ms IS NOT NULL;
"#;

/// Hard upper bound on `matches` filter clauses applied SQL-side.
/// Beyond this we trust the index less than client-side filter; bail
/// out and let the caller's prefix narrow first.
const MAX_SQL_MATCH_FILTERS: usize = 8;

/// Postgres-backed store. Cheap to clone (`Pool` is internally `Arc`'d).
#[derive(Clone)]
pub struct PostgresStore {
    pool: Pool,
}

impl PostgresStore {
    /// Connect via libpq DSN. Schema created idempotently on first use.
    pub async fn connect(dsn: &str) -> Result<Self> {
        let mut cfg = Config::new();
        cfg.url = Some(dsn.to_string());
        cfg.manager = Some(ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        });
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        let this = Self { pool };
        this.init().await?;
        Ok(this)
    }

    /// Reuse an existing pool — preferred when the app already pools
    /// Postgres for the checkpointer / chat-history / cache.
    pub async fn from_pool(pool: Pool) -> Result<Self> {
        let this = Self { pool };
        this.init().await?;
        Ok(this)
    }

    async fn init(&self) -> Result<()> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        client
            .batch_execute(DDL)
            .await
            .map_err(|e| Error::other(format!("pg_store schema: {e}")))?;
        debug!("pg_store schema ready");
        Ok(())
    }

    /// Manually evict expired rows. Called by `evict_expired()`; exposed
    /// publicly so callers can wire it to a cron job. Returns # of rows
    /// removed.
    pub async fn evict_expired(&self) -> Result<u64> {
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        let now = now_ms_i64();
        let n = client
            .execute(
                "DELETE FROM litgraph_kv_store \
                 WHERE expires_at_ms IS NOT NULL AND expires_at_ms <= $1",
                &[&now],
            )
            .await
            .map_err(|e| Error::other(format!("pg_store evict: {e}")))?;
        Ok(n)
    }
}

#[async_trait]
impl Store for PostgresStore {
    async fn put(
        &self,
        namespace: &Namespace,
        key: &str,
        value: &Value,
        ttl_ms: Option<u64>,
    ) -> Result<()> {
        let ns: Vec<String> = namespace.to_vec();
        let now = now_ms_i64();
        let expires_at: Option<i64> = ttl_ms.map(|t| now + t as i64);
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        // Preserve created_at across upserts via DO UPDATE WHERE.
        client
            .execute(
                "INSERT INTO litgraph_kv_store \
                   (namespace, key, value, expires_at_ms, created_at_ms, updated_at_ms) \
                 VALUES ($1, $2, $3, $4, $5, $5) \
                 ON CONFLICT (namespace, key) DO UPDATE SET \
                   value = EXCLUDED.value, \
                   expires_at_ms = EXCLUDED.expires_at_ms, \
                   updated_at_ms = EXCLUDED.updated_at_ms",
                &[&ns, &key, value, &expires_at, &now],
            )
            .await
            .map_err(|e| Error::other(format!("pg_store put: {e}")))?;
        Ok(())
    }

    async fn get(&self, namespace: &Namespace, key: &str) -> Result<Option<StoreItem>> {
        let ns: Vec<String> = namespace.to_vec();
        let now = now_ms_i64();
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        let row_opt = client
            .query_opt(
                "SELECT namespace, key, value, expires_at_ms, created_at_ms, updated_at_ms \
                 FROM litgraph_kv_store \
                 WHERE namespace = $1 AND key = $2",
                &[&ns, &key],
            )
            .await
            .map_err(|e| Error::other(format!("pg_store get: {e}")))?;
        let Some(row) = row_opt else {
            return Ok(None);
        };
        let expires: Option<i64> = row.get("expires_at_ms");
        if matches!(expires, Some(t) if t <= now) {
            // Lazy eviction — same behaviour as InMemoryStore.
            client
                .execute(
                    "DELETE FROM litgraph_kv_store WHERE namespace = $1 AND key = $2",
                    &[&ns, &key],
                )
                .await
                .map_err(|e| Error::other(format!("pg_store evict-on-get: {e}")))?;
            return Ok(None);
        }
        Ok(Some(row_to_item(&row)?))
    }

    async fn delete(&self, namespace: &Namespace, key: &str) -> Result<bool> {
        let ns: Vec<String> = namespace.to_vec();
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;
        let n = client
            .execute(
                "DELETE FROM litgraph_kv_store WHERE namespace = $1 AND key = $2",
                &[&ns, &key],
            )
            .await
            .map_err(|e| Error::other(format!("pg_store delete: {e}")))?;
        Ok(n > 0)
    }

    async fn search(
        &self,
        namespace_prefix: &Namespace,
        filter: &SearchFilter,
    ) -> Result<Vec<StoreItem>> {
        let prefix: Vec<String> = namespace_prefix.to_vec();
        let now = now_ms_i64();
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;

        // We build the query incrementally so each parameter has a
        // stable index. Owned vec of Box<dyn ToSql + Sync> keeps the
        // borrow checker happy across the SQL string.
        let mut sql = String::from(
            "SELECT namespace, key, value, expires_at_ms, created_at_ms, updated_at_ms \
             FROM litgraph_kv_store WHERE ",
        );
        let mut params: Vec<Box<dyn ToSql + Sync + Send>> = Vec::new();

        // Prefix match. Empty prefix = everything.
        if prefix.is_empty() {
            sql.push_str("TRUE");
        } else {
            params.push(Box::new(prefix.clone()));
            // Generated `array_length`-aware prefix slice: namespace[1:N] = prefix.
            sql.push_str("namespace[1:array_length($1, 1)] = $1");
        }

        // TTL: drop expired rows.
        params.push(Box::new(now));
        sql.push_str(&format!(
            " AND (expires_at_ms IS NULL OR expires_at_ms > ${})",
            params.len()
        ));

        // SQL-side query_text — case-insensitive substring match against
        // the JSONB serialised as text. Mirrors InMemoryStore semantics
        // (.to_string().to_lowercase().contains).
        if let Some(q) = &filter.query_text {
            params.push(Box::new(q.clone()));
            sql.push_str(&format!(" AND value::text ILIKE '%' || ${} || '%'", params.len()));
        }

        // Path matches — apply up to N. Postgres `#>` returns the
        // referenced JSON; we compare with literal JSON.
        let n_path_filters = filter.matches.len().min(MAX_SQL_MATCH_FILTERS);
        for (path, expected) in filter.matches.iter().take(n_path_filters) {
            // Convert /a/b → {a,b} for postgres path syntax.
            let path_arr = json_pointer_to_pg_path(path);
            params.push(Box::new(path_arr));
            sql.push_str(&format!(" AND value #> ${}::text[] = ", params.len()));
            params.push(Box::new(expected.clone()));
            sql.push_str(&format!("${}::jsonb", params.len()));
        }

        sql.push_str(" ORDER BY updated_at_ms DESC");

        let limit = filter.limit;
        let offset = filter.offset.unwrap_or(0);
        if let Some(l) = limit {
            params.push(Box::new(l as i64));
            sql.push_str(&format!(" LIMIT ${}", params.len()));
        }
        if offset > 0 {
            params.push(Box::new(offset as i64));
            sql.push_str(&format!(" OFFSET ${}", params.len()));
        }

        let param_refs: Vec<&(dyn ToSql + Sync)> =
            params.iter().map(|p| p.as_ref() as &(dyn ToSql + Sync)).collect();

        let rows = client
            .query(&sql, &param_refs[..])
            .await
            .map_err(|e| Error::other(format!("pg_store search: {e} (sql={sql})")))?;

        // Apply remaining match filters client-side if we exceeded
        // MAX_SQL_MATCH_FILTERS. Same client-side logic as InMemoryStore.
        let extra_matches: Vec<_> = filter.matches.iter().skip(n_path_filters).cloned().collect();
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let item = row_to_item(&row)?;
            if !extra_matches.is_empty()
                && !extra_matches.iter().all(|(p, exp)| {
                    json_pointer_get(&item.value, p) == Some(exp.clone())
                })
            {
                continue;
            }
            out.push(item);
        }
        Ok(out)
    }

    async fn list_namespaces(
        &self,
        prefix: &Namespace,
        limit: Option<usize>,
    ) -> Result<Vec<Vec<String>>> {
        let p: Vec<String> = prefix.to_vec();
        let client = self
            .pool
            .get()
            .await
            .map_err(|e| Error::other(format!("pg_store pool: {e}")))?;

        let mut sql = String::from("SELECT DISTINCT namespace FROM litgraph_kv_store");
        let mut params: Vec<Box<dyn ToSql + Sync + Send>> = Vec::new();
        if !p.is_empty() {
            params.push(Box::new(p.clone()));
            sql.push_str(" WHERE namespace[1:array_length($1, 1)] = $1");
        }
        sql.push_str(" ORDER BY namespace");
        if let Some(l) = limit {
            params.push(Box::new(l as i64));
            sql.push_str(&format!(" LIMIT ${}", params.len()));
        }
        let param_refs: Vec<&(dyn ToSql + Sync)> =
            params.iter().map(|p| p.as_ref() as &(dyn ToSql + Sync)).collect();
        let rows = client
            .query(&sql, &param_refs[..])
            .await
            .map_err(|e| Error::other(format!("pg_store list_namespaces: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let ns: Vec<String> = row.get(0);
            out.push(ns);
        }
        Ok(out)
    }
}

// ---- helpers ---------------------------------------------------------------

fn row_to_item(row: &tokio_postgres::Row) -> Result<StoreItem> {
    let namespace: Vec<String> = row.get("namespace");
    let key: String = row.get("key");
    let value: Value = row.get("value");
    let expires_at_ms: Option<i64> = row.get("expires_at_ms");
    let created_at_ms: i64 = row.get("created_at_ms");
    let updated_at_ms: i64 = row.get("updated_at_ms");
    Ok(StoreItem {
        namespace,
        key,
        value,
        expires_at_ms: expires_at_ms.map(|x| x as u64),
        created_at_ms: created_at_ms as u64,
        updated_at_ms: updated_at_ms as u64,
    })
}

/// Convert RFC-6901 JSON Pointer (`/a/b`) into a Postgres `text[]`
/// path (`{a,b}`) suitable for the `#>` operator. Empty path → empty
/// array which makes `#>` return the whole document.
fn json_pointer_to_pg_path(path: &str) -> Vec<String> {
    let normalized = path.strip_prefix('/').unwrap_or(path);
    if normalized.is_empty() {
        return Vec::new();
    }
    normalized
        .split('/')
        .map(|s| s.replace("~1", "/").replace("~0", "~"))
        .collect()
}

/// Mirror of `litgraph_core::store::json_pointer_get` since that fn is
/// pub(crate). Tiny — re-deriving here keeps the PG crate self-contained.
fn json_pointer_get(value: &Value, path: &str) -> Option<Value> {
    let normalized = path.strip_prefix('/').unwrap_or(path);
    if normalized.is_empty() {
        return Some(value.clone());
    }
    let mut cur = value;
    for seg in normalized.split('/') {
        let unescaped = seg.replace("~1", "/").replace("~0", "~");
        cur = match cur {
            Value::Object(m) => m.get(&unescaped)?,
            Value::Array(a) => {
                let i: usize = unescaped.parse().ok()?;
                a.get(i)?
            }
            _ => return None,
        };
    }
    Some(cur.clone())
}

fn now_ms_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    //! Integration tests gated on `PG_STORE_DSN`. Run with:
    //! ```sh
    //! PG_STORE_DSN=postgres://localhost:5432/litgraph_test \
    //!     cargo test -p litgraph-store-postgres
    //! ```
    //!
    //! Each test uses a unique top-level namespace so they don't
    //! interfere when run in parallel against the same database.

    use super::*;
    use serde_json::json;

    fn try_dsn() -> Option<String> {
        std::env::var("PG_STORE_DSN").ok()
    }

    fn unique_ns(tag: &str) -> Vec<String> {
        use std::sync::atomic::{AtomicU64, Ordering};
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        vec!["test".into(), format!("{tag}-{nanos:x}-{n:x}")]
    }

    async fn connect_or_skip() -> Option<PostgresStore> {
        let dsn = try_dsn()?;
        Some(PostgresStore::connect(&dsn).await.unwrap())
    }

    #[tokio::test]
    async fn put_get_round_trip() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("rt");
        s.put(&ns, "k1", &json!({"a": 1, "b": "two"}), None).await.unwrap();
        let got = s.get(&ns, "k1").await.unwrap().unwrap();
        assert_eq!(got.namespace, ns);
        assert_eq!(got.key, "k1");
        assert_eq!(got.value, json!({"a": 1, "b": "two"}));
        assert!(got.expires_at_ms.is_none());
        assert!(got.created_at_ms > 0);
    }

    #[tokio::test]
    async fn put_upsert_preserves_created_at() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("upsert");
        s.put(&ns, "k", &json!(1), None).await.unwrap();
        let first = s.get(&ns, "k").await.unwrap().unwrap();
        // Sleep just enough to make updated_at differ.
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        s.put(&ns, "k", &json!(2), None).await.unwrap();
        let second = s.get(&ns, "k").await.unwrap().unwrap();
        assert_eq!(second.value, json!(2));
        assert_eq!(second.created_at_ms, first.created_at_ms);
        assert!(second.updated_at_ms >= first.updated_at_ms);
    }

    #[tokio::test]
    async fn missing_key_returns_none() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("missing");
        assert!(s.get(&ns, "nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn delete_returns_true_when_existed() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("del");
        s.put(&ns, "k", &json!("v"), None).await.unwrap();
        assert!(s.delete(&ns, "k").await.unwrap());
        // Idempotent.
        assert!(!s.delete(&ns, "k").await.unwrap());
    }

    #[tokio::test]
    async fn ttl_expires_on_get() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("ttl");
        s.put(&ns, "k", &json!("v"), Some(50)).await.unwrap();
        // Read before expiry.
        assert!(s.get(&ns, "k").await.unwrap().is_some());
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert!(s.get(&ns, "k").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn search_prefix_and_query_text() {
        let Some(s) = connect_or_skip().await else { return };
        let ns_root = unique_ns("search");
        let mut ns_a = ns_root.clone();
        ns_a.push("a".to_string());
        let mut ns_b = ns_root.clone();
        ns_b.push("b".to_string());
        s.put(&ns_a, "k1", &json!({"name": "Alice"}), None).await.unwrap();
        s.put(&ns_a, "k2", &json!({"name": "Bob"}), None).await.unwrap();
        s.put(&ns_b, "k3", &json!({"name": "Alice"}), None).await.unwrap();

        // Prefix search returns all under ns_root.
        let all = s
            .search(&ns_root, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(all.len(), 3);

        // query_text narrows to "Alice" rows.
        let alice = s
            .search(
                &ns_root,
                &SearchFilter {
                    query_text: Some("Alice".into()),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(alice.len(), 2);
        assert!(alice.iter().all(|it| it.value["name"] == "Alice"));

        // Nested prefix narrows further.
        let only_a = s
            .search(&ns_a, &SearchFilter::default())
            .await
            .unwrap();
        assert_eq!(only_a.len(), 2);
    }

    #[tokio::test]
    async fn search_path_matches_filter() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("match");
        s.put(&ns, "k1", &json!({"kind": "preference", "v": 1}), None).await.unwrap();
        s.put(&ns, "k2", &json!({"kind": "fact", "v": 2}), None).await.unwrap();
        s.put(&ns, "k3", &json!({"kind": "preference", "v": 3}), None).await.unwrap();
        let prefs = s
            .search(
                &ns,
                &SearchFilter {
                    matches: vec![("/kind".into(), json!("preference"))],
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(prefs.len(), 2);
        assert!(prefs.iter().all(|it| it.value["kind"] == "preference"));
    }

    #[tokio::test]
    async fn search_limit_offset_obeyed() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("page");
        for i in 0..5 {
            s.put(&ns, &format!("k{i}"), &json!({"i": i}), None).await.unwrap();
        }
        let page = s
            .search(
                &ns,
                &SearchFilter {
                    limit: Some(2),
                    offset: Some(1),
                    ..Default::default()
                },
            )
            .await
            .unwrap();
        assert_eq!(page.len(), 2);
    }

    #[tokio::test]
    async fn search_skips_expired() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("exp-search");
        s.put(&ns, "live", &json!("alive"), None).await.unwrap();
        s.put(&ns, "dead", &json!("dead"), Some(20)).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(80)).await;
        let hits = s
            .search(&ns, &SearchFilter::default())
            .await
            .unwrap();
        // The expired row shouldn't appear in the result set.
        assert!(hits.iter().all(|it| it.key != "dead"));
    }

    #[tokio::test]
    async fn list_namespaces_reports_distinct_paths() {
        let Some(s) = connect_or_skip().await else { return };
        let root = unique_ns("listns");
        let mut ns_a = root.clone();
        ns_a.push("alpha".to_string());
        let mut ns_b = root.clone();
        ns_b.push("beta".to_string());
        s.put(&ns_a, "k", &json!(1), None).await.unwrap();
        s.put(&ns_b, "k", &json!(2), None).await.unwrap();
        let nss = s.list_namespaces(&root, None).await.unwrap();
        assert!(nss.iter().any(|n| n == &ns_a));
        assert!(nss.iter().any(|n| n == &ns_b));
    }

    #[tokio::test]
    async fn evict_expired_returns_count() {
        let Some(s) = connect_or_skip().await else { return };
        let ns = unique_ns("evict");
        s.put(&ns, "live", &json!(1), None).await.unwrap();
        s.put(&ns, "d1", &json!(2), Some(10)).await.unwrap();
        s.put(&ns, "d2", &json!(3), Some(10)).await.unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(60)).await;
        let n = s.evict_expired().await.unwrap();
        assert!(n >= 2, "expected at least 2 evictions, got {n}");
    }

    #[test]
    fn json_pointer_to_pg_path_round_trips() {
        assert_eq!(json_pointer_to_pg_path("/foo/bar"), vec!["foo", "bar"]);
        assert_eq!(json_pointer_to_pg_path("foo"), vec!["foo"]);
        let empty: Vec<String> = vec![];
        assert_eq!(json_pointer_to_pg_path(""), empty);
        assert_eq!(json_pointer_to_pg_path("/"), empty);
    }

    #[test]
    fn json_pointer_to_pg_path_unescapes() {
        // ~0 → ~, ~1 → /, per RFC 6901.
        assert_eq!(json_pointer_to_pg_path("/a~1b/c~0"), vec!["a/b", "c~"]);
    }

    #[test]
    fn json_pointer_get_walks_objects_and_arrays() {
        let v = json!({"a": [{"b": 1}, {"b": 2}]});
        assert_eq!(json_pointer_get(&v, "/a/0/b"), Some(json!(1)));
        assert_eq!(json_pointer_get(&v, "/a/1/b"), Some(json!(2)));
        assert_eq!(json_pointer_get(&v, "/a/9/b"), None);
        assert_eq!(json_pointer_get(&v, "/missing"), None);
    }
}

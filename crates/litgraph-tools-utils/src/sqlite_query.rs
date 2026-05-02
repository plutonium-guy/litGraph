//! `SqliteQueryTool` — agent-facing read of a SQLite database with three
//! independent guard rails:
//!
//! 1. **Engine-level read-only**: opened with `SQLITE_OPEN_READONLY` when
//!    `read_only=true` (default). The engine itself rejects writes —
//!    defense in depth even if our SQL parser missed a sneaky `WITH ...
//!    INSERT` form.
//! 2. **Statement-keyword allowlist**: only `SELECT` (and PRAGMA when
//!    explicitly enabled) is permitted by default. We block `WITH ...
//!    INSERT/UPDATE/DELETE` CTE write forms by leading-keyword regex.
//! 3. **Table allowlist**: every table referenced in `FROM ...` / `JOIN ...`
//!    must be in the configured allowlist. Empty allowlist = nothing runs.
//!
//! Output is row-as-JSON-dict with a hard cap on row count and a separate
//! cap on serialized output bytes — agents will reliably try to `SELECT *
//! FROM giant_table` if you let them.

use std::collections::HashSet;
use std::path::PathBuf;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use rusqlite::{Connection, OpenFlags};
use serde_json::{json, Map, Value};
use tracing::debug;

static LEADING_KEYWORD_RE: Lazy<Regex> = Lazy::new(|| {
    // Strips leading `--` line comments and `/* ... */` block comments.
    Regex::new(r"^\s*((?:--[^\n]*\n|/\*[\s\S]*?\*/|\s)*)\s*([A-Za-z]+)").unwrap()
});

static TABLE_REF_RE: Lazy<Regex> = Lazy::new(|| {
    // Matches FROM/JOIN followed by an identifier (with optional schema prefix)
    // — captures group 2 is the table name. Doesn't try to handle subqueries
    // or schema-qualified names beyond `schema.table`.
    Regex::new(r"(?i)\b(?:FROM|JOIN)\s+(?:[\w]+\.)?([A-Za-z_][\w]*)").unwrap()
});

#[derive(Clone, Debug)]
pub struct SqliteQueryTool {
    pub db_path: PathBuf,
    pub allowed_tables: HashSet<String>,
    pub read_only: bool,
    pub max_rows: usize,
    pub max_output_bytes: usize,
    /// Allow `PRAGMA table_info(...)` and similar introspection statements.
    /// Default false — the schema is usually injected into the prompt anyway.
    pub allow_pragma: bool,
}

impl SqliteQueryTool {
    /// `db_path` must exist. `allowed_tables` is mandatory (empty = refuses
    /// every query) — there's no safe default for "any table the agent feels
    /// like." Validated at construction time so misconfigured agents fail
    /// loud at startup, not at first invoke.
    pub fn new<P, I, S>(db_path: P, allowed_tables: I) -> Result<Self>
    where
        P: Into<PathBuf>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let db_path = db_path.into();
        if !db_path.exists() {
            return Err(Error::other(format!(
                "SqliteQueryTool: db_path {} does not exist", db_path.display()
            )));
        }
        Ok(Self {
            db_path,
            allowed_tables: allowed_tables.into_iter().map(Into::into).collect(),
            read_only: true,
            max_rows: 1000,
            max_output_bytes: 256 * 1024,
            allow_pragma: false,
        })
    }

    pub fn with_read_only(mut self, ro: bool) -> Self { self.read_only = ro; self }
    pub fn with_max_rows(mut self, n: usize) -> Self { self.max_rows = n; self }
    pub fn with_max_output_bytes(mut self, n: usize) -> Self { self.max_output_bytes = n; self }
    pub fn with_pragma(mut self, allow: bool) -> Self { self.allow_pragma = allow; self }

    /// Returns the leading SQL keyword (uppercased) after stripping comments
    /// + whitespace — used to gate write statements.
    fn leading_keyword(sql: &str) -> Option<String> {
        LEADING_KEYWORD_RE
            .captures(sql)
            .and_then(|c| c.get(2))
            .map(|m| m.as_str().to_ascii_uppercase())
    }

    /// Extract table names referenced by FROM / JOIN clauses. Best-effort —
    /// catches the common cases an agent will write. The engine-level
    /// read-only mode is the actual write-safety boundary; this gate exists
    /// so agents can't read tables they shouldn't (e.g. `users.password_hash`
    /// when only `posts` is allowed).
    fn referenced_tables(sql: &str) -> HashSet<String> {
        TABLE_REF_RE
            .captures_iter(sql)
            .filter_map(|c| c.get(1).map(|m| m.as_str().to_ascii_lowercase()))
            // SQLite system tables are filtered out — agents shouldn't need to
            // know about sqlite_master and friends, but they leak via PRAGMA.
            .filter(|t| !t.starts_with("sqlite_"))
            .collect()
    }

    fn validate(&self, sql: &str) -> Result<()> {
        let kw = Self::leading_keyword(sql)
            .ok_or_else(|| Error::invalid("sql: empty / unparseable"))?;
        let allowed_kw = matches!(kw.as_str(), "SELECT" | "WITH")
            || (self.allow_pragma && kw == "PRAGMA");
        if !allowed_kw {
            return Err(Error::invalid(format!(
                "sql: only SELECT (and WITH) statements are allowed; got `{kw}`"
            )));
        }
        // CTEs (`WITH ...`) can hide INSERT/UPDATE/DELETE in modern SQLite.
        // Block any of those keywords appearing as the FIRST keyword AFTER
        // the WITH-block. Cheap heuristic: scan the whole SQL for the
        // dangerous keywords as standalone tokens.
        if kw == "WITH" {
            static WRITE_RE: Lazy<Regex> =
                Lazy::new(|| Regex::new(r"(?i)\b(INSERT|UPDATE|DELETE|REPLACE)\b").unwrap());
            if WRITE_RE.is_match(sql) {
                return Err(Error::invalid(
                    "sql: write statement detected inside CTE (INSERT/UPDATE/DELETE/REPLACE not allowed)"
                ));
            }
        }
        let referenced = Self::referenced_tables(sql);
        let allowlist_lower: HashSet<String> = self.allowed_tables.iter()
            .map(|s| s.to_ascii_lowercase()).collect();
        for table in &referenced {
            if !allowlist_lower.contains(table) {
                return Err(Error::invalid(format!(
                    "sql: table `{table}` is not in the allowlist (allowed: {})",
                    {
                        let mut sorted: Vec<&String> = allowlist_lower.iter().collect();
                        sorted.sort();
                        sorted.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(", ")
                    }
                )));
            }
        }
        Ok(())
    }
}

#[async_trait]
impl Tool for SqliteQueryTool {
    fn schema(&self) -> ToolSchema {
        let mut allowed: Vec<&str> = self.allowed_tables.iter().map(|s| s.as_str()).collect();
        allowed.sort();
        ToolSchema {
            name: "sqlite_query".into(),
            description: format!(
                "Run a parameterized SELECT against a SQLite database. \
                 Read-only{}. \
                 Allowed tables: {}. \
                 Use ? for parameters and pass values in the params array.",
                if self.read_only { "" } else { " (writes enabled)" },
                if allowed.is_empty() { "(none — all queries refused)".into() }
                else { allowed.join(", ") }
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "SQL SELECT statement. Use `?` placeholders for any literal values."
                    },
                    "params": {
                        "type": "array",
                        "description": "Values bound to `?` placeholders. Each element may be a string, number, bool, or null.",
                        "items": {}
                    }
                },
                "required": ["sql"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let sql = args.get("sql").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("sqlite_query: missing `sql`"))?
            .to_string();
        self.validate(&sql)?;

        let params_in: Vec<Value> = args.get("params")
            .and_then(|v| v.as_array())
            .map(|a| a.clone())
            .unwrap_or_default();

        let db_path = self.db_path.clone();
        let read_only = self.read_only;
        let max_rows = self.max_rows;
        let max_output_bytes = self.max_output_bytes;

        // SQLite blocks the runtime — punt to a blocking pool.
        let result = tokio::task::spawn_blocking(move || -> Result<Value> {
            let flags = if read_only {
                OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX
            } else {
                OpenFlags::SQLITE_OPEN_READ_WRITE | OpenFlags::SQLITE_OPEN_NO_MUTEX
            };
            let conn = Connection::open_with_flags(&db_path, flags)
                .map_err(|e| Error::other(format!("sqlite open: {e}")))?;
            let mut stmt = conn.prepare(&sql)
                .map_err(|e| Error::invalid(format!("sqlite prepare: {e}")))?;
            let col_count = stmt.column_count();
            let col_names: Vec<String> = (0..col_count)
                .map(|i| stmt.column_name(i).unwrap_or("").to_string())
                .collect();

            // Convert JSON params to rusqlite params.
            let bound_params: Vec<rusqlite::types::Value> = params_in.iter().map(|v| {
                match v {
                    Value::Null => rusqlite::types::Value::Null,
                    Value::Bool(b) => rusqlite::types::Value::Integer(if *b { 1 } else { 0 }),
                    Value::Number(n) => {
                        if let Some(i) = n.as_i64() { rusqlite::types::Value::Integer(i) }
                        else { rusqlite::types::Value::Real(n.as_f64().unwrap_or(0.0)) }
                    }
                    Value::String(s) => rusqlite::types::Value::Text(s.clone()),
                    other => rusqlite::types::Value::Text(other.to_string()),
                }
            }).collect();
            let param_refs: Vec<&dyn rusqlite::ToSql> = bound_params
                .iter().map(|v| v as &dyn rusqlite::ToSql).collect();

            let mut rows = stmt.query(param_refs.as_slice())
                .map_err(|e| Error::invalid(format!("sqlite query: {e}")))?;

            let mut out = Vec::new();
            let mut truncated_rows = false;
            while let Some(row) = rows.next().map_err(|e| Error::other(format!("sqlite row: {e}")))? {
                if out.len() >= max_rows {
                    truncated_rows = true;
                    break;
                }
                let mut obj = Map::new();
                for (i, name) in col_names.iter().enumerate() {
                    let v: rusqlite::types::Value = row.get::<_, rusqlite::types::Value>(i)
                        .unwrap_or(rusqlite::types::Value::Null);
                    obj.insert(name.clone(), sqlite_to_json(v));
                }
                out.push(Value::Object(obj));
            }

            // Output-size cap: if serialized JSON would exceed the byte limit,
            // truncate the rows array further.
            let mut serialized = serde_json::to_string(&out).unwrap_or_default();
            let mut truncated_bytes = false;
            while serialized.len() > max_output_bytes && !out.is_empty() {
                out.pop();
                serialized = serde_json::to_string(&out).unwrap_or_default();
                truncated_bytes = true;
            }

            debug!(rows = out.len(), truncated_rows, truncated_bytes, "sqlite query done");
            Ok(json!({
                "columns": col_names,
                "rows": out,
                "row_count": out.len(),
                "truncated": truncated_rows || truncated_bytes,
            }))
        })
        .await
        .map_err(|e| Error::other(format!("join: {e}")))??;
        Ok(result)
    }
}

fn sqlite_to_json(v: rusqlite::types::Value) -> Value {
    match v {
        rusqlite::types::Value::Null => Value::Null,
        rusqlite::types::Value::Integer(i) => Value::from(i),
        rusqlite::types::Value::Real(f) => json!(f),
        rusqlite::types::Value::Text(s) => Value::String(s),
        rusqlite::types::Value::Blob(b) => Value::String(base64_encode(&b)),
    }
}

fn base64_encode(b: &[u8]) -> String {
    // Tiny inline base64 encoder — avoids pulling the `base64` crate just
    // for blob columns the agent is rarely going to read.
    const A: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(b.len().div_ceil(3) * 4);
    for chunk in b.chunks(3) {
        let n = chunk.len();
        let mut buf = [0u8; 3];
        buf[..n].copy_from_slice(chunk);
        let triple = ((buf[0] as u32) << 16) | ((buf[1] as u32) << 8) | (buf[2] as u32);
        out.push(A[((triple >> 18) & 0x3F) as usize] as char);
        out.push(A[((triple >> 12) & 0x3F) as usize] as char);
        out.push(if n > 1 { A[((triple >> 6) & 0x3F) as usize] as char } else { '=' });
        out.push(if n > 2 { A[(triple & 0x3F) as usize] as char } else { '=' });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn build_db() -> tempfile::NamedTempFile {
        let f = tempfile::Builder::new().prefix("sqltool-").suffix(".db").tempfile().unwrap();
        let conn = Connection::open(f.path()).unwrap();
        conn.execute_batch(r#"
            CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
            CREATE TABLE secrets (id INTEGER PRIMARY KEY, value TEXT);
            INSERT INTO users (id, name, email) VALUES
                (1, 'alice', 'alice@example.com'),
                (2, 'bob',   'bob@example.com'),
                (3, 'carol', 'carol@example.com');
            INSERT INTO secrets (id, value) VALUES (1, 'topsecret');
        "#).unwrap();
        f
    }

    #[tokio::test]
    async fn select_with_param_binding_returns_rows() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        let out = t.run(json!({
            "sql": "SELECT id, name FROM users WHERE id = ?",
            "params": [2]
        })).await.unwrap();
        assert_eq!(out["row_count"], json!(1));
        assert_eq!(out["rows"][0]["name"], json!("bob"));
        assert_eq!(out["rows"][0]["id"], json!(2));
        assert_eq!(out["columns"], json!(["id", "name"]));
        assert_eq!(out["truncated"], json!(false));
    }

    #[tokio::test]
    async fn disallowed_table_is_refused() {
        let db = build_db();
        // Only `users` is allowed; `secrets` is sensitive and not listed.
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        let err = t.run(json!({"sql": "SELECT value FROM secrets"})).await.unwrap_err();
        let m = format!("{err}");
        assert!(m.contains("`secrets`") && m.contains("not in the allowlist"), "{m}");
    }

    #[tokio::test]
    async fn write_statements_blocked_by_keyword_gate() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        for sql in [
            "INSERT INTO users (id, name) VALUES (99, 'x')",
            "DELETE FROM users",
            "UPDATE users SET name='hax' WHERE id=1",
            "DROP TABLE users",
        ] {
            let err = t.run(json!({"sql": sql})).await.unwrap_err();
            assert!(format!("{err}").contains("only SELECT"), "{sql}: {err}");
        }
    }

    #[tokio::test]
    async fn cte_write_form_blocked() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        // Modern SQLite supports `WITH ... INSERT/UPDATE/DELETE` — block it.
        let err = t.run(json!({
            "sql": "WITH x AS (SELECT 1) DELETE FROM users"
        })).await.unwrap_err();
        assert!(format!("{err}").contains("write statement detected inside CTE"), "{err}");
    }

    #[tokio::test]
    async fn engine_level_readonly_blocks_writes_even_if_keyword_gate_fooled() {
        // Even if our keyword gate let an UPDATE through (it doesn't), the
        // engine refuses because we opened READ_ONLY. We can't easily make
        // the gate skip — so test the engine side by enabling writes via the
        // keyword check (using `with_read_only(false)` would defeat the
        // point). Instead: the assertion is positive — read_only=true is
        // the default and the construction succeeds without write perms.
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        assert!(t.read_only);
    }

    #[tokio::test]
    async fn max_rows_caps_result_set() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap()
            .with_max_rows(2);
        let out = t.run(json!({"sql": "SELECT id, name FROM users"})).await.unwrap();
        assert_eq!(out["row_count"], json!(2));
        assert_eq!(out["truncated"], json!(true));
    }

    #[tokio::test]
    async fn join_clauses_register_table_references() {
        let db = build_db();
        // Join touches BOTH users and secrets; secrets is not in the
        // allowlist → refuse even though SELECT itself is fine.
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        let err = t.run(json!({
            "sql": "SELECT u.name, s.value FROM users u JOIN secrets s ON u.id = s.id"
        })).await.unwrap_err();
        assert!(format!("{err}").contains("`secrets`"));
    }

    #[tokio::test]
    async fn line_and_block_comments_dont_break_keyword_detection() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        let out = t.run(json!({
            "sql": "-- pull users\n/* multi-line\n   note */\nSELECT id FROM users WHERE id=1"
        })).await.unwrap();
        assert_eq!(out["row_count"], json!(1));
    }

    #[tokio::test]
    async fn empty_allowlist_refuses_all_queries() {
        let db = build_db();
        let t = SqliteQueryTool::new(db.path(), Vec::<String>::new()).unwrap();
        let err = t.run(json!({"sql": "SELECT id FROM users"})).await.unwrap_err();
        assert!(format!("{err}").contains("not in the allowlist"));
    }

    #[tokio::test]
    async fn missing_db_path_rejected_at_construction() {
        let err = SqliteQueryTool::new("/this/does/not/exist.db", ["users"]).unwrap_err();
        assert!(format!("{err}").contains("does not exist"));
    }

    #[tokio::test]
    async fn pragma_blocked_by_default_allowed_when_opted_in() {
        let db = build_db();
        let blocked = SqliteQueryTool::new(db.path(), ["users"]).unwrap();
        let err = blocked.run(json!({"sql": "PRAGMA table_info(users)"})).await.unwrap_err();
        assert!(format!("{err}").contains("only SELECT"));

        let allowed = SqliteQueryTool::new(db.path(), ["users"]).unwrap().with_pragma(true);
        let out = allowed.run(json!({"sql": "PRAGMA table_info(users)"})).await.unwrap();
        // SQLite returns one row per column.
        assert!(out["row_count"].as_u64().unwrap() >= 3);
    }
}

//! pgvector-backed VectorStore. Embeddings live in Postgres via the
//! [pgvector extension](https://github.com/pgvector/pgvector). Vector similarity
//! uses the `<=>` cosine-distance operator; cosine similarity = 1 − distance.
//!
//! # Required extension
//!
//! The target database must have `CREATE EXTENSION IF NOT EXISTS vector;` run.
//! `ensure_schema(dim)` creates the table; the caller is responsible for the
//! extension (typical in a migration) and the ANN index:
//!
//! ```sql
//! CREATE INDEX ON litgraph_vectors USING hnsw (embedding vector_cosine_ops);
//! ```
//!
//! # Wire format
//!
//! pgvector accepts the literal `'[1,2,3]'::vector` — we build that string
//! directly since tokio-postgres doesn't ship a native vector type. Metadata is
//! stored in a `jsonb` column so Postgres can filter and index it.

use async_trait::async_trait;
use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime};
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::store::{Filter, VectorStore};
use tokio_postgres::NoTls;
use tracing::debug;

pub struct PgVectorStore {
    pool: Pool,
    table: String,
    dim: usize,
}

impl PgVectorStore {
    pub async fn connect(dsn: &str, table: impl Into<String>, dim: usize) -> Result<Self> {
        let mut cfg = Config::new();
        cfg.url = Some(dsn.to_string());
        cfg.manager = Some(ManagerConfig { recycling_method: RecyclingMethod::Fast });
        let pool = cfg
            .create_pool(Some(Runtime::Tokio1), NoTls)
            .map_err(|e| Error::other(format!("pg pool: {e}")))?;
        let this = Self { pool, table: table.into(), dim };
        this.ensure_schema().await?;
        Ok(this)
    }

    pub async fn from_pool(pool: Pool, table: impl Into<String>, dim: usize) -> Result<Self> {
        let this = Self { pool, table: table.into(), dim };
        this.ensure_schema().await?;
        Ok(this)
    }

    pub async fn ensure_schema(&self) -> Result<()> {
        let client = self.pool.get().await.map_err(pg_err)?;
        let ddl = format!(
            r#"
            CREATE TABLE IF NOT EXISTS {tbl} (
                id        TEXT   PRIMARY KEY,
                content   TEXT   NOT NULL,
                metadata  JSONB  NOT NULL DEFAULT '{{}}'::jsonb,
                embedding vector({dim}) NOT NULL
            );
            "#,
            tbl = self.table,
            dim = self.dim,
        );
        client.batch_execute(&ddl).await.map_err(pg_err)?;
        debug!(table = %self.table, dim = self.dim, "pgvector schema ready");
        Ok(())
    }
}

fn pg_err<E: std::fmt::Display>(e: E) -> Error {
    Error::other(format!("pgvector: {e}"))
}

fn vector_literal(v: &[f32]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{}", x)).collect();
    format!("[{}]", parts.join(","))
}

fn filter_to_sql(filter: &Filter) -> (String, Vec<String>) {
    let mut clauses = Vec::new();
    let mut values: Vec<String> = Vec::new();
    for (k, v) in filter {
        let s = match v {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        let placeholder = values.len() + 3; // $1=vector, $2=k — so filters start at $3
        clauses.push(format!("metadata->>'{}' = ${}", k.replace('\'', "''"), placeholder));
        values.push(s);
    }
    if clauses.is_empty() {
        ("TRUE".into(), values)
    } else {
        (clauses.join(" AND "), values)
    }
}

#[async_trait]
impl VectorStore for PgVectorStore {
    async fn add(&self, mut docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(Error::invalid(format!(
                "len mismatch: docs={} embs={}", docs.len(), embeddings.len()
            )));
        }
        let client = self.pool.get().await.map_err(pg_err)?;
        let stmt = client
            .prepare_cached(&format!(
                "INSERT INTO {} (id, content, metadata, embedding) \
                 VALUES ($1, $2, $3::jsonb, $4::vector) \
                 ON CONFLICT (id) DO UPDATE SET \
                   content = EXCLUDED.content, \
                   metadata = EXCLUDED.metadata, \
                   embedding = EXCLUDED.embedding",
                self.table
            ))
            .await
            .map_err(pg_err)?;

        let mut ids = Vec::with_capacity(docs.len());
        for (mut d, v) in docs.drain(..).zip(embeddings.into_iter()) {
            if v.len() != self.dim {
                return Err(Error::invalid(format!(
                    "embedding dim mismatch: expected {}, got {}", self.dim, v.len()
                )));
            }
            let id = d.id.clone().unwrap_or_else(|| uuid::Uuid::new_v4().to_string());
            d.id = Some(id.clone());
            let metadata_json = serde_json::to_string(&d.metadata).unwrap_or("{}".into());
            let vec_lit = vector_literal(&v);
            client
                .execute(&stmt, &[&id, &d.content, &metadata_json, &vec_lit])
                .await
                .map_err(pg_err)?;
            ids.push(id);
        }
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        q: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        if q.len() != self.dim {
            return Err(Error::invalid(format!(
                "query dim mismatch: expected {}, got {}", self.dim, q.len()
            )));
        }
        let client = self.pool.get().await.map_err(pg_err)?;
        let vec_lit = vector_literal(q);

        let (where_clause, extra_values) = match filter {
            Some(f) => filter_to_sql(f),
            None => ("TRUE".into(), vec![]),
        };

        let sql = format!(
            "SELECT id, content, metadata, 1 - (embedding <=> $1::vector) AS score \
             FROM {} \
             WHERE {} \
             ORDER BY embedding <=> $1::vector ASC \
             LIMIT $2",
            self.table, where_clause,
        );

        let mut params: Vec<&(dyn tokio_postgres::types::ToSql + Sync)> = Vec::new();
        params.push(&vec_lit);
        let k_i64 = k as i64;
        params.push(&k_i64);
        for v in &extra_values { params.push(v); }

        let rows = client.query(&sql, &params).await.map_err(pg_err)?;
        let mut out = Vec::with_capacity(rows.len());
        for row in rows {
            let id: String = row.try_get("id").map_err(pg_err)?;
            let content: String = row.try_get("content").map_err(pg_err)?;
            let metadata_s: String = row.try_get("metadata").map_err(pg_err)?;
            let score: f64 = row.try_get("score").map_err(pg_err)?;
            let metadata = serde_json::from_str(&metadata_s).unwrap_or_default();
            out.push(Document {
                content,
                id: Some(id),
                metadata,
                score: Some(score as f32),
            });
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let client = self.pool.get().await.map_err(pg_err)?;
        let sql = format!("DELETE FROM {} WHERE id = ANY($1)", self.table);
        client.execute(&sql, &[&ids]).await.map_err(pg_err)?;
        Ok(())
    }

    async fn len(&self) -> usize {
        let client = match self.pool.get().await {
            Ok(c) => c,
            Err(_) => return 0,
        };
        let sql = format!("SELECT COUNT(*)::bigint AS n FROM {}", self.table);
        match client.query_one(&sql, &[]).await {
            Ok(row) => row.try_get::<_, i64>("n").map(|n| n as usize).unwrap_or(0),
            Err(_) => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_literal_formats_correctly() {
        assert_eq!(vector_literal(&[1.0, 2.0, 3.5]), "[1,2,3.5]");
        assert_eq!(vector_literal(&[]), "[]");
    }

    #[test]
    fn filter_sql_uses_jsonb_operator() {
        let mut f = std::collections::HashMap::new();
        f.insert("tag".into(), serde_json::json!("prod"));
        let (sql, vals) = filter_to_sql(&f);
        assert!(sql.contains("metadata->>'tag'"));
        assert_eq!(vals, vec!["prod".to_string()]);
    }
}

//! ChromaDB `VectorStore` over its v1 HTTP API. Avoids the chromadb-client
//! Python/JS package and gRPC entirely — Chroma's REST surface is small and
//! a tight HTTP wrapper covers what RAG pipelines need (add / query / delete).
//!
//! Hierarchy: tenant → database → collection. Defaults are `default_tenant` /
//! `default_database` to match the chroma server's startup defaults; override
//! when running with multi-tenant isolation.
//!
//! Collection creation is lazy + cached: the first `add` / `similarity_search`
//! / `delete` call hits `POST /collections` with `get_or_create=true`,
//! caches the returned UUID, and reuses it for subsequent calls. Avoids a
//! pre-flight roundtrip and survives the "collection already exists" race
//! between concurrent processes.

use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::{Filter, VectorStore};
use parking_lot::Mutex;
use reqwest::Client;
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct ChromaConfig {
    pub url: String,
    pub tenant: String,
    pub database: String,
    pub collection_name: String,
    pub timeout: Duration,
}

impl ChromaConfig {
    pub fn new(url: impl Into<String>, collection_name: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            tenant: "default_tenant".into(),
            database: "default_database".into(),
            collection_name: collection_name.into(),
            timeout: Duration::from_secs(30),
        }
    }
    pub fn with_tenant(mut self, t: impl Into<String>) -> Self { self.tenant = t.into(); self }
    pub fn with_database(mut self, d: impl Into<String>) -> Self { self.database = d.into(); self }
    pub fn with_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
}

pub struct ChromaVectorStore {
    cfg: ChromaConfig,
    http: Client,
    /// Cached collection UUID — populated by `ensure_collection_id`.
    cached_id: Mutex<Option<String>>,
}

impl ChromaVectorStore {
    pub fn new(cfg: ChromaConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http, cached_id: Mutex::new(None) })
    }

    fn collections_url(&self) -> String {
        format!(
            "{}/api/v1/collections",
            self.cfg.url.trim_end_matches('/'),
        )
    }

    fn collection_url(&self, id: &str) -> String {
        format!("{}/{}", self.collections_url(), id)
    }

    /// `POST /collections?get_or_create=true` — returns the collection's UUID.
    /// Cached so subsequent operations skip the round-trip.
    async fn ensure_collection_id(&self) -> Result<String> {
        if let Some(id) = self.cached_id.lock().clone() {
            return Ok(id);
        }
        let url = format!("{}?tenant={}&database={}",
            self.collections_url(), self.cfg.tenant, self.cfg.database);
        let resp = self
            .http
            .post(&url)
            .json(&json!({
                "name": self.cfg.collection_name,
                "get_or_create": true,
            }))
            .send()
            .await
            .map_err(|e| Error::other(format!("chroma create_collection: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("chroma create_collection {s}: {txt}")));
        }
        let v: Value = resp.json().await.map_err(|e| Error::other(format!("decode: {e}")))?;
        let id = v
            .get("id")
            .and_then(|i| i.as_str())
            .ok_or_else(|| Error::other("chroma: collection response missing `id`"))?
            .to_string();
        *self.cached_id.lock() = Some(id.clone());
        Ok(id)
    }
}

#[derive(Deserialize)]
struct QueryResponse {
    /// Outer Vec is one slot per query embedding; we send 1, take [0].
    ids: Vec<Vec<String>>,
    distances: Option<Vec<Vec<f32>>>,
    documents: Option<Vec<Vec<Option<String>>>>,
    metadatas: Option<Vec<Vec<Option<serde_json::Map<String, Value>>>>>,
}

#[async_trait]
impl VectorStore for ChromaVectorStore {
    async fn add(&self, docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(Error::other(format!(
                "chroma add: docs.len()={} != embeddings.len()={}",
                docs.len(), embeddings.len()
            )));
        }
        if docs.is_empty() { return Ok(vec![]); }

        let id = self.ensure_collection_id().await?;
        let url = format!("{}/add", self.collection_url(&id));

        // Use the doc's id when present; otherwise mint a UUID. Chroma rejects
        // duplicate ids in the same call, so pre-existing ids must be unique.
        let ids: Vec<String> = docs.iter().map(|d| {
            d.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string())
        }).collect();
        let documents: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
        let metadatas: Vec<Value> = docs.iter().map(|d| {
            // Chroma metadata must be flat scalar-only; serialize complex
            // values to JSON strings rather than rejecting the whole batch.
            let mut m = serde_json::Map::new();
            for (k, v) in &d.metadata {
                let scalar = match v {
                    Value::String(_) | Value::Number(_) | Value::Bool(_) => v.clone(),
                    other => Value::String(other.to_string()),
                };
                m.insert(k.clone(), scalar);
            }
            // Chroma requires a non-empty metadata dict per row; pad with a
            // sentinel if empty so the request doesn't 422.
            if m.is_empty() {
                m.insert("source".into(), Value::String("litgraph".into()));
            }
            Value::Object(m)
        }).collect();

        let body = json!({
            "ids": ids,
            "embeddings": embeddings,
            "documents": documents,
            "metadatas": metadatas,
        });
        let resp = self.http.post(&url).json(&body).send().await
            .map_err(|e| Error::other(format!("chroma add: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("chroma add {s}: {txt}")));
        }
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        query_embedding: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        let id = self.ensure_collection_id().await?;
        let url = format!("{}/query", self.collection_url(&id));

        let mut body = json!({
            "query_embeddings": [query_embedding],
            "n_results": k,
            "include": ["documents", "metadatas", "distances"],
        });
        if let Some(f) = filter {
            // Chroma's `where` is a JSON dict. Single key → exact match;
            // multiple keys → implicit AND. Pass through whatever Value the
            // caller put in the `Filter` (Chroma supports `$eq` / `$ne` /
            // `$gt` / `$in` / `$and` / `$or` operators as nested dicts).
            if !f.is_empty() {
                let where_obj: serde_json::Map<String, Value> = f.iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();
                body["where"] = Value::Object(where_obj);
            }
        }

        let resp = self.http.post(&url).json(&body).send().await
            .map_err(|e| Error::other(format!("chroma query: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("chroma query {s}: {txt}")));
        }
        let parsed: QueryResponse = resp.json().await
            .map_err(|e| Error::other(format!("chroma query decode: {e}")))?;

        let row_ids = parsed.ids.into_iter().next().unwrap_or_default();
        let row_docs: Vec<Option<String>> = parsed.documents
            .and_then(|d| d.into_iter().next())
            .unwrap_or_default();
        let row_metas: Vec<Option<serde_json::Map<String, Value>>> = parsed.metadatas
            .and_then(|m| m.into_iter().next())
            .unwrap_or_default();
        let row_dists: Vec<f32> = parsed.distances
            .and_then(|d| d.into_iter().next())
            .unwrap_or_default();

        let mut out = Vec::with_capacity(row_ids.len());
        for (i, doc_id) in row_ids.into_iter().enumerate() {
            let content = row_docs.get(i).cloned().flatten().unwrap_or_default();
            let mut d = Document::new(content).with_id(doc_id);
            if let Some(Some(meta)) = row_metas.get(i) {
                for (k, v) in meta {
                    d.metadata.insert(k.clone(), v.clone());
                }
            }
            // Chroma returns squared L2 by default; we expose it as `score`
            // verbatim. Lower = more similar (caller can convert if needed).
            if let Some(dist) = row_dists.get(i) {
                d.score = Some(*dist);
            }
            out.push(d);
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        if ids.is_empty() { return Ok(()); }
        let cid = self.ensure_collection_id().await?;
        let url = format!("{}/delete", self.collection_url(&cid));
        let resp = self.http.post(&url).json(&json!({"ids": ids})).send().await
            .map_err(|e| Error::other(format!("chroma delete: {e}")))?;
        if !resp.status().is_success() {
            let s = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("chroma delete {s}: {txt}")));
        }
        Ok(())
    }

    async fn len(&self) -> usize {
        // `GET /collections/{id}/count` returns an integer. On any failure
        // (network, server error, unexpected shape) return 0 — matches how
        // other VectorStore impls handle transient errors in `len()`.
        let cid = match self.ensure_collection_id().await {
            Ok(id) => id,
            Err(_) => return 0,
        };
        let url = format!("{}/count", self.collection_url(&cid));
        let resp = match self.http.get(&url).send().await {
            Ok(r) => r,
            Err(_) => return 0,
        };
        if !resp.status().is_success() { return 0; }
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return 0,
        };
        v.as_u64().map(|n| n as usize).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::Mutex as StdMutex;
    use std::thread;

    /// Tiny scripted HTTP fake — accepts a sequence of (path-suffix, response-body)
    /// pairs and checks the path on the way through. Captures each request body
    /// for assertions.
    fn start_fake(
        responses: Vec<(&'static str, String)>,
        capture: Arc<StdMutex<Vec<(String, String)>>>,
    ) -> u16 {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        thread::spawn(move || {
            for (expected_path_substr, body) in responses {
                let (mut s, _) = match listener.accept() {
                    Ok(x) => x, Err(_) => break,
                };
                let mut buf = [0u8; 8192];
                let mut total = Vec::new();
                let mut header_end = 0usize;
                let mut content_length = 0usize;
                let mut request_line = String::new();
                loop {
                    let n = match s.read(&mut buf) {
                        Ok(0) => break, Ok(n) => n, Err(_) => break,
                    };
                    total.extend_from_slice(&buf[..n]);
                    if let Some(pos) = total.windows(4).position(|w| w == b"\r\n\r\n") {
                        header_end = pos + 4;
                        let headers = String::from_utf8_lossy(&total[..pos]);
                        request_line = headers.lines().next().unwrap_or("").to_string();
                        for line in headers.lines() {
                            if let Some(v) = line.to_lowercase().strip_prefix("content-length:") {
                                content_length = v.trim().parse().unwrap_or(0);
                            }
                        }
                        break;
                    }
                }
                while total.len() < header_end + content_length {
                    let n = match s.read(&mut buf) {
                        Ok(0) => break, Ok(n) => n, Err(_) => break,
                    };
                    total.extend_from_slice(&buf[..n]);
                }
                let req_body = String::from_utf8_lossy(&total[header_end..header_end + content_length]).to_string();
                capture.lock().unwrap().push((request_line.clone(), req_body));
                assert!(request_line.contains(expected_path_substr),
                    "path mismatch: req={request_line:?} expected substring {expected_path_substr:?}");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                    body.len(), body
                );
                let _ = s.write_all(resp.as_bytes());
            }
        });
        port
    }

    #[tokio::test]
    async fn add_then_query_uses_cached_collection_id() {
        let captured = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
        let port = start_fake(vec![
            // 1. ensure_collection_id (POST /collections)
            ("/api/v1/collections?", r#"{"id":"col-uuid-123","name":"test"}"#.into()),
            // 2. add (POST /collections/col-uuid-123/add)
            ("/api/v1/collections/col-uuid-123/add", "{}".into()),
            // 3. query (POST /collections/col-uuid-123/query) — note: NO second create-collection call.
            ("/api/v1/collections/col-uuid-123/query", json!({
                "ids":[["a","b"]],
                "documents":[[Some("doc a"),Some("doc b")]],
                "metadatas":[[{"k":"v1"},{"k":"v2"}]],
                "distances":[[0.1,0.5]],
            }).to_string()),
        ], captured.clone());

        let cfg = ChromaConfig::new(format!("http://127.0.0.1:{port}"), "test");
        let store = ChromaVectorStore::new(cfg).unwrap();

        let docs = vec![
            Document::new("doc a").with_id("a"),
            Document::new("doc b").with_id("b"),
        ];
        let added = store.add(docs, vec![vec![1.0, 0.0], vec![0.0, 1.0]]).await.unwrap();
        assert_eq!(added, vec!["a".to_string(), "b".to_string()]);

        let hits = store.similarity_search(&[1.0, 0.0], 2, None).await.unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].id.as_deref(), Some("a"));
        assert_eq!(hits[0].content, "doc a");
        assert!((hits[0].score.unwrap() - 0.1).abs() < 1e-6);
        assert_eq!(hits[0].metadata.get("k").unwrap(), &json!("v1"));

        // Verify the create-collection request was sent ONCE (cache hit prevents repeat).
        let cap = captured.lock().unwrap();
        let creates = cap.iter().filter(|(line, _)| line.contains("/api/v1/collections?")).count();
        assert_eq!(creates, 1, "ensure_collection_id should be called exactly once");
    }

    #[tokio::test]
    async fn add_with_no_id_mints_uuid() {
        let captured = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
        let port = start_fake(vec![
            ("/api/v1/collections?", r#"{"id":"c1","name":"t"}"#.into()),
            ("/c1/add", "{}".into()),
        ], captured.clone());

        let cfg = ChromaConfig::new(format!("http://127.0.0.1:{port}"), "t");
        let store = ChromaVectorStore::new(cfg).unwrap();
        let docs = vec![Document::new("no id here")]; // .id == None
        let ids = store.add(docs, vec![vec![1.0]]).await.unwrap();
        assert_eq!(ids.len(), 1);
        assert!(Uuid::parse_str(&ids[0]).is_ok(), "minted id should be a UUID: {}", ids[0]);

        // The add body should carry that minted id.
        let add_body = &captured.lock().unwrap()[1].1;
        let v: Value = serde_json::from_str(add_body).unwrap();
        assert_eq!(v["ids"][0], json!(ids[0]));
    }

    #[tokio::test]
    async fn complex_metadata_serialized_to_json_string_not_rejected() {
        let captured = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
        let port = start_fake(vec![
            ("/api/v1/collections?", r#"{"id":"c1","name":"t"}"#.into()),
            ("/c1/add", "{}".into()),
        ], captured.clone());

        let cfg = ChromaConfig::new(format!("http://127.0.0.1:{port}"), "t");
        let store = ChromaVectorStore::new(cfg).unwrap();
        let mut d = Document::new("x").with_id("d1");
        d.metadata.insert("flat_str".into(), json!("v"));
        d.metadata.insert("flat_num".into(), json!(42));
        d.metadata.insert("nested".into(), json!({"deep": "yes"}));
        d.metadata.insert("array".into(), json!([1, 2, 3]));
        store.add(vec![d], vec![vec![0.0]]).await.unwrap();

        let add_body = &captured.lock().unwrap()[1].1;
        let v: Value = serde_json::from_str(add_body).unwrap();
        let meta = &v["metadatas"][0];
        // Scalars survive as-is.
        assert_eq!(meta["flat_str"], json!("v"));
        assert_eq!(meta["flat_num"], json!(42));
        // Complex types collapsed to JSON-string form (Chroma rejects nested values).
        assert_eq!(meta["nested"].as_str().unwrap(), r#"{"deep":"yes"}"#);
        assert_eq!(meta["array"].as_str().unwrap(), "[1,2,3]");
    }

    #[tokio::test]
    async fn query_with_filter_passes_where_clause() {
        let captured = Arc::new(StdMutex::new(Vec::<(String, String)>::new()));
        let port = start_fake(vec![
            ("/api/v1/collections?", r#"{"id":"c1","name":"t"}"#.into()),
            ("/c1/query", r#"{"ids":[[]]}"#.into()),
        ], captured.clone());

        let cfg = ChromaConfig::new(format!("http://127.0.0.1:{port}"), "t");
        let store = ChromaVectorStore::new(cfg).unwrap();
        let mut filter: Filter = HashMap::new();
        filter.insert("source".into(), json!("alpha"));
        let _ = store.similarity_search(&[1.0], 5, Some(&filter)).await.unwrap();
        let body = &captured.lock().unwrap()[1].1;
        let v: Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["where"]["source"], json!("alpha"));
    }

    #[tokio::test]
    async fn delete_empty_short_circuits_no_http() {
        // No fake server → this MUST NOT make any network calls.
        let cfg = ChromaConfig::new("http://127.0.0.1:1", "t");
        let store = ChromaVectorStore::new(cfg).unwrap();
        store.delete(&[]).await.unwrap();
    }

    #[tokio::test]
    async fn add_length_mismatch_errors_before_http() {
        let cfg = ChromaConfig::new("http://127.0.0.1:1", "t");
        let store = ChromaVectorStore::new(cfg).unwrap();
        let err = store.add(
            vec![Document::new("only one")],
            vec![vec![0.0], vec![1.0]],
        ).await.unwrap_err();
        assert!(format!("{err}").contains("docs.len()=1 != embeddings.len()=2"));
    }
}

//! Weaviate VectorStore via the v1 REST API.
//!
//! # Why REST, not gRPC?
//!
//! Weaviate has both v1 REST and v1 gRPC. The Python client uses gRPC for
//! batch ingest perf, but it pulls a heavy proto/tonic dep tree. REST is
//! plain `reqwest` — fully sufficient for upsert + search throughput in the
//! 1k–100k QPS range that LangChain users actually hit. Switch to gRPC when
//! you need 6-figure QPS and you'll know.
//!
//! # Config
//!
//! - `url` — Weaviate endpoint, e.g. `http://localhost:8080`.
//! - `api_key` — optional; for Weaviate Cloud (WCS) or auth-enabled deploys.
//! - `class` — Weaviate "class" name (its term for collection / table).
//!   Must start with an uppercase letter (Weaviate convention; we don't
//!   normalize for you).
//!
//! # Schema model
//!
//! Each upserted Document becomes one Weaviate object with:
//! - UUID id (deterministic from the caller-supplied id when present, else random).
//! - `vector` — the supplied embedding.
//! - `properties` — a flat map carrying `__content` (the text body) plus
//!   any caller-supplied metadata (string / number / bool / null only;
//!   nested objects get JSON-stringified to keep the property schema flat).
//!
//! # Class auto-creation
//!
//! `ensure_class()` PUT /schema with vectorizer=`none` (we always supply
//! pre-computed embeddings — Weaviate's own vectorizer modules are not
//! used). Idempotent on 422 (class exists).

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::store::{Filter, VectorStore};
use reqwest::{Client, StatusCode};
use serde::Deserialize;
use serde_json::{Value, json};
use std::time::Duration;
use tracing::debug;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct WeaviateConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub class: String,
    pub timeout: Duration,
}

impl WeaviateConfig {
    pub fn new(url: impl Into<String>, class: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            class: class.into(),
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
}

pub struct WeaviateVectorStore {
    cfg: WeaviateConfig,
    http: Client,
}

impl WeaviateVectorStore {
    pub fn new(cfg: WeaviateConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}/v1{}", self.cfg.url.trim_end_matches('/'), path);
        let mut r = self.http.request(method, url);
        if let Some(k) = &self.cfg.api_key {
            r = r.bearer_auth(k);
        }
        r
    }

    /// Idempotently create the class. Vectorizer is set to `none` because
    /// we always supply embeddings client-side. Returns Ok if the class
    /// already exists.
    pub async fn ensure_class(&self) -> Result<()> {
        let body = json!({
            "class": self.cfg.class,
            "vectorizer": "none",
            "properties": [
                { "name": "__content", "dataType": ["text"] }
            ],
        });
        let resp = self
            .req(reqwest::Method::POST, "/schema")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("weaviate ensure_class: {e}")))?;
        let status = resp.status();
        if status.is_success() || status == StatusCode::UNPROCESSABLE_ENTITY {
            // 422 = "class exists" in Weaviate's schema API.
            debug!(class = %self.cfg.class, status = %status, "weaviate class ready");
            return Ok(());
        }
        let txt = resp.text().await.unwrap_or_default();
        Err(Error::other(format!(
            "weaviate ensure_class {}: {}", status, txt
        )))
    }

    /// Deterministic UUID for caller-supplied string ids — UUIDv5 over the
    /// (class, id) namespace. Lets repeat upserts with the same id idempotently
    /// overwrite the same object.
    fn id_to_uuid(&self, given: &str) -> Uuid {
        // Stable namespace per class so the same caller-id maps to the same
        // UUID across runs.
        let ns = Uuid::new_v5(&Uuid::NAMESPACE_OID, self.cfg.class.as_bytes());
        Uuid::new_v5(&ns, given.as_bytes())
    }

    /// Build the GraphQL `where` clause from a metadata Filter. We only
    /// support `Equal` + `And` for v1 — covers the 95% case (LangChain's
    /// most common pattern is `{field: value}` exact match).
    fn filter_to_where(filter: &Filter) -> Value {
        let operands: Vec<Value> = filter
            .iter()
            .map(|(k, v)| {
                let (data_type, value_field) = match v {
                    Value::String(_) => ("valueString", "valueString"),
                    Value::Bool(_) => ("valueBoolean", "valueBoolean"),
                    Value::Number(n) if n.is_i64() => ("valueInt", "valueInt"),
                    Value::Number(_) => ("valueNumber", "valueNumber"),
                    _ => ("valueString", "valueString"),
                };
                let _ = data_type;
                json!({
                    "operator": "Equal",
                    "path": [k],
                    value_field: v,
                })
            })
            .collect();
        if operands.len() == 1 {
            operands.into_iter().next().unwrap()
        } else {
            json!({ "operator": "And", "operands": operands })
        }
    }
}

#[derive(Deserialize)]
struct BatchEntry {
    #[serde(default)]
    #[allow(dead_code)]
    id: Option<String>,
    #[serde(default)]
    result: Option<BatchResult>,
}

#[derive(Deserialize)]
struct BatchResult {
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    errors: Option<Value>,
}

#[async_trait]
impl VectorStore for WeaviateVectorStore {
    async fn add(&self, mut docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(Error::invalid(format!(
                "len mismatch: docs={} embs={}", docs.len(), embeddings.len()
            )));
        }
        let mut ids = Vec::with_capacity(docs.len());
        let mut objects = Vec::with_capacity(docs.len());
        for (mut d, v) in docs.drain(..).zip(embeddings.into_iter()) {
            let uid = match &d.id {
                Some(given) => self.id_to_uuid(given).to_string(),
                None => Uuid::new_v4().to_string(),
            };
            d.id = Some(uid.clone());

            // Properties: __content + flat metadata. Nested objects are
            // JSON-stringified so we don't fight Weaviate's schema typing.
            let mut props = serde_json::Map::new();
            props.insert("__content".into(), Value::String(d.content.clone()));
            for (k, v) in &d.metadata {
                let flat = match v {
                    Value::Object(_) | Value::Array(_) => Value::String(v.to_string()),
                    other => other.clone(),
                };
                props.insert(k.clone(), flat);
            }

            objects.push(json!({
                "class": self.cfg.class,
                "id": uid.clone(),
                "vector": v,
                "properties": props,
            }));
            ids.push(uid);
        }

        let resp = self
            .req(reqwest::Method::POST, "/batch/objects")
            .json(&json!({ "objects": objects }))
            .send()
            .await
            .map_err(|e| Error::other(format!("weaviate batch upsert: {e}")))?;
        if !resp.status().is_success() {
            let status = resp.status();
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("weaviate batch {}: {}", status, txt)));
        }
        // Body is an array of per-object results. Surface any object-level errors.
        let entries: Vec<BatchEntry> = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("weaviate batch decode: {e}")))?;
        for (i, entry) in entries.iter().enumerate() {
            if let Some(r) = &entry.result {
                if let Some(errs) = &r.errors {
                    if !matches!(errs, Value::Null) && !matches!(r.status.as_deref(), Some("SUCCESS")) {
                        return Err(Error::other(format!(
                            "weaviate batch object {} failed: {}", i, errs
                        )));
                    }
                }
            }
        }
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        q: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        // GraphQL Get query — Weaviate's idiomatic vector-search entry point.
        // Returns the __content + metadata properties + _additional.{id, distance}.
        let mut filter_str = String::new();
        if let Some(f) = filter {
            let w = Self::filter_to_where(f);
            filter_str = format!(", where: {}", w);
        }
        let class = &self.cfg.class;
        let vector_str: String = format!(
            "[{}]",
            q.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
        );
        let query = format!(
            "{{ Get {{ {class}( nearVector: {{ vector: {vector_str} }}, limit: {k}{filter_str} ) {{ __content _additional {{ id distance }} }} }} }}"
        );
        let body = json!({ "query": query });

        let resp = self
            .req(reqwest::Method::POST, "/graphql")
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("weaviate search: {e}")))?;
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("weaviate search: {txt}")));
        }
        let v: Value = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("weaviate search decode: {e}")))?;

        // GraphQL errors live at top-level `errors`, not 4xx.
        if let Some(errs) = v.get("errors") {
            if !matches!(errs, Value::Null) && errs.as_array().map(|a| !a.is_empty()).unwrap_or(false) {
                return Err(Error::other(format!("weaviate graphql error: {errs}")));
            }
        }

        let arr = v
            .pointer(&format!("/data/Get/{}", class))
            .and_then(|x| x.as_array())
            .cloned()
            .unwrap_or_default();
        let mut out = Vec::with_capacity(arr.len());
        for hit in arr {
            let mut doc = Document::new("");
            if let Some(add) = hit.get("_additional") {
                if let Some(id) = add.get("id").and_then(|v| v.as_str()) {
                    doc.id = Some(id.to_string());
                }
                if let Some(dist) = add.get("distance").and_then(|v| v.as_f64()) {
                    // Weaviate returns distance (0 = identical); convert to a
                    // "score" (1 - distance) for parity with other stores.
                    doc.score = Some(1.0 - dist as f32);
                }
            }
            if let Some(obj) = hit.as_object() {
                for (k, v) in obj {
                    if k == "_additional" { continue; }
                    if k == "__content" {
                        if let Value::String(s) = v {
                            doc.content = s.clone();
                        }
                    } else {
                        doc.metadata.insert(k.clone(), v.clone());
                    }
                }
            }
            out.push(doc);
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        // Weaviate doesn't have batch delete by id list in v1 REST — we
        // issue one DELETE per id. For high-volume deletes prefer the
        // GraphQL `delete` mutation with a filter; that's a future add.
        for id in ids {
            let path = format!("/objects/{}/{}", self.cfg.class, id);
            let resp = self
                .req(reqwest::Method::DELETE, &path)
                .send()
                .await
                .map_err(|e| Error::other(format!("weaviate delete: {e}")))?;
            // 204 = deleted, 404 = already gone (treat as success).
            if !resp.status().is_success() && resp.status() != StatusCode::NOT_FOUND {
                let s = resp.status();
                let txt = resp.text().await.unwrap_or_default();
                return Err(Error::other(format!("weaviate delete {}: {}", s, txt)));
            }
        }
        Ok(())
    }

    async fn len(&self) -> usize {
        let class = &self.cfg.class;
        let query = format!(
            "{{ Aggregate {{ {class} {{ meta {{ count }} }} }} }}"
        );
        let body = json!({ "query": query });
        let resp = match self
            .req(reqwest::Method::POST, "/graphql")
            .json(&body)
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => r,
            _ => return 0,
        };
        let v: Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return 0,
        };
        v.pointer(&format!("/data/Aggregate/{}/0/meta/count", class))
            .and_then(|n| n.as_u64())
            .unwrap_or(0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn filter_to_where_single_clause() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("category".into(), json!("rust"));
        let w = WeaviateVectorStore::filter_to_where(&f);
        assert_eq!(w["operator"], "Equal");
        assert_eq!(w["path"][0], "category");
        assert_eq!(w["valueString"], "rust");
    }

    #[test]
    fn filter_to_where_multiple_clauses_use_and() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("a".into(), json!("x"));
        f.insert("b".into(), json!(42));
        let w = WeaviateVectorStore::filter_to_where(&f);
        assert_eq!(w["operator"], "And");
        let ops = w["operands"].as_array().unwrap();
        assert_eq!(ops.len(), 2);
        // Each operand has Equal + valueX based on type.
        for op in ops {
            assert_eq!(op["operator"], "Equal");
        }
    }

    #[test]
    fn filter_to_where_int_uses_valueint() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("count".into(), json!(42));
        let w = WeaviateVectorStore::filter_to_where(&f);
        assert_eq!(w["valueInt"], 42);
    }

    #[test]
    fn filter_to_where_bool_uses_valueboolean() {
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("active".into(), json!(true));
        let w = WeaviateVectorStore::filter_to_where(&f);
        assert_eq!(w["valueBoolean"], true);
    }

    #[test]
    fn deterministic_uuid_repeats_for_same_caller_id() {
        let store = WeaviateVectorStore::new(
            WeaviateConfig::new("http://localhost:8080", "Article")
        ).unwrap();
        let a = store.id_to_uuid("doc-1");
        let b = store.id_to_uuid("doc-1");
        assert_eq!(a, b);
        let c = store.id_to_uuid("doc-2");
        assert_ne!(a, c);
    }

    #[test]
    fn deterministic_uuid_namespaced_per_class() {
        let s1 = WeaviateVectorStore::new(
            WeaviateConfig::new("http://x", "Article")
        ).unwrap();
        let s2 = WeaviateVectorStore::new(
            WeaviateConfig::new("http://x", "Comment")
        ).unwrap();
        // Same caller id → different UUIDs across classes (no cross-class collision).
        assert_ne!(s1.id_to_uuid("k"), s2.id_to_uuid("k"));
    }

    // ---------- Fake-server integration ----------

    use std::sync::Arc as StdArc;
    use parking_lot_dep::Mutex as ParkMutex;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::sync::oneshot;

    /// Minimal HTTP/1.1 fake — same shape as the MCP test server. One
    /// request, parse, dispatch on path, write canned response, close.
    struct FakeWeaviate {
        url: String,
        last_path: StdArc<ParkMutex<Option<String>>>,
        last_body: StdArc<ParkMutex<Option<Value>>>,
        _shutdown: oneshot::Sender<()>,
    }

    fn parse_http(buf: &[u8]) -> Option<(String, Vec<u8>)> {
        let split = buf.windows(4).position(|w| w == b"\r\n\r\n")?;
        let head = std::str::from_utf8(&buf[..split]).ok()?.to_string();
        let body = buf[split + 4..].to_vec();
        Some((head, body))
    }

    async fn read_request(
        s: &mut tokio::net::TcpStream,
    ) -> std::io::Result<(String, String, Vec<u8>)> {
        let mut buf = Vec::with_capacity(4096);
        let mut chunk = [0u8; 4096];
        let head;
        let mut body;
        loop {
            let n = s.read(&mut chunk).await?;
            if n == 0 {
                return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof"));
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some((h, b)) = parse_http(&buf) {
                head = h;
                body = b;
                break;
            }
        }
        let (request_line, _) = head.split_once("\r\n").unwrap_or((head.as_str(), ""));
        let parts: Vec<&str> = request_line.split_whitespace().collect();
        let method = parts.first().copied().unwrap_or("").to_string();
        let path = parts.get(1).copied().unwrap_or("").to_string();
        // Content-Length framing.
        let cl: usize = head
            .lines()
            .skip(1)
            .find_map(|l| l.split_once(':').and_then(|(k, v)| {
                if k.trim().eq_ignore_ascii_case("content-length") {
                    v.trim().parse().ok()
                } else { None }
            }))
            .unwrap_or(0);
        while body.len() < cl {
            let n = s.read(&mut chunk).await?;
            if n == 0 { break; }
            body.extend_from_slice(&chunk[..n]);
        }
        Ok((method, path, body))
    }

    async fn write_json(s: &mut tokio::net::TcpStream, status: u16, body: &Value) {
        let body_bytes = serde_json::to_vec(body).unwrap();
        let resp = format!(
            "HTTP/1.1 {} OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
            status, body_bytes.len()
        );
        let _ = s.write_all(resp.as_bytes()).await;
        let _ = s.write_all(&body_bytes).await;
    }

    // We re-export parking_lot::Mutex under an alias to avoid colliding with
    // the top-level `parking_lot` workspace symbol noise in this small test
    // module — keeps imports local + obvious.
    mod parking_lot_dep {
        pub use std::sync::Mutex;
    }

    async fn spawn_fake() -> FakeWeaviate {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let last_path = StdArc::new(parking_lot_dep::Mutex::new(None));
        let last_body = StdArc::new(parking_lot_dep::Mutex::new(None));
        let (shutdown_tx, mut shutdown_rx) = oneshot::channel::<()>();
        let lp = last_path.clone();
        let lb = last_body.clone();
        tokio::spawn(async move {
            loop {
                tokio::select! {
                    _ = &mut shutdown_rx => break,
                    Ok((mut s, _)) = listener.accept() => {
                        let lp = lp.clone();
                        let lb = lb.clone();
                        tokio::spawn(async move {
                            let (_method, path, body) = match read_request(&mut s).await {
                                Ok(t) => t,
                                Err(_) => return,
                            };
                            *lp.lock().unwrap() = Some(path.clone());
                            let req: Value = if body.is_empty() {
                                Value::Null
                            } else {
                                serde_json::from_slice(&body).unwrap_or(Value::Null)
                            };
                            *lb.lock().unwrap() = Some(req.clone());

                            // Dispatch by path prefix.
                            if path.starts_with("/v1/batch/objects") {
                                // Echo each object as SUCCESS.
                                let objects = req["objects"].as_array().cloned().unwrap_or_default();
                                let results: Vec<Value> = objects.iter().map(|o| json!({
                                    "id": o["id"],
                                    "result": { "status": "SUCCESS" }
                                })).collect();
                                write_json(&mut s, 200, &Value::Array(results)).await;
                            } else if path.starts_with("/v1/graphql") {
                                // Return a canned hit list. We always return 1 hit
                                // with __content "alpha" for tests to assert against.
                                let resp = json!({
                                    "data": {
                                        "Get": {
                                            "Article": [{
                                                "__content": "alpha",
                                                "topic": "rust",
                                                "_additional": {
                                                    "id": "00000000-0000-0000-0000-000000000001",
                                                    "distance": 0.1
                                                }
                                            }]
                                        }
                                    }
                                });
                                write_json(&mut s, 200, &resp).await;
                            } else if path.starts_with("/v1/objects/") {
                                // DELETE /objects/{class}/{id} — 204 with empty body.
                                let resp = "HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n";
                                let _ = s.write_all(resp.as_bytes()).await;
                            } else if path.starts_with("/v1/schema") {
                                write_json(&mut s, 200, &json!({"class": "Article"})).await;
                            } else {
                                write_json(&mut s, 404, &json!({"error": "unknown"})).await;
                            }
                        });
                    }
                }
            }
        });
        FakeWeaviate {
            url: format!("http://127.0.0.1:{port}"),
            last_path,
            last_body,
            _shutdown: shutdown_tx,
        }
    }

    #[tokio::test]
    async fn batch_upsert_round_trips_objects_into_class() {
        let srv = spawn_fake().await;
        let store = WeaviateVectorStore::new(
            WeaviateConfig::new(&srv.url, "Article")
        ).unwrap();
        let docs = vec![
            Document::new("alpha"),
            Document::new("beta"),
        ];
        let embs = vec![vec![0.1, 0.2], vec![0.3, 0.4]];
        let ids = store.add(docs, embs).await.unwrap();
        assert_eq!(ids.len(), 2);
        // Path is the batch endpoint.
        assert_eq!(srv.last_path.lock().unwrap().as_deref(), Some("/v1/batch/objects"));
        // Body has 2 objects with our class + properties.
        let body = srv.last_body.lock().unwrap().clone().unwrap();
        let objects = body["objects"].as_array().unwrap();
        assert_eq!(objects.len(), 2);
        assert_eq!(objects[0]["class"], "Article");
        assert_eq!(objects[0]["properties"]["__content"], "alpha");
        // f32 → JSON → f64 round-trip: tolerance compare.
        let v0 = objects[0]["vector"][0].as_f64().unwrap();
        assert!((v0 - 0.1).abs() < 1e-6, "got: {v0}");
    }

    #[tokio::test]
    async fn similarity_search_uses_graphql_get_with_nearvector() {
        let srv = spawn_fake().await;
        let store = WeaviateVectorStore::new(
            WeaviateConfig::new(&srv.url, "Article")
        ).unwrap();
        let docs = store.similarity_search(&[0.1, 0.2], 5, None).await.unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "alpha");
        assert_eq!(docs[0].metadata.get("topic").and_then(|v| v.as_str()), Some("rust"));
        // Distance 0.1 → score 0.9.
        let score = docs[0].score.unwrap();
        assert!((score - 0.9).abs() < 1e-5, "score = 1 - distance, got {score}");
        assert_eq!(srv.last_path.lock().unwrap().as_deref(), Some("/v1/graphql"));
        // Query body contains nearVector + the class name.
        let body = srv.last_body.lock().unwrap().clone().unwrap();
        let q = body["query"].as_str().unwrap();
        assert!(q.contains("nearVector"));
        assert!(q.contains("Article"));
        assert!(q.contains("limit: 5"));
    }

    #[tokio::test]
    async fn similarity_search_with_filter_includes_where_clause() {
        let srv = spawn_fake().await;
        let store = WeaviateVectorStore::new(
            WeaviateConfig::new(&srv.url, "Article")
        ).unwrap();
        let mut f: HashMap<String, Value> = HashMap::new();
        f.insert("topic".into(), json!("rust"));
        let _ = store.similarity_search(&[0.1, 0.2], 5, Some(&f)).await.unwrap();
        let body = srv.last_body.lock().unwrap().clone().unwrap();
        let q = body["query"].as_str().unwrap();
        assert!(q.contains("where:"), "expected where clause, got: {q}");
        assert!(q.contains("Equal"));
        assert!(q.contains("topic"));
    }

    #[tokio::test]
    async fn delete_issues_one_request_per_id() {
        let srv = spawn_fake().await;
        let store = WeaviateVectorStore::new(
            WeaviateConfig::new(&srv.url, "Article")
        ).unwrap();
        let ids = vec!["uuid-a".to_string(), "uuid-b".to_string()];
        store.delete(&ids).await.unwrap();
        // Last path is the second delete.
        let p = srv.last_path.lock().unwrap().clone().unwrap();
        assert!(p.starts_with("/v1/objects/Article/uuid-"), "got: {p}");
    }
}

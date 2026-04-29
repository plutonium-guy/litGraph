//! Qdrant VectorStore via the REST HTTP API.
//!
//! # Why REST, not gRPC?
//!
//! The official `qdrant-client` uses tonic/gRPC which pulls a heavy dep tree
//! (hyper-h2, prost, …). REST is plain `reqwest` — same wire formats Qdrant's
//! Python client defaults to, and for control-plane + point CRUD it's plenty fast.
//!
//! # Config
//!
//! - `url` — Qdrant endpoint, e.g. `http://localhost:6333`.
//! - `api_key` — optional; set for Qdrant Cloud.
//! - `collection` — must be created ahead of time (via the Qdrant dashboard or
//!   `ensure_collection`); we don't hide the DDL from you.
//! - `vector_name` — Qdrant supports named vectors per point. Defaults to `""`
//!   (the unnamed single-vector layout).

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
pub struct QdrantConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub collection: String,
    pub vector_name: Option<String>,
    pub timeout: Duration,
}

impl QdrantConfig {
    pub fn new(url: impl Into<String>, collection: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            collection: collection.into(),
            vector_name: None,
            timeout: Duration::from_secs(30),
        }
    }

    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    pub fn with_vector_name(mut self, name: impl Into<String>) -> Self {
        self.vector_name = Some(name.into());
        self
    }
}

pub struct QdrantVectorStore {
    cfg: QdrantConfig,
    http: Client,
}

impl QdrantVectorStore {
    pub fn new(cfg: QdrantConfig) -> Result<Self> {
        let http = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| Error::other(format!("http build: {e}")))?;
        Ok(Self { cfg, http })
    }

    fn req(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.cfg.url.trim_end_matches('/'), path);
        let mut r = self.http.request(method, url);
        if let Some(k) = &self.cfg.api_key {
            r = r.header("api-key", k);
        }
        r
    }

    /// Idempotently create the collection. Useful for CI / test setup.
    pub async fn ensure_collection(&self, dim: u64, distance: &str) -> Result<()> {
        // PUT /collections/{name} creates or updates.
        let path = format!("/collections/{}", self.cfg.collection);
        let body = if let Some(name) = &self.cfg.vector_name {
            json!({
                "vectors": {
                    name: { "size": dim, "distance": distance }
                }
            })
        } else {
            json!({ "vectors": { "size": dim, "distance": distance } })
        };
        let resp = self
            .req(reqwest::Method::PUT, &path)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("qdrant PUT collection: {e}")))?;
        if !resp.status().is_success() && resp.status() != StatusCode::CONFLICT {
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("qdrant collection create failed: {txt}")));
        }
        debug!(collection = %self.cfg.collection, "qdrant collection ready");
        Ok(())
    }

    fn make_vector(&self, v: Vec<f32>) -> Value {
        match &self.cfg.vector_name {
            Some(name) => json!({ name: v }),
            None => json!(v),
        }
    }

    fn filter_to_qdrant(filter: &Filter) -> Value {
        let must: Vec<Value> = filter
            .iter()
            .map(|(k, v)| json!({ "key": k, "match": { "value": v } }))
            .collect();
        json!({ "must": must })
    }
}

#[derive(Deserialize)]
struct UpsertResponse {
    #[serde(default)]
    #[allow(dead_code)]
    status: Option<String>,
}

#[derive(Deserialize)]
struct SearchResponse {
    result: Vec<SearchHit>,
}

#[derive(Deserialize)]
struct SearchHit {
    id: Value,
    score: f32,
    #[serde(default)]
    payload: Option<Value>,
}

#[async_trait]
impl VectorStore for QdrantVectorStore {
    async fn add(&self, mut docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(Error::invalid(format!(
                "len mismatch: docs={} embs={}", docs.len(), embeddings.len()
            )));
        }
        let mut ids = Vec::with_capacity(docs.len());
        let mut points = Vec::with_capacity(docs.len());
        for (mut d, v) in docs.drain(..).zip(embeddings.into_iter()) {
            // Qdrant point IDs must be uint or UUID — we always use UUIDs.
            let uid = match &d.id {
                Some(given) => Uuid::parse_str(given).unwrap_or_else(|_| Uuid::new_v4()),
                None => Uuid::new_v4(),
            }
            .to_string();
            d.id = Some(uid.clone());

            // Payload = metadata + __content so we can reconstruct docs on read.
            let mut payload = serde_json::Map::new();
            payload.insert("__content".into(), Value::String(d.content.clone()));
            for (k, v) in &d.metadata {
                payload.insert(k.clone(), v.clone());
            }

            points.push(json!({
                "id": uid.clone(),
                "vector": self.make_vector(v),
                "payload": payload,
            }));
            ids.push(uid);
        }

        let path = format!("/collections/{}/points?wait=true", self.cfg.collection);
        let resp = self
            .req(reqwest::Method::PUT, &path)
            .json(&json!({ "points": points }))
            .send()
            .await
            .map_err(|e| Error::other(format!("qdrant upsert: {e}")))?;
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("qdrant upsert {}: {}", "failed", txt)));
        }
        let _: UpsertResponse = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("qdrant upsert decode: {e}")))?;
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        q: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        let mut body = json!({
            "vector": self.make_vector(q.to_vec()),
            "limit": k,
            "with_payload": true,
        });
        if let Some(f) = filter {
            body["filter"] = Self::filter_to_qdrant(f);
        }
        if let Some(vname) = &self.cfg.vector_name {
            body["using"] = json!(vname);
        }

        let path = format!("/collections/{}/points/search", self.cfg.collection);
        let resp = self
            .req(reqwest::Method::POST, &path)
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::other(format!("qdrant search: {e}")))?;
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("qdrant search: {txt}")));
        }
        let sr: SearchResponse = resp
            .json()
            .await
            .map_err(|e| Error::other(format!("qdrant search decode: {e}")))?;

        let mut out = Vec::with_capacity(sr.result.len());
        for hit in sr.result {
            let mut doc = Document::new("");
            doc.score = Some(hit.score);
            doc.id = Some(hit.id.to_string().trim_matches('"').to_string());
            if let Some(Value::Object(m)) = hit.payload {
                for (k, v) in m {
                    if k == "__content" {
                        if let Value::String(s) = v { doc.content = s; }
                    } else {
                        doc.metadata.insert(k, v);
                    }
                }
            }
            out.push(doc);
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        let path = format!("/collections/{}/points/delete?wait=true", self.cfg.collection);
        let ids_json: Vec<Value> = ids.iter().map(|s| Value::String(s.clone())).collect();
        let resp = self
            .req(reqwest::Method::POST, &path)
            .json(&json!({ "points": ids_json }))
            .send()
            .await
            .map_err(|e| Error::other(format!("qdrant delete: {e}")))?;
        if !resp.status().is_success() {
            let txt = resp.text().await.unwrap_or_default();
            return Err(Error::other(format!("qdrant delete: {txt}")));
        }
        Ok(())
    }

    async fn len(&self) -> usize {
        #[derive(Deserialize)]
        struct CountResp { result: CountInner }
        #[derive(Deserialize)]
        struct CountInner { count: u64 }

        let path = format!("/collections/{}/points/count", self.cfg.collection);
        let resp = self
            .req(reqwest::Method::POST, &path)
            .json(&json!({ "exact": true }))
            .send()
            .await;
        match resp.ok().and_then(|r| if r.status().is_success() { Some(r) } else { None }) {
            Some(r) => match r.json::<CountResp>().await {
                Ok(c) => c.result.count as usize,
                Err(_) => 0,
            },
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_serializes_to_must_clauses() {
        let mut f = std::collections::HashMap::new();
        f.insert("k".into(), json!("v"));
        let out = QdrantVectorStore::filter_to_qdrant(&f);
        assert_eq!(out["must"][0]["key"], json!("k"));
        assert_eq!(out["must"][0]["match"]["value"], json!("v"));
    }

    #[test]
    fn make_vector_respects_named_mode() {
        let cfg = QdrantConfig::new("http://localhost:6333", "coll")
            .with_vector_name("dense");
        let store = QdrantVectorStore::new(cfg).unwrap();
        let v = store.make_vector(vec![1.0, 2.0, 3.0]);
        assert!(v.get("dense").is_some());
    }
}

//! Embedded HNSW VectorStore backed by `instant-distance` (pure Rust HNSW).
//!
//! # Design
//!
//! `instant-distance`'s `HnswMap` is build-once. We support incremental `add()`
//! by keeping parallel `Vec<FVec>` + `Vec<Document>` buffers and rebuilding the
//! index lazily on first `similarity_search()` after any mutation. For the
//! typical RAG workload — ingest batch, then many queries — this is optimal.
//! Streaming ingest with per-doc query-ability should use `MemoryVectorStore`.
//!
//! # Why not usearch / hnsw_rs?
//!
//! - `usearch`: C++ core adds build-time complexity.
//! - `hnsw_rs`: incremental insert but pulls BLAS.
//! - `instant-distance`: pure Rust, zero C deps, One Signal production-tested.

use async_trait::async_trait;
use instant_distance::{Builder, HnswMap, Point, Search};
use litgraph_core::{Document, Error, Result};
use litgraph_retrieval::store::{Filter, VectorStore};
use parking_lot::RwLock;
use tracing::debug;

#[derive(Debug, Clone)]
struct FVec(Vec<f32>);

impl Point for FVec {
    /// Cosine distance = 1 − cosine similarity. instant-distance wants a metric
    /// where smaller means closer; we convert back at query time.
    fn distance(&self, other: &Self) -> f32 {
        cosine_distance(&self.0, &other.0)
    }
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0f32;
    let mut an = 0f32;
    let mut bn = 0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        an += a[i] * a[i];
        bn += b[i] * b[i];
    }
    let denom = (an.sqrt() * bn.sqrt()).max(1e-12);
    1.0 - (dot / denom)
}

struct Inner {
    points: Vec<FVec>,
    docs: Vec<Document>,
    index: Option<HnswMap<FVec, usize>>,
    dirty: bool,
    ef_search: usize,
    ef_construction: usize,
}

pub struct HnswVectorStore {
    inner: RwLock<Inner>,
}

impl HnswVectorStore {
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(Inner {
                points: Vec::new(),
                docs: Vec::new(),
                index: None,
                dirty: false,
                ef_search: 64,
                ef_construction: 200,
            }),
        }
    }

    pub fn with_ef_search(self, ef: usize) -> Self {
        self.inner.write().ef_search = ef.max(1);
        self
    }

    pub fn with_ef_construction(self, ef: usize) -> Self {
        self.inner.write().ef_construction = ef.max(16);
        self
    }

    fn rebuild(inner: &mut Inner) {
        if !inner.dirty { return; }
        if inner.points.is_empty() {
            inner.index = None;
            inner.dirty = false;
            return;
        }
        let points = inner.points.clone();
        let values: Vec<usize> = (0..inner.docs.len()).collect();
        let map = Builder::default()
            .ef_search(inner.ef_search)
            .ef_construction(inner.ef_construction)
            .build(points, values);
        inner.index = Some(map);
        inner.dirty = false;
        debug!(n = inner.docs.len(), "hnsw index rebuilt");
    }
}

impl Default for HnswVectorStore {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl VectorStore for HnswVectorStore {
    async fn add(&self, mut docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(Error::invalid(format!(
                "len mismatch: docs={} embs={}", docs.len(), embeddings.len()
            )));
        }
        let mut g = self.inner.write();
        let mut ids = Vec::with_capacity(docs.len());
        for (i, mut d) in docs.drain(..).enumerate() {
            let id = d.id.clone().unwrap_or_else(|| format!("h{}", g.docs.len() + i));
            d.id = Some(id.clone());
            g.docs.push(d);
            g.points.push(FVec(embeddings[i].clone()));
            ids.push(id);
        }
        g.dirty = true;
        Ok(ids)
    }

    async fn similarity_search(
        &self,
        q: &[f32],
        k: usize,
        filter: Option<&Filter>,
    ) -> Result<Vec<Document>> {
        // Rebuild if needed under a write lock, then drop and acquire read.
        {
            let mut g = self.inner.write();
            Self::rebuild(&mut g);
        }
        let g = self.inner.read();
        let Some(index) = g.index.as_ref() else { return Ok(vec![]); };

        let query = FVec(q.to_vec());
        let mut search = Search::default();
        // Pull a few extra to survive post-filtering losses.
        let overfetch = if filter.is_some() { k.saturating_mul(4).max(16) } else { k };
        let mut out: Vec<Document> = Vec::new();
        for item in index.search(&query, &mut search).take(overfetch) {
            let idx = *item.value;
            let doc = &g.docs[idx];
            if let Some(f) = filter {
                if !f.iter().all(|(k, v)| doc.metadata.get(k) == Some(v)) {
                    continue;
                }
            }
            let sim = 1.0 - item.distance; // cosine similarity
            let mut d = doc.clone();
            d.score = Some(sim);
            out.push(d);
            if out.len() >= k { break; }
        }
        Ok(out)
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        let mut g = self.inner.write();
        let target: std::collections::HashSet<&String> = ids.iter().collect();
        let points = std::mem::take(&mut g.points);
        let docs = std::mem::take(&mut g.docs);
        let mut keep_points = Vec::with_capacity(docs.len());
        let mut keep_docs = Vec::with_capacity(docs.len());
        for (p, d) in points.into_iter().zip(docs.into_iter()) {
            let keep = d.id.as_ref().map(|id| !target.contains(id)).unwrap_or(true);
            if keep {
                keep_points.push(p);
                keep_docs.push(d);
            }
        }
        g.points = keep_points;
        g.docs = keep_docs;
        g.dirty = true;
        Ok(())
    }

    async fn len(&self) -> usize { self.inner.read().docs.len() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn add_and_search_nearest_first() {
        let s = HnswVectorStore::new();
        let docs = vec![
            Document::new("cat"),
            Document::new("dog"),
            Document::new("car"),
            Document::new("cathode"),
        ];
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.8, 0.2, 0.0],
        ];
        s.add(docs, embs).await.unwrap();
        let r = s.similarity_search(&[1.0, 0.0, 0.0], 2, None).await.unwrap();
        assert_eq!(r.len(), 2);
        assert_eq!(r[0].content, "cat");
        assert!(r[0].score.unwrap() > 0.99);
    }

    #[tokio::test]
    async fn filter_excludes_docs_by_metadata() {
        let s = HnswVectorStore::new();
        let docs = vec![
            Document::new("a").with_metadata("tag", serde_json::json!("keep")).with_id("1"),
            Document::new("b").with_metadata("tag", serde_json::json!("drop")).with_id("2"),
            Document::new("c").with_metadata("tag", serde_json::json!("keep")).with_id("3"),
        ];
        let embs = vec![
            vec![1.0, 0.0],
            vec![0.95, 0.05],
            vec![0.9, 0.1],
        ];
        s.add(docs, embs).await.unwrap();

        let mut f = std::collections::HashMap::new();
        f.insert("tag".into(), serde_json::json!("keep"));
        let r = s.similarity_search(&[1.0, 0.0], 3, Some(&f)).await.unwrap();
        assert_eq!(r.len(), 2);
        assert!(r.iter().all(|d| d.id.as_deref() != Some("2")));
    }

    #[tokio::test]
    async fn delete_removes_docs_from_future_searches() {
        let s = HnswVectorStore::new();
        s.add(
            vec![Document::new("cat").with_id("a"), Document::new("dog").with_id("b")],
            vec![vec![1.0, 0.0], vec![0.9, 0.1]],
        )
        .await
        .unwrap();
        s.delete(&["a".into()]).await.unwrap();
        let r = s.similarity_search(&[1.0, 0.0], 5, None).await.unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].id.as_deref(), Some("b"));
    }
}

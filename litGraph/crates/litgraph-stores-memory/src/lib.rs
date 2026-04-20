//! In-memory vector store. Rayon-parallel brute-force cosine similarity.
//!
//! Reference implementation of the `VectorStore` trait from `litgraph-retrieval`.
//! For production use `litgraph-stores-usearch` or a remote client.

use std::sync::RwLock;

use async_trait::async_trait;
use litgraph_core::{Document, Result};
use litgraph_retrieval::store::{Filter, VectorStore};
use rayon::prelude::*;

struct Entry {
    doc: Document,
    embedding: Vec<f32>,
}

#[derive(Default)]
pub struct MemoryVectorStore {
    inner: RwLock<std::collections::HashMap<String, Entry>>,
}

impl MemoryVectorStore {
    pub fn new() -> Self { Self::default() }
}

#[async_trait]
impl VectorStore for MemoryVectorStore {
    async fn add(&self, mut docs: Vec<Document>, embeddings: Vec<Vec<f32>>) -> Result<Vec<String>> {
        if docs.len() != embeddings.len() {
            return Err(litgraph_core::Error::invalid(format!(
                "len mismatch: docs={} embs={}", docs.len(), embeddings.len()
            )));
        }
        let mut ids = Vec::with_capacity(docs.len());
        let mut g = self
            .inner
            .write()
            .map_err(|_| litgraph_core::Error::other("rwlock poisoned"))?;
        for (i, mut d) in docs.drain(..).enumerate() {
            let id = d.id.clone().unwrap_or_else(|| format!("m{}", g.len() + i));
            d.id = Some(id.clone());
            g.insert(id.clone(), Entry { doc: d, embedding: embeddings[i].clone() });
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
        let g = self
            .inner
            .read()
            .map_err(|_| litgraph_core::Error::other("rwlock poisoned"))?;

        let candidates: Vec<&Entry> = g
            .values()
            .filter(|e| match filter {
                None => true,
                Some(f) => f.iter().all(|(k, v)| e.doc.metadata.get(k) == Some(v)),
            })
            .collect();

        let q_norm = norm(q);
        let mut scored: Vec<(f32, &Entry)> = candidates
            .par_iter()
            .map(|e| (cosine(q, &e.embedding, q_norm), *e))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .map(|(s, e)| { let mut d = e.doc.clone(); d.score = Some(s); d })
            .collect())
    }

    async fn delete(&self, ids: &[String]) -> Result<()> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| litgraph_core::Error::other("rwlock poisoned"))?;
        for id in ids { g.remove(id); }
        Ok(())
    }

    async fn len(&self) -> usize {
        self.inner.read().map(|g| g.len()).unwrap_or(0)
    }
}

fn norm(v: &[f32]) -> f32 { v.iter().map(|x| x * x).sum::<f32>().sqrt() }

fn cosine(a: &[f32], b: &[f32], a_norm: f32) -> f32 {
    let len = a.len().min(b.len());
    let mut dot = 0f32;
    let mut bn = 0f32;
    for i in 0..len {
        dot += a[i] * b[i];
        bn += b[i] * b[i];
    }
    let bn = bn.sqrt();
    if a_norm == 0.0 || bn == 0.0 { return 0.0; }
    dot / (a_norm * bn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn add_and_search() {
        let s = MemoryVectorStore::new();
        let docs = vec![
            Document::new("cat"),
            Document::new("dog"),
            Document::new("car"),
        ];
        let embs = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.9, 0.1, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        s.add(docs, embs).await.unwrap();
        let res = s.similarity_search(&[1.0, 0.0, 0.0], 2, None).await.unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].content, "cat");
    }
}

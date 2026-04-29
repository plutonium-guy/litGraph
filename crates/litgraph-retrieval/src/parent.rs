//! ParentDocumentRetriever — store small chunks for embedding precision,
//! return the larger parent document for context.
//!
//! # The pattern
//!
//! Standard RAG indexing makes a tradeoff: small chunks give precise
//! embedding similarity (each chunk talks about one thing) but lose context;
//! large chunks preserve context but smear the embedding across multiple
//! topics. ParentDocumentRetriever sidesteps the tradeoff by indexing
//! BOTH:
//!
//! - **Children** — small split chunks, embedded + stored in the vector
//!   store with a `__parent_id` metadata pointer back to their parent.
//! - **Parents** — full original documents, stored in a separate `DocStore`
//!   keyed by id.
//!
//! At retrieval time we similarity-search the children, deduplicate by
//! parent_id (preserving best-rank order), and look up the corresponding
//! parents. Result: precise retrieval over small chunks, but the final docs
//! handed to the LLM are the larger parents — better context, no fragmentation.
//!
//! Direct LangChain `ParentDocumentRetriever` parity, plus the standard
//! Retriever trait so it composes with HybridRetriever / RerankingRetriever
//! transparently.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use litgraph_core::{Document, Embeddings, Error, Result};
use parking_lot::Mutex;
use uuid::Uuid;

use crate::retriever::Retriever;
use crate::store::{Filter, VectorStore};

pub const PARENT_ID_META_KEY: &str = "__parent_id";

/// Persistence for full parent documents, keyed by id. Separate from
/// `VectorStore` because parents don't need vector indexing — they're only
/// fetched by id during the second phase of retrieval.
#[async_trait]
pub trait DocStore: Send + Sync {
    async fn put_many(&self, docs: Vec<(String, Document)>) -> Result<()>;

    /// Return docs in the order of `ids`. `None` for missing ids — caller
    /// decides whether to error or skip.
    async fn get_many(&self, ids: &[String]) -> Result<Vec<Option<Document>>>;
}

/// In-process HashMap-backed DocStore. Drops with the process; use a
/// sqlite-backed impl for durability across restarts (TODO future iter).
pub struct MemoryDocStore {
    inner: Mutex<HashMap<String, Document>>,
}

impl MemoryDocStore {
    pub fn new() -> Self { Self { inner: Mutex::new(HashMap::new()) } }
    pub fn len(&self) -> usize { self.inner.lock().len() }
    pub fn is_empty(&self) -> bool { self.inner.lock().is_empty() }
}

impl Default for MemoryDocStore {
    fn default() -> Self { Self::new() }
}

#[async_trait]
impl DocStore for MemoryDocStore {
    async fn put_many(&self, docs: Vec<(String, Document)>) -> Result<()> {
        let mut g = self.inner.lock();
        for (id, doc) in docs {
            g.insert(id, doc);
        }
        Ok(())
    }

    async fn get_many(&self, ids: &[String]) -> Result<Vec<Option<Document>>> {
        let g = self.inner.lock();
        Ok(ids.iter().map(|id| g.get(id).cloned()).collect())
    }
}

/// Splitter trait alias to avoid pulling litgraph-splitters into this crate.
/// `ParentDocumentRetriever` works with anything that splits a Document into
/// chunks; concrete callers pass an `Arc<dyn ChildSplitter>`.
pub trait ChildSplitter: Send + Sync {
    fn split(&self, doc: &Document) -> Vec<Document>;
}

pub struct ParentDocumentRetriever {
    pub child_splitter: Arc<dyn ChildSplitter>,
    pub vector_store: Arc<dyn VectorStore>,
    pub embeddings: Arc<dyn Embeddings>,
    pub parent_store: Arc<dyn DocStore>,
    /// How many child chunks to fetch from the vector store per query.
    /// Typically larger than the requested parent k since multiple children
    /// may map to the same parent. Default: max(k * 4, 16).
    pub child_k_factor: usize,
    pub filter: Option<Filter>,
}

impl ParentDocumentRetriever {
    pub fn new(
        child_splitter: Arc<dyn ChildSplitter>,
        vector_store: Arc<dyn VectorStore>,
        embeddings: Arc<dyn Embeddings>,
        parent_store: Arc<dyn DocStore>,
    ) -> Self {
        Self {
            child_splitter,
            vector_store,
            embeddings,
            parent_store,
            child_k_factor: 4,
            filter: None,
        }
    }

    /// Override the child fan-out multiplier. The retriever requests
    /// `max(k * factor, 16)` child chunks before deduplicating to parents.
    pub fn with_child_k_factor(mut self, factor: usize) -> Self {
        self.child_k_factor = factor.max(1);
        self
    }

    pub fn with_filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Index a batch of parent documents. For each parent:
    /// - Assign a UUID id (or reuse `doc.id` if set).
    /// - Store the parent in the doc store.
    /// - Split into children; attach `__parent_id` metadata to each child.
    /// - Embed children in one batch and upsert to the vector store.
    ///
    /// Returns the parent ids (in input order).
    pub async fn index_documents(&self, mut docs: Vec<Document>) -> Result<Vec<String>> {
        if docs.is_empty() {
            return Ok(Vec::new());
        }

        // Assign parent ids + collect (id, doc) for the parent store.
        let mut parent_ids = Vec::with_capacity(docs.len());
        let mut parents = Vec::with_capacity(docs.len());
        for d in docs.iter_mut() {
            let pid = match &d.id {
                Some(existing) => existing.clone(),
                None => {
                    let new_id = Uuid::new_v4().to_string();
                    d.id = Some(new_id.clone());
                    new_id
                }
            };
            parent_ids.push(pid.clone());
            parents.push((pid, d.clone()));
        }
        self.parent_store.put_many(parents).await?;

        // Split each parent → children. Tag each child with __parent_id.
        let mut all_children: Vec<Document> = Vec::new();
        for (parent_doc, parent_id) in docs.iter().zip(parent_ids.iter()) {
            let mut children = self.child_splitter.split(parent_doc);
            for child in children.iter_mut() {
                child.metadata.insert(
                    PARENT_ID_META_KEY.to_string(),
                    serde_json::Value::String(parent_id.clone()),
                );
                // Children get fresh ids — they live in the vector store
                // independently of their parent.
                child.id = None;
            }
            all_children.extend(children);
        }

        if all_children.is_empty() {
            return Ok(parent_ids);
        }

        // One batch embedding call for all children — major perf win on
        // large indexings (caller doesn't have to pre-batch).
        let texts: Vec<String> = all_children.iter().map(|c| c.content.clone()).collect();
        let embs = self.embeddings.embed_documents(&texts).await?;
        if embs.len() != all_children.len() {
            return Err(Error::other(format!(
                "embed_documents returned {} embeddings for {} children",
                embs.len(), all_children.len()
            )));
        }
        self.vector_store.add(all_children, embs).await?;

        Ok(parent_ids)
    }
}

#[async_trait]
impl Retriever for ParentDocumentRetriever {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let child_k = (k * self.child_k_factor).max(16);
        let q = self.embeddings.embed_query(query).await?;
        let children = self
            .vector_store
            .similarity_search(&q, child_k, self.filter.as_ref())
            .await?;

        // Dedupe parent_ids preserving order-of-first-appearance (best child
        // rank ⇒ parent ranks first).
        let mut seen = std::collections::HashSet::new();
        let mut ordered_parent_ids: Vec<String> = Vec::new();
        for child in &children {
            if let Some(pid) = child
                .metadata
                .get(PARENT_ID_META_KEY)
                .and_then(|v| v.as_str())
            {
                if seen.insert(pid.to_string()) {
                    ordered_parent_ids.push(pid.to_string());
                    if ordered_parent_ids.len() >= k {
                        break;
                    }
                }
            }
        }

        if ordered_parent_ids.is_empty() {
            return Ok(Vec::new());
        }

        let parents = self.parent_store.get_many(&ordered_parent_ids).await?;
        // Drop missing ids silently — a parent could've been evicted while
        // its children remained. The caller still gets the rest.
        Ok(parents.into_iter().flatten().collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Filter;
    use litgraph_core::{Document, Result as LgResult};
    use std::sync::Mutex as StdMutex;

    /// Splitter that breaks on ". " (period+space). Keeps tests deterministic.
    struct PeriodSplitter;
    impl ChildSplitter for PeriodSplitter {
        fn split(&self, doc: &Document) -> Vec<Document> {
            doc.content
                .split(". ")
                .filter(|s| !s.is_empty())
                .map(|s| {
                    let mut d = Document::new(s);
                    d.metadata = doc.metadata.clone();
                    d
                })
                .collect()
        }
    }

    /// Embeddings: token-count vector (length-of-text repeated 4 dims) — keeps
    /// tests deterministic without pulling a real embedding model.
    struct LenEmbeddings;
    #[async_trait]
    impl Embeddings for LenEmbeddings {
        fn name(&self) -> &str { "len" }
        fn dimensions(&self) -> usize { 4 }
        async fn embed_query(&self, text: &str) -> LgResult<Vec<f32>> {
            Ok(vec![text.len() as f32; 4])
        }
        async fn embed_documents(&self, texts: &[String]) -> LgResult<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| vec![t.len() as f32; 4]).collect())
        }
    }

    /// VectorStore that returns ALL stored docs scored by L2-distance from query.
    /// Dead simple for testing; not for production.
    struct TinyStore {
        rows: StdMutex<Vec<(Vec<f32>, Document)>>,
    }
    impl TinyStore {
        fn new() -> Self { Self { rows: StdMutex::new(Vec::new()) } }
    }
    #[async_trait]
    impl VectorStore for TinyStore {
        async fn add(&self, docs: Vec<Document>, embs: Vec<Vec<f32>>) -> LgResult<Vec<String>> {
            let mut g = self.rows.lock().unwrap();
            let mut ids = Vec::new();
            for (mut d, e) in docs.into_iter().zip(embs) {
                let id = d.id.clone().unwrap_or_else(|| Uuid::new_v4().to_string());
                d.id = Some(id.clone());
                g.push((e, d));
                ids.push(id);
            }
            Ok(ids)
        }
        async fn similarity_search(
            &self,
            q: &[f32],
            k: usize,
            _f: Option<&Filter>,
        ) -> LgResult<Vec<Document>> {
            let g = self.rows.lock().unwrap();
            let mut scored: Vec<(f32, Document)> = g
                .iter()
                .map(|(e, d)| {
                    let dist: f32 = e
                        .iter()
                        .zip(q.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f32>()
                        .sqrt();
                    (dist, d.clone())
                })
                .collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            Ok(scored.into_iter().take(k).map(|(_, d)| d).collect())
        }
        async fn delete(&self, _ids: &[String]) -> LgResult<()> { Ok(()) }
        async fn len(&self) -> usize { self.rows.lock().unwrap().len() }
    }

    #[tokio::test]
    async fn index_stores_parents_and_children_separately() {
        let parent_store = Arc::new(MemoryDocStore::new());
        let vec_store = Arc::new(TinyStore::new());
        let r = ParentDocumentRetriever::new(
            Arc::new(PeriodSplitter),
            vec_store.clone(),
            Arc::new(LenEmbeddings),
            parent_store.clone(),
        );
        let docs = vec![
            Document::new("Rust is fast. Rust is safe. Rust has no GC."),
            Document::new("Python is slow. Python is dynamic."),
        ];
        let ids = r.index_documents(docs).await.unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(parent_store.len(), 2);
        // 3 children for parent 1, 2 children for parent 2 = 5 total.
        assert_eq!(vec_store.len().await, 5);
        // Each child carries __parent_id pointing back at its parent.
        let stored = vec_store.rows.lock().unwrap();
        for (_, child) in stored.iter() {
            let pid = child.metadata.get(PARENT_ID_META_KEY).and_then(|v| v.as_str());
            assert!(pid.is_some(), "child missing parent_id metadata");
            assert!(ids.contains(&pid.unwrap().to_string()));
        }
    }

    #[tokio::test]
    async fn retrieve_returns_parent_documents_not_children() {
        let parent_store = Arc::new(MemoryDocStore::new());
        let vec_store = Arc::new(TinyStore::new());
        let r = ParentDocumentRetriever::new(
            Arc::new(PeriodSplitter),
            vec_store.clone(),
            Arc::new(LenEmbeddings),
            parent_store.clone(),
        );
        r.index_documents(vec![
            Document::new("Rust is fast. Rust is safe."),
            Document::new("Python is slow."),
        ])
        .await
        .unwrap();

        let hits = r.retrieve("anything", 2).await.unwrap();
        // Returned docs are the FULL parents (with periods), not split chunks.
        assert_eq!(hits.len(), 2);
        let contents: Vec<&str> = hits.iter().map(|h| h.content.as_str()).collect();
        assert!(contents.iter().any(|c| c.contains("Rust is fast. Rust is safe.")));
        assert!(contents.iter().any(|c| c.contains("Python is slow.")));
    }

    #[tokio::test]
    async fn retrieve_dedupes_parents_when_multiple_children_match() {
        // 1 parent with 3 children → query returns top-3 child hits, all
        // pointing to the SAME parent. Result must be 1 parent, not 3.
        let parent_store = Arc::new(MemoryDocStore::new());
        let vec_store = Arc::new(TinyStore::new());
        let r = ParentDocumentRetriever::new(
            Arc::new(PeriodSplitter),
            vec_store.clone(),
            Arc::new(LenEmbeddings),
            parent_store.clone(),
        );
        r.index_documents(vec![Document::new("Same. Parent. Three. Chunks.")])
            .await
            .unwrap();
        let hits = r.retrieve("query", 5).await.unwrap();
        assert_eq!(hits.len(), 1, "dedup should collapse to 1 parent, got {}", hits.len());
    }

    #[tokio::test]
    async fn retrieve_skips_parents_that_were_evicted() {
        // Parent stored, child indexed, then parent removed. retrieve() must
        // silently drop the missing parent, not error.
        let parent_store = Arc::new(MemoryDocStore::new());
        let vec_store = Arc::new(TinyStore::new());
        let r = ParentDocumentRetriever::new(
            Arc::new(PeriodSplitter),
            vec_store.clone(),
            Arc::new(LenEmbeddings),
            parent_store.clone(),
        );
        let ids = r
            .index_documents(vec![
                Document::new("Alpha doc. With sentences."),
                Document::new("Beta doc. Other sentences."),
            ])
            .await
            .unwrap();
        // Evict the first parent.
        parent_store.inner.lock().remove(&ids[0]);

        let hits = r.retrieve("anything", 5).await.unwrap();
        // Only the second parent's docs survive — first was evicted.
        assert!(hits.iter().all(|h| h.content != "Alpha doc. With sentences."));
        assert!(hits.iter().any(|h| h.content == "Beta doc. Other sentences."));
    }

    #[tokio::test]
    async fn empty_index_call_is_a_no_op() {
        let parent_store = Arc::new(MemoryDocStore::new());
        let vec_store = Arc::new(TinyStore::new());
        let r = ParentDocumentRetriever::new(
            Arc::new(PeriodSplitter),
            vec_store.clone(),
            Arc::new(LenEmbeddings),
            parent_store.clone(),
        );
        let ids = r.index_documents(vec![]).await.unwrap();
        assert!(ids.is_empty());
        assert_eq!(parent_store.len(), 0);
        assert_eq!(vec_store.len().await, 0);
    }
}


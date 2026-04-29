use std::sync::Arc;

use litgraph_core::{Document, Embeddings, Result};
use litgraph_retrieval::{Bm25Index, HybridRetriever, Retriever, VectorRetriever};
use litgraph_retrieval::store::VectorStore;

struct FakeEmbeddings;

#[async_trait::async_trait]
impl Embeddings for FakeEmbeddings {
    fn name(&self) -> &str { "fake" }
    fn dimensions(&self) -> usize { 3 }
    async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        Ok(text_vec(text))
    }
    async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|t| text_vec(t)).collect())
    }
}

fn text_vec(t: &str) -> Vec<f32> {
    // Extremely naive: presence of letters a/b/c maps to 3 dims.
    let l = t.to_lowercase();
    vec![
        if l.contains('a') { 1.0 } else { 0.0 },
        if l.contains('b') { 1.0 } else { 0.0 },
        if l.contains('c') { 1.0 } else { 0.0 },
    ]
}

struct TinyStore { docs: Vec<(Document, Vec<f32>)> }

#[async_trait::async_trait]
impl VectorStore for TinyStore {
    async fn add(&self, _docs: Vec<Document>, _embs: Vec<Vec<f32>>) -> Result<Vec<String>> {
        Ok(vec![])
    }
    async fn similarity_search(
        &self,
        q: &[f32],
        k: usize,
        _f: Option<&litgraph_retrieval::store::Filter>,
    ) -> Result<Vec<Document>> {
        let mut scored: Vec<(f32, Document)> = self
            .docs
            .iter()
            .map(|(d, e)| {
                let dot: f32 = q.iter().zip(e.iter()).map(|(a, b)| a * b).sum();
                let mut d = d.clone();
                d.score = Some(dot);
                (dot, d)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored.into_iter().map(|(_, d)| d).collect())
    }
    async fn delete(&self, _ids: &[String]) -> Result<()> { Ok(()) }
    async fn len(&self) -> usize { self.docs.len() }
}

#[tokio::test]
async fn hybrid_fuses_vector_and_bm25() {
    let docs = vec![
        Document::new("apple banana").with_id("1"),
        Document::new("banana cherry").with_id("2"),
        Document::new("apple cherry").with_id("3"),
    ];
    let embs: Vec<Vec<f32>> = docs.iter().map(|d| text_vec(&d.content)).collect();
    let store = Arc::new(TinyStore {
        docs: docs.iter().cloned().zip(embs.into_iter()).collect(),
    }) as Arc<dyn VectorStore>;

    let bm25 = Arc::new(Bm25Index::from_docs(docs.clone()).unwrap()) as Arc<dyn Retriever>;
    let vec_r = Arc::new(VectorRetriever::new(Arc::new(FakeEmbeddings), store)) as Arc<dyn Retriever>;

    let hybrid = HybridRetriever::new(vec![vec_r, bm25]);
    let r = hybrid.retrieve("apple", 3).await.unwrap();
    assert!(!r.is_empty());
    // Document 1 or 3 (both mention "apple") should rank above doc 2 (no "apple").
    let top_id = r[0].id.clone().unwrap_or_default();
    assert!(top_id == "1" || top_id == "3", "top was {:?}", top_id);
}

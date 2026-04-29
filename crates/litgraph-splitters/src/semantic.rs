//! Embedding-based semantic chunking — splits text at points of high semantic
//! distance between consecutive sentences. Implementation of Greg Kamradt's
//! algorithm (later adopted as LangChain's experimental `SemanticChunker`):
//!
//! 1. Sentence-split the input.
//! 2. For each sentence, build a "combined" window = N preceding + current +
//!    N following sentences. The window provides local context so that very
//!    short sentences ("OK.") don't dominate the distance signal.
//! 3. Embed all windows in a single batched call.
//! 4. Compute cosine distance (1 − cosine_similarity) between consecutive
//!    embeddings.
//! 5. The breakpoint threshold is the `breakpoint_percentile`th percentile of
//!    those distances — adaptive per document, no global hyperparameter to
//!    tune across corpora.
//! 6. Walk sentences; whenever the distance to the next exceeds the
//!    threshold, close the current chunk.
//!
//! Unlike `RecursiveCharacterSplitter`, chunks honor topic shifts even when
//! mid-paragraph, which materially improves retrieval recall on dense docs.
//!
//! The trait stays sync (recursive/markdown splitters are CPU-only); semantic
//! chunking is exposed as a separate `async fn split_text` because it issues
//! HTTP calls.

use std::sync::Arc;

use litgraph_core::{Document, Embeddings, Error, Result};
use unicode_segmentation::UnicodeSegmentation;

#[derive(Clone)]
pub struct SemanticChunker {
    embeddings: Arc<dyn Embeddings>,
    /// Sentences combined on each side of the focal sentence for context.
    /// Default 1 — i.e. window = 3 sentences (prev, focal, next).
    pub buffer_size: usize,
    /// Percentile (0–100) of consecutive-distance distribution above which we
    /// declare a breakpoint. Higher = fewer/longer chunks. Default 95.
    pub breakpoint_percentile: f64,
    /// Hard minimum number of sentences per chunk; avoids 1-sentence chunks
    /// when a document has many small breakpoints clustered together.
    /// Default 1 (off).
    pub min_sentences_per_chunk: usize,
}

impl SemanticChunker {
    pub fn new(embeddings: Arc<dyn Embeddings>) -> Self {
        Self {
            embeddings,
            buffer_size: 1,
            breakpoint_percentile: 95.0,
            min_sentences_per_chunk: 1,
        }
    }

    pub fn with_buffer_size(mut self, n: usize) -> Self {
        self.buffer_size = n;
        self
    }
    pub fn with_breakpoint_percentile(mut self, p: f64) -> Self {
        self.breakpoint_percentile = p.clamp(0.0, 100.0);
        self
    }
    pub fn with_min_sentences_per_chunk(mut self, n: usize) -> Self {
        self.min_sentences_per_chunk = n.max(1);
        self
    }

    /// Sentence-split the input. UAX-29-aware (handles ?, !, ., abbreviations,
    /// non-Latin scripts) and skips whitespace-only fragments.
    fn sentence_split(text: &str) -> Vec<String> {
        text.unicode_sentences()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    fn build_windows(&self, sentences: &[String]) -> Vec<String> {
        let n = sentences.len();
        let buf = self.buffer_size;
        (0..n)
            .map(|i| {
                let lo = i.saturating_sub(buf);
                let hi = (i + buf + 1).min(n);
                sentences[lo..hi].join(" ")
            })
            .collect()
    }

    pub async fn split_text(&self, text: &str) -> Result<Vec<String>> {
        let sentences = Self::sentence_split(text);
        if sentences.len() <= 1 {
            return Ok(if text.trim().is_empty() {
                vec![]
            } else {
                vec![text.to_string()]
            });
        }

        let windows = self.build_windows(&sentences);
        let embs = self.embeddings.embed_documents(&windows).await?;
        if embs.len() != windows.len() {
            return Err(Error::other(format!(
                "embeddings count mismatch: got {}, expected {}",
                embs.len(),
                windows.len()
            )));
        }

        // Cosine distance between consecutive windows.
        let dists: Vec<f64> = embs
            .windows(2)
            .map(|w| 1.0 - cosine_similarity(&w[0], &w[1]) as f64)
            .collect();

        if dists.is_empty() {
            return Ok(vec![sentences.join(" ")]);
        }

        let threshold = percentile(&dists, self.breakpoint_percentile);

        // Walk sentences, accumulating into the current chunk; close when
        // distance to next exceeds threshold AND current chunk has ≥ min size.
        let min_n = self.min_sentences_per_chunk;
        let mut chunks: Vec<String> = Vec::new();
        let mut current: Vec<&str> = Vec::with_capacity(sentences.len());
        for (i, s) in sentences.iter().enumerate() {
            current.push(s.as_str());
            let is_last = i == sentences.len() - 1;
            let should_break = !is_last && dists[i] > threshold && current.len() >= min_n;
            if should_break || is_last {
                chunks.push(current.join(" "));
                current.clear();
            }
        }
        Ok(chunks)
    }

    /// Document variant — preserves metadata, propagates `chunk_index` and
    /// `source_id` like the sync trait does.
    pub async fn split_document(&self, doc: &Document) -> Result<Vec<Document>> {
        let chunks = self.split_text(&doc.content).await?;
        Ok(chunks
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let mut d = Document::new(chunk);
                d.metadata = doc.metadata.clone();
                d.metadata.insert("chunk_index".into(), serde_json::json!(i));
                if let Some(id) = &doc.id {
                    d.id = Some(format!("{id}#{i}"));
                    d.metadata.insert("source_id".into(), serde_json::json!(id));
                }
                d
            })
            .collect())
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-12);
    (dot / denom).clamp(-1.0, 1.0)
}

/// Linear interpolation percentile (matches numpy.percentile default).
fn percentile(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let p = p.clamp(0.0, 100.0);
    let rank = p / 100.0 * (n - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = rank - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;

    /// Toy embedder: produces vectors that are semantically grouped by a
    /// keyword present in the window. Lets us hand-craft known cosine
    /// distances so the chunker's percentile logic is testable.
    struct ToyEmb;

    #[async_trait]
    impl Embeddings for ToyEmb {
        fn name(&self) -> &str { "toy" }
        fn dimensions(&self) -> usize { 3 }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self.embed_documents(&[text.to_string()]).await?.pop().unwrap())
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|t| {
                let lower = t.to_ascii_lowercase();
                if lower.contains("dog") { vec![1.0, 0.0, 0.0] }
                else if lower.contains("car") { vec![0.0, 1.0, 0.0] }
                else if lower.contains("plant") { vec![0.0, 0.0, 1.0] }
                else { vec![0.5, 0.5, 0.5] }
            }).collect())
        }
    }

    #[test]
    fn percentile_handles_known_inputs() {
        // numpy.percentile([1,2,3,4,5], 50) == 3.0 (linear interp default)
        assert!((percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0) - 3.0).abs() < 1e-9);
        // p=0 → min, p=100 → max
        assert_eq!(percentile(&[5.0, 1.0, 9.0], 0.0), 1.0);
        assert_eq!(percentile(&[5.0, 1.0, 9.0], 100.0), 9.0);
        // single-element edge case
        assert_eq!(percentile(&[42.0], 75.0), 42.0);
    }

    #[test]
    fn cosine_similarity_orthogonal_vs_parallel() {
        assert!((cosine_similarity(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
        assert!((cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) + 1.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn splits_at_topic_shift_using_toy_embedder() {
        let chunker = SemanticChunker::new(Arc::new(ToyEmb))
            .with_buffer_size(0)              // no smoothing → distances are exact
            .with_breakpoint_percentile(50.0); // median: split at the largest gap

        // Three topical clusters: dog / car / plant. Two transitions.
        let text = "I love my dog. The dog plays fetch. \
                    My car is fast. The car needs gas. \
                    The plant needs water. My plant grew tall.";
        let chunks = chunker.split_text(text).await.unwrap();
        assert!(chunks.len() >= 2, "got {} chunks: {:?}", chunks.len(), chunks);
        // First chunk should be all dog-themed; last should be all plant-themed.
        assert!(chunks.first().unwrap().to_ascii_lowercase().contains("dog"));
        assert!(chunks.last().unwrap().to_ascii_lowercase().contains("plant"));
    }

    #[tokio::test]
    async fn single_sentence_returns_unchanged() {
        let chunker = SemanticChunker::new(Arc::new(ToyEmb));
        let chunks = chunker.split_text("Just one sentence here.").await.unwrap();
        assert_eq!(chunks, vec!["Just one sentence here.".to_string()]);
    }

    #[tokio::test]
    async fn empty_text_returns_empty() {
        let chunker = SemanticChunker::new(Arc::new(ToyEmb));
        let chunks = chunker.split_text("   \n\t  ").await.unwrap();
        assert!(chunks.is_empty());
    }

    #[tokio::test]
    async fn document_metadata_propagated_with_chunk_index() {
        // 4 sentences → 3 distances; one in-cluster (dist=0) + two between
        // clusters (dist=1) gives p50 ≈ 1.0 with the small one well below it →
        // exactly one break in the middle.
        let chunker = SemanticChunker::new(Arc::new(ToyEmb))
            .with_buffer_size(0)
            .with_breakpoint_percentile(50.0);
        let mut doc = Document::new(
            "I love my dog. The dog plays fetch. My car is fast. The car needs gas."
        ).with_id("src");
        doc.metadata.insert("origin".into(), serde_json::json!("test"));
        let out = chunker.split_document(&doc).await.unwrap();
        assert!(out.len() >= 2, "got {} chunks: {:?}", out.len(), out);
        for (i, d) in out.iter().enumerate() {
            assert_eq!(d.metadata.get("chunk_index").unwrap(), &serde_json::json!(i));
            assert_eq!(d.metadata.get("source_id").unwrap(), &serde_json::json!("src"));
            assert_eq!(d.metadata.get("origin").unwrap(), &serde_json::json!("test"));
            assert_eq!(d.id.as_deref().unwrap(), format!("src#{i}"));
        }
    }
}

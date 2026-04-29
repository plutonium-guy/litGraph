//! Pure-Rust BM25 index — in-memory, Unicode-segmentation tokenizer, case-folded.
//!
//! k1=1.5, b=0.75 (Okapi defaults). Score:
//!
//! ```text
//!   score(q, d) = Σ_t idf(t) * ( tf(t, d) * (k1+1) ) / ( tf(t, d) + k1 * (1 - b + b * |d|/avgdl) )
//!   idf(t)      = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )
//! ```

use std::collections::HashMap;
use std::sync::RwLock;

use async_trait::async_trait;
use litgraph_core::{Document, Error, Result};
use rayon::prelude::*;
use unicode_segmentation::UnicodeSegmentation;

use crate::retriever::Retriever;

const K1: f32 = 1.5;
const B: f32 = 0.75;

struct Doc {
    doc: Document,
    /// token counts per distinct term in this doc
    term_counts: HashMap<String, u32>,
    len: u32,
}

struct Inner {
    docs: Vec<Doc>,
    /// term → document frequency
    df: HashMap<String, u32>,
    avg_len: f32,
}

impl Inner {
    fn new() -> Self { Self { docs: Vec::new(), df: HashMap::new(), avg_len: 0.0 } }
}

pub struct Bm25Index {
    inner: RwLock<Inner>,
}

impl Default for Bm25Index {
    fn default() -> Self { Self::new() }
}

impl Bm25Index {
    pub fn new() -> Self { Self { inner: RwLock::new(Inner::new()) } }

    pub fn from_docs(docs: Vec<Document>) -> Result<Self> {
        let idx = Self::new();
        idx.add(docs)?;
        Ok(idx)
    }

    /// Add a batch of documents to the index. Tokenization + per-doc
    /// term-counting runs **Rayon-parallel** across documents; the
    /// per-doc term-count maps are then merged into the shared DF
    /// table sequentially under the write lock.
    ///
    /// For small batches (<32 docs) the parallel overhead doesn't pay
    /// off. We still go through `par_iter` because Rayon's overhead
    /// is small (~µs) and the implementation stays single-path; for
    /// hot-loop micro-benchmarks the caller can pre-batch.
    pub fn add(&self, docs: Vec<Document>) -> Result<()> {
        if docs.is_empty() {
            return Ok(());
        }

        // Stage 1: parallel CPU work — tokenize + count per doc.
        // Each closure runs independently with no shared state, so
        // this scales linearly with cores up to the docs.len() ceiling.
        let prepped: Vec<(Document, HashMap<String, u32>, u32)> = docs
            .into_par_iter()
            .map(|d| {
                let tokens = tokenize(&d.content);
                let len = tokens.len() as u32;
                let mut tc: HashMap<String, u32> = HashMap::new();
                for t in &tokens {
                    *tc.entry(t.clone()).or_insert(0) += 1;
                }
                (d, tc, len)
            })
            .collect();

        // Stage 2: merge under the write lock. DF aggregation,
        // doc-id assignment, and avg_len recompute all touch shared
        // state, so they're sequential — but they're cheap pure-Rust
        // hashmap ops, not the bottleneck.
        let mut g = self
            .inner
            .write()
            .map_err(|_| Error::other("bm25 rwlock poisoned"))?;
        for (mut doc, tc, len) in prepped {
            for term in tc.keys() {
                *g.df.entry(term.clone()).or_insert(0) += 1;
            }
            if doc.id.is_none() {
                doc.id = Some(format!("bm25_{}", g.docs.len()));
            }
            g.docs.push(Doc {
                doc,
                term_counts: tc,
                len,
            });
        }
        let n: usize = g.docs.len();
        if n > 0 {
            let total: u64 = g.docs.iter().map(|d| d.len as u64).sum();
            g.avg_len = total as f32 / n as f32;
        }
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.inner.read().map(|g| g.docs.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool { self.len() == 0 }

    pub fn search(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        let g = self.inner.read().map_err(|_| Error::other("bm25 rwlock poisoned"))?;
        if g.docs.is_empty() { return Ok(vec![]); }
        let q_terms = tokenize(query);
        let n = g.docs.len() as f32;

        let mut scored: Vec<(f32, &Doc)> = g
            .docs
            .par_iter()
            .map(|d| {
                let score = score_bm25(&q_terms, d, &g.df, g.avg_len, n);
                (score, d)
            })
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored
            .into_iter()
            .filter(|(s, _)| s.is_finite() && *s > 0.0)
            .map(|(s, d)| {
                let mut out = d.doc.clone();
                out.score = Some(s);
                out
            })
            .collect())
    }
}

#[async_trait]
impl Retriever for Bm25Index {
    async fn retrieve(&self, query: &str, k: usize) -> Result<Vec<Document>> {
        self.search(query, k)
    }
}

fn tokenize(text: &str) -> Vec<String> {
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty())
        .collect()
}

fn score_bm25(
    q_terms: &[String],
    d: &Doc,
    df: &HashMap<String, u32>,
    avg_len: f32,
    n: f32,
) -> f32 {
    let mut score = 0.0;
    let dl = d.len as f32;
    for t in q_terms {
        let tf = *d.term_counts.get(t).unwrap_or(&0) as f32;
        if tf == 0.0 { continue; }
        let df_t = *df.get(t).unwrap_or(&0) as f32;
        let idf = ((n - df_t + 0.5) / (df_t + 0.5) + 1.0).ln();
        let denom = tf + K1 * (1.0 - B + B * dl / avg_len.max(1.0));
        score += idf * (tf * (K1 + 1.0)) / denom;
    }
    score
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_exact_match_first() {
        let docs = vec![
            Document::new("the quick brown fox jumps over the lazy dog"),
            Document::new("my cat sleeps a lot"),
            Document::new("foxes are clever animals"),
        ];
        let idx = Bm25Index::from_docs(docs).unwrap();
        let r = idx.search("quick fox", 2).unwrap();
        assert!(!r.is_empty());
        assert!(r[0].content.contains("quick brown fox"));
    }

    #[test]
    fn parallel_add_assigns_distinct_ids() {
        // Auto-id assignment uses `g.docs.len()` at the moment of
        // insertion, so even with parallel tokenization the final
        // ids end up unique and contiguous (0..N).
        let docs: Vec<Document> = (0..200)
            .map(|i| Document::new(format!("doc number {i} contains the word marker_{i}")))
            .collect();
        let idx = Bm25Index::from_docs(docs).unwrap();
        let mut ids: Vec<String> = idx
            .inner
            .read()
            .unwrap()
            .docs
            .iter()
            .map(|d| d.doc.id.clone().unwrap())
            .collect();
        ids.sort();
        ids.dedup();
        assert_eq!(ids.len(), 200, "duplicate auto-ids were assigned");
    }

    #[test]
    fn parallel_add_df_matches_sequential() {
        // Build the same corpus two ways — original sequential reference
        // (computed inline below) and the parallel `add` — DF maps must
        // match exactly.
        let docs: Vec<Document> = vec![
            Document::new("rust memory safety"),
            Document::new("memory leaks in c programs"),
            Document::new("rust async runtime"),
            Document::new("javascript closures"),
            Document::new("python generators"),
        ];
        let idx = Bm25Index::from_docs(docs.clone()).unwrap();
        let par_df = idx.inner.read().unwrap().df.clone();

        // Sequential reference DF construction.
        let mut ref_df: HashMap<String, u32> = HashMap::new();
        for d in &docs {
            let tokens = tokenize(&d.content);
            let mut seen: HashMap<String, u32> = HashMap::new();
            for t in &tokens {
                *seen.entry(t.clone()).or_insert(0) += 1;
            }
            for term in seen.keys() {
                *ref_df.entry(term.clone()).or_insert(0) += 1;
            }
        }
        assert_eq!(par_df, ref_df, "parallel DF diverges from sequential reference");
    }

    #[test]
    fn parallel_add_avg_len_matches_sequential() {
        let docs: Vec<Document> = (0..50)
            .map(|i| Document::new(format!("doc {i} {}", "tok ".repeat(i % 7 + 1))))
            .collect();
        let idx = Bm25Index::from_docs(docs.clone()).unwrap();
        let par_avg = idx.inner.read().unwrap().avg_len;

        let total: u64 = docs.iter().map(|d| tokenize(&d.content).len() as u64).sum();
        let ref_avg = total as f32 / docs.len() as f32;
        assert!((par_avg - ref_avg).abs() < 1e-4, "avg_len drift: par={par_avg} ref={ref_avg}");
    }

    #[test]
    fn parallel_add_search_quality_unchanged() {
        // Ranking must still surface the lexical-best match first
        // for a query — confirms the parallel pipeline didn't
        // shuffle term-counts or DF.
        let docs: Vec<Document> = (0..100)
            .map(|i| {
                if i == 42 {
                    Document::new("this is a fox-only document marker_unique")
                } else {
                    Document::new(format!("filler doc {i} unrelated to the query"))
                }
            })
            .collect();
        let idx = Bm25Index::from_docs(docs).unwrap();
        let hits = idx.search("marker_unique fox-only", 3).unwrap();
        assert!(!hits.is_empty());
        assert!(hits[0].content.contains("marker_unique"));
    }

    #[test]
    fn parallel_add_empty_input_is_noop() {
        let idx = Bm25Index::new();
        idx.add(vec![]).unwrap();
        assert_eq!(idx.len(), 0);
        assert_eq!(idx.inner.read().unwrap().avg_len, 0.0);
    }

    #[test]
    fn parallel_add_preserves_supplied_doc_ids() {
        let docs = vec![
            Document::new("first").with_id("custom-a"),
            Document::new("second").with_id("custom-b"),
        ];
        let idx = Bm25Index::from_docs(docs).unwrap();
        let g = idx.inner.read().unwrap();
        assert_eq!(g.docs[0].doc.id.as_deref(), Some("custom-a"));
        assert_eq!(g.docs[1].doc.id.as_deref(), Some("custom-b"));
    }
}

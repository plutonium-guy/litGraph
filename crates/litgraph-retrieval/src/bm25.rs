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

    pub fn add(&self, docs: Vec<Document>) -> Result<()> {
        let mut g = self
            .inner
            .write()
            .map_err(|_| Error::other("bm25 rwlock poisoned"))?;
        for mut d in docs {
            let tokens = tokenize(&d.content);
            let len = tokens.len() as u32;
            let mut tc: HashMap<String, u32> = HashMap::new();
            for t in &tokens { *tc.entry(t.clone()).or_insert(0) += 1; }
            for term in tc.keys() { *g.df.entry(term.clone()).or_insert(0) += 1; }
            if d.id.is_none() { d.id = Some(format!("bm25_{}", g.docs.len())); }
            g.docs.push(Doc { doc: d, term_counts: tc, len });
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
}

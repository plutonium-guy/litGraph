//! BERTScore-lite — embedding-based F1 for NLG evaluation.
//!
//! # Why BERTScore
//!
//! ROUGE / BLEU / chrF / METEOR / WER are all token-overlap metrics.
//! They miss semantic-equivalence cases where the candidate uses
//! synonyms or paraphrases the reference: "the cat is happy" vs
//! "the feline is content" share zero word n-grams (and minimal
//! char n-grams) but mean roughly the same thing.
//!
//! BERTScore (Zhang et al. 2020) replaces token-overlap with token-
//! embedding cosine similarity:
//! - For each candidate token, take the max cosine to any
//!   reference token → contributes to PRECISION.
//! - For each reference token, take the max cosine to any
//!   candidate token → contributes to RECALL.
//! - F1 = 2·P·R / (P+R).
//!
//! Synonym pairs (cat / feline) score high cosine in any modern
//! embedding space, so the metric credits semantic match where
//! word-level metrics would return zero.
//!
//! # BERTScore-lite vs full BERTScore
//!
//! Full BERTScore uses **contextual** BERT embeddings — each token's
//! vector depends on its surrounding sentence. Most modern Python
//! impls run a transformer over the whole sentence and pluck per-
//! token vectors from the hidden states.
//!
//! `bertscore` here uses the litgraph `Embeddings` trait, which
//! embeds whole strings → one vector. We approximate per-token
//! contextual embeddings by embedding each token *in isolation* (one
//! `embed_documents` call per side, batched). Loses cross-token
//! context (ambiguous words like "bank" get the same vector
//! regardless of "river bank" vs "bank account") but works with any
//! Embeddings backend (OpenAI / Cohere / Voyage / Jina / BGE / a
//! custom embedder), no model-specific contextual-embedding API
//! required.
//!
//! For full-fidelity BERTScore-with-context, callers can implement a
//! custom Embeddings backend that runs sentence-level transformer
//! inference and exposes per-token vectors via a separate API.
//!
//! # Composability
//!
//! BERTScore complements word-level metrics — typical NLG eval
//! reports BLEU + chrF + BERTScore so reviewers see vocabulary
//! overlap (BLEU/chrF) AND semantic match (BERTScore) side-by-side.
//! When BERTScore is high but BLEU is low, the model is
//! paraphrasing accurately. When BERTScore is low but BLEU is
//! high, the model is repeating reference vocabulary without
//! semantic fidelity (rare; usually flags annotation noise).
//!
//! # Performance
//!
//! O(n_ref · n_cand) cosine comparisons after the embed calls.
//! With typical sentences (10-50 tokens each side) the pairwise
//! matrix is small. The dominant cost is the embedding calls —
//! consider caching via iter-292 `CachedEmbeddings` for repeated
//! eval runs over the same gold set.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::{Embeddings, Result};

/// BERTScore result with diagnostic breakdown.
///
/// `f1` is the headline number; `precision` (cand-side max-sim avg)
/// and `recall` (ref-side max-sim avg) surface over/under-generation
/// patterns the same way ROUGE's P/R does.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BertScoreResult {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
    pub n_ref_tokens: u32,
    pub n_cand_tokens: u32,
}

impl BertScoreResult {
    fn zero() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
            n_ref_tokens: 0,
            n_cand_tokens: 0,
        }
    }
}

/// Compute BERTScore-lite F1 for a single (reference, candidate) pair.
///
/// Tokenizes via lowercase + whitespace, embeds each side's tokens
/// in two batched `embed_documents` calls, then computes the standard
/// BERTScore P/R/F1 from the pairwise cosine-similarity matrix.
///
/// # Edge cases
///
/// - Either side empty → `BertScoreResult::zero` without an embed
///   call.
/// - Both single-token → P=R=F1 = cos(ref_emb, cand_emb).
///
/// # Errors
///
/// Propagates errors from the embedder (rate limit, network,
/// unsupported model). Compose with iter-251 `RetryingEmbeddings`
/// + iter-292 `CachedEmbeddings` for production resilience.
pub async fn bertscore(
    reference: &str,
    candidate: &str,
    embeddings: Arc<dyn Embeddings>,
) -> Result<BertScoreResult> {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(BertScoreResult::zero());
    }
    // Two batched embed calls. embed_documents lets the provider
    // exploit batching (OpenAI/Voyage/etc all charge less per token
    // for batched calls).
    let ref_embs = embeddings.embed_documents(&ref_tokens).await?;
    let cand_embs = embeddings.embed_documents(&cand_tokens).await?;
    let n_ref = ref_embs.len();
    let n_cand = cand_embs.len();
    // Per-cand max-cosine → precision contribution.
    let mut p_sum = 0.0_f32;
    for c in &cand_embs {
        let mut best = -1.0_f32;
        for r in &ref_embs {
            let sim = cosine_sim(c, r);
            if sim > best {
                best = sim;
            }
        }
        p_sum += best.max(0.0); // clamp negative cosines (orthogonal/anti) to 0
    }
    // Per-ref max-cosine → recall contribution.
    let mut r_sum = 0.0_f32;
    for r in &ref_embs {
        let mut best = -1.0_f32;
        for c in &cand_embs {
            let sim = cosine_sim(r, c);
            if sim > best {
                best = sim;
            }
        }
        r_sum += best.max(0.0);
    }
    let p = p_sum / n_cand as f32;
    let r = r_sum / n_ref as f32;
    let f1 = if p + r == 0.0 {
        0.0
    } else {
        2.0 * p * r / (p + r)
    };
    Ok(BertScoreResult {
        precision: p,
        recall: r,
        f1,
        n_ref_tokens: n_ref as u32,
        n_cand_tokens: n_cand as u32,
    })
}

fn tokenize_lower(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split_whitespace().map(|t| t.to_lowercase())
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    let mut dot = 0.0_f32;
    let mut na = 0.0_f32;
    let mut nb = 0.0_f32;
    for (x, y) in a.iter().zip(b) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    let denom = (na * nb).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::collections::HashMap;

    /// Embeddings stub: maps each token string to a hand-picked
    /// vector so tests can construct exact pairwise similarity
    /// scenarios without a real provider.
    struct ScriptedEmb {
        map: HashMap<String, Vec<f32>>,
        dim: usize,
    }

    impl ScriptedEmb {
        fn new(pairs: &[(&str, Vec<f32>)]) -> Arc<Self> {
            let mut map = HashMap::new();
            let dim = pairs.first().map(|(_, v)| v.len()).unwrap_or(2);
            for (k, v) in pairs {
                map.insert(k.to_string(), v.clone());
            }
            Arc::new(Self { map, dim })
        }
    }

    #[async_trait]
    impl Embeddings for ScriptedEmb {
        fn name(&self) -> &str {
            "scripted-bert"
        }
        fn dimensions(&self) -> usize {
            self.dim
        }
        async fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
            Ok(self
                .map
                .get(text)
                .cloned()
                .unwrap_or_else(|| vec![0.0; self.dim]))
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(texts
                .iter()
                .map(|t| {
                    self.map
                        .get(t)
                        .cloned()
                        .unwrap_or_else(|| vec![0.0; self.dim])
                })
                .collect())
        }
    }

    #[tokio::test]
    async fn identical_strings_score_one() {
        // Identical inputs → each token's max-sim partner is itself
        // (cosine = 1) → P=R=F1=1.
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
        ]);
        let r = bertscore("the cat", "the cat", emb).await.unwrap();
        assert!((r.f1 - 1.0).abs() < 1e-5);
        assert!((r.precision - 1.0).abs() < 1e-5);
        assert!((r.recall - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn empty_inputs_zero() {
        let emb = ScriptedEmb::new(&[]);
        assert_eq!(bertscore("", "x", emb.clone()).await.unwrap(), BertScoreResult::zero());
        assert_eq!(bertscore("x", "", emb.clone()).await.unwrap(), BertScoreResult::zero());
        assert_eq!(bertscore("", "", emb).await.unwrap(), BertScoreResult::zero());
    }

    #[tokio::test]
    async fn synonym_credit_above_word_overlap() {
        // The marquee feature: word-level overlap is 0 (cat vs feline),
        // but BERTScore credits the semantic match via high cosine.
        let emb = ScriptedEmb::new(&[
            // cat and feline: nearly identical vectors (high cos).
            ("cat", vec![1.0, 0.05]),
            ("feline", vec![0.99, 0.04]),
            // happy / content: also near-synonyms.
            ("happy", vec![0.0, 1.0]),
            ("content", vec![0.05, 0.99]),
        ]);
        let r = bertscore("cat happy", "feline content", emb).await.unwrap();
        // Word-level: 0/2 unigrams match → ROUGE-1 F1 = 0. BERTScore
        // should be high (>0.9) because the semantic pairs match.
        assert!(r.f1 > 0.9, "expected BERTScore-lite > 0.9, got {}", r.f1);
    }

    #[tokio::test]
    async fn disjoint_orthogonal_yields_low_score() {
        // Vectors orthogonal between sides → max cosine is 0
        // (clamped from negative or actual 0) → F1 = 0.
        let emb = ScriptedEmb::new(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![1.0, 0.0]),
            ("x", vec![0.0, 1.0]),
            ("y", vec![0.0, 1.0]),
        ]);
        let r = bertscore("a b", "x y", emb).await.unwrap();
        assert_eq!(r.f1, 0.0);
        assert_eq!(r.precision, 0.0);
        assert_eq!(r.recall, 0.0);
    }

    #[tokio::test]
    async fn token_counts_in_struct() {
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
            ("sat", vec![0.5, 0.5]),
        ]);
        let r = bertscore("the cat sat", "the cat", emb).await.unwrap();
        assert_eq!(r.n_ref_tokens, 3);
        assert_eq!(r.n_cand_tokens, 2);
    }

    #[tokio::test]
    async fn negative_cosine_clamped_to_zero() {
        // Ref token's vector is anti-aligned with cand's. Raw cosine
        // would be -1. The clamp prevents negative max-sim from
        // dragging the average below 0.
        let emb = ScriptedEmb::new(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![-1.0, 0.0]), // anti-aligned with "a"
        ]);
        let r = bertscore("a", "b", emb).await.unwrap();
        // Raw cosine = -1 → clamped to 0 → F1 = 0.
        assert_eq!(r.precision, 0.0);
        assert_eq!(r.f1, 0.0);
    }

    #[tokio::test]
    async fn lowercase_normalization() {
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
        ]);
        // "The Cat" lowercases to "the cat"; should still match
        // perfectly against "the cat".
        let r = bertscore("The Cat", "the cat", emb).await.unwrap();
        assert!((r.f1 - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn p_and_r_can_diverge() {
        // Cand has an extra token not present in ref. Recall stays
        // at 1.0 (every ref token has a perfect match in cand);
        // precision drops because the extra cand token has nothing
        // in ref.
        let emb = ScriptedEmb::new(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![0.0, 1.0]),
            ("c", vec![0.5, 0.5]), // unrelated extra
        ]);
        let r = bertscore("a b", "a b c", emb).await.unwrap();
        // Recall: ref tokens [a, b] both have perfect partner in cand → R=1.0.
        // Precision: cand=[a,b,c]. a→a (1.0), b→b (1.0), c→max(c,a or c,b).
        //   c=[0.5,0.5]; cos(c,a)= 0.5/sqrt(0.5)=√0.5 ≈ 0.707;
        //   cos(c,b)= same. So c contributes ~0.707.
        // P = (1 + 1 + 0.707) / 3 ≈ 0.902.
        assert!((r.recall - 1.0).abs() < 1e-5, "r={}", r.recall);
        assert!(r.precision < 1.0 && r.precision > 0.85, "p={}", r.precision);
    }

    #[tokio::test]
    async fn embed_failure_propagates() {
        struct FailEmb;
        #[async_trait]
        impl Embeddings for FailEmb {
            fn name(&self) -> &str {
                "fail"
            }
            fn dimensions(&self) -> usize {
                2
            }
            async fn embed_query(&self, _: &str) -> Result<Vec<f32>> {
                Err(crate::Error::Provider("embed down".into()))
            }
            async fn embed_documents(&self, _: &[String]) -> Result<Vec<Vec<f32>>> {
                Err(crate::Error::Provider("embed down".into()))
            }
        }
        let r = bertscore("a", "b", Arc::new(FailEmb)).await;
        assert!(r.is_err());
    }
}

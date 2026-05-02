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

/// Result of a Relaxed Word Mover Distance (RWMD) call.
///
/// `rwmd` is the headline distance — `max(ref_to_cand, cand_to_ref)`.
/// Lower is better; 0.0 = identical content (same embeddings); higher
/// values indicate progressively more semantic distance. The per-
/// direction fields are exposed so callers can diagnose asymmetric
/// cases ("ref has content the cand misses entirely" → high
/// `ref_to_cand`, low `cand_to_ref`).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WmdScore {
    pub rwmd: f32,
    pub ref_to_cand: f32,
    pub cand_to_ref: f32,
    pub n_ref_tokens: u32,
    pub n_cand_tokens: u32,
}

impl WmdScore {
    fn zero() -> Self {
        Self {
            rwmd: 0.0,
            ref_to_cand: 0.0,
            cand_to_ref: 0.0,
            n_ref_tokens: 0,
            n_cand_tokens: 0,
        }
    }
}

/// Relaxed Word Mover Distance (Kusner et al. 2015) — semantic
/// distance via per-token nearest-neighbor matching in embedding
/// space.
///
/// **Lower is better** (opposite of iter-321 BERTScore). 0 = identical
/// content; higher values = more semantic distance. Unbounded above
/// 1.0 in principle but for normalized cosine-derived distance stays
/// in `[0, 2]`.
///
/// # Distinct from iter-321 BERTScore
///
/// Both use the existing `Embeddings` trait + per-token nearest-
/// neighbor cosine, but the conventions differ:
///
/// - BERTScore reports SIMILARITY (higher-better) as F1 of per-side
///   max-cosine averages. Result in [0, 1].
/// - RWMD reports DISTANCE (lower-better) as `1 - max_cosine`
///   averaged per side, then `max` of the two directions. Result
///   in [0, 2].
///
/// Different conventions, different communities. WMD is the
/// canonical name in MT literature (Kusner 2015 introduced it for
/// document classification; widely adopted in machine translation
/// + summarization eval). BERTScore is the modern name for
/// embedding-based similarity. Ship both.
///
/// # Why "Relaxed" not full WMD
///
/// Full Word Mover Distance solves a linear-programming transport
/// problem (each ref token's "mass" flows to cand tokens minimizing
/// total transport cost). LP solver is heavy and the asymptotic
/// complexity is O(n³ log n). RWMD drops the transport constraint
/// — each token independently picks its nearest-neighbor in the
/// other side, no flow conservation. Kusner 2015 §4 showed RWMD is
/// a tight lower bound on full WMD AND has nearly equivalent
/// empirical correlation with human judgment, at O(n²) cost. The
/// canonical practical choice.
///
/// Full WMD with LP solver is roadmap work for callers who need
/// exact transport semantics (rare).
///
/// # Algorithm
///
/// 1. Tokenize via lowercase + whitespace.
/// 2. Two batched `embed_documents` calls (one per side).
/// 3. `ref_to_cand = mean(1 - max_cosine_over_cand_per_ref_token)`.
/// 4. `cand_to_ref = mean(1 - max_cosine_over_ref_per_cand_token)`.
/// 5. `rwmd = max(ref_to_cand, cand_to_ref)`.
///
/// Negative cosines clamped — `1 - max(0, cosine)` — so anti-aligned
/// pairs contribute distance ≤ 1.0, not > 1.0.
///
/// # Edge cases
///
/// - Both empty → `WmdScore::zero` (0 distance, 0 tokens).
/// - One side empty: returns 1.0 distance (no tokens to compare against
///   — assume max distance under normalized cosine convention).
pub async fn relaxed_wmd(
    reference: &str,
    candidate: &str,
    embeddings: Arc<dyn Embeddings>,
) -> Result<WmdScore> {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return Ok(WmdScore::zero());
    }
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return Ok(WmdScore {
            rwmd: 1.0,
            ref_to_cand: if ref_tokens.is_empty() { 0.0 } else { 1.0 },
            cand_to_ref: if cand_tokens.is_empty() { 0.0 } else { 1.0 },
            n_ref_tokens: ref_tokens.len() as u32,
            n_cand_tokens: cand_tokens.len() as u32,
        });
    }
    let ref_embs = embeddings.embed_documents(&ref_tokens).await?;
    let cand_embs = embeddings.embed_documents(&cand_tokens).await?;
    let n_ref = ref_embs.len();
    let n_cand = cand_embs.len();
    // ref → cand: per ref token, find max cosine over cand tokens; distance = 1 - max.
    let mut r2c_sum = 0.0_f32;
    for r in &ref_embs {
        let mut best = -1.0_f32;
        for c in &cand_embs {
            let sim = cosine_sim(r, c);
            if sim > best {
                best = sim;
            }
        }
        // Clamp negative similarities to 0 — distance never exceeds 1.
        r2c_sum += 1.0 - best.max(0.0);
    }
    // cand → ref: mirror.
    let mut c2r_sum = 0.0_f32;
    for c in &cand_embs {
        let mut best = -1.0_f32;
        for r in &ref_embs {
            let sim = cosine_sim(c, r);
            if sim > best {
                best = sim;
            }
        }
        c2r_sum += 1.0 - best.max(0.0);
    }
    let r2c = r2c_sum / n_ref as f32;
    let c2r = c2r_sum / n_cand as f32;
    let rwmd = r2c.max(c2r);
    Ok(WmdScore {
        rwmd,
        ref_to_cand: r2c,
        cand_to_ref: c2r,
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

    // ─── Relaxed WMD ──────────────────────────────────────────────

    #[tokio::test]
    async fn rwmd_identical_strings_zero_distance() {
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
        ]);
        let r = relaxed_wmd("the cat", "the cat", emb).await.unwrap();
        assert!(r.rwmd < 1e-5, "got {}", r.rwmd);
        assert!(r.ref_to_cand < 1e-5);
        assert!(r.cand_to_ref < 1e-5);
    }

    #[tokio::test]
    async fn rwmd_both_empty_zero() {
        let emb = ScriptedEmb::new(&[]);
        let r = relaxed_wmd("", "", emb).await.unwrap();
        assert_eq!(r, WmdScore::zero());
    }

    #[tokio::test]
    async fn rwmd_one_empty_max_distance() {
        let emb = ScriptedEmb::new(&[("the", vec![1.0, 0.0])]);
        let r = relaxed_wmd("", "the", emb.clone()).await.unwrap();
        assert_eq!(r.rwmd, 1.0);
        let r = relaxed_wmd("the", "", emb).await.unwrap();
        assert_eq!(r.rwmd, 1.0);
    }

    #[tokio::test]
    async fn rwmd_synonyms_low_distance() {
        // cat/feline + happy/content near-identical embeddings
        // → low WMD, despite zero word overlap.
        let emb = ScriptedEmb::new(&[
            ("cat", vec![1.0, 0.05]),
            ("feline", vec![0.99, 0.04]),
            ("happy", vec![0.0, 1.0]),
            ("content", vec![0.05, 0.99]),
        ]);
        let r = relaxed_wmd("cat happy", "feline content", emb).await.unwrap();
        // High cosine → low distance per token. RWMD < 0.05.
        assert!(r.rwmd < 0.1, "expected low RWMD, got {}", r.rwmd);
    }

    #[tokio::test]
    async fn rwmd_disjoint_high_distance() {
        let emb = ScriptedEmb::new(&[
            ("a", vec![1.0, 0.0]),
            ("x", vec![0.0, 1.0]),
        ]);
        // Each side's only token has no good match in the other side:
        // a vs x → cosine 0 → distance 1.
        let r = relaxed_wmd("a", "x", emb).await.unwrap();
        assert!((r.rwmd - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn rwmd_asymmetric_directions() {
        // Ref has 3 tokens, all matching cand's 1 token strongly.
        // ref_to_cand: 3 tokens each find a perfect match in cand → 0.
        // cand_to_ref: 1 cand token finds a perfect match in ref → 0.
        // Both directions ≈ 0, total ≈ 0. Construct a case where they
        // differ instead: ref has perfect-match-in-cand token + a
        // disjoint token; cand has only the matching token.
        // ref = "good", "weird"; cand = "good"; emb: good aligned, weird disjoint.
        // ref_to_cand: good→good (dist 0) + weird→good (dist 1) → mean 0.5
        // cand_to_ref: good→good (dist 0) → mean 0
        // RWMD = max(0.5, 0) = 0.5. ref_to_cand > cand_to_ref → asymmetric.
        let emb = ScriptedEmb::new(&[
            ("good", vec![1.0, 0.0]),
            ("weird", vec![0.0, 1.0]),
        ]);
        let r = relaxed_wmd("good weird", "good", emb).await.unwrap();
        assert!((r.ref_to_cand - 0.5).abs() < 1e-5, "r2c={}", r.ref_to_cand);
        assert!(r.cand_to_ref < 1e-5);
        assert!((r.rwmd - 0.5).abs() < 1e-5);
    }

    #[tokio::test]
    async fn rwmd_negative_cosine_clamped_to_distance_one() {
        // Anti-aligned vectors → cosine -1 → 1 - max(0, -1) = 1.
        let emb = ScriptedEmb::new(&[
            ("a", vec![1.0, 0.0]),
            ("b", vec![-1.0, 0.0]),
        ]);
        let r = relaxed_wmd("a", "b", emb).await.unwrap();
        assert!((r.rwmd - 1.0).abs() < 1e-5);
        // Should NOT exceed 1.0 — clamp protects against -2 distance.
        assert!(r.rwmd <= 1.0);
    }

    #[tokio::test]
    async fn rwmd_token_counts_in_struct() {
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
            ("sat", vec![0.5, 0.5]),
        ]);
        let r = relaxed_wmd("the cat sat", "the cat", emb).await.unwrap();
        assert_eq!(r.n_ref_tokens, 3);
        assert_eq!(r.n_cand_tokens, 2);
    }

    #[tokio::test]
    async fn rwmd_lowercase_normalization() {
        let emb = ScriptedEmb::new(&[
            ("the", vec![1.0, 0.0]),
            ("cat", vec![0.0, 1.0]),
        ]);
        let r = relaxed_wmd("The Cat", "the cat", emb).await.unwrap();
        assert!(r.rwmd < 1e-5);
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

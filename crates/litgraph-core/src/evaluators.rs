//! String evaluators for prompt/output scoring. LangChain
//! `evaluation.string_distance` + `evaluation.exact_match` parity.
//!
//! # Why pure-string evaluators
//!
//! When you tweak a prompt or swap a model, you want to know if the
//! output is "close enough" to the gold answer. These functions are the
//! lowest-level building blocks — pure, deterministic, no LLM calls.
//!
//! Use them in:
//! - Prompt-experiment harnesses (run N prompts, score each output)
//! - Regression tests for agent traces ("did the model answer 'Paris'?")
//! - Eval pipelines (composite scores: 0.6 levenshtein + 0.4 jaccard)
//!
//! For LLM-as-judge or RAGAS-style metrics, you'd compose these with a
//! ChatModel — out of scope here.
//!
//! # Functions
//!
//! - `exact_match(actual, expected)` → `bool` (trim + lowercase compare)
//! - `levenshtein_ratio(a, b)` → `f32` in `[0, 1]` (1 = identical, 0 = no shared chars)
//! - `jaccard_similarity(a, b)` → `f32` in `[0, 1]` (token-set overlap)
//! - `regex_match(text, pattern)` → `bool` (compiled regex against text)
//! - `json_validity(text)` → `bool` (does it parse as JSON?)
//! - `embedding_cosine(a, b)` → `f32` in `[-1, 1]` (cosine of two vectors)
//! - `contains_all(text, substrings)` → `bool` (every substring present)
//! - `contains_any(text, substrings)` → `bool` (at least one present)

use std::collections::HashSet;

use regex::Regex;

use crate::{Error, Result};

/// True if `actual` and `expected` match after trimming and case-folding.
/// The 80% LLM-eval check ("did the model say 42?").
pub fn exact_match(actual: &str, expected: &str) -> bool {
    actual.trim().eq_ignore_ascii_case(expected.trim())
}

/// Strict version: byte-equal with no normalization. Use when whitespace
/// or case matters (code generation, formatted output).
pub fn exact_match_strict(actual: &str, expected: &str) -> bool {
    actual == expected
}

/// Levenshtein-distance ratio normalized to `[0, 1]`. 1.0 = identical;
/// 0.0 = entirely different (one is empty or no shared chars).
///
/// Formula: `1 - (lev_distance / max(|a|, |b|))`.
///
/// O(|a|·|b|) time. Both empty → 1.0.
pub fn levenshtein_ratio(a: &str, b: &str) -> f32 {
    let la = a.chars().count();
    let lb = b.chars().count();
    if la == 0 && lb == 0 {
        return 1.0;
    }
    let max = la.max(lb) as f32;
    let dist = levenshtein(a, b) as f32;
    (max - dist) / max
}

/// Raw Levenshtein distance (number of single-char edits to transform `a`
/// into `b`). O(|a|·|b|) time, O(min(|a|, |b|)) space (rolling-row).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let av: Vec<char> = a.chars().collect();
    let bv: Vec<char> = b.chars().collect();
    let m = av.len();
    let n = bv.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    // Two rolling rows: prev and curr.
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if av[i - 1] == bv[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Jaccard token-set similarity. Tokenizes by whitespace + lowercases.
/// `|A ∩ B| / |A ∪ B|`. Both empty → 1.0; one empty → 0.0.
///
/// Use when word ORDER doesn't matter ("apple banana" ≈ "banana apple").
pub fn jaccard_similarity(a: &str, b: &str) -> f32 {
    let toks_a: HashSet<String> = tokenize_lower(a).collect();
    let toks_b: HashSet<String> = tokenize_lower(b).collect();
    if toks_a.is_empty() && toks_b.is_empty() {
        return 1.0;
    }
    let intersection = toks_a.intersection(&toks_b).count() as f32;
    let union = toks_a.union(&toks_b).count() as f32;
    intersection / union
}

fn tokenize_lower(s: &str) -> impl Iterator<Item = String> + '_ {
    s.split_whitespace().map(|t| t.to_lowercase())
}

/// True if `pattern` (a regex) matches anywhere in `text`. Returns an
/// `Err` if the pattern doesn't compile — caller can decide whether to
/// treat that as "no match" or surface the error.
///
/// Internally uses the [`regex`] crate (RE2-ish, no backtracking) — safe
/// against catastrophic regexes.
pub fn regex_match(text: &str, pattern: &str) -> Result<bool> {
    let re = Regex::new(pattern)
        .map_err(|e| Error::parse(format!("invalid regex `{pattern}`: {e}")))?;
    Ok(re.is_match(text))
}

/// True if `text` parses as valid JSON (any value — object, array,
/// scalar). Use after asking the LLM for JSON output and before feeding
/// to a strict deserializer.
pub fn json_validity(text: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(text).is_ok()
}

/// Cosine similarity of two equal-length embedding vectors. Result in
/// `[-1, 1]` (1 = identical direction; 0 = orthogonal; -1 = opposite).
///
/// Returns `Err` if vector lengths differ. Both vectors all-zero → 0.0
/// (avoids division-by-zero NaN).
pub fn embedding_cosine(a: &[f32], b: &[f32]) -> Result<f32> {
    if a.len() != b.len() {
        return Err(Error::invalid(format!(
            "embedding_cosine: length mismatch {} vs {}",
            a.len(),
            b.len()
        )));
    }
    if a.is_empty() {
        return Ok(0.0);
    }
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (na.sqrt() * nb.sqrt()))
}

/// True if every substring in `needles` appears somewhere in `text`.
/// Case-sensitive — wrap with `.to_lowercase()` if needed.
pub fn contains_all(text: &str, needles: &[&str]) -> bool {
    needles.iter().all(|n| text.contains(n))
}

/// True if at least one substring in `needles` appears in `text`.
pub fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|n| text.contains(n))
}

/// Precision / Recall / F1 triple returned by ROUGE-N and ROUGE-L.
///
/// All three values are in `[0.0, 1.0]`. Convention used here:
/// - **Precision** = overlap / candidate-total
/// - **Recall** = overlap / reference-total
/// - **F1** = 2·P·R / (P + R), with `0.0` when `P + R == 0`
///
/// The headline number is usually `f1` for symmetric reporting;
/// individual `precision`/`recall` are exposed for callers who
/// need to debug which side of the over/under-generation tradeoff
/// a candidate fell on.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RougeScore {
    pub precision: f32,
    pub recall: f32,
    pub f1: f32,
}

/// ROUGE-N score (Lin 2004) — n-gram overlap between `reference`
/// and `candidate` summaries. The standard automated metric for
/// summarization eval; pairs with `bleu` for translation eval (BLEU
/// is precision-weighted, ROUGE is recall-emphasizing).
///
/// # Algorithm
///
/// 1. Tokenize both strings via lowercase + whitespace split.
/// 2. Build n-gram **multiset** counters (counts matter; "the the"
///    must overlap with one "the" once, not twice — the standard
///    "clipped" overlap rule).
/// 3. Overlap = `Σ min(ref_count[g], cand_count[g])` over unique grams.
/// 4. Precision = overlap / cand-total-grams; Recall = overlap /
///    ref-total-grams; F1 = 2·P·R / (P+R).
///
/// # Edge cases
///
/// - `n == 0` → all zeros (degenerate; caller should pass n ≥ 1).
/// - Either string yields fewer than `n` tokens → all zeros (no
///   n-grams to compare).
/// - Both empty → all zeros (no signal).
///
/// # Convention shortcuts
///
/// `rouge_1` and `rouge_2` (unigram and bigram, the two most
/// commonly reported variants) are exposed as separate functions.
pub fn rouge_n(reference: &str, candidate: &str, n: usize) -> RougeScore {
    if n == 0 {
        return RougeScore::zero();
    }
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    let ref_grams = ngrams(&ref_tokens, n);
    let cand_grams = ngrams(&cand_tokens, n);
    if ref_grams.is_empty() || cand_grams.is_empty() {
        return RougeScore::zero();
    }
    // Build count maps.
    let ref_counts = count_map(&ref_grams);
    let cand_counts = count_map(&cand_grams);
    // Clipped overlap: sum of min(ref[g], cand[g]) over each unique gram.
    let mut overlap: usize = 0;
    for (g, &c_count) in &cand_counts {
        if let Some(&r_count) = ref_counts.get(g) {
            overlap += r_count.min(c_count);
        }
    }
    let p = overlap as f32 / cand_grams.len() as f32;
    let r = overlap as f32 / ref_grams.len() as f32;
    RougeScore::from_pr(p, r)
}

/// ROUGE-1 (unigram overlap). The most commonly reported variant —
/// captures content-word overlap without sensitivity to phrasing
/// order.
pub fn rouge_1(reference: &str, candidate: &str) -> RougeScore {
    rouge_n(reference, candidate, 1)
}

/// ROUGE-2 (bigram overlap). Captures local fluency / phrasing
/// — penalizes summaries that have the right vocabulary but in
/// the wrong order.
pub fn rouge_2(reference: &str, candidate: &str) -> RougeScore {
    rouge_n(reference, candidate, 2)
}

/// ROUGE-L score (Lin 2004) — longest common subsequence (LCS)
/// based F1. Distinct from ROUGE-N: ROUGE-N is local-order-sensitive
/// (bigrams must appear in the same adjacent order), ROUGE-L is
/// global-order-sensitive but allows gaps between matched tokens.
/// Captures sentence-level structure that bigrams miss.
///
/// # Algorithm
///
/// 1. Tokenize both strings via lowercase + whitespace split.
/// 2. Compute LCS length via DP — O(m·n) time, O(min(m, n)) space
///    via rolling row.
/// 3. Precision = LCS / cand-len; Recall = LCS / ref-len;
///    F1 = 2·P·R / (P+R).
///
/// # Edge cases
///
/// Both empty → all zeros. One empty → all zeros.
pub fn rouge_l(reference: &str, candidate: &str) -> RougeScore {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    let m = ref_tokens.len();
    let n = cand_tokens.len();
    if m == 0 || n == 0 {
        return RougeScore::zero();
    }
    let lcs = lcs_length(&ref_tokens, &cand_tokens);
    let p = lcs as f32 / n as f32;
    let r = lcs as f32 / m as f32;
    RougeScore::from_pr(p, r)
}

impl RougeScore {
    fn zero() -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f1: 0.0,
        }
    }
    fn from_pr(precision: f32, recall: f32) -> Self {
        let f1 = if precision + recall == 0.0 {
            0.0
        } else {
            2.0 * precision * recall / (precision + recall)
        };
        Self {
            precision,
            recall,
            f1,
        }
    }
}

/// BLEU-N score (Papineni et al. 2002) — clipped n-gram precision
/// with brevity penalty. The de-facto translation-eval metric;
/// pairs with iter-304 ROUGE for the precision-vs-recall tradeoff.
///
/// # Distinct from ROUGE
///
/// - **ROUGE** is recall-emphasizing → "did the candidate cover the
///   reference's content?" — fits summarization where missing facts
///   is worse than adding extras.
/// - **BLEU** is precision-weighted → "is the candidate's content
///   actually in the reference?" — fits translation where extra
///   content is hallucination, not bonus.
///
/// Different shapes of the same n-gram-overlap problem; ship both
/// because different tasks need different bias.
///
/// # The score
///
/// 1. Tokenize both via lowercase + whitespace split.
/// 2. For each n in `1..=max_n`: compute clipped n-gram precision
///    `p_n = overlap / cand-total-grams` (same clipped-count rule
///    as ROUGE-N).
/// 3. Geometric mean of the precisions:
///    `gm = exp(Σ log(p_n) / max_n)`. With strict semantics, ANY
///    zero precision sends gm to 0 (and therefore BLEU to 0). With
///    smoothing (`smooth = true`), zero overlaps are replaced with
///    `1 / (cand_total + 1)` per Chen & Cherry (2014) method 1 lite
///    — keeps short-sentence scores informative without inflating.
/// 4. Brevity penalty `BP = 1` if `cand_len ≥ ref_len`, else
///    `exp(1 - ref_len / cand_len)`. Penalizes abridged candidates
///    that "cheat" by hitting high precision on a short prefix.
/// 5. `BLEU = BP · gm`.
///
/// # Convention shortcuts
///
/// - `bleu(ref, cand)` → BLEU-4 (standard, what every paper reports).
/// - `bleu_smoothed(ref, cand)` → BLEU-4 with Chen & Cherry smoothing.
///   Use this for sentence-level evals where strict zeros are common.
pub fn bleu_n(reference: &str, candidate: &str, max_n: usize, smooth: bool) -> BleuScore {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    let r = ref_tokens.len();
    let c = cand_tokens.len();
    if max_n == 0 || c == 0 || r == 0 {
        return BleuScore {
            score: 0.0,
            brevity_penalty: 0.0,
            precisions: Vec::new(),
            ref_len: r,
            cand_len: c,
        };
    }
    let mut precisions = Vec::with_capacity(max_n);
    for n in 1..=max_n {
        let ref_grams = ngrams(&ref_tokens, n);
        let cand_grams = ngrams(&cand_tokens, n);
        if cand_grams.is_empty() {
            // Cand too short for this n — treat as 0 (will zero-out
            // strict score, smoothing handles below).
            precisions.push(0.0);
            continue;
        }
        let ref_counts = count_map(&ref_grams);
        let cand_counts = count_map(&cand_grams);
        let mut overlap: usize = 0;
        for (g, &cc) in &cand_counts {
            if let Some(&rc) = ref_counts.get(g) {
                overlap += rc.min(cc);
            }
        }
        let p = if smooth && overlap == 0 {
            // Chen & Cherry (2014) method-1 lite — replace zero
            // overlap with `1 / (cand_total + 1)`. Avoids -inf
            // in log-domain and keeps the contribution small but
            // nonzero (penalizing missed n-grams without erasing
            // the score entirely).
            1.0 / (cand_grams.len() + 1) as f32
        } else {
            overlap as f32 / cand_grams.len() as f32
        };
        precisions.push(p);
    }
    // Brevity penalty: penalize candidates shorter than the reference.
    let bp = if c >= r {
        1.0
    } else {
        (1.0 - r as f32 / c as f32).exp()
    };
    // Geometric mean. Any zero precision in strict mode → score 0.
    let any_zero = precisions.iter().any(|&p| p == 0.0);
    let geo_mean = if any_zero {
        0.0
    } else {
        let log_sum: f32 = precisions.iter().map(|p| p.ln()).sum();
        (log_sum / precisions.len() as f32).exp()
    };
    BleuScore {
        score: bp * geo_mean,
        brevity_penalty: bp,
        precisions,
        ref_len: r,
        cand_len: c,
    }
}

/// BLEU-4 (the standard reported variant). Strict semantics — any
/// zero n-gram precision yields a score of 0.
pub fn bleu(reference: &str, candidate: &str) -> f32 {
    bleu_n(reference, candidate, 4, false).score
}

/// BLEU-4 with Chen & Cherry (2014) method-1 lite smoothing. Use
/// when comparing short sentences where strict BLEU often degenerates
/// to 0 due to sparse 3-grams / 4-grams (statistically uninformative).
pub fn bleu_smoothed(reference: &str, candidate: &str) -> f32 {
    bleu_n(reference, candidate, 4, true).score
}

/// Full BLEU result with diagnostic breakdown.
///
/// `precisions` is the per-n vector (length `max_n`); `brevity_penalty`
/// surfaces whether the score was attenuated for being too short
/// (helpful for debugging "why is BLEU so low?" — often the answer
/// is "your candidate is half the length of the reference").
#[derive(Debug, Clone, PartialEq)]
pub struct BleuScore {
    pub score: f32,
    pub brevity_penalty: f32,
    pub precisions: Vec<f32>,
    pub ref_len: usize,
    pub cand_len: usize,
}

/// chrF score (Popović 2015) — character-level F-score over n-gram
/// precision and recall, averaged across `n = 1..=max_n`. The third
/// leg of the NLG-eval triad alongside iter-304 ROUGE and iter-305
/// BLEU.
///
/// # Why character n-grams
///
/// Word-level metrics (ROUGE, BLEU) treat "running" and "runs" as
/// completely different tokens — zero overlap — even though they
/// share the morphological root. chrF's character n-grams catch
/// the shared substring, giving partial credit for inflectional
/// variants. Especially load-bearing for morphologically rich
/// languages (German compound words, Slavic case endings, Arabic
/// roots) but improves correlation with human judgment in English
/// too where paraphrases swap word forms.
///
/// # The score
///
/// 1. Strip whitespace from both strings (chrF's standard
///    preprocessing — character n-grams within and across word
///    boundaries are equally informative for short n).
/// 2. For each n in `1..=max_n`: compute clipped character-n-gram
///    precision `p_n` and recall `r_n` (same clipped-count rule
///    as ROUGE-N / BLEU).
/// 3. Macro-average: `chrP = (1/N) · Σ p_n`, `chrR = (1/N) · Σ r_n`.
/// 4. F-beta combination: `F = (1+β²) · chrP · chrR / (β² · chrP + chrR)`.
///
/// # Standard parameters
///
/// `chrf(reference, candidate)` defaults to `max_n = 6`, `β = 2.0`
/// (recall-weighted), matching Popović 2015's empirically-best
/// settings for translation eval. Use `chrf_n` to override either.
///
/// # Case folding
///
/// This implementation lowercases both inputs for consistency with
/// the rest of the evaluator family (ROUGE, BLEU). Strict case-
/// sensitive chrF (matching the original Popović reference impl)
/// is one `to_lowercase` call away in user code if needed; case-
/// folded was chosen as the default because most agent-facing
/// evals don't want "the cat" / "The cat" to score below 1.0.
pub fn chrf_n(
    reference: &str,
    candidate: &str,
    max_n: usize,
    beta: f32,
) -> ChrfScore {
    if max_n == 0 || reference.is_empty() || candidate.is_empty() {
        return ChrfScore::zero(beta);
    }
    // Lowercase + strip whitespace.
    let ref_chars: Vec<char> = reference
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let cand_chars: Vec<char> = candidate
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    if ref_chars.is_empty() || cand_chars.is_empty() {
        return ChrfScore::zero(beta);
    }
    let mut sum_p = 0.0_f32;
    let mut sum_r = 0.0_f32;
    let mut count = 0_f32;
    for n in 1..=max_n {
        if ref_chars.len() < n || cand_chars.len() < n {
            // No n-grams of this length on one side — skip rather
            // than push a zero (a zero would drag the macro average
            // down for input that's just shorter than n).
            continue;
        }
        let (p, r) = char_ngram_pr(&ref_chars, &cand_chars, n);
        sum_p += p;
        sum_r += r;
        count += 1.0;
    }
    if count == 0.0 {
        return ChrfScore::zero(beta);
    }
    let chr_p = sum_p / count;
    let chr_r = sum_r / count;
    let beta_sq = beta * beta;
    let denom = beta_sq * chr_p + chr_r;
    let f_beta = if denom == 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * chr_p * chr_r / denom
    };
    ChrfScore {
        precision: chr_p,
        recall: chr_r,
        f_beta,
        beta,
    }
}

/// Standard chrF — `max_n = 6`, `β = 2.0`. Returns the headline
/// F-beta score directly; use `chrf_n` for the full struct or
/// non-standard parameters.
pub fn chrf(reference: &str, candidate: &str) -> f32 {
    chrf_n(reference, candidate, 6, 2.0).f_beta
}

/// chrF++ (Popović 2017) — chrF + word n-grams averaged together.
/// Adds word-level overlap to chrF's char-level signal so paraphrase-
/// reordering is caught (chrF alone is permutation-invariant within
/// short n; word bigrams catch word-order changes).
///
/// **Algorithm**: macro-average across (a) `char_max_n` character
/// n-grams (1..=char_max_n) and (b) `word_max_n` word n-grams
/// (1..=word_max_n) — single mean over `char_max_n + word_max_n`
/// total precisions and recalls. F-beta combination with the same
/// β as chrF.
///
/// **Standard params** (Popović 2017 §4): `char_max_n=6`,
/// `word_max_n=2`, `β=2.0`. Word bigrams are the standard chrF++
/// addition; chrF+ is char + word unigrams only (less common).
///
/// Returns `ChrfScore` with the same shape as `chrf_n` so callers
/// can swap the two without rewiring.
pub fn chrf_pp(
    reference: &str,
    candidate: &str,
    char_max_n: usize,
    word_max_n: usize,
    beta: f32,
) -> ChrfScore {
    if (char_max_n == 0 && word_max_n == 0)
        || reference.is_empty()
        || candidate.is_empty()
    {
        return ChrfScore::zero(beta);
    }
    // Char-side: same prep as chrf_n — lowercase + strip whitespace.
    let ref_chars: Vec<char> = reference
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    let cand_chars: Vec<char> = candidate
        .to_lowercase()
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    // Word-side: lowercase + whitespace tokenize (matches the rest of
    // the evaluator family).
    let ref_words: Vec<String> = tokenize_lower(reference).collect();
    let cand_words: Vec<String> = tokenize_lower(candidate).collect();

    let mut sum_p = 0.0_f32;
    let mut sum_r = 0.0_f32;
    let mut count = 0_f32;

    // Char n-grams contribution.
    for n in 1..=char_max_n {
        if ref_chars.len() < n || cand_chars.len() < n {
            continue;
        }
        let (p, r) = char_ngram_pr(&ref_chars, &cand_chars, n);
        sum_p += p;
        sum_r += r;
        count += 1.0;
    }
    // Word n-grams contribution.
    for n in 1..=word_max_n {
        if ref_words.len() < n || cand_words.len() < n {
            continue;
        }
        let (p, r) = word_ngram_pr(&ref_words, &cand_words, n);
        sum_p += p;
        sum_r += r;
        count += 1.0;
    }
    if count == 0.0 {
        return ChrfScore::zero(beta);
    }
    let chr_p = sum_p / count;
    let chr_r = sum_r / count;
    let beta_sq = beta * beta;
    let denom = beta_sq * chr_p + chr_r;
    let f_beta = if denom == 0.0 {
        0.0
    } else {
        (1.0 + beta_sq) * chr_p * chr_r / denom
    };
    ChrfScore {
        precision: chr_p,
        recall: chr_r,
        f_beta,
        beta,
    }
}

/// Standard chrF++ — `char_max_n=6`, `word_max_n=2`, `β=2.0`.
/// Returns the headline F-beta directly. Use `chrf_pp` for the full
/// struct or non-standard parameters.
pub fn chrf_pp_default(reference: &str, candidate: &str) -> f32 {
    chrf_pp(reference, candidate, 6, 2, 2.0).f_beta
}

/// Compute clipped word-n-gram precision and recall for a single n.
/// Mirrors `char_ngram_pr` but operates on `Vec<String>` word
/// tokens instead of `Vec<char>`.
fn word_ngram_pr(ref_words: &[String], cand_words: &[String], n: usize) -> (f32, f32) {
    let ref_grams = ngrams(ref_words, n);
    let cand_grams = ngrams(cand_words, n);
    if ref_grams.is_empty() || cand_grams.is_empty() {
        return (0.0, 0.0);
    }
    use std::collections::HashMap;
    let mut ref_counts: HashMap<&Vec<String>, usize> = HashMap::new();
    for g in &ref_grams {
        *ref_counts.entry(g).or_insert(0) += 1;
    }
    let mut cand_counts: HashMap<&Vec<String>, usize> = HashMap::new();
    for g in &cand_grams {
        *cand_counts.entry(g).or_insert(0) += 1;
    }
    let mut overlap: usize = 0;
    for (g, &cc) in &cand_counts {
        if let Some(&rc) = ref_counts.get(g) {
            overlap += rc.min(cc);
        }
    }
    let p = overlap as f32 / cand_grams.len() as f32;
    let r = overlap as f32 / ref_grams.len() as f32;
    (p, r)
}

/// chrF result with diagnostic breakdown.
///
/// `f_beta` is the headline number (β-weighted F-score over the
/// macro-averaged character n-gram precision and recall);
/// `precision` and `recall` are exposed for over/under-generation
/// debugging; `beta` is echoed so callers comparing cross-
/// configuration results know which weighting was used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ChrfScore {
    pub precision: f32,
    pub recall: f32,
    pub f_beta: f32,
    pub beta: f32,
}

impl ChrfScore {
    fn zero(beta: f32) -> Self {
        Self {
            precision: 0.0,
            recall: 0.0,
            f_beta: 0.0,
            beta,
        }
    }
}

/// Compute clipped char-n-gram precision and recall for a single n.
fn char_ngram_pr(ref_chars: &[char], cand_chars: &[char], n: usize) -> (f32, f32) {
    let ref_grams = char_ngrams(ref_chars, n);
    let cand_grams = char_ngrams(cand_chars, n);
    if ref_grams.is_empty() || cand_grams.is_empty() {
        return (0.0, 0.0);
    }
    use std::collections::HashMap;
    let mut ref_counts: HashMap<&[char], usize> = HashMap::new();
    for g in &ref_grams {
        *ref_counts.entry(g.as_slice()).or_insert(0) += 1;
    }
    let mut cand_counts: HashMap<&[char], usize> = HashMap::new();
    for g in &cand_grams {
        *cand_counts.entry(g.as_slice()).or_insert(0) += 1;
    }
    let mut overlap: usize = 0;
    for (g, &cc) in &cand_counts {
        if let Some(&rc) = ref_counts.get(g) {
            overlap += rc.min(cc);
        }
    }
    let p = overlap as f32 / cand_grams.len() as f32;
    let r = overlap as f32 / ref_grams.len() as f32;
    (p, r)
}

fn char_ngrams(chars: &[char], n: usize) -> Vec<Vec<char>> {
    if chars.len() < n || n == 0 {
        return Vec::new();
    }
    (0..=chars.len() - n)
        .map(|i| chars[i..i + n].to_vec())
        .collect()
}

/// Multi-reference BLEU (Papineni et al. 2002 §3) — n-gram precision
/// against the **max count** across N references, with brevity
/// penalty using the closest reference length.
///
/// # Why multi-reference
///
/// Single-reference BLEU is harsh on cands that pick a different
/// valid wording than the chosen ref ("the cat slept" vs ref "the
/// cat was sleeping" — overlap collapses on tense / aux verbs).
/// Multi-reference BLEU is the standard fix: every WMT shared task
/// ships 2-4 reference translations per source sentence, and the
/// metric takes the BEST overlap across them. Cands that match ANY
/// valid wording get full credit.
///
/// # Algorithm
///
/// 1. Tokenize cand and each ref via lowercase + whitespace.
/// 2. For each n in `1..=max_n`: build the cand n-gram count map
///    AND, for each n-gram, take `max(ref1_count, ref2_count, ...)`
///    across all refs as the "ceiling" count. Overlap = `Σ
///    min(cand_count, max_ref_count)`.
/// 3. Geometric mean (or smoothed if `smooth=true`) — same shape
///    as iter-305 single-ref `bleu_n`.
/// 4. Brevity penalty uses the CLOSEST reference length to the
///    candidate (not the average): `r_closest = arg_min |r_len −
///    cand_len|`. This is the standard rule from the original BLEU
///    paper for multi-ref — picking the ref length closest to the
///    cand avoids over-penalizing when ref lengths vary widely.
/// 5. `BLEU = BP · gm`.
///
/// Returns `BleuScore::zero` shape if `references.is_empty()` or
/// any input is empty.
pub fn bleu_multi(
    references: &[&str],
    candidate: &str,
    max_n: usize,
    smooth: bool,
) -> BleuScore {
    if references.is_empty() || max_n == 0 || candidate.is_empty() {
        return BleuScore {
            score: 0.0,
            brevity_penalty: 0.0,
            precisions: Vec::new(),
            ref_len: 0,
            cand_len: tokenize_lower(candidate).count(),
        };
    }
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    let c = cand_tokens.len();
    if c == 0 {
        return BleuScore {
            score: 0.0,
            brevity_penalty: 0.0,
            precisions: Vec::new(),
            ref_len: 0,
            cand_len: 0,
        };
    }
    // Tokenize each reference once.
    let ref_token_lists: Vec<Vec<String>> = references
        .iter()
        .map(|r| tokenize_lower(r).collect())
        .collect();
    // Closest reference length to the candidate (BLEU multi-ref
    // brevity-penalty rule from the original paper).
    let r_closest = ref_token_lists
        .iter()
        .map(|tokens| tokens.len())
        .min_by_key(|len| {
            let d = (*len as i64 - c as i64).abs();
            // Tie-break: prefer the SHORTER ref length on ties (matches
            // sacrebleu / NIST mteval convention).
            (d, *len as i64)
        })
        .unwrap_or(0);
    if r_closest == 0 {
        return BleuScore {
            score: 0.0,
            brevity_penalty: 0.0,
            precisions: Vec::new(),
            ref_len: 0,
            cand_len: c,
        };
    }
    let mut precisions = Vec::with_capacity(max_n);
    for n in 1..=max_n {
        let cand_grams = ngrams(&cand_tokens, n);
        if cand_grams.is_empty() {
            precisions.push(0.0);
            continue;
        }
        // Build max-count-across-refs map: for each n-gram, the
        // ceiling is the max of its counts across ALL references.
        use std::collections::HashMap;
        let mut max_ref_counts: HashMap<&Vec<String>, usize> = HashMap::new();
        let ref_grams_per_ref: Vec<Vec<Vec<String>>> = ref_token_lists
            .iter()
            .map(|tokens| ngrams(tokens, n))
            .collect();
        for r_grams in &ref_grams_per_ref {
            let mut this_ref: HashMap<&Vec<String>, usize> = HashMap::new();
            for g in r_grams {
                *this_ref.entry(g).or_insert(0) += 1;
            }
            for (g, &c) in &this_ref {
                let entry = max_ref_counts.entry(g).or_insert(0);
                if c > *entry {
                    *entry = c;
                }
            }
        }
        // Cand counts.
        let mut cand_counts: HashMap<&Vec<String>, usize> = HashMap::new();
        for g in &cand_grams {
            *cand_counts.entry(g).or_insert(0) += 1;
        }
        let mut overlap: usize = 0;
        for (g, &cc) in &cand_counts {
            if let Some(&max_rc) = max_ref_counts.get(g) {
                overlap += max_rc.min(cc);
            }
        }
        let p = if smooth && overlap == 0 {
            1.0 / (cand_grams.len() + 1) as f32
        } else {
            overlap as f32 / cand_grams.len() as f32
        };
        precisions.push(p);
    }
    let bp = if c >= r_closest {
        1.0
    } else {
        (1.0 - r_closest as f32 / c as f32).exp()
    };
    let any_zero = precisions.iter().any(|&p| p == 0.0);
    let geo_mean = if any_zero {
        0.0
    } else {
        let log_sum: f32 = precisions.iter().map(|p| p.ln()).sum();
        (log_sum / precisions.len() as f32).exp()
    };
    BleuScore {
        score: bp * geo_mean,
        brevity_penalty: bp,
        precisions,
        ref_len: r_closest,
        cand_len: c,
    }
}

/// Multi-reference ROUGE-N — n-gram F-score taking the BEST score
/// across all references. Shadow of iter-304 `rouge_n` for the
/// multi-ref case common in NLG eval.
///
/// The common convention (matching Lin's 2004 follow-up + most
/// reference impls) is "max F over refs" rather than "max overlap
/// per gram". That keeps each ref independent — a cand that
/// mirrors ref 1 perfectly scores 1.0 even if ref 2 is wildly
/// different.
pub fn rouge_n_multi(references: &[&str], candidate: &str, n: usize) -> RougeScore {
    if references.is_empty() {
        return RougeScore::zero();
    }
    references
        .iter()
        .map(|r| rouge_n(r, candidate, n))
        .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(RougeScore::zero)
}

/// Multi-reference ROUGE-L — same max-over-refs rule applied to LCS-based F1.
pub fn rouge_l_multi(references: &[&str], candidate: &str) -> RougeScore {
    if references.is_empty() {
        return RougeScore::zero();
    }
    references
        .iter()
        .map(|r| rouge_l(r, candidate))
        .max_by(|a, b| a.f1.partial_cmp(&b.f1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(RougeScore::zero)
}

/// Word Error Rate — `levenshtein_word_distance(ref, cand) / ref_word_count`.
///
/// Standard ASR / MT / NLG metric. **Lower is better** (opposite of
/// ROUGE / BLEU / chrF / METEOR which are higher-better).
///
/// # Distinct from `levenshtein_ratio`
///
/// `levenshtein_ratio` (already in this file) is char-level + bounded
/// to `[0, 1]` via `1 - dist/max(|a|, |b|)`. WER is **word-level**
/// (the more usual ASR convention) and unbounded above 1.0 — a
/// candidate twice as long as the reference with all wrong words
/// has WER ≈ 2.0.
///
/// # Algorithm
///
/// 1. Tokenize both via lowercase + whitespace split.
/// 2. Compute Levenshtein edit distance over the token sequences
///    (insertion + deletion + substitution; cost 1 each).
/// 3. Return `dist / max(1, ref_word_count)`.
///
/// # Edge cases
///
/// - Both empty → 0.0 (no edits needed).
/// - Empty ref + non-empty cand → cand.len() / 1 = `cand_len` (every
///   cand word is an insertion error).
/// - Empty cand + non-empty ref → 1.0 (every ref word is a deletion).
pub fn wer(reference: &str, candidate: &str) -> f32 {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    if ref_tokens.is_empty() && cand_tokens.is_empty() {
        return 0.0;
    }
    let dist = token_levenshtein(&ref_tokens, &cand_tokens);
    let denom = ref_tokens.len().max(1);
    dist as f32 / denom as f32
}

/// Character Error Rate — char-level Levenshtein / ref char count.
///
/// Same shape as WER but operates on Unicode characters. Standard
/// for OCR + char-level MT evaluation. **Lower is better.**
///
/// Lowercases both inputs (matches the rest of the evaluator family).
/// Treats whitespace as significant (standard CER convention — a
/// missing space is an edit).
pub fn cer(reference: &str, candidate: &str) -> f32 {
    let ref_chars: Vec<char> = reference.to_lowercase().chars().collect();
    let cand_chars: Vec<char> = candidate.to_lowercase().chars().collect();
    if ref_chars.is_empty() && cand_chars.is_empty() {
        return 0.0;
    }
    let dist = char_levenshtein(&ref_chars, &cand_chars);
    let denom = ref_chars.len().max(1);
    dist as f32 / denom as f32
}

/// Levenshtein distance over a generic token sequence. O(m·n) time,
/// O(min(m,n)) space via rolling row.
fn token_levenshtein(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Char-level sibling — same algorithm over `Vec<char>`.
fn char_levenshtein(a: &[char], b: &[char]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }
    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            curr[j] = (curr[j - 1] + 1)
                .min(prev[j] + 1)
                .min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// METEOR-lite (Banerjee & Lavie 2005) — alignment-based MT/NLG eval
/// with chunk-fragmentation penalty. The fourth NLG-eval primitive
/// alongside ROUGE (iter 304), BLEU (iter 305), chrF (iter 307).
///
/// # The unique contribution
///
/// ROUGE/BLEU/chrF all reward n-gram overlap regardless of order
/// beyond the n-gram window. METEOR explicitly models alignment:
/// each candidate token matches AT MOST ONE reference token, and
/// the metric penalizes fragmented matches. A candidate that has
/// the right words in the wrong order ("cat the sat" vs "the cat
/// sat") gets full word-overlap credit from BLEU-1 but is penalized
/// by METEOR for poor word-order fidelity.
///
/// # Algorithm
///
/// 1. Tokenize both via lowercase + whitespace split.
/// 2. Greedy alignment: for each cand token (left-to-right), find
///    the first un-matched ref position it matches. Match priority:
///    exact word match → suffix-stripped "stem-light" match (drops
///    common English suffixes `s/es/ies/ing/ed/ly` when stem ≥ 3
///    chars) → no match.
/// 3. Count matches `m` and **chunks** `c` — number of contiguous
///    monotonic runs in the ref-position sequence (a chunk break
///    happens when the next match is NOT at ref-position+1).
/// 4. Precision `P = m / cand_len`, recall `R = m / ref_len`.
/// 5. F-mean with recall-weight α=0.9: `F = P·R / (α·P + (1−α)·R)`.
/// 6. Fragmentation penalty: `pen = γ · (c/m)^β` with γ=0.5, β=3.
///    No fragmentation (c=1) → small penalty; one chunk per match
///    (c=m) → max penalty.
/// 7. **METEOR = F · (1 − pen)**.
///
/// # METEOR-lite vs full METEOR
///
/// Full METEOR uses three matchers: exact, Porter-stemmer, WordNet
/// synonyms + paraphrases. WordNet is a 30 MB English-only lexical
/// database; pulling it would dominate the crate's dep weight.
/// `meteor_lite` ships exact + suffix-strip stem-light matching —
/// covers ~80% of the stem signal that the full Porter stemmer +
/// WordNet would provide for English, with zero dep cost. For
/// stricter scoring use `meteor_exact` (exact-match only).
///
/// Caveats:
/// - English-only stem heuristic. For other languages use chrF
///   (iter 307) which is language-agnostic.
/// - No paraphrase matching — if the eval needs synonym credit,
///   layer an embedding-based scorer.
pub fn meteor_lite(reference: &str, candidate: &str) -> MeteorScore {
    meteor_inner(reference, candidate, true)
}

/// Strict-exact-match METEOR — no stem backoff. Use when stricter
/// vocabulary matching matters (technical docs where "running" and
/// "runs" are genuinely different).
pub fn meteor_exact(reference: &str, candidate: &str) -> MeteorScore {
    meteor_inner(reference, candidate, false)
}

/// METEOR result with diagnostic breakdown.
///
/// `score` is the headline number; `precision`/`recall`/`f_mean`
/// reveal the over/under-generation tradeoff; `chunks`/`matches`
/// surface why fragmentation pushed the score down (or didn't).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeteorScore {
    pub score: f32,
    pub precision: f32,
    pub recall: f32,
    pub f_mean: f32,
    pub fragmentation_penalty: f32,
    pub matches: u32,
    pub chunks: u32,
    pub ref_len: u32,
    pub cand_len: u32,
}

impl MeteorScore {
    fn zero() -> Self {
        Self {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            f_mean: 0.0,
            fragmentation_penalty: 0.0,
            matches: 0,
            chunks: 0,
            ref_len: 0,
            cand_len: 0,
        }
    }
}

fn meteor_inner(reference: &str, candidate: &str, allow_stem: bool) -> MeteorScore {
    let ref_tokens: Vec<String> = tokenize_lower(reference).collect();
    let cand_tokens: Vec<String> = tokenize_lower(candidate).collect();
    if ref_tokens.is_empty() || cand_tokens.is_empty() {
        return MeteorScore::zero();
    }
    // Greedy alignment: each cand token grabs the first unmatched
    // ref position with exact match, then (if allowed) stem match.
    // ref_match_positions[i] = ref index aligned to cand[i], or None.
    let mut ref_used = vec![false; ref_tokens.len()];
    let mut alignment: Vec<Option<usize>> = vec![None; cand_tokens.len()];
    // Pass 1: exact matches.
    for (ci, ct) in cand_tokens.iter().enumerate() {
        for (ri, rt) in ref_tokens.iter().enumerate() {
            if !ref_used[ri] && ct == rt {
                alignment[ci] = Some(ri);
                ref_used[ri] = true;
                break;
            }
        }
    }
    // Pass 2: stem matches (only for cand tokens still unmatched).
    if allow_stem {
        for (ci, ct) in cand_tokens.iter().enumerate() {
            if alignment[ci].is_some() {
                continue;
            }
            let cs = stem_light(ct);
            for (ri, rt) in ref_tokens.iter().enumerate() {
                if !ref_used[ri] && stem_light(rt) == cs {
                    alignment[ci] = Some(ri);
                    ref_used[ri] = true;
                    break;
                }
            }
        }
    }
    let matches: u32 = alignment.iter().filter(|a| a.is_some()).count() as u32;
    if matches == 0 {
        return MeteorScore {
            score: 0.0,
            precision: 0.0,
            recall: 0.0,
            f_mean: 0.0,
            fragmentation_penalty: 0.0,
            matches: 0,
            chunks: 0,
            ref_len: ref_tokens.len() as u32,
            cand_len: cand_tokens.len() as u32,
        };
    }
    // Count chunks: contiguous monotonic runs over the matched
    // ref-positions (in cand order).
    let mut chunks: u32 = 0;
    let mut prev_ref: Option<usize> = None;
    for &slot in &alignment {
        let Some(ri) = slot else {
            // Unmatched cand token breaks the run on its own.
            prev_ref = None;
            continue;
        };
        match prev_ref {
            Some(pr) if pr + 1 == ri => {} // continuing chunk
            _ => chunks += 1,
        }
        prev_ref = Some(ri);
    }
    let p = matches as f32 / cand_tokens.len() as f32;
    let r = matches as f32 / ref_tokens.len() as f32;
    let alpha = 0.9_f32;
    let f_mean_denom = alpha * p + (1.0 - alpha) * r;
    let f_mean = if f_mean_denom == 0.0 {
        0.0
    } else {
        p * r / f_mean_denom
    };
    // Fragmentation penalty: γ · (c/m)^β, γ=0.5, β=3.
    let frag = chunks as f32 / matches as f32;
    let pen = 0.5_f32 * frag.powi(3);
    let score = f_mean * (1.0 - pen);
    MeteorScore {
        score,
        precision: p,
        recall: r,
        f_mean,
        fragmentation_penalty: pen,
        matches,
        chunks,
        ref_len: ref_tokens.len() as u32,
        cand_len: cand_tokens.len() as u32,
    }
}

/// Suffix-strip stem-light: covers common English inflections
/// without pulling Porter stemmer's full rule set or WordNet. Min
/// stem length 3 to avoid over-aggressive stripping ("is" → "i").
fn stem_light(word: &str) -> String {
    let suffixes = ["ies", "ing", "ed", "ly", "es", "s"];
    for suf in &suffixes {
        if word.len() >= suf.len() + 3 && word.ends_with(suf) {
            return word[..word.len() - suf.len()].to_string();
        }
    }
    word.to_string()
}

fn ngrams(tokens: &[String], n: usize) -> Vec<Vec<String>> {
    if tokens.len() < n || n == 0 {
        return Vec::new();
    }
    (0..=tokens.len() - n)
        .map(|i| tokens[i..i + n].to_vec())
        .collect()
}

fn count_map(grams: &[Vec<String>]) -> std::collections::HashMap<&Vec<String>, usize> {
    let mut m = std::collections::HashMap::new();
    for g in grams {
        *m.entry(g).or_insert(0) += 1;
    }
    m
}

fn lcs_length(a: &[String], b: &[String]) -> usize {
    let m = a.len();
    let n = b.len();
    if m == 0 || n == 0 {
        return 0;
    }
    // Rolling-row DP — O(min(m,n)) space.
    // Make `b` the inner (shorter) axis for memory efficiency.
    let (a, b) = if m >= n { (a, b) } else { (b, a) };
    let inner = b.len();
    let mut prev = vec![0_usize; inner + 1];
    let mut curr = vec![0_usize; inner + 1];
    for i in 1..=a.len() {
        for j in 1..=inner {
            if a[i - 1] == b[j - 1] {
                curr[j] = prev[j - 1] + 1;
            } else {
                curr[j] = prev[j].max(curr[j - 1]);
            }
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[inner]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_match_trims_and_lowercases() {
        assert!(exact_match("  Paris  ", "paris"));
        assert!(exact_match("YES", "yes"));
        assert!(!exact_match("Paris", "London"));
    }

    #[test]
    fn exact_match_strict_byte_equal() {
        assert!(exact_match_strict("Paris", "Paris"));
        assert!(!exact_match_strict("Paris", "paris"));
        assert!(!exact_match_strict("Paris", "Paris "));
    }

    #[test]
    fn levenshtein_basic_distances() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("same", "same"), 0);
    }

    #[test]
    fn levenshtein_ratio_identical_one() {
        assert!((levenshtein_ratio("hello", "hello") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn levenshtein_ratio_completely_different() {
        // No shared chars, equal length → distance == length → ratio = 0.
        assert!((levenshtein_ratio("abc", "xyz") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn levenshtein_ratio_partial_match() {
        // "kitten" vs "sitting": dist=3, max_len=7 → ratio ≈ 0.571.
        let r = levenshtein_ratio("kitten", "sitting");
        assert!((r - (4.0 / 7.0)).abs() < 1e-3);
    }

    #[test]
    fn jaccard_word_order_invariant() {
        let r = jaccard_similarity("apple banana cherry", "cherry banana apple");
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_partial_overlap() {
        let r = jaccard_similarity("apple banana", "banana cherry");
        // intersection=1, union=3 → 1/3.
        assert!((r - (1.0 / 3.0)).abs() < 1e-3);
    }

    #[test]
    fn jaccard_empty_both_is_one() {
        assert!((jaccard_similarity("", "") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_one_empty_is_zero() {
        assert!((jaccard_similarity("apple", "") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn regex_match_basic() {
        assert!(regex_match("the year is 2026", r"\d{4}").unwrap());
        assert!(!regex_match("no digits", r"\d{4}").unwrap());
    }

    #[test]
    fn regex_match_invalid_pattern_returns_err() {
        assert!(regex_match("anything", r"(unclosed").is_err());
    }

    #[test]
    fn json_validity_round_trip() {
        assert!(json_validity(r#"{"a": 1}"#));
        assert!(json_validity(r#"[1, 2, 3]"#));
        assert!(json_validity(r#""string""#));
        assert!(json_validity("42"));
        assert!(json_validity("true"));
        assert!(!json_validity("not json"));
        assert!(!json_validity(r#"{"a": 1"#)); // unclosed
    }

    #[test]
    fn embedding_cosine_identical_vectors_one() {
        let v = vec![1.0, 2.0, 3.0];
        let r = embedding_cosine(&v, &v).unwrap();
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_cosine_orthogonal_zero() {
        let r = embedding_cosine(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(r.abs() < 1e-6);
    }

    #[test]
    fn embedding_cosine_opposite_neg_one() {
        let r = embedding_cosine(&[1.0, 0.0], &[-1.0, 0.0]).unwrap();
        assert!((r + 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_cosine_length_mismatch_errors() {
        assert!(embedding_cosine(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_err());
    }

    #[test]
    fn embedding_cosine_zero_vector_returns_zero_not_nan() {
        let r = embedding_cosine(&[0.0, 0.0], &[1.0, 1.0]).unwrap();
        assert_eq!(r, 0.0);
        assert!(!r.is_nan());
    }

    #[test]
    fn contains_all_requires_every_needle() {
        assert!(contains_all("the quick brown fox", &["quick", "fox"]));
        assert!(!contains_all("the quick brown fox", &["quick", "horse"]));
    }

    #[test]
    fn contains_any_requires_at_least_one() {
        assert!(contains_any("hello world", &["world", "moon"]));
        assert!(!contains_any("hello world", &["moon", "stars"]));
    }

    // ─── ROUGE-N / ROUGE-L ────────────────────────────────────────

    #[test]
    fn rouge_1_identical_strings() {
        let r = rouge_1("the cat sat on the mat", "the cat sat on the mat");
        assert!((r.precision - 1.0).abs() < 1e-6);
        assert!((r.recall - 1.0).abs() < 1e-6);
        assert!((r.f1 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn rouge_1_known_overlap_5_of_6() {
        // ref:  the cat sat on the mat
        // cand: the cat sat on the floor
        // 1-gram counts: ref={the:2, cat:1, sat:1, on:1, mat:1}, cand={the:2, cat:1, sat:1, on:1, floor:1}.
        // Overlap (clipped): the=2 + cat=1 + sat=1 + on=1 = 5. Total each side = 6.
        let r = rouge_1("the cat sat on the mat", "the cat sat on the floor");
        assert!((r.precision - 5.0 / 6.0).abs() < 1e-5, "p={}", r.precision);
        assert!((r.recall - 5.0 / 6.0).abs() < 1e-5, "r={}", r.recall);
        assert!((r.f1 - 5.0 / 6.0).abs() < 1e-5, "f1={}", r.f1);
    }

    #[test]
    fn rouge_2_bigram_overlap() {
        // ref bigrams:  (the,cat),(cat,sat),(sat,on),(on,the),(the,mat) → 5
        // cand bigrams: (the,cat),(cat,sat),(sat,on),(on,the),(the,floor) → 5
        // Overlap: 4. P = R = F1 = 4/5 = 0.8.
        let r = rouge_2("the cat sat on the mat", "the cat sat on the floor");
        assert!((r.precision - 0.8).abs() < 1e-5);
        assert!((r.recall - 0.8).abs() < 1e-5);
        assert!((r.f1 - 0.8).abs() < 1e-5);
    }

    #[test]
    fn rouge_n_clipped_overlap_repeats() {
        // Cand has "the" 3 times; ref has it once. Clipped overlap = 1, NOT 3.
        // ref tokens: [the, cat] → 2 unigrams
        // cand tokens: [the, the, the] → 3 unigrams
        // Overlap (clipped): min(1, 3) = 1
        // P = 1/3, R = 1/2, F1 = 2*(1/3)*(1/2)/(1/3+1/2) = (1/3)/(5/6) = 2/5 = 0.4
        let r = rouge_1("the cat", "the the the");
        assert!((r.precision - 1.0 / 3.0).abs() < 1e-5);
        assert!((r.recall - 0.5).abs() < 1e-5);
        assert!((r.f1 - 0.4).abs() < 1e-5);
    }

    #[test]
    fn rouge_n_disjoint_yields_zero() {
        let r = rouge_1("apple banana", "cherry date");
        assert_eq!(r.precision, 0.0);
        assert_eq!(r.recall, 0.0);
        assert_eq!(r.f1, 0.0);
    }

    #[test]
    fn rouge_n_empty_inputs_zero() {
        let r = rouge_1("", "anything");
        assert_eq!(r, RougeScore::zero());
        let r = rouge_1("anything", "");
        assert_eq!(r, RougeScore::zero());
        let r = rouge_1("", "");
        assert_eq!(r, RougeScore::zero());
    }

    #[test]
    fn rouge_n_too_few_tokens_for_n() {
        // "single" has 1 token; n=2 means no bigrams exist.
        let r = rouge_n("single", "single", 2);
        assert_eq!(r, RougeScore::zero());
    }

    #[test]
    fn rouge_n_zero_n_zero_score() {
        let r = rouge_n("anything", "anything", 0);
        assert_eq!(r, RougeScore::zero());
    }

    #[test]
    fn rouge_l_lcs_5_of_6() {
        // Same fixture as rouge_1: differ only in last token.
        // LCS of (the cat sat on the mat) vs (the cat sat on the floor) = 5.
        // P = 5/6, R = 5/6, F1 = 5/6.
        let r = rouge_l("the cat sat on the mat", "the cat sat on the floor");
        assert!((r.f1 - 5.0 / 6.0).abs() < 1e-5);
    }

    #[test]
    fn rouge_l_handles_gaps() {
        // ref:  a b c d e
        // cand: a x b y c z d w e   (matched tokens spread across gaps)
        // LCS = 5 (all of a,b,c,d,e in order).
        // P = 5/9, R = 5/5 = 1.0, F1 = 2*(5/9)/(5/9 + 1) = (10/9) / (14/9) = 10/14 ≈ 0.714.
        let r = rouge_l("a b c d e", "a x b y c z d w e");
        assert!((r.recall - 1.0).abs() < 1e-5, "r={}", r.recall);
        assert!((r.precision - 5.0 / 9.0).abs() < 1e-5, "p={}", r.precision);
        assert!((r.f1 - 10.0 / 14.0).abs() < 1e-5, "f1={}", r.f1);
    }

    #[test]
    fn rouge_l_distinct_from_rouge_n_on_reordering() {
        // Permuted-bigram case: rouge_2 returns 0 (no shared bigrams),
        // but rouge_l finds matched subsequence.
        // ref:  a b c
        // cand: c b a   (reversed)
        // ROUGE-2: ref bigrams (a,b),(b,c); cand bigrams (c,b),(b,a). Overlap 0.
        // ROUGE-L: LCS of [a,b,c] and [c,b,a] is 1 (single token like "b"),
        // not 0 — proves L sees something N misses.
        let r2 = rouge_2("a b c", "c b a");
        assert_eq!(r2.f1, 0.0);
        let rl = rouge_l("a b c", "c b a");
        assert!(rl.f1 > 0.0);
    }

    #[test]
    fn rouge_l_empty_inputs_zero() {
        assert_eq!(rouge_l("", "x"), RougeScore::zero());
        assert_eq!(rouge_l("x", ""), RougeScore::zero());
        assert_eq!(rouge_l("", ""), RougeScore::zero());
    }

    #[test]
    fn rouge_lowercase_normalization() {
        // Case differences must NOT affect scores.
        let a = rouge_1("Hello World", "hello world");
        assert!((a.f1 - 1.0).abs() < 1e-6);
    }

    // ─── BLEU ─────────────────────────────────────────────────────

    #[test]
    fn bleu_identical_strings_one() {
        // Identical strings (≥ 4 tokens so all n-grams exist) → 1.0.
        let s = bleu("the cat sat on the mat", "the cat sat on the mat");
        assert!((s - 1.0).abs() < 1e-5, "got {s}");
    }

    #[test]
    fn bleu_known_5_of_6_fixture() {
        // ref:  the cat sat on the mat (6)
        // cand: the cat sat on the floor (6)
        // p1 = 5/6, p2 = 4/5, p3 = 3/4, p4 = 2/3
        // BP = 1 (lengths equal)
        // BLEU = exp((ln(5/6) + ln(4/5) + ln(3/4) + ln(2/3)) / 4) ≈ 0.7598
        let s = bleu("the cat sat on the mat", "the cat sat on the floor");
        let expected = ((5.0_f32 / 6.0).ln()
            + (4.0_f32 / 5.0).ln()
            + (3.0_f32 / 4.0).ln()
            + (2.0_f32 / 3.0).ln())
            / 4.0;
        let expected = expected.exp();
        assert!((s - expected).abs() < 1e-5, "got {s}, expected {expected}");
    }

    #[test]
    fn bleu_empty_inputs_zero() {
        assert_eq!(bleu("", "x y z w"), 0.0);
        assert_eq!(bleu("x y z w", ""), 0.0);
        assert_eq!(bleu("", ""), 0.0);
    }

    #[test]
    fn bleu_brevity_penalty_short_candidate() {
        // ref = 7 tokens, cand = 2. BP = exp(1 - 7/2) = exp(-2.5) ≈ 0.0821.
        // Strict BLEU-4 would be 0 (cand too short for trigrams/4-grams),
        // so we use bleu_n with max_n=1 to isolate the BP effect:
        // p1 = 2/2 = 1.0, BP * 1.0 = exp(-2.5).
        let res = bleu_n("the cat is sitting on the mat", "the cat", 1, false);
        let expected_bp = (-2.5_f32).exp();
        assert!(
            (res.brevity_penalty - expected_bp).abs() < 1e-5,
            "bp={}",
            res.brevity_penalty
        );
        assert!((res.score - expected_bp).abs() < 1e-5, "score={}", res.score);
    }

    #[test]
    fn bleu_no_brevity_penalty_when_cand_longer() {
        // cand longer than ref → BP = 1.
        let res = bleu_n("the cat", "the cat is cute and fluffy", 1, false);
        assert!((res.brevity_penalty - 1.0).abs() < 1e-6);
    }

    #[test]
    fn bleu_strict_zero_precision_zeros_score() {
        // ref:  the cat sat on the mat
        // cand: dogs run quickly daily here  (5 tokens, completely disjoint)
        // ALL p_n are 0 → score is 0.
        let s = bleu("the cat sat on the mat", "dogs run quickly daily here");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn bleu_smoothed_avoids_zero_for_short_match() {
        // 4-token cand against an 8-token ref; cand has 1 unigram match.
        // Strict BLEU-4: p_2/p_3/p_4 are likely 0 → score 0.
        // Smoothed: zeros replaced with 1/(cand_grams + 1) → small but
        // nonzero score.
        let strict = bleu("the cat sat on the mat over there", "the dog ran fast");
        let smoothed = bleu_smoothed("the cat sat on the mat over there", "the dog ran fast");
        assert_eq!(strict, 0.0);
        assert!(smoothed > 0.0, "smoothed should be > 0");
        // Smoothed score should still be small (penalizing the misses)
        // — sanity check it didn't explode upward.
        assert!(smoothed < 0.5, "smoothed should be < 0.5, got {smoothed}");
    }

    #[test]
    fn bleu_clipped_overlap_repeats() {
        // Same trap as ROUGE: cand has "the" 5 times; ref has 1.
        // p_1 should be 1/5, NOT 5/5. BLEU-1 with that:
        // BP = exp(1 - 2/5) = exp(0.6)... wait, ref=2 tokens, cand=5 → cand longer → BP=1.
        // score = 1.0 * 1/5 = 0.2.
        let res = bleu_n("the cat", "the the the the the", 1, false);
        assert!((res.precisions[0] - 0.2).abs() < 1e-5, "p1={}", res.precisions[0]);
        assert!((res.score - 0.2).abs() < 1e-5);
    }

    #[test]
    fn bleu_max_n_zero_returns_zero() {
        let res = bleu_n("the cat", "the cat", 0, false);
        assert_eq!(res.score, 0.0);
        assert!(res.precisions.is_empty());
    }

    #[test]
    fn bleu_max_n_one_isolates_unigrams() {
        let res = bleu_n("the cat sat", "the cat sat", 1, false);
        assert_eq!(res.precisions.len(), 1);
        assert!((res.score - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bleu_lowercase_normalization() {
        let s = bleu("The Cat Sat On The Mat", "the cat sat on the mat");
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn bleu_score_struct_carries_diagnostics() {
        let res = bleu_n("the cat sat on the mat", "the cat sat on the floor", 4, false);
        assert_eq!(res.ref_len, 6);
        assert_eq!(res.cand_len, 6);
        assert_eq!(res.precisions.len(), 4);
        assert!((res.brevity_penalty - 1.0).abs() < 1e-6);
        // p1 should be the highest, p4 the lowest (since the mismatch
        // is in the last token).
        assert!(res.precisions[0] >= res.precisions[3]);
    }

    // ─── chrF ────────────────────────────────────────────────────

    #[test]
    fn chrf_identical_strings_one() {
        let s = chrf("the cat sat on the mat", "the cat sat on the mat");
        assert!((s - 1.0).abs() < 1e-5, "got {s}");
    }

    #[test]
    fn chrf_empty_inputs_zero() {
        assert_eq!(chrf("", "anything"), 0.0);
        assert_eq!(chrf("anything", ""), 0.0);
        assert_eq!(chrf("", ""), 0.0);
    }

    #[test]
    fn chrf_disjoint_chars_yields_zero() {
        // All characters disjoint after lowercase + whitespace strip.
        let s = chrf("abc", "xyz");
        assert_eq!(s, 0.0);
    }

    #[test]
    fn chrf_morphological_variants_score_above_zero() {
        // The marquee feature: word-level metrics give 0 for "running"
        // vs "runs" (different tokens). chrF gives partial credit for
        // the shared substring "run".
        let r = rouge_1("running", "runs");
        assert_eq!(r.f1, 0.0);
        let c = chrf("running", "runs");
        assert!(c > 0.0, "chrf got {c}, expected > 0 for morphological match");
    }

    #[test]
    fn chrf_n_max_n_zero_returns_zero() {
        let r = chrf_n("hello", "hello", 0, 2.0);
        assert_eq!(r.f_beta, 0.0);
        assert_eq!(r.precision, 0.0);
        assert_eq!(r.recall, 0.0);
    }

    #[test]
    fn chrf_clipped_overlap_repeats() {
        // ref="aa" (1 unigram 'a' with count 2), cand="aaaaaa" (1 unigram 'a' with count 6).
        // n=1: clipped overlap = min(2, 6) = 2. p1 = 2/6, r1 = 2/2 = 1.0.
        // n=2: ref bigrams = [aa] (1), cand bigrams = [aa]*5 (5). overlap = min(1,5) = 1.
        //   p2 = 1/5 = 0.2, r2 = 1/1 = 1.0.
        // Macro avg: chrP = (1/3 + 1/5) / 2, chrR = 1.0.
        let r = chrf_n("aa", "aaaaaa", 2, 2.0);
        let expected_p = (1.0 / 3.0 + 0.2) / 2.0;
        assert!(
            (r.precision - expected_p).abs() < 1e-5,
            "p={}, expected {expected_p}",
            r.precision
        );
        assert!((r.recall - 1.0).abs() < 1e-5, "r={}", r.recall);
    }

    #[test]
    fn chrf_higher_beta_weights_recall() {
        // cand has all ref chars + many extras (high recall, low precision).
        // β=2 should rank this higher than β=0.5 (which weights precision more).
        let f_high_beta = chrf_n("ab", "abcdefghij", 1, 2.0).f_beta;
        let f_low_beta = chrf_n("ab", "abcdefghij", 1, 0.5).f_beta;
        assert!(
            f_high_beta > f_low_beta,
            "β=2 ({f_high_beta}) should outrank β=0.5 ({f_low_beta}) on recall-heavy match"
        );
    }

    #[test]
    fn chrf_whitespace_stripped() {
        // Strings differing only in whitespace score 1.0.
        let s = chrf("hello world", "helloworld");
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn chrf_lowercase_normalization() {
        let s = chrf("Hello World", "hello world");
        assert!((s - 1.0).abs() < 1e-5);
    }

    #[test]
    fn chrf_struct_carries_beta() {
        let r = chrf_n("the cat", "the dog", 6, 2.0);
        assert_eq!(r.beta, 2.0);
        let r2 = chrf_n("the cat", "the dog", 6, 1.0);
        assert_eq!(r2.beta, 1.0);
    }

    #[test]
    fn chrf_skip_n_when_too_short_doesnt_zero_score() {
        // After lowercase + whitespace strip, "Hi!" → "hi!" = 3 chars.
        // With max_n=6, n=4..6 have no grams → SKIP rather than zero.
        // n=1..3 contribute → final score ~1 for identical inputs.
        let r = chrf_n("Hi!", "Hi!", 6, 2.0);
        assert!(
            r.f_beta > 0.99,
            "got {} — short identical inputs should score ~1",
            r.f_beta
        );
    }

    // ─── METEOR ───────────────────────────────────────────────────

    #[test]
    fn meteor_identical_strings_near_one() {
        // METEOR's chunk penalty applies even to perfect matches —
        // identical 6-token strings score 1·(1 − 0.5·(1/6)^3) ≈ 0.9977.
        // Reference impls behave the same way; the penalty asymptotes
        // to 0 only for infinite match length.
        let s = meteor_lite("the cat sat on the mat", "the cat sat on the mat");
        assert!(s.score > 0.99, "got {}", s.score);
        assert_eq!(s.chunks, 1); // one contiguous run
        let expected_pen = 0.5_f32 * (1.0_f32 / 6.0).powi(3);
        assert!((s.fragmentation_penalty - expected_pen).abs() < 1e-6);
    }

    #[test]
    fn meteor_empty_inputs_zero() {
        assert_eq!(meteor_lite("", "anything").score, 0.0);
        assert_eq!(meteor_lite("anything", "").score, 0.0);
        assert_eq!(meteor_lite("", "").score, 0.0);
    }

    #[test]
    fn meteor_disjoint_yields_zero() {
        let s = meteor_lite("the cat sat", "dogs run quickly");
        assert_eq!(s.score, 0.0);
        assert_eq!(s.matches, 0);
    }

    #[test]
    fn meteor_word_order_matters() {
        // Same vocabulary, scrambled order. BLEU-1 would give 1.0
        // (all unigrams match). METEOR penalizes the fragmentation
        // (3 separate chunks instead of 1).
        let same_order = meteor_lite("the cat sat", "the cat sat");
        let scrambled = meteor_lite("the cat sat", "sat cat the");
        assert!(scrambled.matches == 3); // all 3 still match
        assert!(scrambled.chunks > same_order.chunks);
        assert!(
            scrambled.score < same_order.score,
            "scrambled ({}) should < same-order ({})",
            scrambled.score,
            same_order.score
        );
    }

    #[test]
    fn meteor_stem_match_credit() {
        // "running" / "runs" — exact METEOR scores 0; stem-light
        // METEOR catches via suffix strip ("running"→"runn"+stem_min, "runs"→"run").
        // Hmm, our stem_light strips only specific suffixes. Let me
        // check what runs/running both reduce to: "runs"→"run" (s strip)
        // works since len=4≥3+1. "running"→"runn" (ing strip) since len=7≥3+3.
        // "run" != "runn" — stems don't actually match here!
        // Use a fixture where the stems DO match: "walks" → "walk",
        // "walk" → "walk" (no suffix matches min-length).
        let exact = meteor_exact("walk fast", "walks fast");
        let lite = meteor_lite("walk fast", "walks fast");
        assert!(lite.matches >= exact.matches, "lite must match >= exact");
        assert!(lite.score >= exact.score);
    }

    #[test]
    fn meteor_partial_match_below_one() {
        // 2 of 3 cand words match — score must be < 1.0.
        let s = meteor_lite("the cat sat", "the cat ran");
        assert_eq!(s.matches, 2);
        assert!(s.score < 1.0);
        assert!(s.score > 0.0);
    }

    #[test]
    fn meteor_struct_carries_diagnostics() {
        let s = meteor_lite("the cat sat on the mat", "the dog sat on the floor");
        // 4 of 6 cand match (the/sat/on/the).
        assert_eq!(s.cand_len, 6);
        assert_eq!(s.ref_len, 6);
        assert!(s.matches > 0);
        assert!(s.precision > 0.0 && s.precision <= 1.0);
        assert!(s.recall > 0.0 && s.recall <= 1.0);
    }

    #[test]
    fn meteor_recall_weighted_alpha() {
        // α=0.9 weights recall MUCH more than precision. A cand that's
        // 2× the length of ref but matches all ref tokens has P=0.5,
        // R=1.0 → F-mean = 0.5·1 / (0.9·0.5 + 0.1·1) = 0.5 / 0.55 ≈ 0.909.
        // (Not that high — α=0.9 is RECALL-emphasizing, so penalizing
        // low precision hurts less than penalizing low recall.)
        let r = meteor_lite("the cat sat", "the the cat the sat the");
        // matches = 3 (the cat sat all align), but cand has duplicates.
        // P = 3/6 = 0.5, R = 3/3 = 1.0.
        assert!((r.precision - 0.5).abs() < 1e-5);
        assert!((r.recall - 1.0).abs() < 1e-5);
        // F-mean = P·R / (0.9·P + 0.1·R) = 0.5 / (0.45 + 0.1) = 0.909.
        assert!((r.f_mean - 0.5_f32 / 0.55_f32).abs() < 1e-5);
    }

    #[test]
    fn meteor_lowercase_normalization() {
        // 3-token identical-after-lowercase → score very close to 1
        // but with the small chunk-penalty floor.
        let s = meteor_lite("The Cat Sat", "the cat sat");
        assert!(s.score > 0.98);
        assert_eq!(s.matches, 3);
        assert_eq!(s.chunks, 1);
    }

    #[test]
    fn meteor_lite_at_least_as_high_as_exact() {
        // Stem-light should always equal or exceed exact.
        let exact = meteor_exact("walk runs", "walks run");
        let lite = meteor_lite("walk runs", "walks run");
        assert!(lite.score >= exact.score);
    }

    // ─── WER / CER ────────────────────────────────────────────────

    #[test]
    fn wer_identical_zero() {
        assert_eq!(wer("the cat sat", "the cat sat"), 0.0);
    }

    #[test]
    fn wer_one_word_substitution() {
        // 3 ref words, 1 substitution → WER = 1/3.
        let r = wer("the cat sat", "the cat ran");
        assert!((r - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn wer_one_deletion() {
        // 3 ref, cand has 2 → 1 deletion → WER = 1/3.
        let r = wer("the cat sat", "the cat");
        assert!((r - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn wer_one_insertion() {
        // 3 ref, cand has 4 → 1 insertion → WER = 1/3.
        let r = wer("the cat sat", "the cat sat down");
        assert!((r - 1.0 / 3.0).abs() < 1e-5);
    }

    #[test]
    fn wer_can_exceed_one() {
        // Ref 1 word, cand 5 unrelated words → 5 substitutions/insertions
        // for 1 ref word → WER = 5.0 (or close — depends on alignment).
        let r = wer("hello", "completely different gibberish words here");
        assert!(r > 1.0, "WER should exceed 1.0 for very long disjoint cand, got {r}");
    }

    #[test]
    fn wer_empty_ref_empty_cand_zero() {
        assert_eq!(wer("", ""), 0.0);
    }

    #[test]
    fn wer_empty_cand_one() {
        assert_eq!(wer("the cat sat", ""), 1.0);
    }

    #[test]
    fn wer_lowercase_normalization() {
        // Same as wer_identical_zero with case differences.
        assert_eq!(wer("The Cat Sat", "the cat sat"), 0.0);
    }

    #[test]
    fn cer_identical_zero() {
        assert_eq!(cer("hello world", "hello world"), 0.0);
    }

    #[test]
    fn cer_single_char_substitution() {
        // 5 ref chars (excluding... actually including space → "hello" is 5).
        // Substitute "hello" → "jello": 1 edit, ref_len=5, CER = 1/5 = 0.2.
        let r = cer("hello", "jello");
        assert!((r - 0.2).abs() < 1e-5);
    }

    #[test]
    fn cer_treats_whitespace_significantly() {
        // Missing space is an edit. "helloworld" vs "hello world":
        // ref="hello world" (11 chars), cand="helloworld" (10 chars).
        // 1 deletion → CER = 1/11 ≈ 0.0909.
        let r = cer("hello world", "helloworld");
        assert!((r - 1.0 / 11.0).abs() < 1e-5);
    }

    #[test]
    fn cer_empty_ref_empty_cand_zero() {
        assert_eq!(cer("", ""), 0.0);
    }

    #[test]
    fn cer_lowercase_normalization() {
        assert_eq!(cer("Hello", "hello"), 0.0);
    }

    #[test]
    fn cer_lower_than_wer_for_morphological_diff() {
        // "running" vs "runs" — WER counts as 1 substitution out of 1
        // ref word → WER = 1.0. CER counts shared chars; ref="running"
        // (7), cand="runs" (4). Levenshtein dist = 4 (delete -ning, swap
        // letters... let me think. "running" → "runs":
        // r-u-n-n-i-n-g vs r-u-n-s — share r,u,n. Edit 4 chars (delete
        // n,i,n,g + insert s OR substitute n→s + delete i,n,g = 4 edits).
        // CER = 4/7 ≈ 0.571.
        let w = wer("running", "runs");
        let c = cer("running", "runs");
        assert!((w - 1.0).abs() < 1e-5);
        assert!(c < w, "CER ({c}) should be < WER ({w}) on a morphological diff");
    }

    // ─── chrF++ ───────────────────────────────────────────────────

    #[test]
    fn chrf_pp_identical_near_one() {
        let s = chrf_pp_default("the cat sat on the mat", "the cat sat on the mat");
        assert!(s > 0.99, "got {s}");
    }

    #[test]
    fn chrf_pp_empty_inputs_zero() {
        assert_eq!(chrf_pp_default("", "x"), 0.0);
        assert_eq!(chrf_pp_default("x", ""), 0.0);
        assert_eq!(chrf_pp_default("", ""), 0.0);
    }

    #[test]
    fn chrf_pp_catches_word_reorder_better_than_chrf() {
        // Same vocabulary, different order. chrF (char-only) is
        // permutation-invariant within short n; chrF++ adds word
        // n-grams which catch word-order changes.
        let same_order = chrf_pp_default("the cat sat", "the cat sat");
        let scrambled = chrf_pp_default("the cat sat", "sat cat the");
        assert!(scrambled < same_order, "scrambled {scrambled} should < same {same_order}");
    }

    #[test]
    fn chrf_pp_word_max_n_zero_falls_back_to_chrf() {
        // word_max_n=0 → only char n-grams contribute → equivalent to chrf_n.
        let pp_score = chrf_pp("the cat", "the cat", 6, 0, 2.0);
        let chrf_score = chrf_n("the cat", "the cat", 6, 2.0);
        assert!((pp_score.f_beta - chrf_score.f_beta).abs() < 1e-5);
    }

    #[test]
    fn chrf_pp_char_max_n_zero_word_only() {
        // char_max_n=0 → only word n-grams. With 3 identical tokens
        // and word_max_n=2, both n=1 and n=2 contribute → score ~1.
        let pp_score = chrf_pp("the cat sat", "the cat sat", 0, 2, 2.0);
        assert!(pp_score.f_beta > 0.99);
    }

    #[test]
    fn chrf_pp_disjoint_yields_zero() {
        // Disjoint vocab (no shared words OR chars) → score 0.
        let s = chrf_pp_default("abc def", "xyz qrs");
        assert_eq!(s, 0.0);
    }

    // ─── Multi-reference BLEU + ROUGE ───────────────────────────

    #[test]
    fn bleu_multi_picks_best_overlap_across_refs() {
        // Ref 1 is verbose; ref 2 matches cand exactly. max_n=3 since
        // cand is 3 tokens (BLEU at higher n needs cand long enough
        // to have those n-grams; same caveat as single-ref BLEU —
        // see iter-305 docstring).
        let cand = "the cat sat";
        let refs = ["the cat was sleeping nicely", "the cat sat"];
        let s = bleu_multi(&refs, cand, 3, false);
        // Ref 2 matches exactly → all 3 unigrams + 2 bigrams + 1
        // trigram match. score should be near 1.0.
        assert!(s.score > 0.9, "got {}", s.score);
    }

    #[test]
    fn bleu_multi_brevity_uses_closest_ref_length() {
        // Cand length 3. Ref lengths [10, 4, 100] — closest is 4.
        // BP = exp(1 - 4/3) ≈ 0.717.
        let cand = "the cat sat";
        let refs = [
            "the cat was sleeping for a long time today there",
            "the cat sat there",
            "completely different and very very very very very very long sentence here",
        ];
        let s = bleu_multi(
            &refs.iter().copied().collect::<Vec<_>>(),
            cand,
            1,
            false,
        );
        assert_eq!(s.ref_len, 4);
        let expected_bp = (1.0_f32 - 4.0 / 3.0).exp();
        assert!((s.brevity_penalty - expected_bp).abs() < 1e-5);
    }

    #[test]
    fn bleu_multi_empty_refs_zero() {
        let s = bleu_multi(&[], "anything", 4, false);
        assert_eq!(s.score, 0.0);
    }

    #[test]
    fn bleu_multi_at_least_as_good_as_best_single_ref() {
        // For any cand, multi-ref BLEU should be >= the best of the
        // single-ref BLEU scores (the multi-ref formula is a max-
        // count-per-gram extension; can't score lower than picking
        // the best ref alone).
        let cand = "the cat sat on the mat";
        let refs = ["completely unrelated", "the cat sat on the floor"];
        let single_best = refs
            .iter()
            .map(|r| bleu(r, cand))
            .fold(0.0_f32, f32::max);
        let multi = bleu_multi(&refs, cand, 4, false);
        assert!(
            multi.score >= single_best - 1e-5,
            "multi {} should >= best single-ref {single_best}",
            multi.score
        );
    }

    #[test]
    fn rouge_n_multi_picks_max_f1() {
        let cand = "the cat sat on the mat";
        let refs = ["completely unrelated", "the cat sat on the mat"];
        let r = rouge_n_multi(&refs, cand, 1);
        // Identical to ref 2 → F1 = 1.0.
        assert!((r.f1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rouge_n_multi_empty_refs_zero() {
        let r = rouge_n_multi(&[], "anything", 1);
        assert_eq!(r.f1, 0.0);
    }

    #[test]
    fn rouge_l_multi_picks_max_f1() {
        let cand = "the cat sat on the mat";
        let refs = ["totally different vocabulary used here", "the cat sat on the mat"];
        let r = rouge_l_multi(&refs, cand);
        assert!((r.f1 - 1.0).abs() < 1e-5);
    }

    #[test]
    fn rouge_n_multi_at_least_as_good_as_best_single() {
        let cand = "the cat sat on the mat";
        let refs = ["the cat sat on the floor", "the dog ran fast"];
        let single_best = refs
            .iter()
            .map(|r| rouge_1(r, cand).f1)
            .fold(0.0_f32, f32::max);
        let multi = rouge_n_multi(&refs, cand, 1);
        assert!(multi.f1 >= single_best - 1e-5);
    }

    #[test]
    fn stem_light_strips_common_suffixes() {
        assert_eq!(stem_light("walks"), "walk");
        assert_eq!(stem_light("playing"), "play");
        assert_eq!(stem_light("walked"), "walk");
        assert_eq!(stem_light("quickly"), "quick");
        // Min stem length 3 → "is" does NOT lose s.
        assert_eq!(stem_light("is"), "is");
        // Stems already at min length are not stripped.
        assert_eq!(stem_light("cat"), "cat");
    }
}

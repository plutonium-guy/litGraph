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
}

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
}

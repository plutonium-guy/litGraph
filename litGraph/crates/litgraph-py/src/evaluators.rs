//! Python bindings for `litgraph_core::evaluators` — string-distance and
//! semantic-similarity scoring functions for prompt evaluation.

use litgraph_core::{
    contains_all as core_contains_all, contains_any as core_contains_any,
    embedding_cosine as core_embedding_cosine, exact_match as core_exact_match,
    exact_match_strict as core_exact_match_strict, jaccard_similarity as core_jaccard_similarity,
    json_validity as core_json_validity, levenshtein as core_levenshtein,
    levenshtein_ratio as core_levenshtein_ratio, regex_match as core_regex_match,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_match, m)?)?;
    m.add_function(wrap_pyfunction!(exact_match_strict, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(regex_match, m)?)?;
    m.add_function(wrap_pyfunction!(json_validity, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(contains_all, m)?)?;
    m.add_function(wrap_pyfunction!(contains_any, m)?)?;
    Ok(())
}

/// Exact match (trim + case-fold). Returns True if `actual` matches `expected`.
#[pyfunction]
fn exact_match(actual: &str, expected: &str) -> bool {
    core_exact_match(actual, expected)
}

/// Strict byte-equal match. No trimming, no case-folding.
#[pyfunction]
fn exact_match_strict(actual: &str, expected: &str) -> bool {
    core_exact_match_strict(actual, expected)
}

/// Raw Levenshtein edit distance (number of single-char edits).
#[pyfunction]
fn levenshtein(a: &str, b: &str) -> usize {
    core_levenshtein(a, b)
}

/// Levenshtein ratio in `[0.0, 1.0]`. 1.0 = identical.
#[pyfunction]
fn levenshtein_ratio(a: &str, b: &str) -> f32 {
    core_levenshtein_ratio(a, b)
}

/// Jaccard token-set similarity in `[0.0, 1.0]`. Word-order invariant.
#[pyfunction]
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    core_jaccard_similarity(a, b)
}

/// Returns True if the regex pattern matches anywhere in `text`. Raises
/// `ValueError` if the pattern doesn't compile.
#[pyfunction]
fn regex_match(text: &str, pattern: &str) -> PyResult<bool> {
    core_regex_match(text, pattern).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// True if `text` parses as valid JSON.
#[pyfunction]
fn json_validity(text: &str) -> bool {
    core_json_validity(text)
}

/// Cosine similarity of two equal-length embedding vectors. Result in
/// `[-1.0, 1.0]`. Raises `ValueError` if the lengths don't match.
/// Zero-vector inputs return 0.0 (no NaN).
#[pyfunction]
fn embedding_cosine(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    core_embedding_cosine(&a, &b).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// True if every substring in `needles` appears in `text`.
#[pyfunction]
fn contains_all(text: &str, needles: Vec<String>) -> bool {
    let refs: Vec<&str> = needles.iter().map(|s| s.as_str()).collect();
    core_contains_all(text, &refs)
}

/// True if at least one substring in `needles` appears in `text`.
#[pyfunction]
fn contains_any(text: &str, needles: Vec<String>) -> bool {
    let refs: Vec<&str> = needles.iter().map(|s| s.as_str()).collect();
    core_contains_any(text, &refs)
}

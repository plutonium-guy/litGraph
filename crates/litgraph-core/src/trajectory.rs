//! Trajectory evaluation — score an agent's tool-call path against a
//! reference. Sits alongside `eval_harness`'s output-only scorers and answers
//! a different question: did the agent take the *right path*, regardless of
//! whether it landed on the right answer?
//!
//! Three policies, all pure-Rust, no LLM dependency:
//!
//! * `ContainsAll` — every expected tool name appears in the actual run
//!   (order ignored, duplicates collapsed). Great for "did the agent at
//!   least *try* the right tools".
//! * `ExactOrder` — the actual tool sequence matches the expected one
//!   exactly. Use sparingly: real agents take legitimate detours.
//! * `Subsequence` — the expected sequence is a (possibly non-contiguous)
//!   subsequence of the actual one. Sweet spot: enforces ordering of the
//!   moves that matter without forbidding extra exploration.
//! * `Levenshtein` — 1 - editdistance/max(len), edit distance over tool
//!   names. Smooth gradient for partial-credit grading.

use serde::{Deserialize, Serialize};

use crate::evaluators::levenshtein;

/// One tool call from an agent's trajectory. `input` is optional; today the
/// scoring policies only inspect tool names, but downstream callers can read
/// `input` themselves for finer-grained checks.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TrajectoryStep {
    pub tool: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input: Option<serde_json::Value>,
}

impl TrajectoryStep {
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            input: None,
        }
    }

    pub fn with_input(tool: impl Into<String>, input: serde_json::Value) -> Self {
        Self {
            tool: tool.into(),
            input: Some(input),
        }
    }
}

/// Policy for `evaluate_trajectory`. See module-level docs for semantics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrajectoryPolicy {
    ContainsAll,
    ExactOrder,
    Subsequence,
    Levenshtein,
}

/// Score in `[0.0, 1.0]` — 1.0 = perfect match under the policy.
pub fn evaluate_trajectory(
    actual: &[TrajectoryStep],
    expected: &[TrajectoryStep],
    policy: TrajectoryPolicy,
) -> f64 {
    let act: Vec<&str> = actual.iter().map(|s| s.tool.as_str()).collect();
    let exp: Vec<&str> = expected.iter().map(|s| s.tool.as_str()).collect();
    match policy {
        TrajectoryPolicy::ContainsAll => contains_all(&act, &exp),
        TrajectoryPolicy::ExactOrder => exact_order(&act, &exp),
        TrajectoryPolicy::Subsequence => subsequence(&act, &exp),
        TrajectoryPolicy::Levenshtein => levenshtein_ratio(&act, &exp),
    }
}

fn contains_all(actual: &[&str], expected: &[&str]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    let actual_set: std::collections::HashSet<&str> = actual.iter().copied().collect();
    let hit = expected
        .iter()
        .filter(|t| actual_set.contains(*t))
        .count();
    hit as f64 / expected.len() as f64
}

fn exact_order(actual: &[&str], expected: &[&str]) -> f64 {
    if actual.len() != expected.len() {
        return 0.0;
    }
    if actual.is_empty() {
        return 1.0;
    }
    let matches = actual.iter().zip(expected.iter()).filter(|(a, b)| a == b).count();
    if matches == actual.len() { 1.0 } else { 0.0 }
}

fn subsequence(actual: &[&str], expected: &[&str]) -> f64 {
    if expected.is_empty() {
        return 1.0;
    }
    // Longest common subsequence between expected and actual — this is the
    // count of expected steps that can be matched in order while permitting
    // gaps in `actual`. We deliberately use LCS over greedy because greedy
    // gives 0 credit for "search → ??? → shell" if the middle expected step
    // ("calc") didn't fire — but the agent did still execute search-then-shell
    // in the correct relative order, which is real partial credit.
    let n = actual.len();
    let m = expected.len();
    let mut prev = vec![0usize; m + 1];
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        for j in 1..=m {
            cur[j] = if actual[i - 1] == expected[j - 1] {
                prev[j - 1] + 1
            } else {
                prev[j].max(cur[j - 1])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        for v in cur.iter_mut() {
            *v = 0;
        }
    }
    prev[m] as f64 / m as f64
}

fn levenshtein_ratio(actual: &[&str], expected: &[&str]) -> f64 {
    // Reuse the char-level levenshtein on tool names joined with a separator
    // unlikely to appear in tool names. The result is a smooth ratio.
    if actual.is_empty() && expected.is_empty() {
        return 1.0;
    }
    let sep = "\u{1f}"; // ASCII unit separator — won't appear in tool names
    let a = actual.join(sep);
    let b = expected.join(sep);
    let max = a.chars().count().max(b.chars().count());
    if max == 0 {
        return 1.0;
    }
    let d = levenshtein(&a, &b);
    1.0 - (d as f64 / max as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steps(names: &[&str]) -> Vec<TrajectoryStep> {
        names.iter().map(|n| TrajectoryStep::new(*n)).collect()
    }

    #[test]
    fn contains_all_full_score_when_every_expected_tool_appears() {
        let a = steps(&["search", "calc", "search", "shell"]);
        let e = steps(&["search", "shell"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ContainsAll),
            1.0
        );
    }

    #[test]
    fn contains_all_partial_credit() {
        let a = steps(&["search"]);
        let e = steps(&["search", "shell"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ContainsAll),
            0.5
        );
    }

    #[test]
    fn contains_all_perfect_score_for_empty_expected() {
        let a = steps(&["x"]);
        let e: Vec<TrajectoryStep> = vec![];
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ContainsAll),
            1.0
        );
    }

    #[test]
    fn exact_order_zero_on_length_mismatch() {
        let a = steps(&["a", "b", "c"]);
        let e = steps(&["a", "b"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ExactOrder),
            0.0
        );
    }

    #[test]
    fn exact_order_one_for_identical_sequence() {
        let a = steps(&["a", "b", "c"]);
        let e = steps(&["a", "b", "c"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ExactOrder),
            1.0
        );
    }

    #[test]
    fn exact_order_zero_for_swapped_step() {
        let a = steps(&["a", "c", "b"]);
        let e = steps(&["a", "b", "c"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::ExactOrder),
            0.0
        );
    }

    #[test]
    fn subsequence_full_when_actual_contains_expected_in_order_with_extras() {
        let a = steps(&["search", "noise", "calc", "more_noise", "shell"]);
        let e = steps(&["search", "calc", "shell"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::Subsequence),
            1.0
        );
    }

    #[test]
    fn subsequence_partial_when_one_expected_step_missing() {
        let a = steps(&["search", "noise", "shell"]);
        let e = steps(&["search", "calc", "shell"]);
        // 2 of 3 expected matched in order.
        let s = evaluate_trajectory(&a, &e, TrajectoryPolicy::Subsequence);
        assert!((s - 2.0 / 3.0).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn subsequence_zero_when_order_wrong() {
        let a = steps(&["shell", "search"]);
        let e = steps(&["search", "shell"]);
        // First expected step "search" found at index 1; second expected
        // "shell" must come after — none. → 1/2.
        let s = evaluate_trajectory(&a, &e, TrajectoryPolicy::Subsequence);
        assert!((s - 0.5).abs() < 1e-9, "got {s}");
    }

    #[test]
    fn levenshtein_perfect_on_identical() {
        let a = steps(&["a", "b", "c"]);
        let e = steps(&["a", "b", "c"]);
        assert_eq!(
            evaluate_trajectory(&a, &e, TrajectoryPolicy::Levenshtein),
            1.0
        );
    }

    #[test]
    fn levenshtein_smooth_gradient_with_minor_changes() {
        let a = steps(&["search", "calc", "shell"]);
        let e = steps(&["search", "shell"]);
        let s = evaluate_trajectory(&a, &e, TrajectoryPolicy::Levenshtein);
        assert!(s > 0.5, "should be high partial credit, got {s}");
        assert!(s < 1.0);
    }

    #[test]
    fn empty_actual_and_empty_expected_score_one() {
        let a: Vec<TrajectoryStep> = vec![];
        let e: Vec<TrajectoryStep> = vec![];
        for p in [
            TrajectoryPolicy::ContainsAll,
            TrajectoryPolicy::ExactOrder,
            TrajectoryPolicy::Subsequence,
            TrajectoryPolicy::Levenshtein,
        ] {
            assert_eq!(evaluate_trajectory(&a, &e, p), 1.0, "policy {p:?}");
        }
    }

    #[test]
    fn step_with_input_serializes_round_trip() {
        let s = TrajectoryStep::with_input("calc", serde_json::json!({"x": 1}));
        let blob = serde_json::to_string(&s).unwrap();
        let back: TrajectoryStep = serde_json::from_str(&blob).unwrap();
        assert_eq!(back, s);
    }
}

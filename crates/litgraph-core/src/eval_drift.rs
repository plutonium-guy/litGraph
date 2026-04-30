//! `EvalDriftDetector` — compare two eval reports and surface
//! per-case regressions / improvements / stable failures.
//!
//! # Why
//!
//! After upgrading a model, changing a prompt, or shipping a
//! retrieval refactor, an aggregate score change ("win rate
//! went from 87% to 88%") hides the case-level story:
//!
//! - **Regressions**: cases that previously passed and now fail.
//!   These are usually the most actionable — the new system
//!   broke something the old one handled.
//! - **Improvements**: cases that previously failed and now
//!   pass. Confirms the change is helping where it should.
//! - **Stable failures**: cases that fail in both. Probably
//!   need a different fix — model-agnostic dataset issue.
//! - **Aggregate score deltas** per scorer.
//!
//! Drift detection feeds CI gates ("block merge if any case
//! regressed") and changelogs ("this PR fixed N cases, broke
//! M cases").
//!
//! # Threshold
//!
//! `threshold` defines what counts as a regression / improvement
//! — score drop strictly greater than `threshold` is a
//! regression; score gain strictly greater is an improvement.
//! Default `0.5` works for binary pass/fail scorers (a 0→1 or
//! 1→0 flip both register). For continuous scorers, tune to
//! taste (e.g. 0.1 for "10-percentage-point shift counts").

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval_harness::EvalReport;

/// Per-case drift signal. One drift entry per (case, scorer)
/// pair where the score moved by more than the threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseDrift {
    pub input: String,
    pub scorer: String,
    pub baseline_score: f64,
    pub current_score: f64,
    /// `current - baseline`. Negative = regression. Positive =
    /// improvement.
    pub delta: f64,
}

/// Full drift report.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct DriftReport {
    /// Cases where the score dropped by more than the threshold.
    pub regressions: Vec<CaseDrift>,
    /// Cases where the score rose by more than the threshold.
    pub improvements: Vec<CaseDrift>,
    /// Cases that scored 0 in both baseline AND current — likely
    /// dataset issues rather than model issues.
    pub stable_failures: Vec<CaseDrift>,
    /// Aggregate per-scorer mean deltas (current.mean -
    /// baseline.mean). Useful for the headline "win rate changed
    /// by X" summary.
    pub aggregate_deltas: HashMap<String, f64>,
    /// Cases present in baseline but not in current (input not
    /// matched). Informational.
    pub missing_in_current: Vec<String>,
    /// Cases present in current but not in baseline.
    /// Informational.
    pub new_in_current: Vec<String>,
}

impl DriftReport {
    /// `true` if there are any regressions. Useful for CI gates:
    /// `if drift.has_regressions() { fail_build() }`.
    pub fn has_regressions(&self) -> bool {
        !self.regressions.is_empty()
    }
}

/// Compare two eval reports and produce a drift report.
///
/// Cases are matched by `input` string. Scores are matched by
/// scorer name. A case appearing in both reports with the same
/// scorer triggers exactly one of:
/// - Regression: `baseline_score - current_score > threshold`.
/// - Improvement: `current_score - baseline_score > threshold`.
/// - Stable failure: both scores are 0.0 (the score-0 baseline +
///   score-0 current case — typically what "still failing" means).
/// - Otherwise: not surfaced (within-threshold movement, or both
///   pass).
pub fn detect_drift(
    baseline: &EvalReport,
    current: &EvalReport,
    threshold: f64,
) -> DriftReport {
    let mut report = DriftReport::default();

    // Index baseline by input for fast lookup.
    let baseline_by_input: HashMap<&str, &crate::eval_harness::EvalCaseResult> = baseline
        .per_case
        .iter()
        .map(|c| (c.input.as_str(), c))
        .collect();
    let current_by_input: HashMap<&str, &crate::eval_harness::EvalCaseResult> = current
        .per_case
        .iter()
        .map(|c| (c.input.as_str(), c))
        .collect();

    // Per-case drift.
    for cur_case in &current.per_case {
        let Some(base_case) = baseline_by_input.get(cur_case.input.as_str()) else {
            report.new_in_current.push(cur_case.input.clone());
            continue;
        };
        // Compare per-scorer.
        for (scorer, cur_val) in &cur_case.scores {
            let cur_score = score_to_f64(cur_val);
            let base_score = base_case
                .scores
                .get(scorer)
                .map(score_to_f64)
                .unwrap_or(0.0);
            let delta = cur_score - base_score;
            let entry = CaseDrift {
                input: cur_case.input.clone(),
                scorer: scorer.clone(),
                baseline_score: base_score,
                current_score: cur_score,
                delta,
            };
            if delta < -threshold {
                report.regressions.push(entry);
            } else if delta > threshold {
                report.improvements.push(entry);
            } else if base_score == 0.0 && cur_score == 0.0 {
                // Both failed; flag as stable failure.
                report.stable_failures.push(entry);
            }
        }
    }
    // Cases present in baseline but missing in current.
    for base_case in &baseline.per_case {
        if !current_by_input.contains_key(base_case.input.as_str()) {
            report.missing_in_current.push(base_case.input.clone());
        }
    }

    // Aggregate per-scorer mean deltas.
    let base_means: HashMap<&str, f64> = baseline
        .aggregate
        .means
        .iter()
        .map(|(k, v)| (k.as_str(), score_to_f64(v)))
        .collect();
    for (scorer, cur_val) in &current.aggregate.means {
        let cur_mean = score_to_f64(cur_val);
        let base_mean = base_means.get(scorer.as_str()).copied().unwrap_or(0.0);
        report
            .aggregate_deltas
            .insert(scorer.clone(), cur_mean - base_mean);
    }

    report
}

fn score_to_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .unwrap_or(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_harness::{AggregateScores, EvalCaseResult};
    use serde_json::{json, Map};

    fn make_case(input: &str, scores: &[(&str, f64)]) -> EvalCaseResult {
        let mut score_map = Map::new();
        for (k, v) in scores {
            score_map.insert(k.to_string(), json!(v));
        }
        EvalCaseResult {
            input: input.to_string(),
            expected: None,
            output: Some("output".to_string()),
            scores: score_map,
            error: None,
            metadata: json!({}),
        }
    }

    fn make_report(cases: Vec<EvalCaseResult>, means: &[(&str, f64)]) -> EvalReport {
        let n_cases = cases.len();
        let mut means_map = Map::new();
        for (k, v) in means {
            means_map.insert(k.to_string(), json!(v));
        }
        EvalReport {
            per_case: cases,
            aggregate: AggregateScores {
                n_cases,
                n_errors: 0,
                means: means_map,
            },
        }
    }

    #[test]
    fn detects_regression() {
        let baseline = make_report(
            vec![
                make_case("q1", &[("exact", 1.0)]),
                make_case("q2", &[("exact", 1.0)]),
            ],
            &[("exact", 1.0)],
        );
        let current = make_report(
            vec![
                make_case("q1", &[("exact", 1.0)]),
                make_case("q2", &[("exact", 0.0)]),
            ],
            &[("exact", 0.5)],
        );
        let drift = detect_drift(&baseline, &current, 0.5);
        assert_eq!(drift.regressions.len(), 1);
        assert_eq!(drift.regressions[0].input, "q2");
        assert_eq!(drift.regressions[0].delta, -1.0);
        assert!(drift.improvements.is_empty());
        assert!(drift.has_regressions());
    }

    #[test]
    fn detects_improvement() {
        let baseline = make_report(
            vec![
                make_case("q1", &[("exact", 0.0)]),
                make_case("q2", &[("exact", 1.0)]),
            ],
            &[("exact", 0.5)],
        );
        let current = make_report(
            vec![
                make_case("q1", &[("exact", 1.0)]),
                make_case("q2", &[("exact", 1.0)]),
            ],
            &[("exact", 1.0)],
        );
        let drift = detect_drift(&baseline, &current, 0.5);
        assert!(drift.regressions.is_empty());
        assert_eq!(drift.improvements.len(), 1);
        assert_eq!(drift.improvements[0].input, "q1");
        assert_eq!(drift.improvements[0].delta, 1.0);
    }

    #[test]
    fn flags_stable_failures() {
        let baseline = make_report(
            vec![
                make_case("hard_q", &[("exact", 0.0)]),
                make_case("ok_q", &[("exact", 1.0)]),
            ],
            &[("exact", 0.5)],
        );
        let current = make_report(
            vec![
                make_case("hard_q", &[("exact", 0.0)]),
                make_case("ok_q", &[("exact", 1.0)]),
            ],
            &[("exact", 0.5)],
        );
        let drift = detect_drift(&baseline, &current, 0.5);
        assert!(drift.regressions.is_empty());
        assert!(drift.improvements.is_empty());
        assert_eq!(drift.stable_failures.len(), 1);
        assert_eq!(drift.stable_failures[0].input, "hard_q");
    }

    #[test]
    fn aggregate_deltas_per_scorer() {
        let baseline = make_report(
            vec![make_case("q1", &[("exact", 1.0), ("jaccard", 0.5)])],
            &[("exact", 1.0), ("jaccard", 0.5)],
        );
        let current = make_report(
            vec![make_case("q1", &[("exact", 0.0), ("jaccard", 0.7)])],
            &[("exact", 0.0), ("jaccard", 0.7)],
        );
        let drift = detect_drift(&baseline, &current, 0.5);
        assert_eq!(drift.aggregate_deltas["exact"], -1.0);
        assert!((drift.aggregate_deltas["jaccard"] - 0.2).abs() < 1e-9);
    }

    #[test]
    fn within_threshold_movement_not_surfaced() {
        let baseline = make_report(
            vec![make_case("q1", &[("score", 0.5)])],
            &[("score", 0.5)],
        );
        let current = make_report(
            vec![make_case("q1", &[("score", 0.7)])],
            &[("score", 0.7)],
        );
        // Threshold 0.5; delta is 0.2 which is within. Not surfaced.
        let drift = detect_drift(&baseline, &current, 0.5);
        assert!(drift.regressions.is_empty());
        assert!(drift.improvements.is_empty());
        assert!(drift.stable_failures.is_empty());
    }

    #[test]
    fn small_threshold_catches_continuous_drift() {
        let baseline = make_report(
            vec![make_case("q1", &[("score", 0.7)])],
            &[("score", 0.7)],
        );
        let current = make_report(
            vec![make_case("q1", &[("score", 0.5)])],
            &[("score", 0.5)],
        );
        // Threshold 0.1; delta is -0.2 → regression.
        let drift = detect_drift(&baseline, &current, 0.1);
        assert_eq!(drift.regressions.len(), 1);
    }

    #[test]
    fn missing_and_new_cases_listed() {
        let baseline = make_report(
            vec![
                make_case("q1", &[("exact", 1.0)]),
                make_case("q2", &[("exact", 1.0)]),
                make_case("q3", &[("exact", 0.0)]),
            ],
            &[("exact", 0.67)],
        );
        let current = make_report(
            vec![
                make_case("q1", &[("exact", 1.0)]),
                make_case("q4", &[("exact", 1.0)]),
            ],
            &[("exact", 1.0)],
        );
        let drift = detect_drift(&baseline, &current, 0.5);
        assert!(drift.missing_in_current.contains(&"q2".to_string()));
        assert!(drift.missing_in_current.contains(&"q3".to_string()));
        assert_eq!(drift.new_in_current, vec!["q4".to_string()]);
    }

    #[test]
    fn empty_reports_return_empty_drift() {
        let baseline = make_report(vec![], &[]);
        let current = make_report(vec![], &[]);
        let drift = detect_drift(&baseline, &current, 0.5);
        assert!(drift.regressions.is_empty());
        assert!(drift.improvements.is_empty());
        assert!(drift.stable_failures.is_empty());
        assert!(drift.aggregate_deltas.is_empty());
        assert!(!drift.has_regressions());
    }
}

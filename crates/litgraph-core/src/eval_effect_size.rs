//! Paired effect size for two eval reports — Cohen's d_z and Hedges' g.
//!
//! # Why effect size on top of significance
//!
//! Significance tests (`mcnemar_test`, `wilcoxon_signed_rank_test`)
//! answer "is the difference real?" — they tell you whether the
//! observed change is likely to be noise. They do NOT tell you
//! whether the change is *big enough to care about*.
//!
//! With a large enough eval set, a 0.001 shift in average cosine
//! similarity is statistically significant. That's also practically
//! meaningless. Effect size is the antidote: a normalized magnitude
//! that doesn't grow with sample size.
//!
//! Use both: significance to filter out noise, effect size to filter
//! out trivia.
//!
//! # The metric
//!
//! For **paired** continuous data (the typical eval case — same input
//! scored under baseline + current), the paired-sample equivalent of
//! Cohen's d is:
//!
//! ```text
//! d_z = mean(diff) / std(diff)
//! ```
//!
//! where `diff = current_score - baseline_score`. `std(diff)` is the
//! sample standard deviation of those diffs (n-1 denominator).
//!
//! # Why d_z and not d_av
//!
//! Two paired-d definitions exist in the literature: `d_z` (uses std
//! of diffs) and `d_av` (uses average of the within-pair stds).
//! `d_z` is the right match for Wilcoxon-style "same case under two
//! treatments" eval comparisons — it's the effect size the paired
//! t-test's noncentrality parameter is built on. `d_av` is more
//! appropriate for between-groups comparisons.
//!
//! # Hedges' g
//!
//! Cohen's d is biased upward for small n. Hedges' g multiplies by
//! `(1 - 3/(4n - 5))` — the standard small-sample correction.
//! For n=10, g ≈ 0.91·d. For n>50, the correction is < 2% and
//! d ≈ g.
//!
//! # Magnitude bands (Cohen, 1988)
//!
//! |d| | label |
//! |---|-------|
//! | < 0.2 | negligible |
//! | 0.2 .. 0.5 | small |
//! | 0.5 .. 0.8 | medium |
//! | ≥ 0.8 | large |
//!
//! These are conventions, not laws. Prefer domain-specific anchors
//! when you have them.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval_harness::EvalReport;

/// One paired-effect-size result for a (scorer, paired-cases) pair.
///
/// All effect-size values are signed: positive = `current` scored
/// higher than `baseline`, negative = regression. The `magnitude`
/// field reports the absolute-value Cohen band as a label.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairedEffectSizeResult {
    pub scorer: String,
    /// Total paired-case count (zeros included — different convention
    /// from Wilcoxon, which drops zeros). All `n` cases contribute
    /// to mean_diff and std_diff.
    pub n: u64,
    /// Mean of `current_score - baseline_score`.
    pub mean_diff: f64,
    /// Sample standard deviation of the diffs (n-1 denominator).
    pub std_diff: f64,
    /// Cohen's d_z = mean_diff / std_diff. NaN when n < 2 or
    /// std_diff = 0 (no variance — every diff identical, including
    /// the all-zero case).
    pub cohens_d: f64,
    /// Hedges' g = d_z * (1 - 3/(4n - 5)). Bias-corrected for small n.
    /// NaN under the same conditions as `cohens_d`.
    pub hedges_g: f64,
    /// Cohen-band label of `|cohens_d|`: "negligible" / "small" /
    /// "medium" / "large", or "undefined" when d is NaN.
    pub magnitude: String,
}

/// Compute paired effect size on every scorer that appears in both
/// reports. Returns one result per scorer, sorted by scorer name.
///
/// # Pairing
///
/// Cases are paired by `input` string. Cases present in only one
/// report are silently skipped (same convention as `mcnemar_test`
/// and `wilcoxon_signed_rank_test`).
///
/// # Edge cases
///
/// - **n < 2**: `std_diff` is undefined (sample-std needs ≥ 2
///   observations) → `cohens_d` and `hedges_g` are NaN, magnitude
///   is `"undefined"`.
/// - **All diffs identical** (typically all-zero — no change at all):
///   `std_diff == 0` → division by zero → NaN → `"undefined"`. The
///   "no shift detected" case is genuinely "effect size undefined,"
///   not "effect size = 0," because Cohen's d is mean-relative-to-
///   spread and there's no spread.
/// - **n in (2..)** but std_diff = 0: same — uniform shift with
///   zero variance → undefined effect size. (Significance tests
///   handle this differently — Wilcoxon would call it highly
///   significant.)
pub fn paired_effect_size(
    baseline: &EvalReport,
    current: &EvalReport,
) -> Vec<PairedEffectSizeResult> {
    let baseline_by_input: HashMap<&str, &crate::eval_harness::EvalCaseResult> = baseline
        .per_case
        .iter()
        .map(|c| (c.input.as_str(), c))
        .collect();
    let mut per_scorer: HashMap<String, Vec<f64>> = HashMap::new();
    for cur_case in &current.per_case {
        let Some(base_case) = baseline_by_input.get(cur_case.input.as_str()) else {
            continue;
        };
        for (scorer, cur_val) in &cur_case.scores {
            let cur_score = score_to_f64(cur_val);
            let base_score = base_case
                .scores
                .get(scorer)
                .map(score_to_f64)
                .unwrap_or(0.0);
            per_scorer
                .entry(scorer.clone())
                .or_default()
                .push(cur_score - base_score);
        }
    }
    let mut out = Vec::with_capacity(per_scorer.len());
    for (scorer, diffs) in per_scorer {
        let n = diffs.len() as u64;
        let mean_diff = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f64>() / diffs.len() as f64
        };
        let std_diff = sample_std(&diffs, mean_diff);
        let (cohens_d, hedges_g, magnitude) = if n < 2 || std_diff == 0.0 {
            (f64::NAN, f64::NAN, "undefined".to_string())
        } else {
            let d = mean_diff / std_diff;
            // Hedges' small-sample correction.
            let g_correction = 1.0 - 3.0 / (4.0 * n as f64 - 5.0);
            let g = d * g_correction;
            let mag = magnitude_label(d.abs());
            (d, g, mag.to_string())
        };
        out.push(PairedEffectSizeResult {
            scorer,
            n,
            mean_diff,
            std_diff,
            cohens_d,
            hedges_g,
            magnitude,
        });
    }
    out.sort_by(|x, y| x.scorer.cmp(&y.scorer));
    out
}

fn score_to_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .unwrap_or(0.0)
}

fn sample_std(xs: &[f64], mean: f64) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (xs.len() - 1) as f64;
    var.sqrt()
}

/// Cohen-band label for `|d|`. Cohen (1988) thresholds.
fn magnitude_label(abs_d: f64) -> &'static str {
    if abs_d < 0.2 {
        "negligible"
    } else if abs_d < 0.5 {
        "small"
    } else if abs_d < 0.8 {
        "medium"
    } else {
        "large"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval_harness::{AggregateScores, EvalCaseResult};
    use serde_json::{json, Map};

    fn make_case(input: &str, scores: &[(&str, f64)]) -> EvalCaseResult {
        let mut m = Map::new();
        for (k, v) in scores {
            m.insert(k.to_string(), json!(v));
        }
        EvalCaseResult {
            input: input.to_string(),
            expected: None,
            output: Some("o".to_string()),
            scores: m,
            error: None,
            metadata: json!({}),
        }
    }

    fn make_report(cases: Vec<EvalCaseResult>) -> EvalReport {
        let n = cases.len();
        EvalReport {
            per_case: cases,
            aggregate: AggregateScores {
                n_cases: n,
                n_errors: 0,
                means: Map::new(),
            },
        }
    }

    #[test]
    fn no_change_undefined_effect_size() {
        // 5 cases, identical scores → all diffs 0 → std=0 → undefined.
        let cases: Vec<_> = (0..5)
            .map(|i| make_case(&format!("q{i}"), &[("cosine", 0.5)]))
            .collect();
        let baseline = make_report(cases);
        let current = baseline.clone();
        let res = paired_effect_size(&baseline, &current);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        assert_eq!(r.n, 5);
        assert_eq!(r.mean_diff, 0.0);
        assert_eq!(r.std_diff, 0.0);
        assert!(r.cohens_d.is_nan());
        assert!(r.hedges_g.is_nan());
        assert_eq!(r.magnitude, "undefined");
    }

    #[test]
    fn uniform_shift_undefined_effect_size() {
        // 10 cases, every diff = +0.1 → mean=0.1 but std=0 → d undefined.
        // Cohen's d is mean-relative-to-spread; uniform shift has no spread.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..10 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
        }
        let res = paired_effect_size(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!((r.mean_diff - 0.1).abs() < 1e-9);
        assert_eq!(r.std_diff, 0.0);
        assert!(r.cohens_d.is_nan());
        assert_eq!(r.magnitude, "undefined");
    }

    #[test]
    fn known_d_recovered() {
        // Hand-construct: diffs = [0.0, 0.1, 0.2, 0.3, 0.4] → mean=0.2.
        // var = (0.04+0.01+0+0.01+0.04)/4 = 0.025 → std ≈ 0.158.
        // d = 0.2/0.158 ≈ 1.265 → "large".
        let pairs = [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4)];
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for (i, &(b, c)) in pairs.iter().enumerate() {
            base.push(make_case(&format!("q{i}"), &[("cosine", b)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", c)]));
        }
        let res = paired_effect_size(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!((r.mean_diff - 0.2).abs() < 1e-9);
        assert!((r.std_diff - 0.025_f64.sqrt()).abs() < 1e-9);
        // d_z = 0.2 / sqrt(0.025) = 0.2 / 0.158... ≈ 1.2649
        assert!(
            (r.cohens_d - 0.2 / 0.025_f64.sqrt()).abs() < 1e-9,
            "d={}",
            r.cohens_d
        );
        assert_eq!(r.magnitude, "large");
    }

    #[test]
    fn negative_d_for_regression() {
        // Mirror — diffs all negative.
        let pairs = [(0.4, 0.0), (0.3, 0.0), (0.2, 0.0), (0.1, 0.0), (0.0, 0.0)];
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for (i, &(b, c)) in pairs.iter().enumerate() {
            base.push(make_case(&format!("q{i}"), &[("cosine", b)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", c)]));
        }
        let res = paired_effect_size(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!(r.mean_diff < 0.0);
        assert!(r.cohens_d < 0.0);
        // |d| is the same as the improvement-mirror case.
        assert_eq!(r.magnitude, "large");
    }

    #[test]
    fn hedges_g_correction_shrinks_d() {
        // Same fixture as known_d_recovered. n=5 → correction =
        // 1 - 3/(4*5-5) = 1 - 3/15 = 0.8. So g = 0.8 * d.
        let pairs = [(0.0, 0.0), (0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (0.0, 0.4)];
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for (i, &(b, c)) in pairs.iter().enumerate() {
            base.push(make_case(&format!("q{i}"), &[("cosine", b)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", c)]));
        }
        let res = paired_effect_size(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!((r.hedges_g - 0.8 * r.cohens_d).abs() < 1e-9);
    }

    #[test]
    fn hedges_correction_negligible_for_large_n() {
        // n=100 → correction = 1 - 3/395 ≈ 0.9924 → g/d ≈ 0.9924.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        // Mostly small +0.05 diffs, with a few outliers to give nonzero std.
        for i in 0..100 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
            let bump = 0.05 + (i as f64) * 0.001; // diffs spread 0.05..0.149
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.5 + bump)]));
        }
        let res = paired_effect_size(&make_report(base), &make_report(cur));
        let r = &res[0];
        let ratio = r.hedges_g / r.cohens_d;
        assert!((ratio - 0.9924).abs() < 1e-3, "ratio={ratio}");
    }

    #[test]
    fn magnitude_bands() {
        assert_eq!(magnitude_label(0.0), "negligible");
        assert_eq!(magnitude_label(0.19), "negligible");
        assert_eq!(magnitude_label(0.2), "small");
        assert_eq!(magnitude_label(0.49), "small");
        assert_eq!(magnitude_label(0.5), "medium");
        assert_eq!(magnitude_label(0.79), "medium");
        assert_eq!(magnitude_label(0.8), "large");
        assert_eq!(magnitude_label(5.0), "large");
    }

    #[test]
    fn missing_in_current_skipped() {
        let baseline = make_report(vec![
            make_case("q1", &[("cosine", 0.5)]),
            make_case("q2", &[("cosine", 0.5)]),
            make_case("q3", &[("cosine", 0.9)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("cosine", 0.6)]),
            make_case("q2", &[("cosine", 0.7)]),
        ]);
        let res = paired_effect_size(&baseline, &current);
        assert_eq!(res[0].n, 2);
    }

    #[test]
    fn small_n_undefined() {
        // Single paired case → n=1, std undefined → result undefined.
        let baseline = make_report(vec![make_case("q1", &[("cosine", 0.5)])]);
        let current = make_report(vec![make_case("q1", &[("cosine", 0.7)])]);
        let res = paired_effect_size(&baseline, &current);
        let r = &res[0];
        assert_eq!(r.n, 1);
        assert!(r.cohens_d.is_nan());
        assert_eq!(r.magnitude, "undefined");
    }

    #[test]
    fn per_scorer_sorted() {
        let baseline = make_report(vec![
            make_case("q1", &[("alpha", 0.5), ("zeta", 0.5)]),
            make_case("q2", &[("alpha", 0.5), ("zeta", 0.5)]),
            make_case("q3", &[("alpha", 0.5), ("zeta", 0.5)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("alpha", 0.6), ("zeta", 0.4)]),
            make_case("q2", &[("alpha", 0.7), ("zeta", 0.3)]),
            make_case("q3", &[("alpha", 0.8), ("zeta", 0.2)]),
        ]);
        let res = paired_effect_size(&baseline, &current);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].scorer, "alpha");
        assert_eq!(res[1].scorer, "zeta");
        assert!(res[0].mean_diff > 0.0); // alpha improved
        assert!(res[1].mean_diff < 0.0); // zeta regressed
    }
}

//! Statistical-significance tests for paired eval outcomes between
//! two eval runs. Pairs with iter-289 `detect_drift`: drift says
//! "X regressions, Y improvements"; these say "is the change
//! statistically meaningful or just noise?"
//!
//! - [`mcnemar_test`] — paired *binary* outcomes (pass/fail). McNemar's
//!   chi-squared test (1947). Fits scorers like `exact_match`,
//!   `regex_match`, threshold-coerced LLM judges.
//! - [`wilcoxon_signed_rank_test`] — paired *continuous* outcomes
//!   (any 0..1 / 0..N float). Wilcoxon (1945) signed-rank test, the
//!   non-parametric counterpart to the paired t-test, makes no
//!   normality assumption. Fits scorers like cosine similarity,
//!   BLEU, raw LLM-judge scores, embedding-recall@k.
//!
//! Pick by scorer type:
//! - Binary (0 or 1): McNemar.
//! - Continuous + symmetric-around-zero diffs assumed: Wilcoxon.
//! - Continuous + diffs ≈ Gaussian: a paired t-test would be more
//!   powerful (out of scope here — bring your own).
//!
//! Both tests output `significant_at_05: bool` for direct CI gating
//! and a `small_sample: bool` flag when the asymptotic
//! approximation gets unreliable.
//!
//! # The test
//!
//! McNemar's chi-squared test (Quinn McNemar, 1947) for paired
//! binary outcomes. Each case appears in both baseline and
//! current with a pass/fail outcome. The 2×2 contingency table
//! cell counts:
//!
//! - `a`: pass in both.
//! - `b`: pass in baseline, fail in current (regression).
//! - `c`: fail in baseline, pass in current (improvement).
//! - `d`: fail in both.
//!
//! McNemar's chi-squared statistic with continuity correction:
//!
//! ```text
//! χ² = (|b - c| - 1)² / (b + c)
//! ```
//!
//! Under the null hypothesis (the two systems perform equally
//! well), χ² follows a chi-squared distribution with 1 degree
//! of freedom. The p-value is `P(X > χ²)` from that
//! distribution.
//!
//! # Why this not a paired t-test?
//!
//! Paired t-tests assume continuous, normally-distributed
//! outcomes. Eval scores from binary scorers (`exact_match`,
//! `regex_match`) are bernoulli — McNemar's is the right test.
//! For continuous scorers, use the structured drift report
//! plus a domain-appropriate test (paired t-test, Wilcoxon
//! signed-rank).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval_harness::EvalReport;

/// One McNemar test result for a (scorer, paired-cases) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarResult {
    pub scorer: String,
    /// Pass in baseline, pass in current.
    pub a: u64,
    /// Pass in baseline, fail in current (regression).
    pub b: u64,
    /// Fail in baseline, pass in current (improvement).
    pub c: u64,
    /// Fail in both.
    pub d: u64,
    /// McNemar's chi-squared statistic (continuity-corrected).
    pub chi_squared: f64,
    /// Two-tailed p-value approximation.
    pub p_value: f64,
    /// `true` if `p_value < 0.05`.
    pub significant_at_05: bool,
    /// `true` if `b + c < 25`. McNemar's chi-squared approximation
    /// is unreliable in that regime; users should prefer an exact
    /// binomial test (out of scope here — flagged so callers know).
    pub small_sample: bool,
}

/// Run McNemar's test on every scorer that appears in both
/// reports. Returns one result per scorer.
///
/// Pass/fail threshold is `score >= 0.5` — matches the convention
/// used elsewhere (see iter-289 `detect_drift`). For continuous
/// scorers where binary thresholding loses information, users
/// should run a proper continuous test on the raw `EvalReport`s
/// instead.
pub fn mcnemar_test(
    baseline: &EvalReport,
    current: &EvalReport,
) -> Vec<McNemarResult> {
    // Index baseline by input.
    let baseline_by_input: HashMap<&str, &crate::eval_harness::EvalCaseResult> = baseline
        .per_case
        .iter()
        .map(|c| (c.input.as_str(), c))
        .collect();
    // Collect per-scorer cell counts.
    let mut cells: HashMap<String, [u64; 4]> = HashMap::new(); // [a, b, c, d]
    for cur_case in &current.per_case {
        let Some(base_case) = baseline_by_input.get(cur_case.input.as_str()) else {
            continue;
        };
        for (scorer, cur_val) in &cur_case.scores {
            let cur_pass = score_to_f64(cur_val) >= 0.5;
            let base_pass = base_case
                .scores
                .get(scorer)
                .map(score_to_f64)
                .map(|s| s >= 0.5)
                .unwrap_or(false);
            let entry = cells.entry(scorer.clone()).or_insert([0, 0, 0, 0]);
            match (base_pass, cur_pass) {
                (true, true) => entry[0] += 1,
                (true, false) => entry[1] += 1,
                (false, true) => entry[2] += 1,
                (false, false) => entry[3] += 1,
            }
        }
    }
    let mut out: Vec<McNemarResult> = Vec::with_capacity(cells.len());
    for (scorer, [a, b, c, d]) in cells {
        let bc = b + c;
        let chi_squared = if bc == 0 {
            0.0
        } else {
            let diff = (b as i64 - c as i64).unsigned_abs() as f64;
            // Continuity correction: -1 inside the |...|.
            let corrected = (diff - 1.0).max(0.0);
            (corrected * corrected) / (bc as f64)
        };
        let p_value = chi_squared_df1_p_value(chi_squared);
        out.push(McNemarResult {
            scorer,
            a,
            b,
            c,
            d,
            chi_squared,
            p_value,
            significant_at_05: p_value < 0.05,
            small_sample: bc < 25,
        });
    }
    // Stable sort by scorer name for test reproducibility.
    out.sort_by(|x, y| x.scorer.cmp(&y.scorer));
    out
}

fn score_to_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .unwrap_or(0.0)
}

/// Two-tailed p-value for chi-squared with 1 degree of freedom.
/// Uses the relation `χ²(1) = Z²`, so `p = 2 * (1 - Φ(√χ²))`
/// where Φ is the standard-normal CDF (Abramowitz & Stegun
/// 26.2.17 erf approximation).
fn chi_squared_df1_p_value(chi_squared: f64) -> f64 {
    if chi_squared <= 0.0 {
        return 1.0;
    }
    let z = chi_squared.sqrt();
    let p_one_tail = 1.0 - normal_cdf(z);
    (2.0 * p_one_tail).clamp(0.0, 1.0)
}

/// Standard-normal CDF via Abramowitz & Stegun 7.1.26 (erf
/// approximation; max error ~1.5e-7).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

fn erf(x: f64) -> f64 {
    // A&S 7.1.26: erf(x) ≈ 1 - (a1*t + a2*t² + a3*t³ + a4*t⁴ + a5*t⁵) * exp(-x²)
    // for x ≥ 0, with t = 1/(1+p*x).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let p = 0.327_591_1;
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

// ─────────────────────────────────────────────────────────────────────────────
// Wilcoxon signed-rank test — paired continuous outcomes
// ─────────────────────────────────────────────────────────────────────────────

/// One Wilcoxon signed-rank test result for a (scorer, paired-cases)
/// pair.
///
/// `n` is the count *after* dropping pairs with zero difference
/// (the standard Wilcoxon convention — ties contribute no signed
/// rank). The `mean_diff` field reports the average across ALL
/// paired cases (including zeros) since that's what callers
/// usually want as a directional summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WilcoxonResult {
    pub scorer: String,
    /// Number of non-zero paired diffs (zeros dropped per Wilcoxon convention).
    pub n: u64,
    /// Sum of ranks of positive differences (current > baseline).
    pub w_plus: f64,
    /// Sum of ranks of negative differences (current < baseline).
    pub w_minus: f64,
    /// Test statistic — `min(w_plus, w_minus)`.
    pub w: f64,
    /// Mean (current - baseline) across all paired cases (including zeros).
    /// Sign tells you the *direction* of the change, magnitude the size.
    pub mean_diff: f64,
    /// z-score under normal approximation with continuity + tie correction.
    pub z: f64,
    /// Two-tailed p-value approximation.
    pub p_value: f64,
    /// `true` if `p_value < 0.05`.
    pub significant_at_05: bool,
    /// `true` if `n < 20`. The normal approximation is unreliable
    /// in that regime; users should consult an exact Wilcoxon
    /// table or a permutation test for small n.
    pub small_sample: bool,
}

/// Wilcoxon signed-rank test on every scorer that appears in both
/// reports. Returns one result per scorer, sorted by scorer name.
///
/// # Algorithm
///
/// 1. For each paired (baseline, current) case, compute
///    `diff = current_score - baseline_score` per scorer.
/// 2. Drop pairs where `diff == 0` (Wilcoxon convention — zeros
///    have no signed rank).
/// 3. Rank the remaining `|diff|` values from 1 (smallest) to `n`
///    (largest). Ties get the **average** of the ranks they would
///    otherwise span (e.g. two values tied for ranks 3 and 4 each
///    receive 3.5).
/// 4. Sum the ranks for positive diffs (`w_plus`) and negative
///    diffs (`w_minus`). Test statistic `w = min(w_plus, w_minus)`.
/// 5. Under H₀ (no shift), `w` has expected value
///    `μ = n(n+1)/4` and variance
///    `σ² = n(n+1)(2n+1)/24 − Σ(t³−t)/48` where the sum is over
///    tie-group sizes `t`. Continuity correction +0.5 toward μ.
/// 6. Two-tailed p-value via standard-normal CDF (Abramowitz &
///    Stegun 7.1.26 erf approximation, shared with `mcnemar_test`).
///
/// # Why this not a paired t-test
///
/// Wilcoxon is non-parametric — it doesn't assume diffs are
/// normally distributed, only that they're symmetric around the
/// median under H₀. Eval-score diffs are often skewed (most
/// cases unchanged, a handful regressed sharply); Wilcoxon stays
/// well-calibrated under skew where a paired t-test inflates
/// false positives.
pub fn wilcoxon_signed_rank_test(
    baseline: &EvalReport,
    current: &EvalReport,
) -> Vec<WilcoxonResult> {
    // Index baseline by input.
    let baseline_by_input: HashMap<&str, &crate::eval_harness::EvalCaseResult> = baseline
        .per_case
        .iter()
        .map(|c| (c.input.as_str(), c))
        .collect();
    // Collect per-scorer paired diffs.
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
        let mean_diff = if diffs.is_empty() {
            0.0
        } else {
            diffs.iter().sum::<f64>() / diffs.len() as f64
        };
        // Drop zeros — Wilcoxon convention.
        let nz: Vec<f64> = diffs.iter().copied().filter(|d| *d != 0.0).collect();
        let n = nz.len() as u64;
        if n == 0 {
            out.push(WilcoxonResult {
                scorer,
                n,
                w_plus: 0.0,
                w_minus: 0.0,
                w: 0.0,
                mean_diff,
                z: 0.0,
                p_value: 1.0,
                significant_at_05: false,
                small_sample: true,
            });
            continue;
        }
        // Rank by |diff| with average-rank ties.
        let mut indexed: Vec<(usize, f64)> = nz
            .iter()
            .enumerate()
            .map(|(i, d)| (i, d.abs()))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut ranks = vec![0.0_f64; nz.len()];
        let mut tie_correction = 0.0_f64;
        let mut i = 0;
        while i < indexed.len() {
            let mut j = i + 1;
            while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
                j += 1;
            }
            // Average rank for ties [i..j) is ((i+1) + j) / 2 (1-indexed).
            let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
            let group_size = (j - i) as f64;
            for k in i..j {
                ranks[indexed[k].0] = avg_rank;
            }
            if group_size > 1.0 {
                tie_correction += group_size.powi(3) - group_size;
            }
            i = j;
        }
        let mut w_plus = 0.0;
        let mut w_minus = 0.0;
        for (idx, &diff) in nz.iter().enumerate() {
            if diff > 0.0 {
                w_plus += ranks[idx];
            } else {
                w_minus += ranks[idx];
            }
        }
        let w = w_plus.min(w_minus);
        let n_f = n as f64;
        let mean_w = n_f * (n_f + 1.0) / 4.0;
        let var_w = n_f * (n_f + 1.0) * (2.0 * n_f + 1.0) / 24.0 - tie_correction / 48.0;
        let std_w = var_w.max(0.0).sqrt();
        // Continuity correction: pull toward μ by 0.5 (less significant).
        // Since w = min, w ≤ μ → corrected = (w - μ + 0.5), capped at 0.
        let z = if std_w == 0.0 {
            0.0
        } else {
            let raw = w - mean_w;
            let corrected = (raw + 0.5).min(0.0); // raw ≤ 0 always; clamp on small-n overshoot
            corrected / std_w
        };
        let z_abs = z.abs();
        let p_one_tail = 1.0 - normal_cdf(z_abs);
        let p_value = (2.0 * p_one_tail).clamp(0.0, 1.0);
        out.push(WilcoxonResult {
            scorer,
            n,
            w_plus,
            w_minus,
            w,
            mean_diff,
            z,
            p_value,
            significant_at_05: p_value < 0.05,
            small_sample: n < 20,
        });
    }
    out.sort_by(|x, y| x.scorer.cmp(&y.scorer));
    out
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

    fn make_report(cases: Vec<EvalCaseResult>) -> EvalReport {
        let n_cases = cases.len();
        EvalReport {
            per_case: cases,
            aggregate: AggregateScores {
                n_cases,
                n_errors: 0,
                means: Map::new(),
            },
        }
    }

    #[test]
    fn no_change_chi_squared_zero() {
        // 5 cases, all pass in both → b=c=0, χ²=0.
        let cases = (0..5)
            .map(|i| make_case(&format!("q{i}"), &[("exact", 1.0)]))
            .collect();
        let baseline = make_report(cases);
        let current = baseline.clone();
        let res = mcnemar_test(&baseline, &current);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        assert_eq!(r.b, 0);
        assert_eq!(r.c, 0);
        assert_eq!(r.chi_squared, 0.0);
        assert_eq!(r.p_value, 1.0);
        assert!(!r.significant_at_05);
    }

    #[test]
    fn unbalanced_change_significant() {
        // 100 cases. In baseline, 50 pass + 50 fail. In current, the
        // 50 that failed in baseline now all pass; the 50 that
        // passed in baseline now all fail. Symmetric flip → b=c=50.
        // But that's NOT significant under McNemar's (b == c is the
        // null). Let's do an asymmetric change instead:
        // 50 stay pass; 30 flip from fail→pass; 0 flip pass→fail.
        // → a=50, b=0, c=30, d=20. χ² = (|0-30|-1)²/30 = 841/30 ≈ 28.03.
        // That's well above the 0.05 critical value of 3.841.
        let mut base_cases = Vec::new();
        let mut cur_cases = Vec::new();
        for i in 0..50 {
            // Pass in both.
            base_cases.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
            cur_cases.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
        }
        for i in 50..80 {
            // Improvement: fail in baseline, pass in current.
            base_cases.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
            cur_cases.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
        }
        for i in 80..100 {
            // Stable failure.
            base_cases.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
            cur_cases.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
        }
        let res = mcnemar_test(&make_report(base_cases), &make_report(cur_cases));
        let r = &res[0];
        assert_eq!(r.a, 50);
        assert_eq!(r.b, 0);
        assert_eq!(r.c, 30);
        assert_eq!(r.d, 20);
        assert!((r.chi_squared - 841.0 / 30.0).abs() < 1e-9);
        assert!(r.significant_at_05);
        assert!(!r.small_sample); // b+c = 30 >= 25
    }

    #[test]
    fn balanced_flip_not_significant() {
        // 10 flip pass→fail, 10 flip fail→pass. b == c → χ² ≈ 0
        // (with continuity correction: (|10-10|-1)² → max(0, -1)² = 0).
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..10 {
            base.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
            cur.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
        }
        for i in 10..20 {
            base.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
            cur.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
        }
        let res = mcnemar_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert_eq!(r.b, 10);
        assert_eq!(r.c, 10);
        assert!(!r.significant_at_05);
    }

    #[test]
    fn small_sample_flag_set_under_25() {
        // Only 4 flips total → b+c=4 < 25 → small_sample=true.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..4 {
            base.push(make_case(&format!("q{i}"), &[("exact", 0.0)]));
            cur.push(make_case(&format!("q{i}"), &[("exact", 1.0)]));
        }
        let res = mcnemar_test(&make_report(base), &make_report(cur));
        assert!(res[0].small_sample);
    }

    #[test]
    fn missing_in_current_skipped() {
        // q3 is in baseline but not in current — skipped silently.
        let baseline = make_report(vec![
            make_case("q1", &[("exact", 1.0)]),
            make_case("q2", &[("exact", 0.0)]),
            make_case("q3", &[("exact", 1.0)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("exact", 1.0)]),
            make_case("q2", &[("exact", 1.0)]),
        ]);
        let res = mcnemar_test(&baseline, &current);
        let r = &res[0];
        // Only q1 (a=1) and q2 (c=1) counted.
        assert_eq!(r.a, 1);
        assert_eq!(r.c, 1);
        assert_eq!(r.b, 0);
        assert_eq!(r.d, 0);
    }

    #[test]
    fn p_value_in_valid_range() {
        // Sanity: p_value always in [0, 1].
        let cases = (0..30)
            .map(|i| {
                make_case(
                    &format!("q{i}"),
                    &[(
                        "exact",
                        if i < 15 { 1.0 } else { 0.0 },
                    )],
                )
            })
            .collect();
        let res = mcnemar_test(&make_report(cases), &make_report(make_cases(30)));
        for r in &res {
            assert!(r.p_value >= 0.0);
            assert!(r.p_value <= 1.0);
        }
    }

    fn make_cases(n: usize) -> Vec<EvalCaseResult> {
        (0..n)
            .map(|i| make_case(&format!("q{i}"), &[("exact", 1.0)]))
            .collect()
    }

    #[test]
    fn multi_scorer_independent_results() {
        let baseline = make_report(vec![
            make_case("q1", &[("exact", 1.0), ("jaccard", 0.0)]),
            make_case("q2", &[("exact", 0.0), ("jaccard", 1.0)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("exact", 0.0), ("jaccard", 0.0)]),
            make_case("q2", &[("exact", 0.0), ("jaccard", 1.0)]),
        ]);
        let res = mcnemar_test(&baseline, &current);
        assert_eq!(res.len(), 2);
        // Sorted by scorer name.
        assert_eq!(res[0].scorer, "exact");
        assert_eq!(res[1].scorer, "jaccard");
        // exact: q1 was pass, now fail (b=1); q2 was fail, still fail (d=1).
        assert_eq!(res[0].b, 1);
        assert_eq!(res[0].d, 1);
        // jaccard: q1 was fail, still fail (d=1); q2 was pass, still pass (a=1).
        assert_eq!(res[1].a, 1);
        assert_eq!(res[1].d, 1);
    }

    #[test]
    fn normal_cdf_known_values() {
        // Φ(0) = 0.5, Φ(1.96) ≈ 0.975, Φ(-1.96) ≈ 0.025.
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7);
        assert!((normal_cdf(1.96) - 0.975).abs() < 5e-4);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 5e-4);
    }

    #[test]
    fn chi_squared_critical_value_matches_p05() {
        // χ²(1) critical value at p=0.05 is 3.841. Our function
        // should give p ≈ 0.05 there.
        let p = chi_squared_df1_p_value(3.841);
        assert!((p - 0.05).abs() < 0.01, "got {p}");
    }

    // ─── Wilcoxon signed-rank tests ────────────────────────────────

    #[test]
    fn wilcoxon_no_diffs_p_one() {
        // 5 cases, identical scores → all diffs = 0 → n=0 → p=1.
        let cases: Vec<EvalCaseResult> = (0..5)
            .map(|i| make_case(&format!("q{i}"), &[("cosine", 0.7)]))
            .collect();
        let baseline = make_report(cases);
        let current = baseline.clone();
        let res = wilcoxon_signed_rank_test(&baseline, &current);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        assert_eq!(r.n, 0);
        assert_eq!(r.w, 0.0);
        assert_eq!(r.p_value, 1.0);
        assert!(!r.significant_at_05);
        assert!(r.small_sample);
        assert!((r.mean_diff - 0.0).abs() < 1e-12);
    }

    #[test]
    fn wilcoxon_uniform_improvement_significant() {
        // 25 cases, every diff = +0.1. n=25 (≥ 20 → not small_sample).
        // All abs diffs equal → all ranks = (1 + 25)/2 = 13. W- = 0, W+ = 25*13 = 325.
        // W = 0. Hugely significant.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..25 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert_eq!(r.n, 25);
        assert!(!r.small_sample);
        assert!((r.w_minus - 0.0).abs() < 1e-9);
        assert!((r.w_plus - 325.0).abs() < 1e-9);
        assert_eq!(r.w, 0.0);
        assert!(r.significant_at_05);
        assert!(r.mean_diff > 0.099 && r.mean_diff < 0.101);
    }

    #[test]
    fn wilcoxon_uniform_regression_significant() {
        // Mirror of the above with diffs all -0.1. W+ = 0, W = 0, significant,
        // mean_diff = -0.1.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..25 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!(r.significant_at_05);
        assert!(r.mean_diff < -0.099 && r.mean_diff > -0.101);
        assert!((r.w_plus - 0.0).abs() < 1e-9);
    }

    #[test]
    fn wilcoxon_balanced_change_not_significant() {
        // 10 cases improve by +0.1, 10 regress by -0.1 (same magnitude).
        // All 20 abs diffs tied → all rank = 10.5. W+ = 105, W- = 105, W = 105.
        // mean_w = 20*21/4 = 105 → z = 0 → p = 1.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..10 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
        }
        for i in 10..20 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.7)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert!(!r.significant_at_05);
        assert!((r.w - 105.0).abs() < 1e-9);
        assert!((r.mean_diff - 0.0).abs() < 1e-9);
    }

    #[test]
    fn wilcoxon_small_sample_flag() {
        // n=10 (< 20) → small_sample=true.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..10 {
            base.push(make_case(&format!("q{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", 0.6)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        assert_eq!(res[0].n, 10);
        assert!(res[0].small_sample);
    }

    #[test]
    fn wilcoxon_textbook_example() {
        // Diffs: [+3, -1, +5, -2, +4, +2, -7]. n=7.
        // Abs sorted with ranks (handle ties for the two |2|s):
        //   |1|=1, |2|=2.5, |2|=2.5, |3|=4, |4|=5, |5|=6, |7|=7.
        // Original index → rank:
        //   diff[0]=+3→4, diff[1]=-1→1, diff[2]=+5→6, diff[3]=-2→2.5,
        //   diff[4]=+4→5, diff[5]=+2→2.5, diff[6]=-7→7.
        // W+ = 4 + 6 + 5 + 2.5 = 17.5
        // W- = 1 + 2.5 + 7 = 10.5
        // W = min = 10.5.
        let pairs = [
            (0.5, 0.8), // +0.3 → scaled diff +3 (multiply both by 10 below)
            (0.5, 0.4), // -0.1 → -1
            (0.0, 0.5), // +0.5 → +5
            (0.5, 0.3), // -0.2 → -2
            (0.0, 0.4), // +0.4 → +4
            (0.0, 0.2), // +0.2 → +2
            (0.7, 0.0), // -0.7 → -7
        ];
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for (i, &(b, c)) in pairs.iter().enumerate() {
            base.push(make_case(&format!("q{i}"), &[("cosine", b)]));
            cur.push(make_case(&format!("q{i}"), &[("cosine", c)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert_eq!(r.n, 7);
        assert!((r.w_plus - 17.5).abs() < 1e-9, "w_plus={}", r.w_plus);
        assert!((r.w_minus - 10.5).abs() < 1e-9, "w_minus={}", r.w_minus);
        assert!((r.w - 10.5).abs() < 1e-9);
        assert!(r.small_sample);
        // Not significant at 0.05 (textbook example).
        assert!(!r.significant_at_05);
    }

    #[test]
    fn wilcoxon_missing_in_current_skipped() {
        // q3 only in baseline — skipped silently.
        let baseline = make_report(vec![
            make_case("q1", &[("cosine", 0.5)]),
            make_case("q2", &[("cosine", 0.5)]),
            make_case("q3", &[("cosine", 0.9)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("cosine", 0.6)]),
            make_case("q2", &[("cosine", 0.6)]),
        ]);
        let res = wilcoxon_signed_rank_test(&baseline, &current);
        assert_eq!(res[0].n, 2); // only the two pairs that match
    }

    #[test]
    fn wilcoxon_zero_diffs_dropped_n_excludes_them() {
        // 3 zero-diff cases + 4 +0.1-diff cases. n should be 4, not 7.
        let mut base = Vec::new();
        let mut cur = Vec::new();
        for i in 0..3 {
            base.push(make_case(&format!("z{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("z{i}"), &[("cosine", 0.5)]));
        }
        for i in 0..4 {
            base.push(make_case(&format!("p{i}"), &[("cosine", 0.5)]));
            cur.push(make_case(&format!("p{i}"), &[("cosine", 0.6)]));
        }
        let res = wilcoxon_signed_rank_test(&make_report(base), &make_report(cur));
        let r = &res[0];
        assert_eq!(r.n, 4);
        // mean_diff includes ALL 7 cases: (3*0 + 4*0.1) / 7 ≈ 0.0571
        assert!((r.mean_diff - 0.4 / 7.0).abs() < 1e-9, "got {}", r.mean_diff);
    }

    #[test]
    fn wilcoxon_per_scorer_sorted() {
        // Two scorers — output sorted by name (alpha < beta).
        let baseline = make_report(vec![
            make_case("q1", &[("alpha", 0.5), ("beta", 0.5)]),
            make_case("q2", &[("alpha", 0.5), ("beta", 0.5)]),
        ]);
        let current = make_report(vec![
            make_case("q1", &[("alpha", 0.6), ("beta", 0.4)]),
            make_case("q2", &[("alpha", 0.7), ("beta", 0.3)]),
        ]);
        let res = wilcoxon_signed_rank_test(&baseline, &current);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].scorer, "alpha");
        assert_eq!(res[1].scorer, "beta");
        assert!(res[0].mean_diff > 0.0); // alpha improved
        assert!(res[1].mean_diff < 0.0); // beta regressed
    }
}

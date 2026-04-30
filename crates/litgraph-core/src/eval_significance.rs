//! `mcnemar_test` — statistical-significance test for paired
//! binary outcomes between two eval runs. Pairs with iter-289
//! `detect_drift`: drift says "X regressions, Y improvements";
//! this says "is X-Y statistically meaningful or just noise?"
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
}

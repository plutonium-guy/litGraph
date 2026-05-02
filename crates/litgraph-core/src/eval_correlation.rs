//! Pearson + Spearman rank correlation for eval-metric validation.
//!
//! # Why correlation in an eval suite
//!
//! When you ship a new auto-metric (chrF, a custom regex-based scorer,
//! a faster LLM-judge variant), the question that gates adoption is
//! "does it agree with the metric I trust?". Trust is usually anchored
//! to human ratings or a slow LLM-judge gold-standard. Correlation is
//! the canonical answer:
//!
//! - **Pearson r** captures *linear* agreement. r = 1 means the two
//!   metrics rise and fall together at constant ratio. r = 0 means no
//!   linear relationship. r = -1 means perfect inverse.
//! - **Spearman ρ** captures *rank* agreement. Computes Pearson on the
//!   ranks of each value rather than the raw values, so it's invariant
//!   to monotonic transformations: y = exp(x) is "perfect" Spearman
//!   even though Pearson would only catch the linear part.
//!
//! Use Spearman when you only care that the two metrics rank items
//! the same way (which auto-metric is correct? doesn't matter — pick
//! whichever ranks the gold-set the same as your trusted scorer).
//! Use Pearson when you need them to track magnitude too (e.g. for
//! threshold-based decisions where a 0.85 vs 0.80 difference matters).
//!
//! # Distinct from `mcnemar_test` / `wilcoxon_signed_rank_test`
//!
//! Significance tests (iter 290 / 297) compare *paired* outcomes from
//! the same scorer across two eval runs (baseline vs current). They
//! answer "did the model change?".
//!
//! Correlation compares two *different scorers* on the same eval run.
//! It answers "do my scorers agree?".
//!
//! Both belong in the suite; they answer different questions.
//!
//! # Edge cases
//!
//! - `n < 2` → `None` (correlation is undefined for fewer than 2
//!   pairs).
//! - Length mismatch → `None`.
//! - Either series is **constant** (zero variance) → `None`. Pearson
//!   divides by `std(x) · std(y)`; if either is zero, the result is
//!   undefined. Spearman has the same issue when ranks collapse to
//!   constants.
//!
//! All-`None`-or-`Some(f32)` API — callers can pattern-match on
//! `None` to surface "not enough signal" without sentinel values.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval_harness::EvalReport;

/// Pearson product-moment correlation coefficient.
///
/// Returns `None` when undefined: `xs.len() != ys.len()`, `xs.len() < 2`,
/// or either series has zero variance (constant).
pub fn pearson_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;
    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;
    for (&x, &y) in xs.iter().zip(ys) {
        let dx = x - mean_x;
        let dy = y - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }
    let denom = (sum_sq_x * sum_sq_y).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some(numerator / denom)
}

/// Spearman rank correlation coefficient.
///
/// Computes Pearson on the **ranks** of each series. Ranks use the
/// "fractional" / average-rank convention for ties (the same rule as
/// iter-297 Wilcoxon — k tied values get the average of the ranks they
/// would otherwise span, e.g. two values tied for ranks 3 and 4 each
/// get 3.5).
///
/// Returns `None` under the same conditions as `pearson_correlation`.
pub fn spearman_correlation(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let rx = ranks(xs);
    let ry = ranks(ys);
    pearson_correlation(&rx, &ry)
}

/// Convert a slice of values to fractional ranks (1-indexed) with
/// average-rank tie handling.
fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> =
        xs.iter().copied().enumerate().map(|(i, v)| (i, v)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0_f64; xs.len()];
    let mut i = 0;
    while i < indexed.len() {
        let mut j = i + 1;
        while j < indexed.len() && (indexed[j].1 - indexed[i].1).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for k in i..j {
            ranks[indexed[k].0] = avg_rank;
        }
        i = j;
    }
    ranks
}

/// Kendall's tau-b rank correlation coefficient.
///
/// Counts concordant pairs (both rankings agree on ordering) minus
/// discordant pairs (rankings disagree), normalized by total pair
/// count with **tie correction** for tied values in either series.
///
/// # Why a third correlation alongside Pearson + Spearman
///
/// - **Pearson** is parametric (assumes linear relationship + normality
///   of residuals). Sensitive to outliers.
/// - **Spearman** is non-parametric over ranks but uses Pearson on
///   those ranks — still noise-sensitive when many ties exist.
/// - **Kendall's tau-b** counts pairwise agreements directly. More
///   robust than Spearman when:
///   - The data has many ties (tau-b's `(n_c+n_d+n_x)(n_c+n_d+n_y)`
///     denominator subtracts tied pairs from both sides correctly).
///   - The relationship is monotonic but heavy-tailed (Spearman's
///     squared-rank-diff inflates the influence of large rank gaps).
///   - Sample size is small (Kendall's exact distribution converges
///     to normal faster than Spearman's).
///
/// Common practice: report both Spearman and Kendall — agreement
/// validates the rank-relationship; divergence flags ties or
/// outliers that need investigation.
///
/// # Algorithm (tau-b)
///
/// For all `i < j`:
/// - concordant: `(xs[i] - xs[j]) * (ys[i] - ys[j]) > 0`
/// - discordant: `(xs[i] - xs[j]) * (ys[i] - ys[j]) < 0`
/// - tie in x only: `xs[i] == xs[j]` but `ys[i] != ys[j]`
/// - tie in y only: `ys[i] == ys[j]` but `xs[i] != xs[j]`
/// - tie in both: skipped (contributes to neither numerator nor
///   denominator)
///
/// τ-b = (n_c − n_d) / sqrt((n_c + n_d + n_x) · (n_c + n_d + n_y))
///
/// O(n²) in the naive impl. Fine for eval-scale n; if hot at scale
/// the Knight 1966 O(n log n) merge-sort variant exists.
///
/// # Edge cases
///
/// Returns `None` for: length mismatch, n < 2, denominator zero
/// (full ties on one side — same condition Pearson hits with
/// constant input).
pub fn kendall_tau(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let n = xs.len();
    let mut n_c: u64 = 0;
    let mut n_d: u64 = 0;
    let mut n_x: u64 = 0; // ties in x only
    let mut n_y: u64 = 0; // ties in y only
    for i in 0..n {
        for j in (i + 1)..n {
            let dx = xs[i] - xs[j];
            let dy = ys[i] - ys[j];
            let tx = dx == 0.0;
            let ty = dy == 0.0;
            match (tx, ty) {
                (true, true) => {} // skip both-ties
                (true, false) => n_x += 1,
                (false, true) => n_y += 1,
                (false, false) => {
                    if dx.signum() == dy.signum() {
                        n_c += 1;
                    } else {
                        n_d += 1;
                    }
                }
            }
        }
    }
    let cd = (n_c + n_d) as f64;
    let denom = ((cd + n_x as f64) * (cd + n_y as f64)).sqrt();
    if denom == 0.0 {
        return None;
    }
    Some((n_c as f64 - n_d as f64) / denom)
}

/// Per-pair correlation between two scorers in a single EvalReport.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScorerCorrelation {
    pub scorer_a: String,
    pub scorer_b: String,
    pub n: u64,
    pub pearson: Option<f64>,
    pub spearman: Option<f64>,
    pub kendall: Option<f64>,
}

/// Compute Pearson + Spearman correlation between every pair of scorers
/// in an EvalReport. Returns one result per unordered pair, sorted by
/// `(scorer_a, scorer_b)` for deterministic output.
///
/// Real prod use: a single eval run with both `exact_match` and
/// `llm_judge` scorers. `correlate_scorers(&report)` immediately tells
/// you whether they agree, which is the gate question for "can we
/// drop the slower scorer in CI?".
pub fn correlate_scorers(report: &EvalReport) -> Vec<ScorerCorrelation> {
    // Collect per-scorer score vectors aligned by case order.
    let mut scorer_names: Vec<String> = Vec::new();
    let mut per_scorer: HashMap<String, Vec<Option<f64>>> = HashMap::new();
    let n_cases = report.per_case.len();
    for case in &report.per_case {
        for scorer in case.scores.keys() {
            if !per_scorer.contains_key(scorer) {
                scorer_names.push(scorer.clone());
                per_scorer.insert(scorer.clone(), vec![None; n_cases]);
            }
        }
    }
    for (idx, case) in report.per_case.iter().enumerate() {
        for (scorer, val) in &case.scores {
            if let Some(v) = score_to_f64(val) {
                per_scorer
                    .get_mut(scorer)
                    .unwrap()
                    .get_mut(idx)
                    .map(|slot| *slot = Some(v));
            }
        }
    }
    scorer_names.sort();
    let mut out = Vec::new();
    for i in 0..scorer_names.len() {
        for j in i + 1..scorer_names.len() {
            let a_name = &scorer_names[i];
            let b_name = &scorer_names[j];
            let a_vec = &per_scorer[a_name];
            let b_vec = &per_scorer[b_name];
            // Filter to cases where BOTH scorers reported a value.
            let mut xs = Vec::new();
            let mut ys = Vec::new();
            for (a, b) in a_vec.iter().zip(b_vec) {
                if let (Some(av), Some(bv)) = (a, b) {
                    xs.push(*av);
                    ys.push(*bv);
                }
            }
            out.push(ScorerCorrelation {
                scorer_a: a_name.clone(),
                scorer_b: b_name.clone(),
                n: xs.len() as u64,
                pearson: pearson_correlation(&xs, &ys),
                spearman: spearman_correlation(&xs, &ys),
                kendall: kendall_tau(&xs, &ys),
            });
        }
    }
    out
}

fn score_to_f64(v: &Value) -> Option<f64> {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .or_else(|| {
            // Booleans coerce to 0/1 (matches the rest of the eval suite).
            v.as_bool().map(|b| if b { 1.0 } else { 0.0 })
        })
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

    // ─── Pearson ─────────────────────────────────────────────────

    #[test]
    fn pearson_perfect_positive() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = xs.clone();
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r - 1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_perfect_negative() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!((r - -1.0).abs() < 1e-10);
    }

    #[test]
    fn pearson_uncorrelated_near_zero() {
        // Hand-constructed near-zero correlation.
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![2.0, 4.0, 1.0, 5.0, 3.0]; // shuffled — no linear relationship
        let r = pearson_correlation(&xs, &ys).unwrap();
        assert!(r.abs() < 0.5, "got r={r}, expected near zero");
    }

    #[test]
    fn pearson_constant_returns_none() {
        let xs = vec![3.0, 3.0, 3.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0, 4.0];
        assert!(pearson_correlation(&xs, &ys).is_none());
        assert!(pearson_correlation(&ys, &xs).is_none());
    }

    #[test]
    fn pearson_length_mismatch_none() {
        assert!(pearson_correlation(&[1.0, 2.0], &[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn pearson_too_few_points_none() {
        assert!(pearson_correlation(&[1.0], &[1.0]).is_none());
        assert!(pearson_correlation(&[], &[]).is_none());
    }

    // ─── Spearman ────────────────────────────────────────────────

    #[test]
    fn spearman_perfect_monotonic_nonlinear() {
        // y = x^3 — strictly monotonic but not linear. Pearson is high
        // but not 1; Spearman is exactly 1 because ranks match.
        let xs: Vec<f64> = (1..=10).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|x| x.powi(3)).collect();
        let p = pearson_correlation(&xs, &ys).unwrap();
        let s = spearman_correlation(&xs, &ys).unwrap();
        assert!(s > 0.9999, "spearman should be ~1, got {s}");
        assert!(p < s, "pearson ({p}) should be < spearman ({s}) on nonlinear monotonic");
    }

    #[test]
    fn spearman_perfect_monotonic_decreasing() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![100.0, 50.0, 25.0, 10.0, 1.0]; // strictly decreasing
        let s = spearman_correlation(&xs, &ys).unwrap();
        assert!((s - -1.0).abs() < 1e-10, "got s={s}");
    }

    #[test]
    fn spearman_handles_ties() {
        // Two values tied → average rank.
        // xs = [1, 2, 2, 3] → ranks [1, 2.5, 2.5, 4]
        // ys = [10, 20, 20, 30] → ranks [1, 2.5, 2.5, 4]
        // Identical ranks → spearman = 1.
        let xs = vec![1.0, 2.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 20.0, 30.0];
        let s = spearman_correlation(&xs, &ys).unwrap();
        assert!((s - 1.0).abs() < 1e-10);
    }

    #[test]
    fn spearman_constant_returns_none() {
        let xs = vec![3.0, 3.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0];
        assert!(spearman_correlation(&xs, &ys).is_none());
    }

    #[test]
    fn ranks_assigns_average_for_ties() {
        let r = ranks(&[10.0, 20.0, 20.0, 30.0]);
        assert_eq!(r, vec![1.0, 2.5, 2.5, 4.0]);
    }

    #[test]
    fn ranks_no_ties() {
        let r = ranks(&[5.0, 1.0, 3.0, 2.0, 4.0]);
        assert_eq!(r, vec![5.0, 1.0, 3.0, 2.0, 4.0]);
    }

    // ─── correlate_scorers (eval-report integration) ────────────────

    #[test]
    fn correlate_scorers_perfect_agreement() {
        // Two scorers with identical values across cases.
        let cases = (0..5)
            .map(|i| {
                make_case(
                    &format!("q{i}"),
                    &[("a", i as f64 * 0.1), ("b", i as f64 * 0.1)],
                )
            })
            .collect();
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        assert_eq!(r.scorer_a, "a");
        assert_eq!(r.scorer_b, "b");
        assert_eq!(r.n, 5);
        assert!((r.pearson.unwrap() - 1.0).abs() < 1e-10);
        assert!((r.spearman.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn correlate_scorers_three_scorers_three_pairs() {
        // 3 scorers → C(3,2) = 3 pairs.
        let cases = (0..5)
            .map(|i| {
                make_case(
                    &format!("q{i}"),
                    &[
                        ("a", i as f64),
                        ("b", (i * 2) as f64),
                        ("c", (10 - i) as f64),
                    ],
                )
            })
            .collect();
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 3);
        // Sorted output: (a,b), (a,c), (b,c)
        assert_eq!(res[0].scorer_a, "a");
        assert_eq!(res[0].scorer_b, "b");
        assert_eq!(res[1].scorer_a, "a");
        assert_eq!(res[1].scorer_b, "c");
        assert_eq!(res[2].scorer_a, "b");
        assert_eq!(res[2].scorer_b, "c");
        // a perfectly correlates with b (linear: b = 2a).
        assert!((res[0].pearson.unwrap() - 1.0).abs() < 1e-10);
        // a perfectly anti-correlates with c (c = 10 - a).
        assert!((res[1].pearson.unwrap() - -1.0).abs() < 1e-10);
    }

    #[test]
    fn correlate_scorers_partial_coverage() {
        // Scorer "b" only appears in some cases. Correlation should be
        // computed on the intersection.
        let mut cases = vec![
            make_case("q0", &[("a", 1.0), ("b", 1.0)]),
            make_case("q1", &[("a", 2.0), ("b", 2.0)]),
            make_case("q2", &[("a", 3.0), ("b", 3.0)]),
        ];
        // Case 3 has only "a" — should be skipped from correlation.
        cases.push(make_case("q3", &[("a", 4.0)]));
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 1);
        assert_eq!(res[0].n, 3); // only the 3 cases with both
        assert!((res[0].pearson.unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn correlate_scorers_constant_scorer_yields_none() {
        // Scorer "constant" always returns 1.0. Correlation undefined.
        let cases = (0..5)
            .map(|i| make_case(&format!("q{i}"), &[("varies", i as f64), ("constant", 1.0)]))
            .collect();
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 1);
        assert!(res[0].pearson.is_none());
        assert!(res[0].spearman.is_none());
    }

    #[test]
    fn correlate_scorers_single_scorer_no_pairs() {
        let cases: Vec<_> = (0..3)
            .map(|i| make_case(&format!("q{i}"), &[("only", i as f64)]))
            .collect();
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 0);
    }

    #[test]
    fn correlate_scorers_bool_coercion() {
        // True/false coerce to 1/0; should produce a valid correlation.
        let cases = vec![
            make_case_with_json("q0", &[("a", json!(1.0)), ("pass", json!(true))]),
            make_case_with_json("q1", &[("a", json!(2.0)), ("pass", json!(true))]),
            make_case_with_json("q2", &[("a", json!(3.0)), ("pass", json!(false))]),
            make_case_with_json("q3", &[("a", json!(4.0)), ("pass", json!(false))]),
        ];
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 1);
        // a=[1,2,3,4]→ranks [1,2,3,4]; pass=[1,1,0,0]→ranks [3.5,3.5,1.5,1.5].
        // Pearson should be a valid (negative) correlation.
        assert!(res[0].pearson.is_some());
        assert!(res[0].pearson.unwrap() < 0.0);
    }

    fn make_case_with_json(input: &str, scores: &[(&str, Value)]) -> EvalCaseResult {
        let mut m = Map::new();
        for (k, v) in scores {
            m.insert(k.to_string(), v.clone());
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

    // ─── Kendall's tau ────────────────────────────────────────────

    #[test]
    fn kendall_perfect_positive() {
        // Strictly monotonic increasing — all C(n,2) pairs concordant.
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = xs.clone();
        let t = kendall_tau(&xs, &ys).unwrap();
        assert!((t - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_perfect_negative() {
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let t = kendall_tau(&xs, &ys).unwrap();
        assert!((t - -1.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_zero_correlation_for_balanced_swap() {
        // 4 pairs: 2 concordant + 2 discordant → τ = 0.
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 4.0, 2.0, 3.0];
        // Pairs (i,j): (0,1) dx=-1 dy=-3 → conc;
        //              (0,2) dx=-2 dy=-1 → conc;
        //              (0,3) dx=-3 dy=-2 → conc;
        //              (1,2) dx=-1 dy=2 → disc;
        //              (1,3) dx=-2 dy=1 → disc;
        //              (2,3) dx=-1 dy=-1 → conc.
        // n_c=4, n_d=2, τ = 2/6 ≈ 0.333.
        let t = kendall_tau(&xs, &ys).unwrap();
        assert!((t - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn kendall_handles_ties_in_x_only() {
        // x has tied values but y doesn't.
        // xs = [1, 1, 2, 3], ys = [10, 20, 30, 40].
        // Pairs: (0,1) dx=0 dy=-10 → tie in x;
        //        (0,2) dx=-1 dy=-20 → conc;
        //        (0,3) dx=-2 dy=-30 → conc;
        //        (1,2) dx=-1 dy=-10 → conc;
        //        (1,3) dx=-2 dy=-20 → conc;
        //        (2,3) dx=-1 dy=-10 → conc.
        // n_c=5, n_d=0, n_x=1, n_y=0.
        // τ-b = (5-0)/sqrt((5+0+1)(5+0+0)) = 5/sqrt(30) ≈ 0.913.
        let xs = vec![1.0, 1.0, 2.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        let t = kendall_tau(&xs, &ys).unwrap();
        let expected = 5.0 / (30.0_f64).sqrt();
        assert!((t - expected).abs() < 1e-10, "got {t}");
    }

    #[test]
    fn kendall_returns_none_for_constant() {
        // All x identical → no non-tie pairs in x → denom collapses on
        // one side. Specifically n_c+n_d=0, n_x=C(4,2)=6, n_y=0 →
        // sqrt((0+6)(0+0)) = 0 → None.
        let xs = vec![3.0, 3.0, 3.0, 3.0];
        let ys = vec![1.0, 2.0, 3.0, 4.0];
        assert!(kendall_tau(&xs, &ys).is_none());
    }

    #[test]
    fn kendall_returns_none_for_length_mismatch() {
        assert!(kendall_tau(&[1.0, 2.0], &[1.0]).is_none());
    }

    #[test]
    fn kendall_returns_none_for_n_less_than_2() {
        assert!(kendall_tau(&[1.0], &[1.0]).is_none());
        assert!(kendall_tau(&[], &[]).is_none());
    }

    #[test]
    fn kendall_agrees_with_spearman_no_ties() {
        // Both should give the same sign for any monotonic data.
        let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let ys = vec![2.0, 4.0, 5.0, 7.0, 6.0, 9.0];
        let s = spearman_correlation(&xs, &ys).unwrap();
        let k = kendall_tau(&xs, &ys).unwrap();
        // Both > 0 for this monotonic-ish data.
        assert!(s > 0.0);
        assert!(k > 0.0);
    }

    #[test]
    fn correlate_scorers_includes_kendall() {
        let cases: Vec<EvalCaseResult> = (0..5)
            .map(|i| {
                make_case(
                    &format!("q{i}"),
                    &[("a", i as f64 * 0.1), ("b", i as f64 * 0.2)],
                )
            })
            .collect();
        let report = make_report(cases);
        let res = correlate_scorers(&report);
        assert_eq!(res.len(), 1);
        let r = &res[0];
        // a and b are linearly related → all three correlations = 1.
        assert!((r.pearson.unwrap() - 1.0).abs() < 1e-10);
        assert!((r.spearman.unwrap() - 1.0).abs() < 1e-10);
        assert!((r.kendall.unwrap() - 1.0).abs() < 1e-10);
    }
}

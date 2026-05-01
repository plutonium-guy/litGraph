//! Bootstrap confidence intervals for eval scores.
//!
//! # The problem with point estimates
//!
//! `EvalReport.aggregate.means.cosine = 0.83` is a single number.
//! Whether that's "0.83 ± 0.001" (rock-solid) or "0.83 ± 0.15"
//! (one or two cases swing the whole average) is invisible from
//! the point estimate alone. The user reading the changelog has
//! no way to tell.
//!
//! Bootstrap solves this without distributional assumptions: resample
//! the eval-case scores with replacement N times, compute the mean of
//! each resample, take the (α/2, 1-α/2) percentiles of those bootstrap
//! means as the CI bounds.
//!
//! # Why this not a t-distribution CI
//!
//! A t-CI assumes the underlying score distribution is roughly Normal.
//! Eval scores often aren't (LLM-judge scores cluster near 0 or 1;
//! exact-match is binary; cosine is bounded [-1, 1] with skewed
//! tails). Bootstrap is non-parametric — works for any score
//! distribution, including skewed and bounded ones.
//!
//! # Why this not analytical SE-of-the-mean
//!
//! `SE = std/√n` works for the Normal mean. For functionals of the
//! data (median, percentile, Cohen's d, AUC, …) the analytical SE
//! gets messy. Bootstrap gives a CI for ANY statistic via the same
//! recipe — though this iter ships only the mean to keep the API
//! tight. Future iters can extend to median / percentile bootstraps
//! using the same xorshift PRNG.
//!
//! # Why xorshift and not the `rand` crate
//!
//! Bootstrap doesn't need cryptographic randomness — only enough
//! statistical quality to make resamples uncorrelated. xorshift64
//! has a 2^64−1 period, passes the standard chi-squared / runs
//! tests, fits in 4 lines of code, and avoids pulling `rand` into
//! `litgraph-core`'s dependency tree.
//!
//! # Reproducibility
//!
//! Seed-based. Same `(values, n_resamples, confidence, seed)` →
//! bit-identical CI bounds on every run. CI eval reports stay
//! diffable across runs.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::eval_harness::EvalReport;

/// One bootstrap-CI result for a (scorer, paired-cases) pair.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub scorer: String,
    pub n: u64,
    /// Sample mean (the point estimate the CI is around).
    pub mean: f64,
    /// Lower bound at `(1 - confidence) / 2` percentile.
    pub lower: f64,
    /// Upper bound at `1 - (1 - confidence) / 2` percentile.
    pub upper: f64,
    /// Confidence level (0..1). Echoed for downstream tooling.
    pub confidence: f64,
    /// Number of bootstrap resamples used.
    pub n_resamples: u64,
}

/// Bootstrap CI for the mean of every scorer in an EvalReport.
///
/// `confidence` is clamped to `(0.0, 1.0)`. `n_resamples` is clamped
/// to `>= 1`. The default suggestion for production reports is
/// `n_resamples = 1000` and `confidence = 0.95`. Reduce resamples
/// for snapshot-test speed (100 is enough to verify the API works);
/// raise to 10_000 for tight bounds on critical reports.
pub fn bootstrap_eval_ci(
    report: &EvalReport,
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> Vec<ConfidenceInterval> {
    let confidence = confidence.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    let n_resamples = n_resamples.max(1);
    // Collect per-scorer scores.
    let mut per_scorer: HashMap<String, Vec<f64>> = HashMap::new();
    for case in &report.per_case {
        for (scorer, val) in &case.scores {
            per_scorer
                .entry(scorer.clone())
                .or_default()
                .push(score_to_f64(val));
        }
    }
    let mut out = Vec::with_capacity(per_scorer.len());
    for (scorer, scores) in per_scorer {
        let (mean, lower, upper) = bootstrap_mean_ci(&scores, n_resamples, confidence, seed);
        out.push(ConfidenceInterval {
            scorer,
            n: scores.len() as u64,
            mean,
            lower,
            upper,
            confidence,
            n_resamples: n_resamples as u64,
        });
    }
    out.sort_by(|a, b| a.scorer.cmp(&b.scorer));
    out
}

/// Bootstrap CI for the mean of a `&[f64]` slice. Returns `(mean,
/// lower, upper)`. Public for callers that want a CI on a metric
/// outside the eval harness — e.g. latencies from a tracing run,
/// per-request token counts, etc.
///
/// # Behavior at edges
///
/// - Empty slice: returns `(0.0, 0.0, 0.0)`.
/// - Single value: bootstrap collapses to that value → CI is
///   `(value, value, value)`. The CI doesn't lie about uncertainty;
///   it just has nothing to draw from.
/// - All-identical values: same — CI collapses to the shared value.
pub fn bootstrap_mean_ci(
    values: &[f64],
    n_resamples: usize,
    confidence: f64,
    seed: u64,
) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let n_resamples = n_resamples.max(1);
    let confidence = confidence.clamp(f64::EPSILON, 1.0 - f64::EPSILON);
    // Run resamples.
    let mut rng = Xorshift64::new(seed);
    let mut means: Vec<f64> = Vec::with_capacity(n_resamples);
    let n = values.len();
    for _ in 0..n_resamples {
        let mut sum = 0.0;
        for _ in 0..n {
            let idx = rng.next_in(n);
            sum += values[idx];
        }
        means.push(sum / n as f64);
    }
    // Percentile method.
    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let alpha = 1.0 - confidence;
    let lo_idx = ((alpha / 2.0) * (n_resamples as f64 - 1.0)).round() as usize;
    let hi_idx = ((1.0 - alpha / 2.0) * (n_resamples as f64 - 1.0)).round() as usize;
    let lower = means[lo_idx.min(n_resamples - 1)];
    let upper = means[hi_idx.min(n_resamples - 1)];
    (mean, lower, upper)
}

fn score_to_f64(v: &Value) -> f64 {
    v.as_f64()
        .or_else(|| v.as_i64().map(|i| i as f64))
        .or_else(|| v.as_u64().map(|u| u as f64))
        .unwrap_or(0.0)
}

/// xorshift64 PRNG. Marsaglia 2003 "Xorshift RNGs".
/// Period 2^64 - 1; passes standard statistical quality tests.
/// Good enough for bootstrap resampling; not cryptographically secure.
struct Xorshift64(u64);

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Seed of 0 is degenerate for xorshift (stays at 0). Substitute
        // a non-zero seed (the golden-ratio mix constant) so callers
        // can pass `seed: 0` without surprise.
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn next_in(&mut self, n: usize) -> usize {
        // n > 0 enforced by caller (always ≥ 1).
        (self.next_u64() as usize) % n
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
    fn empty_input_returns_zeros() {
        let (mean, lo, hi) = bootstrap_mean_ci(&[], 100, 0.95, 42);
        assert_eq!(mean, 0.0);
        assert_eq!(lo, 0.0);
        assert_eq!(hi, 0.0);
    }

    #[test]
    fn all_identical_collapses_ci() {
        // 10 copies of 0.5 → bootstrap means are all 0.5 → CI = (0.5, 0.5, 0.5).
        let xs = vec![0.5; 10];
        let (mean, lo, hi) = bootstrap_mean_ci(&xs, 200, 0.95, 7);
        assert!((mean - 0.5).abs() < 1e-12);
        assert!((lo - 0.5).abs() < 1e-12);
        assert!((hi - 0.5).abs() < 1e-12);
    }

    #[test]
    fn single_value_ci_is_point() {
        let (mean, lo, hi) = bootstrap_mean_ci(&[0.7], 50, 0.95, 1);
        assert_eq!(mean, 0.7);
        assert_eq!(lo, 0.7);
        assert_eq!(hi, 0.7);
    }

    #[test]
    fn ci_brackets_true_mean_for_uniform_data() {
        // 100 evenly-spaced values 0.0, 0.01, …, 0.99 → mean = 0.495.
        // 1000 resamples, 95% CI should bracket the true mean.
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let (mean, lo, hi) = bootstrap_mean_ci(&xs, 1000, 0.95, 12345);
        assert!((mean - 0.495).abs() < 1e-9);
        assert!(lo < mean, "lo={} mean={}", lo, mean);
        assert!(hi > mean, "hi={} mean={}", hi, mean);
        // CI should be reasonably tight (not the full 0..1 range).
        assert!((hi - lo) < 0.2, "ci width={}", hi - lo);
    }

    #[test]
    fn same_seed_reproducible() {
        let xs: Vec<f64> = (0..50).map(|i| (i as f64) / 50.0).collect();
        let a = bootstrap_mean_ci(&xs, 500, 0.95, 999);
        let b = bootstrap_mean_ci(&xs, 500, 0.95, 999);
        assert_eq!(a, b);
    }

    #[test]
    fn different_seeds_different_bounds() {
        let xs: Vec<f64> = (0..50).map(|i| (i as f64) / 50.0).collect();
        let (m1, lo1, hi1) = bootstrap_mean_ci(&xs, 500, 0.95, 1);
        let (m2, lo2, hi2) = bootstrap_mean_ci(&xs, 500, 0.95, 2);
        // Same data → same point mean.
        assert!((m1 - m2).abs() < 1e-12);
        // Different seeds → bounds differ at least slightly (huge sample size
        // could converge, but with 500 resamples the percentile picks differ).
        assert!(lo1 != lo2 || hi1 != hi2);
    }

    #[test]
    fn higher_confidence_widens_ci() {
        let xs: Vec<f64> = (0..50).map(|i| (i as f64) / 50.0).collect();
        let (_, lo90, hi90) = bootstrap_mean_ci(&xs, 1000, 0.90, 7);
        let (_, lo99, hi99) = bootstrap_mean_ci(&xs, 1000, 0.99, 7);
        let w90 = hi90 - lo90;
        let w99 = hi99 - lo99;
        assert!(w99 > w90, "99% CI ({w99}) should be wider than 90% ({w90})");
    }

    #[test]
    fn n_resamples_zero_clamps_to_one() {
        let (mean, lo, hi) = bootstrap_mean_ci(&[0.4, 0.6], 0, 0.95, 5);
        // Clamped to 1 resample → still produces a valid result.
        // Mean should be 0.5; CI is one of [0.4, 0.6, average].
        assert!((mean - 0.5).abs() < 1e-12);
        assert!((0.4..=0.6).contains(&lo));
        assert!((0.4..=0.6).contains(&hi));
    }

    #[test]
    fn confidence_clamped_in_range() {
        let xs = vec![0.5; 5];
        // confidence > 1 should clamp to ~1; confidence < 0 should clamp to ~0.
        let (_, lo1, hi1) = bootstrap_mean_ci(&xs, 100, 1.5, 1);
        let (_, lo2, hi2) = bootstrap_mean_ci(&xs, 100, -0.5, 1);
        assert!((lo1 - 0.5).abs() < 1e-12 && (hi1 - 0.5).abs() < 1e-12);
        assert!((lo2 - 0.5).abs() < 1e-12 && (hi2 - 0.5).abs() < 1e-12);
    }

    #[test]
    fn eval_ci_per_scorer_sorted() {
        let cases = vec![
            make_case("q1", &[("alpha", 0.8), ("zeta", 0.2)]),
            make_case("q2", &[("alpha", 0.9), ("zeta", 0.3)]),
            make_case("q3", &[("alpha", 0.7), ("zeta", 0.4)]),
        ];
        let r = make_report(cases);
        let res = bootstrap_eval_ci(&r, 200, 0.95, 42);
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].scorer, "alpha");
        assert_eq!(res[1].scorer, "zeta");
        assert_eq!(res[0].n, 3);
        assert!(res[0].mean > res[1].mean); // alpha mean > zeta mean
        assert!(res[0].lower <= res[0].mean && res[0].mean <= res[0].upper);
        assert!(res[1].lower <= res[1].mean && res[1].mean <= res[1].upper);
        assert_eq!(res[0].confidence, 0.95);
        assert_eq!(res[0].n_resamples, 200);
    }

    #[test]
    fn xorshift_seed_zero_substitutes_default() {
        // seed=0 is degenerate for xorshift; the constructor should
        // substitute a non-zero seed instead of producing all-zero
        // output forever.
        let mut rng = Xorshift64::new(0);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, 0);
        assert_ne!(a, b);
    }
}

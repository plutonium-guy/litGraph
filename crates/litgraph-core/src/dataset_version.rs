//! Dataset versioning + run history — tracks regressions across eval
//! runs without requiring an external eval platform.
//!
//! # The problem
//!
//! Once you have an eval harness, the next thing you want is "did the
//! latest prompt make us worse on dataset X?". That requires three
//! things this module provides:
//!
//! 1. **A stable fingerprint of the dataset** so a comparison knows
//!    it's apples-to-apples (same cases, same expected answers).
//! 2. **A persisted history** of (fingerprint, scorer means, timestamp)
//!    tuples so a new run can compare against the last green one.
//! 3. **A regression check** that flags scorers whose mean dropped
//!    beyond a configurable tolerance.
//!
//! # Fingerprint
//!
//! [`dataset_fingerprint`] hashes the canonical JSON of the case list
//! (input + expected, sorted-by-input for stability) with BLAKE3 and
//! returns the hex digest. Re-ordering cases doesn't change the hash;
//! adding/removing/changing a case does. Metadata is intentionally
//! excluded — case IDs and tags shouldn't make two semantically-
//! identical datasets compare unequal.
//!
//! # On-disk format ([`JsonlRunStore`])
//!
//! One JSON object per line. Append-only. Append is atomic per line
//! (writes the whole record in one syscall on POSIX), so concurrent
//! processes on the same file produce well-formed JSONL even without
//! locking — at worst, line ordering can interleave.
//!
//! # Why this lives in `litgraph-core` and not a separate crate
//!
//! Because the data shapes (`AggregateScores`, `EvalCase`) it depends
//! on already live here, and the implementation is pure-Rust + pure-
//! `serde` with no external service. A future "litgraph-eval-postgres"
//! crate can implement [`RunStore`] for durable team-shared history.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use crate::eval_harness::{AggregateScores, EvalCase, EvalReport};
use crate::{Error, Result};

/// Stable identifier for a dataset's *content*. Computed by hashing the
/// canonical JSON of `(input, expected)` pairs sorted by input. Same
/// content → same hash, regardless of order or metadata changes.
pub fn dataset_fingerprint(cases: &[EvalCase]) -> String {
    let mut canon: Vec<(String, String)> = cases
        .iter()
        .map(|c| (c.input.clone(), c.expected.clone().unwrap_or_default()))
        .collect();
    // Sort so order doesn't affect the hash. Stable per (input, expected).
    canon.sort();
    let bytes = serde_json::to_vec(&canon).expect("canon serialize");
    blake3::hash(&bytes).to_hex().to_string()
}

/// Manifest describing a versioned dataset. Written to disk alongside
/// the dataset JSONL so reviewers can confirm what's being run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct DatasetManifest {
    pub name: String,
    pub version: String,
    pub fingerprint: String,
    pub n_cases: usize,
    pub created_at_unix_ms: i64,
}

impl DatasetManifest {
    pub fn new(
        name: impl Into<String>,
        version: impl Into<String>,
        cases: &[EvalCase],
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            fingerprint: dataset_fingerprint(cases),
            n_cases: cases.len(),
            created_at_unix_ms: now_ms(),
        }
    }
}

/// One persisted eval run. Compact — no per-case rows, just the
/// information needed for regression comparisons.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RunRecord {
    pub dataset_name: String,
    pub dataset_version: String,
    pub dataset_fingerprint: String,
    /// Free-form caller-supplied label (e.g. git SHA, prompt id, model).
    /// Lets the consumer attribute regressions to a code change.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub run_label: Option<String>,
    pub n_cases: usize,
    pub n_errors: usize,
    /// Per-scorer mean as f64. Mirrors `AggregateScores::means` but
    /// flattened to numbers (we drop non-numeric mean values like
    /// arrays/strings since regression maths only makes sense on numbers).
    pub scorer_means: Map<String, Value>,
    pub ts_ms: i64,
}

impl RunRecord {
    /// Build a record from an [`EvalReport`] + manifest. Convenience —
    /// most callers don't need to assemble fields by hand.
    pub fn from_report(
        manifest: &DatasetManifest,
        report: &EvalReport,
        run_label: Option<String>,
    ) -> Self {
        // Filter to numeric means only. Non-numeric means (e.g. an
        // LLM-judge that wrote a string) get dropped — they can't be
        // diffed.
        let mut means = Map::new();
        for (k, v) in &report.aggregate.means {
            if v.is_number() {
                means.insert(k.clone(), v.clone());
            }
        }
        Self {
            dataset_name: manifest.name.clone(),
            dataset_version: manifest.version.clone(),
            dataset_fingerprint: manifest.fingerprint.clone(),
            run_label,
            n_cases: report.aggregate.n_cases,
            n_errors: report.aggregate.n_errors,
            scorer_means: means,
            ts_ms: now_ms(),
        }
    }
}

/// One regression alert per dropped scorer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RegressionAlert {
    pub scorer: String,
    pub baseline: f64,
    pub current: f64,
    pub delta: f64,
}

/// Compare a new run against a baseline. Returns alerts for every
/// scorer whose mean dropped by more than `tolerance` (e.g. 0.02 = 2pp
/// drop). Scorers that improved or stayed flat produce no alert.
///
/// Scorers present in baseline but missing in current ARE flagged
/// (current=0, delta=baseline). Scorers added in current and not in
/// baseline are silently ignored — there's no "before" to regress
/// against.
pub fn regression_check(
    baseline: &RunRecord,
    current: &RunRecord,
    tolerance: f64,
) -> Vec<RegressionAlert> {
    let mut out = Vec::new();
    for (scorer, base_v) in &baseline.scorer_means {
        let base = base_v.as_f64().unwrap_or(0.0);
        let curr = current
            .scorer_means
            .get(scorer)
            .and_then(|v| v.as_f64())
            .unwrap_or(0.0);
        let delta = curr - base;
        if delta < -tolerance {
            out.push(RegressionAlert {
                scorer: scorer.clone(),
                baseline: base,
                current: curr,
                delta,
            });
        }
    }
    // Stable order — alphabetical by scorer name.
    out.sort_by(|a, b| a.scorer.cmp(&b.scorer));
    out
}

// -- Run store trait + built-ins --------------------------------------------

#[async_trait]
pub trait RunStore: Send + Sync {
    /// Persist a record. Implementations are free to dedup or not.
    async fn record(&self, run: RunRecord) -> Result<()>;
    /// Most recent record for `dataset_name`, or `None` if none yet.
    async fn latest_for(&self, dataset_name: &str) -> Result<Option<RunRecord>>;
    /// All runs for `dataset_name` in chronological order.
    async fn history(&self, dataset_name: &str) -> Result<Vec<RunRecord>>;
}

/// In-memory store. Process-local, lossy across restarts. Right for
/// tests + short-lived eval suites where you only need same-process
/// regression detection.
#[derive(Default)]
pub struct InMemoryRunStore {
    inner: Mutex<Vec<RunRecord>>,
}

impl InMemoryRunStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.inner.lock().expect("poisoned").len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.lock().expect("poisoned").is_empty()
    }
}

#[async_trait]
impl RunStore for InMemoryRunStore {
    async fn record(&self, run: RunRecord) -> Result<()> {
        self.inner.lock().expect("poisoned").push(run);
        Ok(())
    }

    async fn latest_for(&self, dataset_name: &str) -> Result<Option<RunRecord>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .iter()
            .rev()
            .find(|r| r.dataset_name == dataset_name)
            .cloned())
    }

    async fn history(&self, dataset_name: &str) -> Result<Vec<RunRecord>> {
        Ok(self
            .inner
            .lock()
            .expect("poisoned")
            .iter()
            .filter(|r| r.dataset_name == dataset_name)
            .cloned()
            .collect())
    }
}

/// JSONL on-disk store. One file holds runs for any number of datasets;
/// `dataset_name` is part of the record so you can colocate. Append is
/// atomic per line (single `write` syscall) so concurrent writers don't
/// corrupt records — ordering may interleave but each line stays
/// well-formed JSON.
#[derive(Clone)]
pub struct JsonlRunStore {
    path: PathBuf,
}

impl JsonlRunStore {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    fn read_all_sync(&self) -> Result<Vec<RunRecord>> {
        match std::fs::read_to_string(&self.path) {
            Ok(s) => parse_jsonl(&s),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(Error::other(format!("run store read: {e}"))),
        }
    }
}

fn parse_jsonl(s: &str) -> Result<Vec<RunRecord>> {
    let mut out = Vec::new();
    for (line_no, line) in s.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let r: RunRecord = serde_json::from_str(trimmed)
            .map_err(|e| Error::other(format!("run store parse line {}: {}", line_no + 1, e)))?;
        out.push(r);
    }
    Ok(out)
}

#[async_trait]
impl RunStore for JsonlRunStore {
    async fn record(&self, run: RunRecord) -> Result<()> {
        let path = self.path.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::io::Write;
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| Error::other(format!("run store mkdir: {e}")))?;
            }
            let line = serde_json::to_string(&run)
                .map_err(|e| Error::other(format!("run store encode: {e}")))?;
            let mut f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&path)
                .map_err(|e| Error::other(format!("run store open: {e}")))?;
            // Single write call → atomic for typical record sizes (<4 KiB
            // on POSIX, which is well under PIPE_BUF). Includes trailing
            // newline so multiple writers can't share a line.
            let mut buf = line.into_bytes();
            buf.push(b'\n');
            f.write_all(&buf)
                .map_err(|e| Error::other(format!("run store write: {e}")))?;
            Ok(())
        })
        .await
        .map_err(|e| Error::other(format!("run store join: {e}")))?
    }

    async fn latest_for(&self, dataset_name: &str) -> Result<Option<RunRecord>> {
        let me = self.clone();
        let name = dataset_name.to_string();
        tokio::task::spawn_blocking(move || -> Result<Option<RunRecord>> {
            let all = me.read_all_sync()?;
            Ok(all.into_iter().rev().find(|r| r.dataset_name == name))
        })
        .await
        .map_err(|e| Error::other(format!("run store join: {e}")))?
    }

    async fn history(&self, dataset_name: &str) -> Result<Vec<RunRecord>> {
        let me = self.clone();
        let name = dataset_name.to_string();
        tokio::task::spawn_blocking(move || -> Result<Vec<RunRecord>> {
            let all = me.read_all_sync()?;
            Ok(all.into_iter().filter(|r| r.dataset_name == name).collect())
        })
        .await
        .map_err(|e| Error::other(format!("run store join: {e}")))?
    }
}

// -- Higher-level helpers ---------------------------------------------------

/// Record a run AND check for regressions vs the most recent prior run.
/// Returns alerts (empty = green). Convenience that wraps the common
/// CI-gate pattern.
pub async fn record_and_check(
    store: &dyn RunStore,
    manifest: &DatasetManifest,
    report: &EvalReport,
    run_label: Option<String>,
    tolerance: f64,
) -> Result<Vec<RegressionAlert>> {
    let new = RunRecord::from_report(manifest, report, run_label);
    let prev = store.latest_for(&manifest.name).await?;
    store.record(new.clone()).await?;
    Ok(match prev {
        Some(b) if b.dataset_fingerprint == new.dataset_fingerprint => {
            regression_check(&b, &new, tolerance)
        }
        // Different fingerprint = different dataset under the same name.
        // Skip regression check; the means aren't comparable.
        _ => Vec::new(),
    })
}

#[allow(dead_code)]
pub(crate) fn aggregate_means_to_record_means(a: &AggregateScores) -> Map<String, Value> {
    let mut out = Map::new();
    for (k, v) in &a.means {
        if v.is_number() {
            out.insert(k.clone(), v.clone());
        }
    }
    out
}

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_cases() -> Vec<EvalCase> {
        vec![
            EvalCase::new("q1").with_expected("a1"),
            EvalCase::new("q2").with_expected("a2"),
            EvalCase::new("q3").with_expected("a3"),
        ]
    }

    fn report_with_means(em: f64, jaccard: f64, n: usize, errs: usize) -> EvalReport {
        let mut means = Map::new();
        means.insert("exact_match".into(), json!(em));
        means.insert("jaccard".into(), json!(jaccard));
        EvalReport {
            per_case: Vec::new(),
            aggregate: AggregateScores {
                n_cases: n,
                n_errors: errs,
                means,
            },
        }
    }

    #[test]
    fn fingerprint_is_stable_under_reorder() {
        let a = vec![
            EvalCase::new("q1").with_expected("a1"),
            EvalCase::new("q2").with_expected("a2"),
        ];
        let b = vec![
            EvalCase::new("q2").with_expected("a2"),
            EvalCase::new("q1").with_expected("a1"),
        ];
        assert_eq!(dataset_fingerprint(&a), dataset_fingerprint(&b));
    }

    #[test]
    fn fingerprint_changes_when_expected_changes() {
        let a = vec![EvalCase::new("q1").with_expected("a1")];
        let b = vec![EvalCase::new("q1").with_expected("CHANGED")];
        assert_ne!(dataset_fingerprint(&a), dataset_fingerprint(&b));
    }

    #[test]
    fn fingerprint_ignores_metadata() {
        let a = vec![EvalCase::new("q1").with_expected("a1")];
        let b = vec![EvalCase::new("q1")
            .with_expected("a1")
            .with_metadata(json!({"tag": "x"}))];
        assert_eq!(dataset_fingerprint(&a), dataset_fingerprint(&b));
    }

    #[test]
    fn fingerprint_changes_when_case_added() {
        let a = vec![EvalCase::new("q1").with_expected("a1")];
        let b = vec![
            EvalCase::new("q1").with_expected("a1"),
            EvalCase::new("q2").with_expected("a2"),
        ];
        assert_ne!(dataset_fingerprint(&a), dataset_fingerprint(&b));
    }

    #[test]
    fn manifest_captures_n_cases_and_fingerprint() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        assert_eq!(m.name, "toy");
        assert_eq!(m.version, "v1");
        assert_eq!(m.n_cases, 3);
        assert_eq!(m.fingerprint, dataset_fingerprint(&cases));
        assert!(m.created_at_unix_ms > 0);
    }

    #[test]
    fn run_record_drops_non_numeric_means() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let mut means = Map::new();
        means.insert("exact_match".into(), json!(0.85));
        means.insert("notes".into(), json!("non-numeric"));
        let report = EvalReport {
            per_case: Vec::new(),
            aggregate: AggregateScores {
                n_cases: 3,
                n_errors: 0,
                means,
            },
        };
        let r = RunRecord::from_report(&m, &report, Some("git-abc".into()));
        assert!(r.scorer_means.contains_key("exact_match"));
        assert!(!r.scorer_means.contains_key("notes"));
        assert_eq!(r.run_label.as_deref(), Some("git-abc"));
        assert_eq!(r.n_cases, 3);
    }

    #[test]
    fn regression_check_flags_drops_beyond_tolerance() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let base = RunRecord::from_report(&m, &report_with_means(0.90, 0.80, 3, 0), None);
        let curr = RunRecord::from_report(&m, &report_with_means(0.85, 0.81, 3, 0), None);
        let alerts = regression_check(&base, &curr, 0.02);
        // exact_match dropped 0.05 (> 0.02); jaccard improved.
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].scorer, "exact_match");
        assert!((alerts[0].baseline - 0.90).abs() < 1e-9);
        assert!((alerts[0].current - 0.85).abs() < 1e-9);
        assert!((alerts[0].delta + 0.05).abs() < 1e-9);
    }

    #[test]
    fn regression_check_within_tolerance_returns_empty() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let base = RunRecord::from_report(&m, &report_with_means(0.90, 0.80, 3, 0), None);
        let curr = RunRecord::from_report(&m, &report_with_means(0.89, 0.80, 3, 0), None);
        let alerts = regression_check(&base, &curr, 0.02);
        assert!(alerts.is_empty());
    }

    #[test]
    fn regression_check_handles_missing_current_scorer_as_zero() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let base = RunRecord::from_report(&m, &report_with_means(0.95, 0.80, 3, 0), None);
        let mut curr_means = Map::new();
        curr_means.insert("jaccard".into(), json!(0.80));
        let curr = RunRecord {
            dataset_name: m.name.clone(),
            dataset_version: m.version.clone(),
            dataset_fingerprint: m.fingerprint.clone(),
            run_label: None,
            n_cases: 3,
            n_errors: 0,
            scorer_means: curr_means,
            ts_ms: 1,
        };
        let alerts = regression_check(&base, &curr, 0.02);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].scorer, "exact_match");
        assert_eq!(alerts[0].current, 0.0);
    }

    #[test]
    fn regression_check_ignores_new_scorers() {
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let base_means = {
            let mut x = Map::new();
            x.insert("exact_match".into(), json!(0.90));
            x
        };
        let base = RunRecord {
            dataset_name: "toy".into(),
            dataset_version: "v1".into(),
            dataset_fingerprint: m.fingerprint.clone(),
            run_label: None,
            n_cases: 3,
            n_errors: 0,
            scorer_means: base_means,
            ts_ms: 0,
        };
        let curr = RunRecord::from_report(&m, &report_with_means(0.91, 0.80, 3, 0), None);
        let alerts = regression_check(&base, &curr, 0.02);
        assert!(alerts.is_empty());
    }

    #[tokio::test]
    async fn inmemory_store_round_trips() {
        let s = InMemoryRunStore::new();
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let r = RunRecord::from_report(&m, &report_with_means(0.9, 0.8, 3, 0), None);
        s.record(r.clone()).await.unwrap();
        assert_eq!(s.len(), 1);
        let latest = s.latest_for("toy").await.unwrap().unwrap();
        assert_eq!(latest, r);
        let hist = s.history("toy").await.unwrap();
        assert_eq!(hist.len(), 1);
        assert!(s.latest_for("nope").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn inmemory_latest_returns_most_recent() {
        let s = InMemoryRunStore::new();
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        for em in [0.7, 0.8, 0.9] {
            let r = RunRecord::from_report(&m, &report_with_means(em, 0.5, 3, 0), None);
            s.record(r).await.unwrap();
        }
        let latest = s.latest_for("toy").await.unwrap().unwrap();
        assert!((latest.scorer_means["exact_match"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn jsonl_store_round_trips_to_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.jsonl");
        let s = JsonlRunStore::new(&path);
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        let r = RunRecord::from_report(&m, &report_with_means(0.9, 0.8, 3, 0), Some("v0".into()));
        s.record(r.clone()).await.unwrap();
        let latest = s.latest_for("toy").await.unwrap().unwrap();
        assert_eq!(latest, r);
        let hist = s.history("toy").await.unwrap();
        assert_eq!(hist.len(), 1);
    }

    #[tokio::test]
    async fn jsonl_store_appends_multiple_runs() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("runs.jsonl");
        let s = JsonlRunStore::new(&path);
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        for em in [0.7, 0.8, 0.9] {
            let r = RunRecord::from_report(&m, &report_with_means(em, 0.5, 3, 0), None);
            s.record(r).await.unwrap();
        }
        let hist = s.history("toy").await.unwrap();
        assert_eq!(hist.len(), 3);
        let latest = s.latest_for("toy").await.unwrap().unwrap();
        assert!((latest.scorer_means["exact_match"].as_f64().unwrap() - 0.9).abs() < 1e-9);
    }

    #[tokio::test]
    async fn jsonl_store_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nope.jsonl");
        let s = JsonlRunStore::new(&path);
        assert!(s.history("any").await.unwrap().is_empty());
        assert!(s.latest_for("any").await.unwrap().is_none());
    }

    #[tokio::test]
    async fn record_and_check_returns_alerts_when_dropping() {
        let s = InMemoryRunStore::new();
        let cases = make_cases();
        let m = DatasetManifest::new("toy", "v1", &cases);
        // First run — no baseline yet, no alerts.
        let r0 = report_with_means(0.90, 0.80, 3, 0);
        let alerts0 = record_and_check(&s, &m, &r0, None, 0.02).await.unwrap();
        assert!(alerts0.is_empty());
        // Second run — drops below tolerance.
        let r1 = report_with_means(0.80, 0.80, 3, 0);
        let alerts1 = record_and_check(&s, &m, &r1, None, 0.02).await.unwrap();
        assert_eq!(alerts1.len(), 1);
        assert_eq!(alerts1[0].scorer, "exact_match");
        // Both runs persisted.
        assert_eq!(s.len(), 2);
    }

    #[tokio::test]
    async fn record_and_check_skips_alerts_on_fingerprint_change() {
        let s = InMemoryRunStore::new();
        let m1 = DatasetManifest::new("toy", "v1", &make_cases());
        let r1 = report_with_means(0.95, 0.80, 3, 0);
        record_and_check(&s, &m1, &r1, None, 0.02).await.unwrap();

        // Different cases → different fingerprint, even with same name.
        let m2 = DatasetManifest::new(
            "toy",
            "v2",
            &[EvalCase::new("totally-different").with_expected("x")],
        );
        let r2 = report_with_means(0.50, 0.40, 1, 0); // catastrophically lower
        let alerts = record_and_check(&s, &m2, &r2, None, 0.02).await.unwrap();
        assert!(
            alerts.is_empty(),
            "fingerprint mismatch should suppress alerts: {alerts:?}"
        );
    }

    #[test]
    fn parse_jsonl_handles_blank_lines() {
        let text = "\n{\"dataset_name\":\"x\",\"dataset_version\":\"v1\",\
                    \"dataset_fingerprint\":\"f\",\"n_cases\":1,\"n_errors\":0,\
                    \"scorer_means\":{},\"ts_ms\":0}\n\n";
        let parsed = parse_jsonl(text).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].dataset_name, "x");
    }

    #[test]
    fn parse_jsonl_surfaces_bad_lines() {
        let text = "not json\n";
        let err = parse_jsonl(text).unwrap_err();
        assert!(format!("{err}").contains("parse line 1"));
    }
}

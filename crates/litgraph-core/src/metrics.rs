//! `MetricsRegistry` — in-process metrics aggregation.
//!
//! Three primitive types, all atomic / lock-free on the hot
//! path:
//!
//! | Primitive   | Semantics                                      |
//! |-------------|------------------------------------------------|
//! | [`Counter`] | monotonically-increasing `u64` (e.g. requests) |
//! | [`Gauge`]   | settable `i64` (e.g. in-flight calls, queue len) |
//! | [`Histogram`] | bucketed observation stream + sum + count    |
//!
//! # Distinct from the tracing / OTel layer
//!
//! `tracing` + `opentelemetry-otlp` (already wired via
//! `litgraph-tracing-otel`) are for *distributed* spans —
//! request-scoped events that propagate `trace_id` across
//! services. `MetricsRegistry` is for *in-process aggregation*:
//! `/metrics`-endpoint counters that don't need per-request
//! correlation but do need cheap concurrent updates from many
//! tasks.
//!
//! Both are useful; they answer different questions. Tracing
//! answers "what did this one request do?"; metrics answer
//! "what's the rate of X across the whole process right now?".
//!
//! # Real prod use
//!
//! - **`/metrics` endpoint** for Prometheus scraping via
//!   [`MetricsRegistry::to_prometheus`].
//! - **Cheap counters** in agent loops: every tool call bumps
//!   `tool_calls_total{name="…"}` without crossing a Mutex.
//! - **In-flight gauge** for live dashboards: a wrapper sets
//!   gauge on entry, decrements on exit.
//! - **Latency histogram** for request timing buckets.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Arc;

use parking_lot::RwLock;

/// Monotonically-increasing `u64`. Cheap clone (Arc inside).
#[derive(Debug)]
pub struct Counter(AtomicU64);

impl Counter {
    fn new() -> Self {
        Self(AtomicU64::new(0))
    }
    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    pub fn add(&self, n: u64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    pub fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Settable `i64`. Useful for in-flight counts, queue lengths,
/// buffer fill ratios, etc.
#[derive(Debug)]
pub struct Gauge(AtomicI64);

impl Gauge {
    fn new() -> Self {
        Self(AtomicI64::new(0))
    }
    pub fn set(&self, v: i64) {
        self.0.store(v, Ordering::Relaxed);
    }
    pub fn inc(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
    pub fn dec(&self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
    pub fn add(&self, n: i64) {
        self.0.fetch_add(n, Ordering::Relaxed);
    }
    pub fn get(&self) -> i64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// Bucketed observation histogram (Prometheus-style cumulative
/// buckets). Construct with sorted ascending bucket bounds; an
/// implicit `+Inf` bucket covers everything beyond the largest
/// explicit bound.
#[derive(Debug)]
pub struct Histogram {
    /// Sorted ascending upper bounds. Last bucket holds count
    /// of observations <= that bound; observations beyond go to
    /// `inf_count`.
    bounds: Vec<f64>,
    /// Cumulative-bucket counts. `bucket_counts[i]` = count of
    /// observations <= `bounds[i]`. Prometheus convention.
    bucket_counts: Vec<AtomicU64>,
    inf_count: AtomicU64,
    sum_bits: AtomicU64, // f64 stored as bits for atomic CAS
    count: AtomicU64,
}

impl Histogram {
    fn new(buckets: &[f64]) -> Self {
        let mut bounds: Vec<f64> = buckets.to_vec();
        // Ensure sorted ascending. Strip duplicates and NaN.
        bounds.retain(|b| !b.is_nan());
        bounds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        bounds.dedup();
        let n = bounds.len();
        let bucket_counts = (0..n).map(|_| AtomicU64::new(0)).collect();
        Self {
            bounds,
            bucket_counts,
            inf_count: AtomicU64::new(0),
            sum_bits: AtomicU64::new(0),
            count: AtomicU64::new(0),
        }
    }

    /// Record an observation. Updates every bucket whose upper
    /// bound is >= `value` (cumulative-bucket convention).
    pub fn observe(&self, value: f64) {
        if value.is_nan() {
            return;
        }
        // Add to each cumulative bucket that contains this value.
        let mut hit = false;
        for (i, &b) in self.bounds.iter().enumerate() {
            if value <= b {
                self.bucket_counts[i].fetch_add(1, Ordering::Relaxed);
                if !hit {
                    hit = true;
                }
            }
        }
        if !hit {
            // Beyond the largest explicit bound — only +Inf bucket
            // and the cumulative inf_count get this observation.
        }
        self.inf_count.fetch_add(1, Ordering::Relaxed);
        self.count.fetch_add(1, Ordering::Relaxed);
        // Atomic add to sum via f64 CAS-loop on bits.
        let mut prev = self.sum_bits.load(Ordering::Relaxed);
        loop {
            let new = f64::from_bits(prev) + value;
            match self.sum_bits.compare_exchange_weak(
                prev,
                new.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => prev = actual,
            }
        }
    }

    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    pub fn sum(&self) -> f64 {
        f64::from_bits(self.sum_bits.load(Ordering::Relaxed))
    }

    /// Returns `(bound, cumulative_count)` for each explicit
    /// bucket plus a final `(f64::INFINITY, total_count)`.
    pub fn snapshot(&self) -> Vec<(f64, u64)> {
        let mut out: Vec<(f64, u64)> = self
            .bounds
            .iter()
            .zip(self.bucket_counts.iter())
            .map(|(&b, c)| (b, c.load(Ordering::Relaxed)))
            .collect();
        out.push((f64::INFINITY, self.inf_count.load(Ordering::Relaxed)));
        out
    }
}

/// In-process metrics registry. Cheap clone (Arc inside).
/// Names may include any UTF-8; the Prometheus exporter
/// rewrites disallowed characters to `_`.
#[derive(Default)]
pub struct MetricsRegistry {
    counters: RwLock<HashMap<String, Arc<Counter>>>,
    gauges: RwLock<HashMap<String, Arc<Gauge>>>,
    histograms: RwLock<HashMap<String, Arc<Histogram>>>,
}

impl MetricsRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Get-or-create a counter. Repeated calls with the same
    /// name return the same Arc — values accumulate across
    /// callers.
    pub fn counter(&self, name: &str) -> Arc<Counter> {
        if let Some(c) = self.counters.read().get(name) {
            return c.clone();
        }
        let mut w = self.counters.write();
        w.entry(name.to_string())
            .or_insert_with(|| Arc::new(Counter::new()))
            .clone()
    }

    pub fn gauge(&self, name: &str) -> Arc<Gauge> {
        if let Some(g) = self.gauges.read().get(name) {
            return g.clone();
        }
        let mut w = self.gauges.write();
        w.entry(name.to_string())
            .or_insert_with(|| Arc::new(Gauge::new()))
            .clone()
    }

    /// Get-or-create a histogram. The first caller's `buckets`
    /// wins; subsequent calls with different buckets ignore the
    /// new buckets and return the existing histogram. Use
    /// distinct names for distinct bucket sets.
    pub fn histogram(&self, name: &str, buckets: &[f64]) -> Arc<Histogram> {
        if let Some(h) = self.histograms.read().get(name) {
            return h.clone();
        }
        let mut w = self.histograms.write();
        w.entry(name.to_string())
            .or_insert_with(|| Arc::new(Histogram::new(buckets)))
            .clone()
    }

    /// Render all metrics in Prometheus text exposition format.
    /// Suitable for an HTTP `/metrics` handler.
    pub fn to_prometheus(&self) -> String {
        let mut s = String::new();

        // Counters
        let counters = self.counters.read();
        let mut names: Vec<&String> = counters.keys().collect();
        names.sort();
        for name in names {
            let safe = sanitize(name);
            s.push_str(&format!("# TYPE {safe} counter\n"));
            s.push_str(&format!("{safe} {}\n", counters[name].get()));
        }

        // Gauges
        let gauges = self.gauges.read();
        let mut names: Vec<&String> = gauges.keys().collect();
        names.sort();
        for name in names {
            let safe = sanitize(name);
            s.push_str(&format!("# TYPE {safe} gauge\n"));
            s.push_str(&format!("{safe} {}\n", gauges[name].get()));
        }

        // Histograms
        let histos = self.histograms.read();
        let mut names: Vec<&String> = histos.keys().collect();
        names.sort();
        for name in names {
            let safe = sanitize(name);
            let h = &histos[name];
            s.push_str(&format!("# TYPE {safe} histogram\n"));
            for (bound, count) in h.snapshot() {
                let le = if bound.is_infinite() {
                    "+Inf".to_string()
                } else {
                    format_f64(bound)
                };
                s.push_str(&format!("{safe}_bucket{{le=\"{le}\"}} {count}\n"));
            }
            s.push_str(&format!("{safe}_sum {}\n", format_f64(h.sum())));
            s.push_str(&format!("{safe}_count {}\n", h.count()));
        }

        s
    }
}

/// Replace characters disallowed by Prometheus metric naming
/// rules with `_`. Allowed: `[a-zA-Z_][a-zA-Z0-9_:]*`.
fn sanitize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, c) in name.chars().enumerate() {
        let ok = if i == 0 {
            c.is_ascii_alphabetic() || c == '_'
        } else {
            c.is_ascii_alphanumeric() || c == '_' || c == ':'
        };
        out.push(if ok { c } else { '_' });
    }
    if out.is_empty() {
        "_".into()
    } else {
        out
    }
}

/// Format an f64 in a way Prometheus parses cleanly. Avoids
/// scientific notation for small integers.
fn format_f64(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "+Inf".into() } else { "-Inf".into() };
    }
    if v.fract() == 0.0 && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counter_inc_and_add() {
        let r = MetricsRegistry::new();
        let c = r.counter("requests_total");
        c.inc();
        c.add(5);
        assert_eq!(c.get(), 6);
    }

    #[test]
    fn counter_get_or_create_returns_same_arc() {
        let r = MetricsRegistry::new();
        let c1 = r.counter("foo");
        let c2 = r.counter("foo");
        c1.inc();
        c2.inc();
        // Same underlying counter, both increments visible from either handle.
        assert_eq!(c1.get(), 2);
        assert_eq!(c2.get(), 2);
    }

    #[test]
    fn gauge_inc_dec_set() {
        let r = MetricsRegistry::new();
        let g = r.gauge("in_flight");
        g.inc();
        g.inc();
        g.dec();
        assert_eq!(g.get(), 1);
        g.set(42);
        assert_eq!(g.get(), 42);
        g.add(-10);
        assert_eq!(g.get(), 32);
    }

    #[test]
    fn histogram_observe_updates_buckets_sum_count() {
        let r = MetricsRegistry::new();
        let h = r.histogram("latency_ms", &[10.0, 50.0, 100.0]);
        h.observe(5.0);
        h.observe(20.0);
        h.observe(75.0);
        h.observe(200.0); // beyond largest explicit bound
        assert_eq!(h.count(), 4);
        assert!((h.sum() - 300.0).abs() < 1e-9);
        let snap = h.snapshot();
        // bucket le=10 → 1 (just the 5.0)
        // bucket le=50 → 2 (5.0, 20.0)
        // bucket le=100 → 3 (5.0, 20.0, 75.0)
        // bucket +Inf → 4
        assert_eq!(snap[0], (10.0, 1));
        assert_eq!(snap[1], (50.0, 2));
        assert_eq!(snap[2], (100.0, 3));
        assert_eq!(snap[3].1, 4);
        assert!(snap[3].0.is_infinite());
    }

    #[test]
    fn histogram_buckets_unsorted_input_is_sorted() {
        let r = MetricsRegistry::new();
        let h = r.histogram("h", &[100.0, 10.0, 50.0]);
        let snap = h.snapshot();
        let bounds: Vec<f64> = snap.iter().map(|(b, _)| *b).collect();
        // Should be sorted ascending, with +Inf last.
        assert_eq!(bounds[0], 10.0);
        assert_eq!(bounds[1], 50.0);
        assert_eq!(bounds[2], 100.0);
        assert!(bounds[3].is_infinite());
    }

    #[test]
    fn prometheus_output_renders_all_three_types() {
        let r = MetricsRegistry::new();
        r.counter("requests_total").add(7);
        r.gauge("queue_len").set(3);
        let h = r.histogram("rt_seconds", &[0.1, 1.0]);
        h.observe(0.05);
        h.observe(2.0);
        let out = r.to_prometheus();
        assert!(out.contains("# TYPE requests_total counter"));
        assert!(out.contains("requests_total 7"));
        assert!(out.contains("# TYPE queue_len gauge"));
        assert!(out.contains("queue_len 3"));
        assert!(out.contains("# TYPE rt_seconds histogram"));
        assert!(out.contains("rt_seconds_bucket{le=\"0.1\"} 1"));
        assert!(out.contains("rt_seconds_bucket{le=\"1\"} 1"));
        assert!(out.contains("rt_seconds_bucket{le=\"+Inf\"} 2"));
        assert!(out.contains("rt_seconds_count 2"));
        assert!(out.contains("rt_seconds_sum 2.05"));
    }

    #[test]
    fn prometheus_sanitizes_disallowed_chars_in_names() {
        let r = MetricsRegistry::new();
        r.counter("requests-total").inc();
        let out = r.to_prometheus();
        // Hyphen rewritten to underscore.
        assert!(out.contains("requests_total 1"));
    }

    #[test]
    fn concurrent_counter_inc_is_atomic() {
        use std::thread;
        let r = MetricsRegistry::new();
        let c = r.counter("hits");
        let mut handles = Vec::new();
        for _ in 0..16 {
            let c = c.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..1000 {
                    c.inc();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        assert_eq!(c.get(), 16_000);
    }

    #[test]
    fn concurrent_histogram_observe_count_correct() {
        use std::thread;
        let r = MetricsRegistry::new();
        let h = r.histogram("vals", &[10.0, 100.0]);
        let mut handles = Vec::new();
        for t in 0..8 {
            let h = h.clone();
            handles.push(thread::spawn(move || {
                for i in 0..1000 {
                    h.observe((t * 1000 + i) as f64 % 200.0);
                }
            }));
        }
        for jh in handles {
            jh.join().unwrap();
        }
        assert_eq!(h.count(), 8_000);
    }
}

//! Bounded-concurrency multi-loader fan-out — given N loaders, run
//! their (blocking) `load()` calls in parallel via Tokio
//! `spawn_blocking` capped by a `Semaphore`. Symmetric in spirit to
//! [`litgraph_core::batch_concurrent`] for chat models, but for
//! document ingestion.
//!
//! # Why this exists
//!
//! `Loader::load()` is **synchronous** by design — most file/IO
//! loaders are blocking I/O, and forcing them through an async trait
//! complicates pure-Rust embed paths. The trade-off is that
//! ingesting from many sources (10 webpages + 50 PDFs + an S3
//! prefix) without this helper means either:
//!
//! - calling each `load()` sequentially (slow), or
//! - hand-rolling `tokio::task::spawn_blocking` + a semaphore + a
//!   join loop every time you want concurrency.
//!
//! `load_concurrent` does the second for you, with the right error
//! handling and ordering guarantees baked in.
//!
//! # Guarantees
//!
//! 1. **Ordered output** — `output[i]` corresponds to `loaders[i]`,
//!    no matter the order of completion.
//! 2. **Per-loader `Result`** — one failure doesn't abort the rest.
//!    Use `load_concurrent_flat` for fail-fast / flatten semantics.
//! 3. **Bounded concurrency** — `max_concurrency` caps the number of
//!    OS threads tied up in `spawn_blocking` at any instant. Sized
//!    correctly: too many blocking tasks starve the runtime's
//!    blocking-thread pool (default 512), which can wedge unrelated
//!    `spawn_blocking` consumers in the same process.

use std::sync::Arc;

use litgraph_core::Document;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::{Loader, LoaderError, LoaderResult};

/// Default in-flight concurrency. Conservative because each spawned
/// loader holds a thread from the Tokio blocking pool; 4 keeps
/// throughput high without crowding out other consumers.
pub const DEFAULT_LOAD_CONCURRENCY: usize = 4;

/// Run `loaders[i].load()` for every `i` in parallel, capped at
/// `max_concurrency` in flight. Returns a `Vec<Result>` aligned with
/// the input — slot `i` holds the outcome of `loaders[i]`.
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
///
/// ```no_run
/// # use std::sync::Arc;
/// # use litgraph_loaders::{Loader, TextLoader, load_concurrent};
/// # async fn run(paths: &[std::path::PathBuf]) {
/// let loaders: Vec<Arc<dyn Loader>> = paths
///     .iter()
///     .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
///     .collect();
/// let results = load_concurrent(loaders, 8).await;
/// for r in results {
///     match r {
///         Ok(docs) => println!("loaded {} docs", docs.len()),
///         Err(e) => eprintln!("loader failed: {e}"),
///     }
/// }
/// # }
/// ```
pub async fn load_concurrent(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
) -> Vec<LoaderResult<Vec<Document>>> {
    if loaders.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, LoaderResult<Vec<Document>>)> = JoinSet::new();

    for (idx, loader) in loaders.into_iter().enumerate() {
        let sem = sem.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return (
                        idx,
                        Err(LoaderError::Other("loader semaphore closed".into())),
                    )
                }
            };
            // `Loader::load` is blocking — hop to the dedicated
            // blocking-thread pool so we don't park an async worker.
            let join = tokio::task::spawn_blocking(move || loader.load()).await;
            let r = match join {
                Ok(r) => r,
                Err(e) => Err(LoaderError::Other(format!(
                    "loader task panicked: {e}",
                ))),
            };
            (idx, r)
        });
    }

    let n = set.len();
    let mut out: Vec<Option<LoaderResult<Vec<Document>>>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => out[idx] = Some(r),
            Err(e) => {
                if let Some(slot) = out.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(LoaderError::Other(format!(
                        "loader join: {e}",
                    ))));
                }
            }
        }
    }

    out.into_iter()
        .map(|s| s.unwrap_or_else(|| Err(LoaderError::Other("loader slot lost".into()))))
        .collect()
}

/// Like `load_concurrent` but flattens successful results into a
/// single `Vec<Document>` and short-circuits on the first error.
/// Use when partial-results aren't useful (e.g. a downstream embed
/// pipeline that needs every chunk for a deterministic dataset).
pub async fn load_concurrent_flat(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
) -> LoaderResult<Vec<Document>> {
    let results = load_concurrent(loaders, max_concurrency).await;
    let mut out: Vec<Document> = Vec::new();
    for r in results {
        out.extend(r?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::TextLoader;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tempfile::TempDir;

    fn write_files(prefix: &str, n: usize) -> (TempDir, Vec<std::path::PathBuf>) {
        let tmp = TempDir::new().unwrap();
        let mut paths = Vec::with_capacity(n);
        for i in 0..n {
            let p = tmp.path().join(format!("{prefix}_{i}.txt"));
            std::fs::write(&p, format!("body of file {i}")).unwrap();
            paths.push(p);
        }
        (tmp, paths)
    }

    /// Sleeps then returns one synthetic document. Lets us measure
    /// concurrency by inspecting `peak`.
    struct DelayProbe {
        delay_ms: u64,
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        label: String,
    }

    impl Loader for DelayProbe {
        fn load(&self) -> LoaderResult<Vec<Document>> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut p = self.peak.load(Ordering::SeqCst);
            while now > p {
                match self
                    .peak
                    .compare_exchange(p, now, Ordering::SeqCst, Ordering::SeqCst)
                {
                    Ok(_) => break,
                    Err(actual) => p = actual,
                }
            }
            std::thread::sleep(Duration::from_millis(self.delay_ms));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(vec![Document::new(self.label.clone()).with_id(self.label.clone())])
        }
    }

    /// Always errors — for the per-loader-error-isolation test.
    struct AlwaysFail;
    impl Loader for AlwaysFail {
        fn load(&self) -> LoaderResult<Vec<Document>> {
            Err(LoaderError::Other("synthetic failure".into()))
        }
    }

    #[tokio::test]
    async fn empty_loaders_returns_empty() {
        let out = load_concurrent(vec![], 4).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_aligned_to_input_order() {
        let (_tmp, paths) = write_files("aligned", 5);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let out = load_concurrent(loaders, 4).await;
        assert_eq!(out.len(), 5);
        for (i, r) in out.iter().enumerate() {
            let docs = r.as_ref().expect("ok");
            assert_eq!(docs.len(), 1);
            assert_eq!(docs[0].content, format!("body of file {i}"));
        }
    }

    #[tokio::test]
    async fn concurrency_cap_honoured() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let loaders: Vec<Arc<dyn Loader>> = (0..12)
            .map(|i| {
                Arc::new(DelayProbe {
                    delay_ms: 25,
                    in_flight: in_flight.clone(),
                    peak: peak.clone(),
                    label: format!("p{i}"),
                }) as Arc<dyn Loader>
            })
            .collect();
        let _ = load_concurrent(loaders, 3).await;
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 3, "peak {observed} > cap 3");
        assert!(observed >= 2, "peak {observed} — concurrency never engaged");
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let loaders: Vec<Arc<dyn Loader>> = (0..4)
            .map(|i| {
                Arc::new(DelayProbe {
                    delay_ms: 5,
                    in_flight: in_flight.clone(),
                    peak: peak.clone(),
                    label: format!("z{i}"),
                }) as Arc<dyn Loader>
            })
            .collect();
        let _ = load_concurrent(loaders, 0).await;
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn per_loader_error_isolated() {
        let (_tmp, paths) = write_files("iso", 1);
        let good: Arc<dyn Loader> = Arc::new(TextLoader::new(&paths[0]));
        let bad: Arc<dyn Loader> = Arc::new(AlwaysFail);
        let out = load_concurrent(vec![good, bad], 4).await;
        assert_eq!(out.len(), 2);
        assert!(out[0].is_ok());
        assert!(out[1].is_err());
    }

    #[tokio::test]
    async fn flat_variant_concatenates() {
        let (_tmp, paths) = write_files("flat", 3);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let docs = load_concurrent_flat(loaders, 4).await.unwrap();
        assert_eq!(docs.len(), 3);
        let texts: Vec<String> = docs.iter().map(|d| d.content.clone()).collect();
        assert!(texts.iter().any(|t| t.contains("body of file 0")));
        assert!(texts.iter().any(|t| t.contains("body of file 1")));
        assert!(texts.iter().any(|t| t.contains("body of file 2")));
    }

    #[tokio::test]
    async fn flat_variant_short_circuits_on_error() {
        let (_tmp, paths) = write_files("err", 1);
        let good: Arc<dyn Loader> = Arc::new(TextLoader::new(&paths[0]));
        let bad: Arc<dyn Loader> = Arc::new(AlwaysFail);
        let r = load_concurrent_flat(vec![good, bad], 4).await;
        assert!(r.is_err());
    }
}

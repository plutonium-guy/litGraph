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

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use litgraph_core::{Document, Progress, ShutdownSignal};
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;

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

/// One item from [`load_concurrent_stream`] — the input loader
/// index plus that loader's outcome, emitted in completion order.
pub type LoadStreamItem = (usize, LoaderResult<Vec<Document>>);

/// Counters maintained by [`load_concurrent_with_progress`] and
/// [`load_concurrent_stream_with_progress`]. Snapshot from any
/// `Progress<LoadProgress>` observer to drive an ingest dashboard.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct LoadProgress {
    /// Total loaders submitted (set on entry).
    pub total: u64,
    /// Loaders whose `load()` has finished, success or failure.
    pub completed: u64,
    /// Total documents returned across all successful loaders.
    pub docs_loaded: u64,
    /// Subset of `completed` that returned `Err`.
    pub errors: u64,
}

/// Same as [`load_concurrent`] but updates `progress` as each
/// loader completes. Sixth and final progress-aware sibling
/// across the parallel-batch family (the loader axis was missing
/// it; this iter closes the gap retroactively).
///
/// Real prod use: an ingestion dashboard rendering "8 / 50
/// loaders done, 1.2k docs loaded, 0 errors" while a long-running
/// fan-out is in flight.
pub async fn load_concurrent_with_progress(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
    progress: Progress<LoadProgress>,
) -> Vec<LoaderResult<Vec<Document>>> {
    if loaders.is_empty() {
        return Vec::new();
    }
    let total = loaders.len() as u64;
    let _ = progress.update(|p| LoadProgress {
        total,
        ..p.clone()
    });

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
            Ok((idx, r)) => {
                let (n_docs, is_err) = match &r {
                    Ok(docs) => (docs.len() as u64, false),
                    Err(_) => (0, true),
                };
                out[idx] = Some(r);
                let _ = progress.update(|p| LoadProgress {
                    completed: p.completed + 1,
                    docs_loaded: p.docs_loaded + n_docs,
                    errors: p.errors + if is_err { 1 } else { 0 },
                    ..p.clone()
                });
            }
            Err(e) => {
                if let Some(slot) = out.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(LoaderError::Other(format!("loader join: {e}"))));
                }
                let _ = progress.update(|p| LoadProgress {
                    completed: p.completed + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
            }
        }
    }

    out.into_iter()
        .map(|s| s.unwrap_or_else(|| Err(LoaderError::Other("loader slot lost".into()))))
        .collect()
}

/// Combined streaming + progress-watcher variant. Yields the same
/// `(loader_idx, LoaderResult)` items as
/// [`load_concurrent_stream`] AND updates the supplied
/// `Progress<LoadProgress>` watcher per-loader. Closes the
/// four-quadrant consumer matrix for the loader axis — the last of
/// the six parallel-batch axes to ship the full set.
pub fn load_concurrent_stream_with_progress(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
    progress: Progress<LoadProgress>,
) -> Pin<Box<dyn Stream<Item = LoadStreamItem> + Send>> {
    if loaders.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let total = loaders.len() as u64;
    let _ = progress.update(|p| LoadProgress {
        total,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let n = loaders.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<LoadStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<LoadStreamItem> = JoinSet::new();
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
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(LoaderError::Other(format!("loader join: {e}"))),
                ),
            };
            let (n_docs, is_err) = match &item.1 {
                Ok(docs) => (docs.len() as u64, false),
                Err(_) => (0, true),
            };
            let _ = progress.update(|p| LoadProgress {
                completed: p.completed + 1,
                docs_loaded: p.docs_loaded + n_docs,
                errors: p.errors + if is_err { 1 } else { 0 },
                ..p.clone()
            });
            if tx.send(item).await.is_err() {
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// Streaming variant of [`load_concurrent`]. Yields
/// `(loader_idx, LoaderResult<Vec<Document>>)` pairs as each
/// loader's blocking `load()` finishes — caller drains in
/// completion order, can index documents into a vector store as
/// they land, and dropping the stream aborts in-flight loaders.
///
/// Streaming-variant pattern from iters 210/211/212/213/214
/// extended to the loader axis. The blocking loader call still
/// runs on `spawn_blocking` like [`load_concurrent`].
///
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub fn load_concurrent_stream(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
) -> Pin<Box<dyn Stream<Item = LoadStreamItem> + Send>> {
    if loaders.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let n = loaders.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<LoadStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<LoadStreamItem> = JoinSet::new();
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
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(LoaderError::Other(format!("loader join: {e}"))),
                ),
            };
            if tx.send(item).await.is_err() {
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// `load_concurrent` plus graceful cancellation via
/// [`ShutdownSignal`]. Output aligned 1:1 with `loaders`:
///
/// - `Ok(docs)` — loader completed before shutdown.
/// - `Err(loader error)` — loader returned an error.
/// - `Err("cancelled by shutdown")` — loader was still in flight
///   (or its `spawn_blocking` thread was still owned) when
///   shutdown fired.
///
/// Mechanical extension of the bridge pattern (iters 227-231) to
/// the loader axis — the sixth and final parallel-batch axis.
/// Real prod use: an ingestion job that crawls hundreds of S3
/// keys + sitemap URLs banks completed loaders on Ctrl+C — the
/// half-loaded corpus stays valid, so downstream embed/index
/// stages can ship the partial result.
///
/// Note: `Loader::load()` runs on `spawn_blocking`, so the OS
/// thread is held until the call returns naturally. `abort_all()`
/// only cancels the *waiting-on-permit* phase plus the join await
/// — already-blocking threads finish their current call. The
/// output slot is filled with `Err("cancelled by shutdown")`
/// regardless once the await is dropped.
pub async fn load_concurrent_with_shutdown(
    loaders: Vec<Arc<dyn Loader>>,
    max_concurrency: usize,
    shutdown: &ShutdownSignal,
) -> Vec<LoaderResult<Vec<Document>>> {
    if loaders.is_empty() {
        return Vec::new();
    }
    if shutdown.is_signaled() {
        return (0..loaders.len())
            .map(|_| Err(LoaderError::Other("cancelled by shutdown".into())))
            .collect();
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

    loop {
        tokio::select! {
            joined = set.join_next() => {
                match joined {
                    Some(Ok((idx, r))) => out[idx] = Some(r),
                    Some(Err(e)) => {
                        if let Some(slot) = out.iter_mut().find(|s| s.is_none()) {
                            *slot = Some(Err(LoaderError::Other(format!(
                                "loader join: {e}",
                            ))));
                        }
                    }
                    None => break,
                }
            }
            _ = shutdown.wait() => {
                set.abort_all();
                break;
            }
        }
    }

    out.into_iter()
        .map(|s| s.unwrap_or_else(|| Err(LoaderError::Other("cancelled by shutdown".into()))))
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

    // ---- load_concurrent_with_progress tests --------------------------

    #[tokio::test]
    async fn progress_total_set_and_completed_counts_advance() {
        let (_tmp, paths) = write_files("p_advance", 4);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let _ = load_concurrent_with_progress(loaders, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 4);
        assert_eq!(snap.completed, 4);
        assert_eq!(snap.docs_loaded, 4); // TextLoader → 1 doc per file
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn progress_records_loader_failures() {
        let (_tmp, paths) = write_files("p_err", 1);
        let good: Arc<dyn Loader> = Arc::new(TextLoader::new(&paths[0]));
        let bad: Arc<dyn Loader> = Arc::new(AlwaysFail);
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let _ = load_concurrent_with_progress(vec![good, bad], 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 2);
        assert_eq!(snap.completed, 2);
        assert_eq!(snap.errors, 1);
        assert_eq!(snap.docs_loaded, 1);
    }

    #[tokio::test]
    async fn progress_empty_loaders_no_updates() {
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let out = load_concurrent_with_progress(vec![], 4, progress).await;
        assert!(out.is_empty());
        assert_eq!(obs.snapshot(), LoadProgress::default());
    }

    // ---- load_concurrent_stream_with_progress tests -------------------

    #[tokio::test]
    async fn stream_with_progress_yields_items_and_advances_counters() {
        let (_tmp, paths) = write_files("sp_one", 5);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let mut s = load_concurrent_stream_with_progress(loaders, 3, progress);
        let mut count = 0;
        while let Some((_idx, r)) = s.next().await {
            assert!(r.is_ok());
            count += 1;
        }
        assert_eq!(count, 5);
        let snap = obs.snapshot();
        assert_eq!(snap.total, 5);
        assert_eq!(snap.completed, 5);
        assert_eq!(snap.docs_loaded, 5);
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn stream_with_progress_records_loader_failures() {
        let (_tmp, paths) = write_files("sp_err", 1);
        let good: Arc<dyn Loader> = Arc::new(TextLoader::new(&paths[0]));
        let bad: Arc<dyn Loader> = Arc::new(AlwaysFail);
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let mut s = load_concurrent_stream_with_progress(vec![good, bad], 4, progress);
        let mut errors_in_stream = 0;
        while let Some((_idx, r)) = s.next().await {
            if r.is_err() {
                errors_in_stream += 1;
            }
        }
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 2);
        assert_eq!(snap.errors, 1);
        assert_eq!(errors_in_stream, snap.errors as usize);
    }

    #[tokio::test]
    async fn stream_with_progress_empty_loaders_no_updates() {
        let progress = Progress::new(LoadProgress::default());
        let obs = progress.observer();
        let mut s = load_concurrent_stream_with_progress(vec![], 4, progress);
        assert!(s.next().await.is_none());
        assert_eq!(obs.snapshot(), LoadProgress::default());
    }

    // ---- load_concurrent_stream tests ---------------------------------

    use futures::StreamExt;

    #[tokio::test]
    async fn stream_yields_one_item_per_loader() {
        let (_tmp, paths) = write_files("stream_one", 5);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let mut s = load_concurrent_stream(loaders, 3);
        let mut indices: Vec<usize> = Vec::new();
        while let Some((idx, r)) = s.next().await {
            assert!(r.is_ok());
            indices.push(idx);
        }
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2, 3, 4]);
    }

    #[tokio::test]
    async fn stream_idx_aligns_with_input_loader() {
        let (_tmp, paths) = write_files("stream_align", 4);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let mut s = load_concurrent_stream(loaders, 2);
        while let Some((idx, r)) = s.next().await {
            let docs = r.unwrap();
            assert_eq!(docs.len(), 1);
            assert_eq!(docs[0].content, format!("body of file {idx}"));
        }
    }

    #[tokio::test]
    async fn stream_per_loader_failure_arrives_as_err_item() {
        let (_tmp, paths) = write_files("stream_err", 1);
        let good: Arc<dyn Loader> = Arc::new(TextLoader::new(&paths[0]));
        let bad: Arc<dyn Loader> = Arc::new(AlwaysFail);
        let mut s = load_concurrent_stream(vec![good, bad], 2);
        let mut count = 0;
        let mut errors = 0;
        while let Some((_idx, r)) = s.next().await {
            count += 1;
            if r.is_err() {
                errors += 1;
            }
        }
        assert_eq!(count, 2);
        assert_eq!(errors, 1);
    }

    #[tokio::test]
    async fn stream_empty_loaders_yields_empty() {
        let mut s = load_concurrent_stream(vec![], 4);
        assert!(s.next().await.is_none());
    }

    // ---- load_concurrent_with_shutdown tests --------------------------

    #[tokio::test]
    async fn shutdown_no_signal_completes_normally() {
        let (_tmp, paths) = write_files("sd_ok", 4);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let shutdown = ShutdownSignal::new();
        let out = load_concurrent_with_shutdown(loaders, 4, &shutdown).await;
        assert_eq!(out.len(), 4);
        assert!(out.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn shutdown_pre_fired_returns_all_cancelled() {
        let (_tmp, paths) = write_files("sd_pre", 3);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let shutdown = ShutdownSignal::new();
        shutdown.signal();
        let out = load_concurrent_with_shutdown(loaders, 4, &shutdown).await;
        assert_eq!(out.len(), 3);
        for r in &out {
            assert!(r
                .as_ref()
                .err()
                .unwrap()
                .to_string()
                .contains("cancelled by shutdown"));
        }
    }

    #[tokio::test]
    async fn shutdown_mid_run_preserves_completed_loaders() {
        // 20 loaders × 50ms × cap=2 — sequential ~500ms. Fire
        // shutdown at 80ms; completed loaders should bank Ok,
        // remaining should resolve as cancelled fast.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let loaders: Vec<Arc<dyn Loader>> = (0..20)
            .map(|i| {
                Arc::new(DelayProbe {
                    delay_ms: 50,
                    in_flight: in_flight.clone(),
                    peak: peak.clone(),
                    label: format!("p{i}"),
                }) as Arc<dyn Loader>
            })
            .collect();
        let shutdown = ShutdownSignal::new();
        let s2 = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(80)).await;
            s2.signal();
        });
        let started = std::time::Instant::now();
        let out = load_concurrent_with_shutdown(loaders, 2, &shutdown).await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < Duration::from_millis(400),
            "elapsed {elapsed:?} — shutdown didn't abort early",
        );
        assert_eq!(out.len(), 20);
        let ok = out.iter().filter(|r| r.is_ok()).count();
        let cancelled = out
            .iter()
            .filter(|r| {
                r.as_ref()
                    .err()
                    .map(|e| e.to_string().contains("cancelled by shutdown"))
                    .unwrap_or(false)
            })
            .count();
        assert!(ok >= 1, "no completed loaders banked");
        assert!(cancelled >= 1, "no in-flight loaders marked cancelled");
    }

    #[tokio::test]
    async fn shutdown_empty_loaders_returns_empty() {
        let shutdown = ShutdownSignal::new();
        let out = load_concurrent_with_shutdown(vec![], 4, &shutdown).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn stream_caller_drop_aborts_in_flight_loaders() {
        // 30 sleep-50ms loaders × cap=2 — full sequential ~750ms.
        // Drop after first item; total wall-clock should be far less.
        let in_flight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let loaders: Vec<Arc<dyn Loader>> = (0..30)
            .map(|i| {
                Arc::new(DelayProbe {
                    delay_ms: 50,
                    in_flight: in_flight.clone(),
                    peak: peak.clone(),
                    label: format!("p{i}"),
                }) as Arc<dyn Loader>
            })
            .collect();
        let started = std::time::Instant::now();
        {
            let mut s = load_concurrent_stream(loaders, 2);
            let _first = s.next().await.unwrap();
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 400,
            "elapsed {elapsed_ms}ms — caller-drop didn't abort remaining loaders",
        );
    }
}

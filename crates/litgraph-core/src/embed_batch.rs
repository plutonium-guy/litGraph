//! Bounded-concurrency `Embeddings::embed_documents` — chunk the input,
//! fan chunks out across a Tokio task pool with a semaphore cap,
//! re-assemble in input order. Symmetric to [`batch_concurrent`] for
//! `ChatModel`.
//!
//! [`batch_concurrent`]: crate::batch_concurrent
//!
//! # Why this exists
//!
//! Every provider's `embed_documents` is one HTTP round trip per call.
//! Pass it 50 000 texts and you either:
//!
//! - hit the provider's per-request input cap (OpenAI: 2048; Cohere:
//!   96; Voyage: 128 — varies wildly) and the call fails, or
//! - send them all and wait for one giant blocking request, which
//!   melts ingestion latency.
//!
//! `embed_documents_concurrent` splits the input into fixed-size
//! chunks and dispatches them concurrently, capped at `max_concurrency`
//! in flight. The returned `Vec<Vec<f32>>` is aligned 1:1 with the
//! input, so callers can swap it in wherever they were calling
//! `embed_documents` directly.
//!
//! # Choosing `chunk_size`
//!
//! Pick the *provider's* per-request batch ceiling, not your total
//! input size. OpenAI's `text-embedding-3-*` accepts up to 2048 inputs
//! per request — `chunk_size = 1024` is a safe default that leaves
//! headroom for retries and respects token-count caps too.
//!
//! # Failure mode
//!
//! Fail-fast — the first chunk that errors aborts the whole call.
//! Embedding pipelines are usually all-or-nothing (you need every
//! row's vector to insert into a vector store). For partial-result
//! semantics, drop down to per-chunk concurrency yourself.

use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde::{Deserialize, Serialize};
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;

use crate::{Embeddings, Error, Progress, Result, ShutdownSignal};

/// Default chunk size — sized for OpenAI's per-request input cap with
/// generous headroom. Override per-call if your provider is stricter
/// (Cohere = 96, Voyage = 128, Bedrock Titan = 25).
pub const DEFAULT_EMBED_CHUNK_SIZE: usize = 1024;

/// Default in-flight concurrency. Most providers throttle aggressively
/// at the per-key level; 4 parallel requests is the sweet spot for
/// throughput without tripping rate limits.
pub const DEFAULT_EMBED_CONCURRENCY: usize = 4;

/// Embed `texts` in concurrent chunks. Output index `i` is the embedding
/// of input index `i`.
///
/// `chunk_size = 0` is normalised to `texts.len().max(1)` (one chunk).
/// `max_concurrency = 0` is normalised to 1 (sequential).
pub async fn embed_documents_concurrent(
    embedder: Arc<dyn Embeddings>,
    texts: &[String],
    chunk_size: usize,
    max_concurrency: usize,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let cap = max_concurrency.max(1);
    let chunk = if chunk_size == 0 { texts.len() } else { chunk_size };

    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Vec<f32>>>)> = JoinSet::new();

    for (idx, slice) in texts.chunks(chunk).enumerate() {
        let sem = sem.clone();
        let embedder = embedder.clone();
        let owned: Vec<String> = slice.to_vec();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("embed semaphore closed"))),
            };
            let r = embedder.embed_documents(&owned).await;
            (idx, r)
        });
    }

    // Collect chunk results, indexed for stable reassembly.
    let n_chunks = set.len();
    let mut chunks: Vec<Option<Vec<Vec<f32>>>> = (0..n_chunks).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, Ok(v))) => chunks[idx] = Some(v),
            Ok((_, Err(e))) => return Err(e),
            Err(e) => return Err(Error::other(format!("embed task join: {e}"))),
        }
    }

    // Flatten in chunk-index order — preserves input alignment.
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for c in chunks.into_iter() {
        let v = c.ok_or_else(|| Error::other("embed chunk lost"))?;
        out.extend(v);
    }

    if out.len() != texts.len() {
        return Err(Error::other(format!(
            "embed_documents_concurrent: expected {} embeddings, got {}",
            texts.len(),
            out.len(),
        )));
    }
    Ok(out)
}

/// `embed_documents_concurrent` plus graceful cancellation via
/// [`ShutdownSignal`]. Output is per-chunk: `Vec<Result<Vec<Vec<f32>>>>`
/// aligned with the chunked input.
///
/// - `Ok(embeddings)` — chunk completed successfully before shutdown.
/// - `Err(provider error)` — chunk completed with the provider's failure.
/// - `Err("cancelled by shutdown")` — chunk was still in flight when
///   shutdown fired.
///
/// Distinct from wrapping `embed_documents_concurrent` in
/// `until_shutdown` (which would discard everything on shutdown):
/// **partial progress preserved**. A long bulk-indexing run that
/// embedded 60% of its chunks before Ctrl+C banks those 60% as
/// `Ok` so they can be flushed to the vector store before exit.
///
/// Composition: extends iter 227's bridge pattern from the chat
/// axis to the embeddings axis. Note the per-chunk granularity —
/// caller can flatten only the `Ok` slots into a partial-but-valid
/// embedding result.
///
/// `chunk_size = 0` is normalised to one chunk; `max_concurrency = 0`
/// to 1 (sequential).
pub async fn embed_documents_concurrent_with_shutdown(
    embedder: Arc<dyn Embeddings>,
    texts: &[String],
    chunk_size: usize,
    max_concurrency: usize,
    shutdown: &ShutdownSignal,
) -> Vec<Result<Vec<Vec<f32>>>> {
    if texts.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let chunk = if chunk_size == 0 { texts.len() } else { chunk_size };
    let n_chunks = texts.chunks(chunk).count();

    if shutdown.is_signaled() {
        return (0..n_chunks)
            .map(|_| Err(Error::other("cancelled by shutdown")))
            .collect();
    }

    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, Result<Vec<Vec<f32>>>)> = JoinSet::new();

    for (idx, slice) in texts.chunks(chunk).enumerate() {
        let sem = sem.clone();
        let embedder = embedder.clone();
        let owned: Vec<String> = slice.to_vec();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => return (idx, Err(Error::other("embed semaphore closed"))),
            };
            let r = embedder.embed_documents(&owned).await;
            (idx, r)
        });
    }

    let mut results: Vec<Option<Result<Vec<Vec<f32>>>>> =
        (0..n_chunks).map(|_| None).collect();

    loop {
        tokio::select! {
            joined = set.join_next() => {
                match joined {
                    Some(Ok((idx, r))) => results[idx] = Some(r),
                    Some(Err(e)) => {
                        if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                            *slot = Some(Err(Error::other(format!("embed task join: {e}"))));
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

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("cancelled by shutdown"))))
        .collect()
}

/// One emitted result from [`embed_documents_concurrent_stream`] —
/// the chunk index plus that chunk's outcome. The chunk index is the
/// chunk's position when `texts` was split by `chunk_size`, so the
/// `i`-th chunk's embeddings cover `texts[i*chunk_size .. min(end, len)]`.
pub type EmbedStreamItem = (usize, Result<Vec<Vec<f32>>>);

/// Streaming variant of [`embed_documents_concurrent`]. Yields
/// `(chunk_idx, Result)` pairs **in completion order** as each
/// chunk's `embed_documents` call finishes — caller can start
/// writing embeddings to a vector store / on-disk index as soon as
/// the first chunk lands instead of waiting for the slowest.
///
/// # When to use this vs `embed_documents_concurrent`
///
/// - `embed_documents_concurrent`: caller wants the **whole aligned
///   `Vec<Vec<f32>>`** when finished. Simpler integration with
///   downstream code that expects a complete result.
/// - `embed_documents_concurrent_stream` (this): caller wants to
///   **upsert chunks to a vector store as they arrive**, **render
///   bulk-ingestion progress live**, or **drop the stream early** to
///   abort remaining chunks.
///
/// The chunk index lets callers reassemble in input order if
/// needed; otherwise the per-chunk `Vec<Vec<f32>>` is internally
/// aligned with the chunk's slice.
///
/// `chunk_size = 0` is normalised to one chunk; `max_concurrency = 0`
/// to 1 (sequential). Empty input yields an empty stream.
pub fn embed_documents_concurrent_stream(
    embedder: Arc<dyn Embeddings>,
    texts: Vec<String>,
    chunk_size: usize,
    max_concurrency: usize,
) -> Pin<Box<dyn Stream<Item = EmbedStreamItem> + Send>> {
    if texts.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let chunk = if chunk_size == 0 { texts.len() } else { chunk_size };
    let chunks: Vec<Vec<String>> = texts.chunks(chunk).map(|s| s.to_vec()).collect();
    let n_chunks = chunks.len();
    let buf = n_chunks.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<EmbedStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<EmbedStreamItem> = JoinSet::new();
        for (idx, chunk_texts) in chunks.into_iter().enumerate() {
            let sem = sem.clone();
            let embedder = embedder.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => return (idx, Err(Error::other("embed semaphore closed"))),
                };
                let r = embedder.embed_documents(&chunk_texts).await;
                (idx, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(Error::other(format!("embed task join: {e}"))),
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

/// Combined streaming + progress-watcher variant. Yields the same
/// `(chunk_idx, Result)` items as
/// [`embed_documents_concurrent_stream`] AND updates the supplied
/// [`Progress<EmbedProgress>`] watcher as each chunk completes.
///
/// Per-chunk accounting matches
/// [`embed_documents_concurrent_with_progress`]; per-row stream
/// accounting matches [`embed_documents_concurrent_stream`].
/// Useful when callers want both a per-chunk row-view (for live
/// vector-store upserts) AND a summary `{total_texts, total_chunks,
/// completed_chunks, completed_texts, errors}` watcher driving a UI
/// progress bar.
///
/// Composition: extends the combined consumer shape from iter 216
/// (chat batch) to the embeddings axis.
pub fn embed_documents_concurrent_stream_with_progress(
    embedder: Arc<dyn Embeddings>,
    texts: Vec<String>,
    chunk_size: usize,
    max_concurrency: usize,
    progress: Progress<EmbedProgress>,
) -> Pin<Box<dyn Stream<Item = EmbedStreamItem> + Send>> {
    if texts.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let chunk = if chunk_size == 0 { texts.len() } else { chunk_size };
    let chunks: Vec<Vec<String>> = texts.chunks(chunk).map(|s| s.to_vec()).collect();
    let n_chunks = chunks.len() as u64;
    let total_texts = texts.len() as u64;
    let _ = progress.update(|p| EmbedProgress {
        total_texts,
        total_chunks: n_chunks,
        ..p.clone()
    });

    let buf = (chunks.len()).min(cap.max(8));
    let (tx, rx) = mpsc::channel::<EmbedStreamItem>(buf);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<(usize, usize, Result<Vec<Vec<f32>>>)> = JoinSet::new();
        for (idx, chunk_texts) in chunks.into_iter().enumerate() {
            let sem = sem.clone();
            let embedder = embedder.clone();
            let chunk_len = chunk_texts.len();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => {
                        return (
                            idx,
                            chunk_len,
                            Err(Error::other("embed semaphore closed")),
                        )
                    }
                };
                let r = embedder.embed_documents(&chunk_texts).await;
                (idx, chunk_len, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let (idx, chunk_len, item) = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    0,
                    Err(Error::other(format!("embed task join: {e}"))),
                ),
            };
            // Update progress BEFORE forwarding — observers always
            // see the counter tick before / at-the-same-time as
            // the stream item. Same consistency contract as iter 216.
            let is_err = item.is_err();
            let _ = progress.update(|p| EmbedProgress {
                completed_chunks: p.completed_chunks + 1,
                completed_texts: p.completed_texts
                    + if is_err { 0 } else { chunk_len as u64 },
                errors: p.errors + if is_err { 1 } else { 0 },
                ..p.clone()
            });
            if tx.send((idx, item)).await.is_err() {
                set.abort_all();
                break;
            }
        }
    });

    Box::pin(ReceiverStream::new(rx))
}

/// Counters maintained by [`embed_documents_concurrent_with_progress`].
/// All fields monotonic. Snapshot from any
/// [`Progress<EmbedProgress>`] observer to drive a progress bar over
/// a long-running ingestion.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct EmbedProgress {
    /// Total texts submitted (set once on entry).
    pub total_texts: u64,
    /// Total chunks the input was split into (set once on entry).
    pub total_chunks: u64,
    /// Chunks whose embed call has finished, success or failure.
    pub completed_chunks: u64,
    /// Texts successfully embedded (sum of completed chunks' lengths).
    pub completed_texts: u64,
    /// Subset of `completed_chunks` that returned `Err`.
    pub errors: u64,
}

/// Same as [`embed_documents_concurrent`] but updates `progress` as
/// each chunk completes. Real use case: a 100k-text bulk indexing
/// run rendering a tqdm-style counter / ETA.
///
/// Composition: third progress-aware sibling after iter 200's
/// `ingest_to_stream_with_progress` and iter 205's
/// `batch_concurrent_with_progress`. Same shape, different domain
/// (Embeddings batch).
///
/// Failure semantics inherit from [`embed_documents_concurrent`]:
/// fail-fast on any chunk error. The `errors` counter still ticks
/// before the call returns `Err`, so observers see the bad chunk's
/// failure even though the result is `Err(_)` overall.
pub async fn embed_documents_concurrent_with_progress(
    embedder: Arc<dyn Embeddings>,
    texts: &[String],
    chunk_size: usize,
    max_concurrency: usize,
    progress: Progress<EmbedProgress>,
) -> Result<Vec<Vec<f32>>> {
    if texts.is_empty() {
        return Ok(Vec::new());
    }
    let cap = max_concurrency.max(1);
    let chunk = if chunk_size == 0 { texts.len() } else { chunk_size };
    let total_chunks = texts.chunks(chunk).count() as u64;

    // Set totals up front so observers see run shape before any
    // chunk completion arrives.
    let total_texts = texts.len() as u64;
    let _ = progress.update(|p| EmbedProgress {
        total_texts,
        total_chunks,
        ..p.clone()
    });

    let sem = Arc::new(Semaphore::new(cap));
    let mut set: JoinSet<(usize, usize, Result<Vec<Vec<f32>>>)> = JoinSet::new();

    for (idx, slice) in texts.chunks(chunk).enumerate() {
        let sem = sem.clone();
        let embedder = embedder.clone();
        let owned: Vec<String> = slice.to_vec();
        let chunk_len = owned.len();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return (
                        idx,
                        chunk_len,
                        Err(Error::other("embed semaphore closed")),
                    )
                }
            };
            let r = embedder.embed_documents(&owned).await;
            (idx, chunk_len, r)
        });
    }

    let n_chunks = set.len();
    let mut chunks: Vec<Option<Vec<Vec<f32>>>> = (0..n_chunks).map(|_| None).collect();
    let mut first_err: Option<Error> = None;

    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, chunk_len, Ok(v))) => {
                chunks[idx] = Some(v);
                let _ = progress.update(|p| EmbedProgress {
                    completed_chunks: p.completed_chunks + 1,
                    completed_texts: p.completed_texts + chunk_len as u64,
                    ..p.clone()
                });
            }
            Ok((_, _chunk_len, Err(e))) => {
                let _ = progress.update(|p| EmbedProgress {
                    completed_chunks: p.completed_chunks + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
            Err(e) => {
                let _ = progress.update(|p| EmbedProgress {
                    completed_chunks: p.completed_chunks + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
                if first_err.is_none() {
                    first_err = Some(Error::other(format!("embed task join: {e}")));
                }
            }
        }
    }

    if let Some(e) = first_err {
        return Err(e);
    }

    let mut out: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for c in chunks.into_iter() {
        let v = c.ok_or_else(|| Error::other("embed chunk lost"))?;
        out.extend(v);
    }
    if out.len() != texts.len() {
        return Err(Error::other(format!(
            "embed_documents_concurrent_with_progress: expected {} embeddings, got {}",
            texts.len(),
            out.len(),
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::Embeddings;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Returns a deterministic embedding per text (length-encoded) and
    /// records peak concurrent invocations.
    struct ProbeEmbed {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
        chunk_sizes_seen: Arc<std::sync::Mutex<Vec<usize>>>,
        delay_ms: u64,
    }

    #[async_trait]
    impl Embeddings for ProbeEmbed {
        fn name(&self) -> &str {
            "probe"
        }
        fn dimensions(&self) -> usize {
            2
        }
        async fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![0.0, 0.0])
        }
        async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.chunk_sizes_seen.lock().unwrap().push(texts.len());
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            let mut peak = self.peak.load(Ordering::SeqCst);
            while now > peak {
                match self.peak.compare_exchange(peak, now, Ordering::SeqCst, Ordering::SeqCst) {
                    Ok(_) => break,
                    Err(actual) => peak = actual,
                }
            }
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            // Embedding for "tN" is [N, len("tN")] — uniquely identifies the input.
            Ok(texts
                .iter()
                .map(|t| {
                    let n: f32 = t
                        .strip_prefix('t')
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(-1.0);
                    vec![n, t.len() as f32]
                })
                .collect())
        }
    }

    fn probe(delay_ms: u64) -> (Arc<dyn Embeddings>, Arc<AtomicUsize>, Arc<std::sync::Mutex<Vec<usize>>>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let chunk_sizes_seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let p = ProbeEmbed {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: peak.clone(),
            chunk_sizes_seen: chunk_sizes_seen.clone(),
            delay_ms,
        };
        (Arc::new(p), peak, chunk_sizes_seen)
    }

    fn ts(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("t{i}")).collect()
    }

    #[tokio::test]
    async fn empty_input_returns_empty() {
        let (e, _peak, _seen) = probe(0);
        let out = embed_documents_concurrent(e, &[], 10, 4).await.unwrap();
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_aligned_to_input_order() {
        let (e, _peak, _seen) = probe(2);
        let texts = ts(11); // chunk_size=4 → 3 chunks of [4,4,3]
        let out = embed_documents_concurrent(e, &texts, 4, 4).await.unwrap();
        assert_eq!(out.len(), 11);
        for (i, v) in out.iter().enumerate() {
            assert_eq!(v[0] as usize, i, "input {i} got embedding for {:?}", v[0]);
        }
    }

    #[tokio::test]
    async fn chunks_split_by_chunk_size() {
        let (e, _peak, seen) = probe(0);
        let texts = ts(11);
        embed_documents_concurrent(e, &texts, 4, 4).await.unwrap();
        let mut sizes = seen.lock().unwrap().clone();
        sizes.sort();
        assert_eq!(sizes, vec![3, 4, 4]);
    }

    #[tokio::test]
    async fn concurrency_cap_honoured() {
        let (e, peak, _seen) = probe(20);
        let texts = ts(40);
        embed_documents_concurrent(e, &texts, 5, 3).await.unwrap();
        let p = peak.load(Ordering::SeqCst);
        assert!(p <= 3, "peak {p} > cap 3");
        assert!(p >= 2, "peak {p} < 2 — concurrency never engaged");
    }

    #[tokio::test]
    async fn zero_chunk_size_means_single_chunk() {
        let (e, _peak, seen) = probe(0);
        let texts = ts(7);
        embed_documents_concurrent(e, &texts, 0, 4).await.unwrap();
        let sizes = seen.lock().unwrap().clone();
        assert_eq!(sizes, vec![7]);
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let (e, peak, _seen) = probe(2);
        let texts = ts(8);
        embed_documents_concurrent(e, &texts, 2, 0).await.unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn chunk_failure_aborts() {
        struct FlakyEmbed {
            seen: Arc<AtomicUsize>,
            fail_on: usize,
        }
        #[async_trait]
        impl Embeddings for FlakyEmbed {
            fn name(&self) -> &str {
                "flaky"
            }
            fn dimensions(&self) -> usize {
                1
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst);
                if n == self.fail_on {
                    return Err(Error::other("synthetic chunk fail"));
                }
                Ok(texts.iter().map(|_| vec![0.0]).collect())
            }
        }
        let e: Arc<dyn Embeddings> = Arc::new(FlakyEmbed {
            seen: Arc::new(AtomicUsize::new(0)),
            fail_on: 1,
        });
        let texts = ts(20);
        let r = embed_documents_concurrent(e, &texts, 5, 4).await;
        assert!(r.is_err());
        let msg = format!("{}", r.err().unwrap());
        assert!(msg.contains("synthetic chunk fail"), "got: {msg}");
    }

    #[tokio::test]
    async fn chunk_size_larger_than_input_yields_one_chunk() {
        let (e, _peak, seen) = probe(0);
        let texts = ts(3);
        embed_documents_concurrent(e, &texts, 1000, 4).await.unwrap();
        let sizes = seen.lock().unwrap().clone();
        assert_eq!(sizes, vec![3]);
    }

    // ---- embed_documents_concurrent_with_shutdown tests ---------------

    #[tokio::test]
    async fn shutdown_no_signal_completes_normally() {
        let (e, _peak, _seen) = probe(0);
        let texts = ts(11);
        let shutdown = ShutdownSignal::new();
        let out = embed_documents_concurrent_with_shutdown(e, &texts, 4, 4, &shutdown)
            .await;
        assert_eq!(out.len(), 3);
        assert!(out.iter().all(|r| r.is_ok()));
    }

    #[tokio::test]
    async fn shutdown_pre_fired_returns_all_cancelled() {
        let (e, _peak, _seen) = probe(0);
        let texts = ts(10); // chunk_size=4 → 3 chunks
        let shutdown = ShutdownSignal::new();
        shutdown.signal();
        let out = embed_documents_concurrent_with_shutdown(e, &texts, 4, 4, &shutdown)
            .await;
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
    async fn shutdown_mid_run_preserves_completed_chunks() {
        // 30 texts, chunk_size=2 → 15 chunks. delay_ms=30, cap=2 →
        // sequential ~225ms. Fire shutdown after 80ms; some chunks
        // complete, rest are cancelled.
        let (e, _peak, _seen) = probe(30);
        let texts = ts(30);
        let shutdown = ShutdownSignal::new();
        let s2 = shutdown.clone();
        tokio::spawn(async move {
            tokio::time::sleep(std::time::Duration::from_millis(80)).await;
            s2.signal();
        });
        let started = std::time::Instant::now();
        let out = embed_documents_concurrent_with_shutdown(e, &texts, 2, 2, &shutdown)
            .await;
        let elapsed = started.elapsed();
        assert!(
            elapsed < std::time::Duration::from_millis(250),
            "elapsed {elapsed:?} — shutdown didn't abort early",
        );
        assert_eq!(out.len(), 15);
        let ok_count = out.iter().filter(|r| r.is_ok()).count();
        let cancelled = out
            .iter()
            .filter(|r| {
                r.as_ref()
                    .err()
                    .map(|e| e.to_string().contains("cancelled by shutdown"))
                    .unwrap_or(false)
            })
            .count();
        assert!(ok_count >= 1, "expected at least one chunk to complete");
        assert!(cancelled >= 1, "expected at least one chunk cancelled");
    }

    #[tokio::test]
    async fn shutdown_empty_texts_returns_empty() {
        let (e, _peak, _seen) = probe(0);
        let shutdown = ShutdownSignal::new();
        let out =
            embed_documents_concurrent_with_shutdown(e, &[], 4, 4, &shutdown).await;
        assert!(out.is_empty());
    }

    // ---- embed_documents_concurrent_with_progress tests ----------------

    #[tokio::test]
    async fn progress_totals_set_at_start() {
        let (e, _peak, _seen) = probe(0);
        let texts = ts(11); // chunk_size=4 → 3 chunks
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let _ = embed_documents_concurrent_with_progress(e, &texts, 4, 4, progress)
            .await
            .unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.total_texts, 11);
        assert_eq!(snap.total_chunks, 3);
        assert_eq!(snap.completed_chunks, 3);
        assert_eq!(snap.completed_texts, 11);
        assert_eq!(snap.errors, 0);
    }

    #[tokio::test]
    async fn progress_records_chunk_failure() {
        struct FailEmbed {
            seen: Arc<AtomicUsize>,
            fail_on: usize,
        }
        #[async_trait]
        impl Embeddings for FailEmbed {
            fn name(&self) -> &str {
                "fail"
            }
            fn dimensions(&self) -> usize {
                1
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst);
                if n == self.fail_on {
                    return Err(Error::other("synthetic"));
                }
                Ok(texts.iter().map(|_| vec![0.0]).collect())
            }
        }
        let e: Arc<dyn Embeddings> = Arc::new(FailEmbed {
            seen: Arc::new(AtomicUsize::new(0)),
            fail_on: 1,
        });
        let texts = ts(15); // 3 chunks of 5
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let r = embed_documents_concurrent_with_progress(e, &texts, 5, 1, progress).await;
        assert!(r.is_err());
        let snap = obs.snapshot();
        assert_eq!(snap.total_chunks, 3);
        // Failed chunk still increments completed_chunks; errors=1.
        assert!(snap.errors >= 1);
    }

    #[tokio::test]
    async fn progress_observer_polls_mid_run() {
        let (e, _peak, _seen) = probe(15);
        let texts = ts(20); // chunk=4 → 5 chunks; with delay we see partial.
        let progress = Progress::new(EmbedProgress::default());
        let mut obs = progress.observer();
        // Spawn into an owning closure so the borrowed `&[String]`
        // arg lives long enough for the spawned task.
        let progress_clone = progress.clone();
        let h = tokio::spawn(async move {
            embed_documents_concurrent_with_progress(e, &texts, 4, 2, progress_clone)
                .await
        });
        // Wait for any progress update.
        let _ = obs.changed().await;
        let mid = obs.snapshot();
        assert_eq!(mid.total_texts, 20);
        assert_eq!(mid.total_chunks, 5);
        let _ = h.await.unwrap().unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.completed_chunks, 5);
        assert_eq!(snap.completed_texts, 20);
    }

    // ---- embed_documents_concurrent_stream tests ---------------------

    use futures::StreamExt;

    // ---- embed_documents_concurrent_stream_with_progress tests --------

    #[tokio::test]
    async fn stream_with_progress_yields_items_and_advances_counters() {
        let (e, _peak, _seen) = probe(5);
        let texts = ts(11);
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let mut s = embed_documents_concurrent_stream_with_progress(
            e, texts, 4, 4, progress,
        );
        let mut count = 0;
        let mut total_embeddings = 0;
        while let Some((_idx, r)) = s.next().await {
            let v = r.unwrap();
            total_embeddings += v.len();
            count += 1;
        }
        assert_eq!(count, 3); // 11 ÷ 4 → 3 chunks
        assert_eq!(total_embeddings, 11);
        let snap = obs.snapshot();
        assert_eq!(snap.total_texts, 11);
        assert_eq!(snap.total_chunks, 3);
        assert_eq!(snap.completed_chunks, 3);
        assert_eq!(snap.completed_texts, 11);
    }

    #[tokio::test]
    async fn stream_with_progress_records_chunk_failure() {
        struct FailEmbed {
            seen: Arc<AtomicUsize>,
            fail_on: usize,
        }
        #[async_trait]
        impl Embeddings for FailEmbed {
            fn name(&self) -> &str {
                "fail"
            }
            fn dimensions(&self) -> usize {
                1
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst);
                if n == self.fail_on {
                    return Err(Error::other("synthetic"));
                }
                Ok(texts.iter().map(|_| vec![0.0]).collect())
            }
        }
        let e: Arc<dyn Embeddings> = Arc::new(FailEmbed {
            seen: Arc::new(AtomicUsize::new(0)),
            fail_on: 1,
        });
        let texts = ts(15);
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let mut s = embed_documents_concurrent_stream_with_progress(
            e, texts, 5, 1, progress,
        );
        let mut errors_in_stream = 0;
        while let Some((_idx, r)) = s.next().await {
            if r.is_err() {
                errors_in_stream += 1;
            }
        }
        let snap = obs.snapshot();
        assert!(snap.errors >= 1);
        assert_eq!(errors_in_stream, snap.errors as usize);
    }

    #[tokio::test]
    async fn stream_with_progress_total_set_at_start() {
        let (e, _peak, _seen) = probe(0);
        let texts = ts(10);
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let _s = embed_documents_concurrent_stream_with_progress(
            e, texts, 4, 2, progress,
        );
        let snap = obs.snapshot();
        assert_eq!(snap.total_texts, 10);
        assert_eq!(snap.total_chunks, 3); // 4 + 4 + 2
    }

    #[tokio::test]
    async fn stream_with_progress_empty_input_no_updates() {
        let (e, _peak, _seen) = probe(0);
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let mut s = embed_documents_concurrent_stream_with_progress(
            e,
            vec![],
            4,
            4,
            progress,
        );
        assert!(s.next().await.is_none());
        assert_eq!(obs.snapshot(), EmbedProgress::default());
    }

    #[tokio::test]
    async fn stream_yields_one_item_per_chunk() {
        let (e, _peak, _seen) = probe(0);
        let texts = ts(11); // chunk_size=4 → 3 chunks (4 + 4 + 3)
        let mut s = embed_documents_concurrent_stream(e, texts, 4, 4);
        let mut got_chunks: Vec<usize> = Vec::new();
        let mut total_embeddings = 0usize;
        while let Some((idx, r)) = s.next().await {
            let v = r.unwrap();
            total_embeddings += v.len();
            got_chunks.push(idx);
        }
        got_chunks.sort();
        assert_eq!(got_chunks, vec![0, 1, 2]);
        assert_eq!(total_embeddings, 11);
    }

    #[tokio::test]
    async fn stream_chunk_idx_aligns_with_chunk_size() {
        // chunk_size=5 → texts[0..5] is chunk 0, texts[5..10] is chunk 1.
        // The probe encodes input idx in dim 0 of the embedding;
        // chunk 0's first embedding therefore has dim0 = 0.
        let (e, _peak, _seen) = probe(0);
        let texts = ts(10);
        let mut s = embed_documents_concurrent_stream(e, texts, 5, 4);
        // Reassemble into a Vec keyed by chunk idx.
        let mut by_chunk: std::collections::HashMap<usize, Vec<Vec<f32>>> =
            std::collections::HashMap::new();
        while let Some((idx, r)) = s.next().await {
            by_chunk.insert(idx, r.unwrap());
        }
        let chunk_0 = by_chunk.remove(&0).unwrap();
        let chunk_1 = by_chunk.remove(&1).unwrap();
        // Chunk 0 starts at input idx 0: dim 0 = 0.
        assert_eq!(chunk_0[0][0] as usize, 0);
        // Chunk 1 starts at input idx 5: dim 0 = 5.
        assert_eq!(chunk_1[0][0] as usize, 5);
    }

    #[tokio::test]
    async fn stream_per_chunk_failure_arrives_as_err_item() {
        struct FailEmbed {
            seen: Arc<AtomicUsize>,
            fail_on: usize,
        }
        #[async_trait]
        impl Embeddings for FailEmbed {
            fn name(&self) -> &str {
                "fail"
            }
            fn dimensions(&self) -> usize {
                1
            }
            async fn embed_query(&self, _t: &str) -> Result<Vec<f32>> {
                Ok(vec![0.0])
            }
            async fn embed_documents(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
                let n = self.seen.fetch_add(1, Ordering::SeqCst);
                if n == self.fail_on {
                    return Err(Error::other("synthetic"));
                }
                Ok(texts.iter().map(|_| vec![0.0]).collect())
            }
        }
        let e: Arc<dyn Embeddings> = Arc::new(FailEmbed {
            seen: Arc::new(AtomicUsize::new(0)),
            fail_on: 1,
        });
        let texts = ts(15); // 3 chunks of 5
        let mut s = embed_documents_concurrent_stream(e, texts, 5, 1);
        let mut count = 0;
        let mut errors = 0;
        while let Some((_idx, r)) = s.next().await {
            count += 1;
            if r.is_err() {
                errors += 1;
            }
        }
        assert_eq!(count, 3);
        assert_eq!(errors, 1);
    }

    #[tokio::test]
    async fn stream_empty_input_yields_empty() {
        let (e, _peak, _seen) = probe(0);
        let mut s = embed_documents_concurrent_stream(e, vec![], 4, 4);
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_caller_drop_aborts_in_flight_chunks() {
        // 30 chunks each delayed 50ms with cap=2 — full sequential
        // would take ~750ms. Drop after 1 item; total wall-clock
        // should be far less.
        let (e, _peak, _seen) = probe(50);
        let texts = ts(30);
        let started = std::time::Instant::now();
        {
            let mut s = embed_documents_concurrent_stream(e, texts, 1, 2);
            let _first = s.next().await.unwrap();
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 400,
            "elapsed {elapsed_ms}ms — caller-drop didn't abort remaining chunks",
        );
    }

    #[tokio::test]
    async fn progress_empty_texts_no_updates() {
        let (e, _peak, _seen) = probe(0);
        let progress = Progress::new(EmbedProgress::default());
        let obs = progress.observer();
        let out = embed_documents_concurrent_with_progress(e, &[], 4, 4, progress)
            .await
            .unwrap();
        assert!(out.is_empty());
        assert_eq!(obs.snapshot(), EmbedProgress::default());
    }
}

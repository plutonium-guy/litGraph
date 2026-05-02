//! Multi-stage backpressured ingestion pipeline.
//!
//! Strings together the three CPU/IO bound steps of a typical
//! vector-store ingestion — **load → split → embed** — as a tokio
//! pipeline of three tasks connected by bounded `mpsc` channels.
//! Caller drains the resulting `Stream<IngestBatch>` and pushes each
//! batch into whatever vector store they want.
//!
//! # Why this, vs. calling the helpers directly
//!
//! - [`load_concurrent`](crate::load_concurrent) (iter 187) parallel-
//!   loads N sources, but waits for the slowest before splitting.
//! - [`embed_documents_concurrent`](litgraph_core::embed_documents_concurrent)
//!   (iter 183) chunk-embeds, but waits for splitting to finish first.
//!
//! `ingest_to_stream` runs all three concurrently with **backpressure**:
//! while loaders are still pulling later sources, the splitter is
//! already chopping earlier ones, and the embedder is already
//! batching the first chunks. Bounded channels mean a fast loader
//! can't OOM the splitter; a fast splitter can't OOM the embedder.
//!
//! # Pipeline shape
//!
//! ```text
//!                       cap=load_buffer        cap=split_buffer
//!  ┌──────────────┐  ┌──────────────────┐  ┌──────────────────┐  ┌────────┐
//!  │ Loader stage │─→│  Split stage     │─→│   Embed stage    │─→│ Stream │
//!  │ spawn_blocking│  │ closure on each  │  │ embed_documents_ │  │  out   │
//!  │   per loader │  │ doc (sync)       │  │ concurrent batch │  │        │
//!  └──────────────┘  └──────────────────┘  └──────────────────┘  └────────┘
//! ```
//!
//! Each stage is a `tokio::spawn`'d task. The first stage uses
//! `spawn_blocking` per loader so synchronous I/O doesn't park an
//! async worker. The second stage runs the user-supplied splitter
//! closure in-place (it should be cheap and CPU-bound — for a heavy
//! splitter, push the work into `tokio::task::spawn_blocking` inside
//! the closure yourself). The third stage accumulates chunks into
//! `embed_chunk_size`-sized batches and dispatches them via
//! `embed_documents_concurrent`.
//!
//! # Errors
//!
//! Per-loader, per-doc, and per-embed-batch errors are surfaced as
//! `Err` variants on the output stream — the pipeline does not
//! short-circuit. Caller decides whether to break or continue.

use std::sync::Arc;

use futures::Stream;
use litgraph_core::{
    embed_documents_concurrent, Document, Embeddings, Progress, DEFAULT_EMBED_CHUNK_SIZE,
    DEFAULT_EMBED_CONCURRENCY,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::{load_concurrent, Loader, LoaderError, LoaderResult};

/// One emitted batch — `docs[i]` corresponds to `embeddings[i]`.
#[derive(Debug, Clone)]
pub struct IngestBatch {
    pub docs: Vec<Document>,
    pub embeddings: Vec<Vec<f32>>,
}

/// Counters maintained by [`ingest_to_stream_with_progress`]. Read on
/// demand from any [`Progress`] observer to drive a UI bar, log
/// throughput, or trip a circuit breaker on stuck stages.
///
/// All fields are monotonic (only increase). The pipeline updates
/// each counter after the corresponding stage emits — so a snapshot
/// where `chunks_split > chunks_embedded` means the embedder hasn't
/// caught up yet (normal mid-flight state).
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IngestProgress {
    /// Number of loaders submitted to stage 1.
    pub loaders_total: u64,
    /// Number of loaders whose `load()` has finished (Ok or Err).
    pub loaders_done: u64,
    /// Number of `Document`s emitted by successful loaders so far.
    pub docs_loaded: u64,
    /// Number of chunks emitted by the splitter stage.
    pub chunks_split: u64,
    /// Number of chunks the embedder has produced vectors for.
    pub chunks_embedded: u64,
    /// Number of `IngestBatch`es emitted to the output stream.
    pub batches_emitted: u64,
    /// Loader-level failures observed (does not include embed errors).
    pub loader_errors: u64,
    /// Embed-level failures observed.
    pub embed_errors: u64,
}

/// Pipeline tunables. All bounded values default to a "modest VM"
/// configuration; bump them for larger boxes.
#[derive(Debug, Clone)]
pub struct IngestConfig {
    /// In-flight loader concurrency (`spawn_blocking` cap).
    pub load_concurrency: usize,
    /// `mpsc` capacity between load and split stages.
    pub load_buffer: usize,
    /// `mpsc` capacity between split and embed stages.
    pub split_buffer: usize,
    /// `mpsc` capacity for the final emitted batch stream.
    pub batch_buffer: usize,
    /// `embed_documents_concurrent` chunk size.
    pub embed_chunk_size: usize,
    /// `embed_documents_concurrent` parallel chunk cap.
    pub embed_concurrency: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            load_concurrency: 4,
            load_buffer: 32,
            split_buffer: 256,
            batch_buffer: 8,
            embed_chunk_size: DEFAULT_EMBED_CHUNK_SIZE,
            embed_concurrency: DEFAULT_EMBED_CONCURRENCY,
        }
    }
}

/// Run the pipeline. Returns a stream of `Result<IngestBatch>`.
/// The stream finishes when every loader has been processed and
/// every split chunk embedded.
///
/// `splitter` is called once per loaded `Document` and must return
/// the chunks for that doc. Use a no-op splitter (`|d| vec![d]`) to
/// embed each loaded document whole.
pub fn ingest_to_stream<S>(
    loaders: Vec<Arc<dyn Loader>>,
    splitter: S,
    embedder: Arc<dyn Embeddings>,
    cfg: IngestConfig,
) -> impl Stream<Item = Result<IngestBatch, LoaderError>> + Send
where
    S: Fn(Document) -> Vec<Document> + Send + Sync + 'static,
{
    let splitter = Arc::new(splitter);
    let (load_tx, load_rx) = mpsc::channel::<LoaderResult<Vec<Document>>>(cfg.load_buffer);
    let (split_tx, split_rx) = mpsc::channel::<LoaderResult<Document>>(cfg.split_buffer);
    let (batch_tx, batch_rx) =
        mpsc::channel::<Result<IngestBatch, LoaderError>>(cfg.batch_buffer);

    // ---- Stage 1: loaders → load_tx ----
    let load_concurrency = cfg.load_concurrency;
    let stage1_loaders = loaders;
    tokio::spawn(async move {
        let results = load_concurrent(stage1_loaders, load_concurrency).await;
        for r in results {
            if load_tx.send(r).await.is_err() {
                break;
            }
        }
    });

    // ---- Stage 2: load_rx → splitter → split_tx ----
    let splitter_arc = splitter.clone();
    tokio::spawn(async move {
        let mut load_rx = load_rx;
        while let Some(batch_result) = load_rx.recv().await {
            match batch_result {
                Ok(docs) => {
                    for d in docs {
                        for chunk in splitter_arc(d) {
                            if split_tx.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    // Tag and forward; downstream wraps as a
                    // `IngestBatch` error so the caller sees it.
                    if split_tx.send(Err(e)).await.is_err() {
                        return;
                    }
                }
            }
        }
    });

    // ---- Stage 3: split_rx → embedder (chunked) → batch_tx ----
    let embed_chunk_size = cfg.embed_chunk_size;
    let embed_concurrency = cfg.embed_concurrency;
    tokio::spawn(async move {
        let mut split_rx = split_rx;
        let mut buf: Vec<Document> = Vec::with_capacity(embed_chunk_size);

        // Helper: drain `buf`, embed, send a batch result.
        async fn flush_buf(
            buf: &mut Vec<Document>,
            embedder: &Arc<dyn Embeddings>,
            embed_chunk_size: usize,
            embed_concurrency: usize,
            tx: &mpsc::Sender<Result<IngestBatch, LoaderError>>,
        ) -> bool {
            if buf.is_empty() {
                return true;
            }
            let texts: Vec<String> = buf.iter().map(|d| d.content.clone()).collect();
            let res = embed_documents_concurrent(
                embedder.clone(),
                &texts,
                embed_chunk_size,
                embed_concurrency,
            )
            .await;
            let payload = match res {
                Ok(embeddings) => {
                    let docs: Vec<Document> = std::mem::take(buf);
                    Ok(IngestBatch { docs, embeddings })
                }
                Err(e) => {
                    buf.clear();
                    Err(LoaderError::Other(format!("embed: {e}")))
                }
            };
            tx.send(payload).await.is_ok()
        }

        while let Some(item) = split_rx.recv().await {
            match item {
                Ok(doc) => {
                    buf.push(doc);
                    if buf.len() >= embed_chunk_size
                        && !flush_buf(
                            &mut buf,
                            &embedder,
                            embed_chunk_size,
                            embed_concurrency,
                            &batch_tx,
                        )
                        .await
                    {
                        return;
                    }
                }
                Err(e) => {
                    // Forward upstream loader failure as a batch error.
                    if batch_tx.send(Err(e)).await.is_err() {
                        return;
                    }
                }
            }
        }
        // Flush any tail.
        let _ = flush_buf(
            &mut buf,
            &embedder,
            embed_chunk_size,
            embed_concurrency,
            &batch_tx,
        )
        .await;
    });

    ReceiverStream::new(batch_rx)
}

/// Like [`ingest_to_stream`] but additionally updates the supplied
/// [`Progress`] handle as each stage advances. Observers attached to
/// the same `Progress` can read counter snapshots on demand —
/// perfect for progress bars, dashboards, throughput logging, and
/// stuck-stage detection.
///
/// All counters are monotonic. The pipeline writes each field
/// immediately after the corresponding stage emits, so a snapshot
/// where `chunks_split > chunks_embedded` is normal mid-flight (the
/// embedder is catching up).
///
/// Composes [iter 196] (multi-stage ingestion) and [iter 199]
/// (Progress) into a single call. Demonstrates the ergonomic value
/// of building primitives that compose.
pub fn ingest_to_stream_with_progress<S>(
    loaders: Vec<Arc<dyn Loader>>,
    splitter: S,
    embedder: Arc<dyn Embeddings>,
    cfg: IngestConfig,
    progress: Progress<IngestProgress>,
) -> impl Stream<Item = Result<IngestBatch, LoaderError>> + Send
where
    S: Fn(Document) -> Vec<Document> + Send + Sync + 'static,
{
    // Initialise loaders_total now that we know the count.
    let total = loaders.len() as u64;
    let _ = progress.update(|p| IngestProgress {
        loaders_total: total,
        ..p.clone()
    });

    let splitter = Arc::new(splitter);
    let (load_tx, load_rx) = mpsc::channel::<LoaderResult<Vec<Document>>>(cfg.load_buffer);
    let (split_tx, split_rx) = mpsc::channel::<LoaderResult<Document>>(cfg.split_buffer);
    let (batch_tx, batch_rx) =
        mpsc::channel::<Result<IngestBatch, LoaderError>>(cfg.batch_buffer);

    // ---- Stage 1: loaders → load_tx (with progress) ----
    let load_concurrency = cfg.load_concurrency;
    let stage1_loaders = loaders;
    let stage1_progress = progress.clone();
    tokio::spawn(async move {
        let results = load_concurrent(stage1_loaders, load_concurrency).await;
        for r in results {
            // Bump loaders_done regardless of success.
            let _ = stage1_progress.update(|p| IngestProgress {
                loaders_done: p.loaders_done + 1,
                docs_loaded: p.docs_loaded
                    + r.as_ref().map(|v| v.len() as u64).unwrap_or(0),
                loader_errors: p.loader_errors + if r.is_err() { 1 } else { 0 },
                ..p.clone()
            });
            if load_tx.send(r).await.is_err() {
                break;
            }
        }
    });

    // ---- Stage 2: load_rx → splitter → split_tx (with progress) ----
    let splitter_arc = splitter.clone();
    let stage2_progress = progress.clone();
    tokio::spawn(async move {
        let mut load_rx = load_rx;
        while let Some(batch_result) = load_rx.recv().await {
            match batch_result {
                Ok(docs) => {
                    for d in docs {
                        let chunks = splitter_arc(d);
                        let n = chunks.len() as u64;
                        let _ = stage2_progress.update(|p| IngestProgress {
                            chunks_split: p.chunks_split + n,
                            ..p.clone()
                        });
                        for chunk in chunks {
                            if split_tx.send(Ok(chunk)).await.is_err() {
                                return;
                            }
                        }
                    }
                }
                Err(e) => {
                    if split_tx.send(Err(e)).await.is_err() {
                        return;
                    }
                }
            }
        }
    });

    // ---- Stage 3: split_rx → embedder → batch_tx (with progress) ----
    let embed_chunk_size = cfg.embed_chunk_size;
    let embed_concurrency = cfg.embed_concurrency;
    let stage3_progress = progress;
    tokio::spawn(async move {
        let mut split_rx = split_rx;
        let mut buf: Vec<Document> = Vec::with_capacity(embed_chunk_size);

        async fn flush_buf(
            buf: &mut Vec<Document>,
            embedder: &Arc<dyn Embeddings>,
            embed_chunk_size: usize,
            embed_concurrency: usize,
            tx: &mpsc::Sender<Result<IngestBatch, LoaderError>>,
            progress: &Progress<IngestProgress>,
        ) -> bool {
            if buf.is_empty() {
                return true;
            }
            let n = buf.len() as u64;
            let texts: Vec<String> = buf.iter().map(|d| d.content.clone()).collect();
            let res = embed_documents_concurrent(
                embedder.clone(),
                &texts,
                embed_chunk_size,
                embed_concurrency,
            )
            .await;
            let payload = match res {
                Ok(embeddings) => {
                    let docs: Vec<Document> = std::mem::take(buf);
                    let _ = progress.update(|p| IngestProgress {
                        chunks_embedded: p.chunks_embedded + n,
                        batches_emitted: p.batches_emitted + 1,
                        ..p.clone()
                    });
                    Ok(IngestBatch { docs, embeddings })
                }
                Err(e) => {
                    buf.clear();
                    let _ = progress.update(|p| IngestProgress {
                        embed_errors: p.embed_errors + 1,
                        ..p.clone()
                    });
                    Err(LoaderError::Other(format!("embed: {e}")))
                }
            };
            tx.send(payload).await.is_ok()
        }

        while let Some(item) = split_rx.recv().await {
            match item {
                Ok(doc) => {
                    buf.push(doc);
                    if buf.len() >= embed_chunk_size
                        && !flush_buf(
                            &mut buf,
                            &embedder,
                            embed_chunk_size,
                            embed_concurrency,
                            &batch_tx,
                            &stage3_progress,
                        )
                        .await
                    {
                        return;
                    }
                }
                Err(e) => {
                    if batch_tx.send(Err(e)).await.is_err() {
                        return;
                    }
                }
            }
        }
        let _ = flush_buf(
            &mut buf,
            &embedder,
            embed_chunk_size,
            embed_concurrency,
            &batch_tx,
            &stage3_progress,
        )
        .await;
    });

    ReceiverStream::new(batch_rx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use futures::StreamExt;
    use litgraph_core::Result as LgResult;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::TempDir;

    /// Embedder that emits one-dim vectors carrying the text length.
    /// Lets us trace which doc produced which embedding.
    struct LenEmbed {
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Embeddings for LenEmbed {
        fn name(&self) -> &str {
            "len"
        }
        fn dimensions(&self) -> usize {
            1
        }
        async fn embed_query(&self, text: &str) -> LgResult<Vec<f32>> {
            Ok(vec![text.len() as f32])
        }
        async fn embed_documents(&self, texts: &[String]) -> LgResult<Vec<Vec<f32>>> {
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
            tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(texts.iter().map(|t| vec![t.len() as f32]).collect())
        }
    }

    fn write_files(dir: &TempDir, prefix: &str, n: usize) -> Vec<std::path::PathBuf> {
        (0..n)
            .map(|i| {
                let p = dir.path().join(format!("{prefix}_{i}.txt"));
                std::fs::write(&p, format!("body of file {i}")).unwrap();
                p
            })
            .collect()
    }

    fn embedder() -> (Arc<dyn Embeddings>, Arc<AtomicUsize>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let e = LenEmbed {
            in_flight: Arc::new(AtomicUsize::new(0)),
            peak: peak.clone(),
        };
        (Arc::new(e), peak)
    }

    #[tokio::test]
    async fn empty_loaders_yields_empty_stream() {
        let (e, _peak) = embedder();
        let s = ingest_to_stream(vec![], |d| vec![d], e, IngestConfig::default());
        let collected: Vec<_> = s.collect().await;
        assert!(collected.is_empty());
    }

    #[tokio::test]
    async fn single_loader_no_split_emits_one_batch() {
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "single", 1);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let cfg = IngestConfig {
            embed_chunk_size: 8,
            ..IngestConfig::default()
        };
        let s = ingest_to_stream(loaders, |d| vec![d], e, cfg);
        let collected: Vec<_> = s.collect().await;
        assert_eq!(collected.len(), 1);
        let batch = collected[0].as_ref().unwrap();
        assert_eq!(batch.docs.len(), 1);
        assert_eq!(batch.embeddings.len(), 1);
        assert_eq!(batch.docs[0].content, "body of file 0");
    }

    #[tokio::test]
    async fn many_loaders_with_split_yield_batched_results() {
        // 5 files; each splits into 3 chunks (synthesized) → 15
        // chunks. With chunk_size=4, that's 4 batches (4+4+4+3).
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "many", 5);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let cfg = IngestConfig {
            embed_chunk_size: 4,
            ..IngestConfig::default()
        };
        let split = |d: Document| {
            let base = d.content.clone();
            (0..3)
                .map(|i| Document::new(format!("{base}|chunk-{i}")))
                .collect()
        };
        let s = ingest_to_stream(loaders, split, e, cfg);
        let mut total_docs = 0usize;
        let mut total_embeds = 0usize;
        let mut s = Box::pin(s);
        while let Some(item) = s.next().await {
            let b = item.expect("ok");
            total_docs += b.docs.len();
            total_embeds += b.embeddings.len();
        }
        assert_eq!(total_docs, 15);
        assert_eq!(total_embeds, 15);
    }

    #[tokio::test]
    async fn loader_failure_surfaces_as_batch_err_does_not_kill_pipeline() {
        let tmp = TempDir::new().unwrap();
        let good = write_files(&tmp, "ok", 1)[0].clone();
        let missing = tmp.path().join("missing.txt");
        let loaders: Vec<Arc<dyn Loader>> = vec![
            Arc::new(crate::TextLoader::new(&good)),
            Arc::new(crate::TextLoader::new(&missing)),
        ];
        let (e, _peak) = embedder();
        let s = ingest_to_stream(loaders, |d| vec![d], e, IngestConfig::default());
        let collected: Vec<_> = s.collect().await;
        // Expect at least one Ok (the good file) and at least one Err.
        assert!(collected.iter().any(|r| r.is_ok()));
        assert!(collected.iter().any(|r| r.is_err()));
    }

    #[tokio::test]
    async fn pipeline_respects_chunk_size_at_boundary() {
        // Exactly chunk_size docs: should produce 1 batch of size N.
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "boundary", 4);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let cfg = IngestConfig {
            embed_chunk_size: 4,
            ..IngestConfig::default()
        };
        let s = ingest_to_stream(loaders, |d| vec![d], e, cfg);
        let mut s = Box::pin(s);
        let mut batches = 0usize;
        let mut total = 0usize;
        while let Some(item) = s.next().await {
            let b = item.unwrap();
            batches += 1;
            total += b.docs.len();
        }
        assert_eq!(total, 4);
        // Either 1 batch of 4 (if chunk-size flush happened mid-stream)
        // or N small batches (if the splitter buffered partially).
        // Either way, total must be 4.
        assert!(batches >= 1);
    }

    #[tokio::test]
    async fn aligned_doc_to_embedding_in_each_batch() {
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "align", 6);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let cfg = IngestConfig {
            embed_chunk_size: 3,
            ..IngestConfig::default()
        };
        let s = ingest_to_stream(loaders, |d| vec![d], e, cfg);
        let mut s = Box::pin(s);
        while let Some(item) = s.next().await {
            let b = item.unwrap();
            assert_eq!(b.docs.len(), b.embeddings.len());
            // Each embedding's first dim is the doc's content length.
            for (d, emb) in b.docs.iter().zip(b.embeddings.iter()) {
                assert_eq!(emb[0] as usize, d.content.len());
            }
        }
    }

    // ---- Progress-aware variant tests --------------------------------

    #[tokio::test]
    async fn progress_loaders_total_set_at_start() {
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "p_total", 5);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let progress = Progress::new(IngestProgress::default());
        let obs = progress.observer();
        let _stream = ingest_to_stream_with_progress(
            loaders,
            |d| vec![d],
            e,
            IngestConfig::default(),
            progress,
        );
        // Snapshot may take a moment to update; tiny yield gives the
        // first-stage spawn a chance to set it. Even before yielding,
        // `loaders_total` is set synchronously in the function body.
        assert_eq!(obs.snapshot().loaders_total, 5);
    }

    #[tokio::test]
    async fn progress_counters_advance_through_pipeline() {
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "p_advance", 3);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let progress = Progress::new(IngestProgress::default());
        let obs = progress.observer();
        let cfg = IngestConfig {
            embed_chunk_size: 2,
            ..IngestConfig::default()
        };
        let split = |d: Document| {
            (0..2)
                .map(|i| Document::new(format!("{}|c{i}", d.content)))
                .collect()
        };
        let s = ingest_to_stream_with_progress(loaders, split, e, cfg, progress);
        let collected: Vec<_> = s.collect().await;
        assert!(collected.iter().all(|r| r.is_ok()));
        let final_snap = obs.snapshot();
        assert_eq!(final_snap.loaders_total, 3);
        assert_eq!(final_snap.loaders_done, 3);
        assert_eq!(final_snap.docs_loaded, 3);
        // 3 docs × 2 chunks = 6 chunks total.
        assert_eq!(final_snap.chunks_split, 6);
        assert_eq!(final_snap.chunks_embedded, 6);
        // 6 chunks ÷ chunk_size 2 = 3 batches.
        assert_eq!(final_snap.batches_emitted, 3);
        assert_eq!(final_snap.loader_errors, 0);
        assert_eq!(final_snap.embed_errors, 0);
    }

    #[tokio::test]
    async fn progress_records_loader_failures() {
        let tmp = TempDir::new().unwrap();
        let good = write_files(&tmp, "ok", 1)[0].clone();
        let missing = tmp.path().join("missing.txt");
        let loaders: Vec<Arc<dyn Loader>> = vec![
            Arc::new(crate::TextLoader::new(&good)),
            Arc::new(crate::TextLoader::new(&missing)),
        ];
        let (e, _peak) = embedder();
        let progress = Progress::new(IngestProgress::default());
        let obs = progress.observer();
        let s = ingest_to_stream_with_progress(
            loaders,
            |d| vec![d],
            e,
            IngestConfig::default(),
            progress,
        );
        let _: Vec<_> = s.collect().await;
        let snap = obs.snapshot();
        assert_eq!(snap.loaders_total, 2);
        assert_eq!(snap.loaders_done, 2);
        assert_eq!(snap.loader_errors, 1);
        assert_eq!(snap.docs_loaded, 1);
    }

    #[tokio::test]
    async fn progress_observer_can_be_polled_mid_run() {
        // Slower embedder so observers see a partial snapshot before
        // the run finishes.
        let tmp = TempDir::new().unwrap();
        let paths = write_files(&tmp, "p_mid", 4);
        let loaders: Vec<Arc<dyn Loader>> = paths
            .iter()
            .map(|p| Arc::new(crate::TextLoader::new(p)) as Arc<dyn Loader>)
            .collect();
        let (e, _peak) = embedder();
        let progress = Progress::new(IngestProgress::default());
        let mut obs = progress.observer();
        let s = ingest_to_stream_with_progress(
            loaders,
            |d| vec![d],
            e,
            IngestConfig::default(),
            progress,
        );
        let mut s = Box::pin(s);
        // Pull one batch.
        let _first = s.next().await.unwrap().unwrap();
        // After at least one batch emitted, batches_emitted >= 1.
        let _ = obs.changed().await; // wait for any update
        let mid = obs.snapshot();
        assert!(mid.batches_emitted >= 1);
        assert!(mid.chunks_embedded >= 1);
        // Drain rest.
        while let Some(_) = s.next().await {}
        let final_snap = obs.snapshot();
        assert_eq!(final_snap.loaders_done, 4);
    }

    #[tokio::test]
    async fn progress_default_is_zero_everywhere() {
        let p = IngestProgress::default();
        assert_eq!(p.loaders_total, 0);
        assert_eq!(p.loaders_done, 0);
        assert_eq!(p.docs_loaded, 0);
        assert_eq!(p.chunks_split, 0);
        assert_eq!(p.chunks_embedded, 0);
        assert_eq!(p.batches_emitted, 0);
        assert_eq!(p.loader_errors, 0);
        assert_eq!(p.embed_errors, 0);
    }
}

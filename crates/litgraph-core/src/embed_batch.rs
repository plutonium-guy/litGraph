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

use std::sync::Arc;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::{Embeddings, Error, Result};

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
}

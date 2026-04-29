//! `tool_dispatch_concurrent` — bounded-concurrency parallel dispatch
//! across heterogeneous `(tool_name, args)` calls.
//!
//! Fourth in the parallel-batch family:
//!
//! | Iter | Helper                          | Domain                               |
//! |------|---------------------------------|--------------------------------------|
//! | 182  | [`batch_concurrent`]            | `ChatModel::invoke` (one model, N inputs) |
//! | 183  | [`embed_documents_concurrent`]  | `Embeddings::embed_documents` (one embedder, N chunks) |
//! | 190  | `retrieve_concurrent` (in `litgraph-retrieval`) | `Retriever::retrieve` (one retriever, N queries) |
//! | 191  | `tool_dispatch_concurrent` (this) | `Tool::run` (HETEROGENEOUS tools, N calls) |
//!
//! [`batch_concurrent`]: crate::batch_concurrent
//! [`embed_documents_concurrent`]: crate::embed_documents_concurrent
//!
//! # Why this exists
//!
//! `ReactAgent` already dispatches its own tool calls in parallel
//! within the agent loop. But callers who plan their own tool
//! sequences outside the loop — Plan-and-Execute agents, custom
//! orchestrators, batch CLI tools, eval harnesses — kept hand-rolling
//! the same `JoinSet + Semaphore` boilerplate. This is the standalone
//! helper.
//!
//! # Guarantees
//!
//! 1. Output index `i` matches `calls[i]` regardless of completion order.
//! 2. Per-call `Result` so a single failing tool doesn't tank the rest.
//! 3. `max_concurrency` is enforced via [`tokio::sync::Semaphore`] so
//!    bursting 10k calls with `max=8` stays bounded.
//! 4. Unknown tool names produce a per-call `Err(Error::other(...))`,
//!    not a crash.
//!
//! # Heterogeneous tools
//!
//! Unlike the trio above, each call routes to a *different* tool by
//! name — the input is a `HashMap<String, Arc<dyn Tool>>` registry
//! plus a list of `(name, args)`. Same parallelism shape, different
//! input topology.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::Arc;

use futures::Stream;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{mpsc, Semaphore};
use tokio::task::JoinSet;
use tokio_stream::wrappers::ReceiverStream;

use crate::tool::Tool;
use crate::{Error, Progress, Result};

/// Run `tools[name].run(args)` for every `(name, args)` in `calls`,
/// capped at `max_concurrency` in flight. Output is aligned 1:1 with
/// `calls` — slot `i` holds the outcome of `calls[i]`.
///
/// `max_concurrency = 0` is normalised to 1 (sequential). Unknown
/// tool names land in their slot as `Err`, not a global crash.
pub async fn tool_dispatch_concurrent(
    tools: HashMap<String, Arc<dyn Tool>>,
    calls: Vec<(String, Value)>,
    max_concurrency: usize,
) -> Vec<Result<Value>> {
    if calls.is_empty() {
        return Vec::new();
    }
    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let tools = Arc::new(tools);
    let mut set: JoinSet<(usize, Result<Value>)> = JoinSet::new();

    for (idx, (name, args)) in calls.into_iter().enumerate() {
        let sem = sem.clone();
        let tools = tools.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return (
                        idx,
                        Err(Error::other("tool dispatch semaphore closed")),
                    )
                }
            };
            let r = match tools.get(&name) {
                Some(tool) => tool.run(args).await,
                None => Err(Error::other(format!("unknown tool `{name}`"))),
            };
            (idx, r)
        });
    }

    let n = set.len();
    let mut results: Vec<Option<Result<Value>>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r)) => results[idx] = Some(r),
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("tool task join: {e}"))));
                }
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("tool slot lost"))))
        .collect()
}

/// Counters maintained by [`tool_dispatch_concurrent_with_progress`].
/// `unknown_tool_errors` is broken out from the generic `errors` count
/// because unknown-tool failures usually indicate a routing /
/// registry-mismatch bug rather than a transient runtime error —
/// callers may want to alert on it differently.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ToolDispatchProgress {
    /// Total calls submitted (set once on entry).
    pub total: u64,
    /// Calls whose `run` (or unknown-tool resolution) has finished.
    pub completed: u64,
    /// Subset of `completed` that returned `Err`. Includes the
    /// `unknown_tool_errors` count.
    pub errors: u64,
    /// Subset of `errors` whose failure was an "unknown tool name"
    /// — exposed separately so dashboards can distinguish a routing
    /// bug from a tool-execution failure.
    pub unknown_tool_errors: u64,
}

/// Same as [`tool_dispatch_concurrent`] but updates `progress` as
/// each call completes. Real prod use: a Plan-and-Execute agent
/// dispatching dozens of tool calls for a single plan, with a
/// dashboard rendering live progress.
///
/// Composition: fifth progress-aware sibling after iters 200, 205,
/// 206, 207. Unknown-tool errors get their own counter so a routing
/// regression (the LLM emitted a tool name your registry doesn't
/// know) shows up distinctly from a tool-runtime failure.
pub async fn tool_dispatch_concurrent_with_progress(
    tools: HashMap<String, Arc<dyn Tool>>,
    calls: Vec<(String, Value)>,
    max_concurrency: usize,
    progress: Progress<ToolDispatchProgress>,
) -> Vec<Result<Value>> {
    if calls.is_empty() {
        return Vec::new();
    }
    let total = calls.len() as u64;
    let _ = progress.update(|p| ToolDispatchProgress {
        total,
        ..p.clone()
    });

    let cap = max_concurrency.max(1);
    let sem = Arc::new(Semaphore::new(cap));
    let tools = Arc::new(tools);
    // Each task carries an `is_unknown` flag so the receiver can
    // attribute errors precisely without re-parsing error messages.
    let mut set: JoinSet<(usize, Result<Value>, bool)> = JoinSet::new();

    for (idx, (name, args)) in calls.into_iter().enumerate() {
        let sem = sem.clone();
        let tools = tools.clone();
        set.spawn(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return (
                        idx,
                        Err(Error::other("tool dispatch semaphore closed")),
                        false,
                    )
                }
            };
            match tools.get(&name) {
                Some(tool) => (idx, tool.run(args).await, false),
                None => (
                    idx,
                    Err(Error::other(format!("unknown tool `{name}`"))),
                    true,
                ),
            }
        });
    }

    let n = set.len();
    let mut results: Vec<Option<Result<Value>>> = (0..n).map(|_| None).collect();
    while let Some(joined) = set.join_next().await {
        match joined {
            Ok((idx, r, is_unknown)) => {
                let is_err = r.is_err();
                results[idx] = Some(r);
                let _ = progress.update(|p| ToolDispatchProgress {
                    completed: p.completed + 1,
                    errors: p.errors + if is_err { 1 } else { 0 },
                    unknown_tool_errors: p.unknown_tool_errors
                        + if is_unknown { 1 } else { 0 },
                    ..p.clone()
                });
            }
            Err(e) => {
                if let Some(slot) = results.iter_mut().find(|s| s.is_none()) {
                    *slot = Some(Err(Error::other(format!("tool task join: {e}"))));
                }
                let _ = progress.update(|p| ToolDispatchProgress {
                    completed: p.completed + 1,
                    errors: p.errors + 1,
                    ..p.clone()
                });
            }
        }
    }

    results
        .into_iter()
        .map(|s| s.unwrap_or_else(|| Err(Error::other("tool slot lost"))))
        .collect()
}

/// One item from [`tool_dispatch_concurrent_stream`] — the input
/// call index plus that call's outcome, emitted in completion order.
pub type ToolDispatchStreamItem = (usize, Result<Value>);

/// Streaming variant of [`tool_dispatch_concurrent`]. Yields
/// `(call_idx, Result<Value>)` pairs as each tool call completes —
/// caller drains in completion order, can react to fast tool
/// results immediately (e.g., feed each into a follow-up LLM
/// turn), and dropping the stream aborts in-flight tool calls.
///
/// Streaming-variant pattern from iters 210/211/212 extended to
/// the heterogeneous tool dispatch axis.
///
/// `max_concurrency = 0` is normalised to 1. Unknown tool names
/// produce per-call `Err` items the same way as
/// `tool_dispatch_concurrent`.
pub fn tool_dispatch_concurrent_stream(
    tools: HashMap<String, Arc<dyn Tool>>,
    calls: Vec<(String, Value)>,
    max_concurrency: usize,
) -> Pin<Box<dyn Stream<Item = ToolDispatchStreamItem> + Send>> {
    if calls.is_empty() {
        return Box::pin(futures::stream::empty());
    }
    let cap = max_concurrency.max(1);
    let n = calls.len();
    let buf = n.min(cap.max(8));
    let (tx, rx) = mpsc::channel::<ToolDispatchStreamItem>(buf);
    let tools = Arc::new(tools);

    tokio::spawn(async move {
        let sem = Arc::new(Semaphore::new(cap));
        let mut set: JoinSet<ToolDispatchStreamItem> = JoinSet::new();
        for (idx, (name, args)) in calls.into_iter().enumerate() {
            let sem = sem.clone();
            let tools = tools.clone();
            set.spawn(async move {
                let _permit = match sem.acquire_owned().await {
                    Ok(p) => p,
                    Err(_) => {
                        return (
                            idx,
                            Err(Error::other("tool dispatch semaphore closed")),
                        )
                    }
                };
                let r = match tools.get(&name) {
                    Some(tool) => tool.run(args).await,
                    None => Err(Error::other(format!("unknown tool `{name}`"))),
                };
                (idx, r)
            });
        }
        while let Some(joined) = set.join_next().await {
            let item = match joined {
                Ok(it) => it,
                Err(e) => (
                    usize::MAX,
                    Err(Error::other(format!("tool task join: {e}"))),
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

/// Like `tool_dispatch_concurrent` but fail-fast: returns `Err` on
/// the first failed call. Output aligned to inputs only on success.
pub async fn tool_dispatch_concurrent_fail_fast(
    tools: HashMap<String, Arc<dyn Tool>>,
    calls: Vec<(String, Value)>,
    max_concurrency: usize,
) -> Result<Vec<Value>> {
    let n = calls.len();
    let results = tool_dispatch_concurrent(tools, calls, max_concurrency).await;
    let mut out = Vec::with_capacity(n);
    for r in results {
        out.push(r?);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool::ToolSchema;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// Echoes its args, after a configurable sleep. Records peak
    /// concurrent invocations across all instances sharing the
    /// `peak`/`in_flight` atomics.
    struct EchoTool {
        name: &'static str,
        delay_ms: u64,
        in_flight: Arc<AtomicUsize>,
        peak: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for EchoTool {
        fn schema(&self) -> ToolSchema {
            ToolSchema {
                name: self.name.into(),
                description: format!("echo tool {}", self.name),
                parameters: json!({"type": "object"}),
            }
        }
        async fn run(&self, args: Value) -> Result<Value> {
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
            tokio::time::sleep(Duration::from_millis(self.delay_ms)).await;
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            Ok(json!({"tool": self.name, "args": args}))
        }
    }

    fn build_registry(
        names: &[&'static str],
        delay_ms: u64,
    ) -> (HashMap<String, Arc<dyn Tool>>, Arc<AtomicUsize>) {
        let peak = Arc::new(AtomicUsize::new(0));
        let in_flight = Arc::new(AtomicUsize::new(0));
        let mut tools: HashMap<String, Arc<dyn Tool>> = HashMap::new();
        for n in names {
            tools.insert(
                (*n).into(),
                Arc::new(EchoTool {
                    name: *n,
                    delay_ms,
                    in_flight: in_flight.clone(),
                    peak: peak.clone(),
                }) as Arc<dyn Tool>,
            );
        }
        (tools, peak)
    }

    #[tokio::test]
    async fn empty_calls_returns_empty() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let out = tool_dispatch_concurrent(tools, vec![], 4).await;
        assert!(out.is_empty());
    }

    #[tokio::test]
    async fn output_aligned_to_input_order() {
        let (tools, _peak) = build_registry(&["a", "b", "c"], 5);
        let calls = vec![
            ("a".into(), json!({"i": 0})),
            ("b".into(), json!({"i": 1})),
            ("c".into(), json!({"i": 2})),
            ("a".into(), json!({"i": 3})),
        ];
        let out = tool_dispatch_concurrent(tools, calls, 4).await;
        assert_eq!(out.len(), 4);
        for (i, r) in out.iter().enumerate() {
            let v = r.as_ref().unwrap();
            assert_eq!(v["args"]["i"], i);
        }
    }

    #[tokio::test]
    async fn concurrency_cap_honoured() {
        let (tools, peak) = build_registry(&["a"], 25);
        let calls: Vec<_> = (0..15)
            .map(|i| ("a".into(), json!({"i": i})))
            .collect();
        let _ = tool_dispatch_concurrent(tools, calls, 3).await;
        let observed = peak.load(Ordering::SeqCst);
        assert!(observed <= 3, "peak {observed} > cap 3");
        assert!(observed >= 2, "peak {observed} — concurrency never engaged");
    }

    #[tokio::test]
    async fn zero_concurrency_normalised_to_one() {
        let (tools, peak) = build_registry(&["a"], 2);
        let calls: Vec<_> = (0..4).map(|i| ("a".into(), json!({"i": i}))).collect();
        let _ = tool_dispatch_concurrent(tools, calls, 0).await;
        assert_eq!(peak.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unknown_tool_isolated_per_slot() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let calls = vec![
            ("a".into(), json!({})),
            ("missing".into(), json!({})),
            ("a".into(), json!({})),
        ];
        let out = tool_dispatch_concurrent(tools, calls, 4).await;
        assert!(out[0].is_ok());
        assert!(out[1].is_err());
        assert!(out[2].is_ok());
        let err = format!("{}", out[1].as_ref().err().unwrap());
        assert!(err.contains("unknown tool"), "got: {err}");
    }

    #[tokio::test]
    async fn per_tool_failure_isolated() {
        struct Failer;
        #[async_trait]
        impl Tool for Failer {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "fail".into(),
                    description: "always errors".into(),
                    parameters: json!({"type": "object"}),
                }
            }
            async fn run(&self, _args: Value) -> Result<Value> {
                Err(Error::other("synthetic"))
            }
        }
        let (mut tools, _peak) = build_registry(&["good"], 0);
        tools.insert("fail".into(), Arc::new(Failer) as Arc<dyn Tool>);
        let calls = vec![
            ("good".into(), json!({})),
            ("fail".into(), json!({})),
            ("good".into(), json!({})),
        ];
        let out = tool_dispatch_concurrent(tools, calls, 4).await;
        assert!(out[0].is_ok());
        assert!(out[1].is_err());
        assert!(out[2].is_ok());
    }

    #[tokio::test]
    async fn fail_fast_raises_on_first_error() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let calls = vec![
            ("missing".into(), json!({})),
            ("a".into(), json!({})),
        ];
        let r = tool_dispatch_concurrent_fail_fast(tools, calls, 4).await;
        assert!(r.is_err());
    }

    #[tokio::test]
    async fn fail_fast_succeeds_on_all_ok() {
        let (tools, _peak) = build_registry(&["a", "b"], 0);
        let calls = vec![
            ("a".into(), json!({"x": 1})),
            ("b".into(), json!({"x": 2})),
        ];
        let out = tool_dispatch_concurrent_fail_fast(tools, calls, 4)
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0]["tool"], "a");
        assert_eq!(out[1]["tool"], "b");
    }

    #[tokio::test]
    async fn heterogeneous_tools_dispatch_correctly() {
        let (tools, _peak) = build_registry(&["alpha", "beta", "gamma"], 1);
        let calls = vec![
            ("alpha".into(), json!({"v": "a"})),
            ("beta".into(), json!({"v": "b"})),
            ("gamma".into(), json!({"v": "c"})),
            ("alpha".into(), json!({"v": "d"})),
        ];
        let out = tool_dispatch_concurrent(tools, calls, 8).await;
        let names: Vec<String> = out
            .iter()
            .map(|r| r.as_ref().unwrap()["tool"].as_str().unwrap().into())
            .collect();
        assert_eq!(names, vec!["alpha", "beta", "gamma", "alpha"]);
    }

    // ---- tool_dispatch_concurrent_with_progress tests ------------------

    #[tokio::test]
    async fn progress_total_set_and_completed_counts_advance() {
        let (tools, _peak) = build_registry(&["a", "b"], 0);
        let calls: Vec<_> = vec![
            ("a".into(), json!({"i": 0})),
            ("b".into(), json!({"i": 1})),
            ("a".into(), json!({"i": 2})),
        ];
        let progress = Progress::new(ToolDispatchProgress::default());
        let obs = progress.observer();
        let _ = tool_dispatch_concurrent_with_progress(tools, calls, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 3);
        assert_eq!(snap.completed, 3);
        assert_eq!(snap.errors, 0);
        assert_eq!(snap.unknown_tool_errors, 0);
    }

    #[tokio::test]
    async fn progress_records_unknown_tool_distinctly() {
        let (tools, _peak) = build_registry(&["a"], 0);
        // 1 known + 2 unknown + 1 known.
        let calls: Vec<_> = vec![
            ("a".into(), json!({})),
            ("missing-1".into(), json!({})),
            ("missing-2".into(), json!({})),
            ("a".into(), json!({})),
        ];
        let progress = Progress::new(ToolDispatchProgress::default());
        let obs = progress.observer();
        let _ = tool_dispatch_concurrent_with_progress(tools, calls, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 4);
        assert_eq!(snap.completed, 4);
        assert_eq!(snap.errors, 2);
        assert_eq!(snap.unknown_tool_errors, 2);
    }

    #[tokio::test]
    async fn progress_records_runtime_failure_without_unknown_flag() {
        // A tool that always errors at runtime — counts as `errors`
        // but NOT `unknown_tool_errors`.
        struct Failer;
        #[async_trait]
        impl Tool for Failer {
            fn schema(&self) -> ToolSchema {
                ToolSchema {
                    name: "fail".into(),
                    description: "always errors".into(),
                    parameters: json!({"type": "object"}),
                }
            }
            async fn run(&self, _args: Value) -> Result<Value> {
                Err(Error::other("synthetic"))
            }
        }
        let (mut tools, _peak) = build_registry(&["good"], 0);
        tools.insert("fail".into(), Arc::new(Failer) as Arc<dyn Tool>);
        let calls: Vec<_> = vec![
            ("good".into(), json!({})),
            ("fail".into(), json!({})),
            ("good".into(), json!({})),
        ];
        let progress = Progress::new(ToolDispatchProgress::default());
        let obs = progress.observer();
        let _ = tool_dispatch_concurrent_with_progress(tools, calls, 4, progress).await;
        let snap = obs.snapshot();
        assert_eq!(snap.total, 3);
        assert_eq!(snap.completed, 3);
        assert_eq!(snap.errors, 1);
        // Runtime error, NOT unknown-tool.
        assert_eq!(snap.unknown_tool_errors, 0);
    }

    #[tokio::test]
    async fn progress_observer_polls_mid_run() {
        let (tools, _peak) = build_registry(&["a"], 15);
        let calls: Vec<_> = (0..6)
            .map(|i| ("a".into(), json!({"i": i})))
            .collect();
        let progress = Progress::new(ToolDispatchProgress::default());
        let mut obs = progress.observer();
        let progress_clone = progress.clone();
        let h = tokio::spawn(async move {
            tool_dispatch_concurrent_with_progress(tools, calls, 2, progress_clone).await
        });
        let _ = obs.changed().await;
        let mid = obs.snapshot();
        assert_eq!(mid.total, 6);
        let _ = h.await.unwrap();
        let snap = obs.snapshot();
        assert_eq!(snap.completed, 6);
    }

    // ---- tool_dispatch_concurrent_stream tests ------------------------

    use futures::StreamExt;

    #[tokio::test]
    async fn stream_yields_one_item_per_call() {
        let (tools, _peak) = build_registry(&["a", "b"], 0);
        let calls: Vec<_> = vec![
            ("a".into(), json!({"i": 0})),
            ("b".into(), json!({"i": 1})),
            ("a".into(), json!({"i": 2})),
        ];
        let mut s = tool_dispatch_concurrent_stream(tools, calls, 4);
        let mut indices: Vec<usize> = Vec::new();
        while let Some((idx, r)) = s.next().await {
            assert!(r.is_ok());
            indices.push(idx);
        }
        indices.sort();
        assert_eq!(indices, vec![0, 1, 2]);
    }

    #[tokio::test]
    async fn stream_idx_aligns_with_input_call() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let calls: Vec<_> = (0..5)
            .map(|i| ("a".into(), json!({"i": i})))
            .collect();
        let mut s = tool_dispatch_concurrent_stream(tools, calls, 2);
        while let Some((idx, r)) = s.next().await {
            let v = r.unwrap();
            assert_eq!(v["args"]["i"], idx);
        }
    }

    #[tokio::test]
    async fn stream_unknown_tool_arrives_as_err_item() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let calls: Vec<_> = vec![
            ("a".into(), json!({})),
            ("missing".into(), json!({})),
            ("a".into(), json!({})),
        ];
        let mut s = tool_dispatch_concurrent_stream(tools, calls, 4);
        let mut errors = 0;
        let mut count = 0;
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
    async fn stream_empty_calls_yields_empty() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let mut s = tool_dispatch_concurrent_stream(tools, vec![], 4);
        assert!(s.next().await.is_none());
    }

    #[tokio::test]
    async fn stream_caller_drop_aborts_in_flight_calls() {
        // 50 calls × 50ms / cap=2 — full sequential ~1.25s. Drop
        // after 1 item; total wall-clock should be far less.
        let (tools, _peak) = build_registry(&["a"], 50);
        let calls: Vec<_> = (0..50)
            .map(|i| ("a".into(), json!({"i": i})))
            .collect();
        let started = std::time::Instant::now();
        {
            let mut s = tool_dispatch_concurrent_stream(tools, calls, 2);
            let _first = s.next().await.unwrap();
        }
        let elapsed_ms = started.elapsed().as_millis() as u64;
        assert!(
            elapsed_ms < 400,
            "elapsed {elapsed_ms}ms — caller-drop didn't abort remaining calls",
        );
    }

    #[tokio::test]
    async fn progress_empty_calls_no_updates() {
        let (tools, _peak) = build_registry(&["a"], 0);
        let progress = Progress::new(ToolDispatchProgress::default());
        let obs = progress.observer();
        let out =
            tool_dispatch_concurrent_with_progress(tools, vec![], 4, progress).await;
        assert!(out.is_empty());
        assert_eq!(obs.snapshot(), ToolDispatchProgress::default());
    }
}

# Architecture

This document explains how litGraph is laid out and the design constraints
behind each piece. Read this before changing crate boundaries, the GIL/runtime
model, or the StateGraph executor.

## Workspace shape

```
crates/
├── litgraph-core            # Types, traits, errors. ZERO PyO3.
├── litgraph-graph           # StateGraph executor + Kahn scheduler + checkpoints
├── litgraph-agents          # ReactAgent + SupervisorAgent on top of -graph
├── litgraph-retrieval       # VectorStore/Retriever traits + BM25 + RRF + Reranker
├── litgraph-splitters       # Recursive char + Markdown header (rayon batch)
├── litgraph-loaders         # text/jsonl/md/dir, rayon-parallel ingest
├── litgraph-observability   # Callback bus, CostTracker, OTel (feature-gated)
├── litgraph-cache           # Memory + SQLite + Semantic caches, CachedModel
├── litgraph-macros          # Proc-macros (#[tool] derives schemars JSON Schema)
├── litgraph-providers-{openai,anthropic,gemini}
├── litgraph-stores-{memory,hnsw,qdrant,pgvector}
├── litgraph-checkpoint-{sqlite,postgres,redis}
├── litgraph-bench           # criterion micro-benches
└── litgraph-py              # PyO3 bindings — the ONLY crate that imports pyo3
```

## Five non-negotiables

1. **No PyO3 in non-py crates.** `litgraph-py` is the single shim. Every other
   crate is usable as a pure Rust dependency. Violating this couples the entire
   workspace to the Python ABI.
2. **GIL released around all Rust work.** Every `#[pymethods]` function that
   does real work wraps the work in `py.allow_threads(...)`. Only the JSON ↔
   Python conversion at the entry/exit holds the GIL. Failing this serializes
   parallel branches through Python's GIL — kills the entire perf story.
3. **One shared Tokio runtime per process.** Lives in `litgraph-py::runtime`,
   built lazily via `OnceCell`. Provider methods, graph scheduler, and
   checkpointers all `block_on` it. Spinning a fresh runtime per call costs
   ~1ms and breaks tokio::spawn callers (e.g. graph stream).
4. **Bincode for checkpoints, JSON for messages.** Snapshots roundtrip through
   `bincode` for speed; the message wire format stays JSON for provider
   compatibility. Don't migrate snapshots to JSON — that was LangGraph's perf
   drag at 10k+ message histories.
5. **Zero default features.** Each store/checkpoint/provider is a separate
   crate. Users pay for what they import. Workspace-level `Cargo.toml` has no
   `default = ["..."]` lists pulling adapters.

## StateGraph execution model

The graph executor in `litgraph-graph::scheduler` is a textbook Kahn
super-step scheduler:

1. Compute initial frontier from `START`.
2. Per super-step:
   - Dedup the frontier.
   - Check `interrupt_before` — if hit, persist checkpoint + raise.
   - Spawn each frontier node on `JoinSet`, bounded by `Semaphore(max_parallel)`.
   - Each node runs under `tokio::select!` with a child `CancellationToken`.
3. As nodes complete:
   - Apply the partial update through the user-supplied reducer.
   - Compute successors: explicit `goto` overrides static edges; `sends`
     emits N parallel sub-invocations.
   - If `interrupt_after` matches, persist + raise.
4. Persist post-super-step checkpoint.
5. Loop until frontier is empty or `recursion_limit` is reached.

Resume re-enters at step 2 with `skip_interrupt_before_once = true` so the
same interrupt doesn't re-fire.

## Observability

`Callback` is a single trait that batches events:

- Producers (providers, graph, agent loops) emit `Event` values into an
  unbounded `mpsc::UnboundedSender<Event>`.
- A drain task batches up to `max_batch` events or flushes every `flush_every`
  (default 16ms), then dispatches one batch to each subscriber.
- Why batched? Python subscribers cost one GIL acquisition per call. Streaming
  100 token events/sec/thread × 5 subscribers = 500 GIL acquisitions/sec/thread
  — enough to stall the runtime. Batching collapses to ~60 calls/sec.

`InstrumentedChatModel` wraps any `ChatModel` and emits Start/End/Error events.
`CostTracker` is a `Callback` that accumulates token usage × `PriceSheet` to
running USD totals.

## Cache architecture

Two orthogonal caching strategies:

- **Hash cache** (`CachedModel`): blake3 of `(model, messages, opts)` — exact
  match. Fast, deterministic. Backends: in-memory moka, SQLite.
- **Semantic cache** (`SemanticCachedModel`): cosine similarity of last user
  message embedding. Lookups via rayon-parallel cosine over the cache set.
  Threshold ≥ 0.95 in production. Use for FAQ-style traffic; never for tool
  calls.

Both bypass cache for `.stream()` — token streams don't roundtrip cleanly.

## Vector store dispatch

`VectorStore` is one async trait in `litgraph-retrieval`. Stores implement it
once; `VectorRetriever` composes any store with any `Embeddings`. Python
bindings use a `dyn VectorStore` so `VectorRetriever(embeddings, store)`
accepts Memory / HNSW / Qdrant / PgVector through one constructor (`Bound<'_,
PyAny>` extraction).

## Python ↔ Rust bridge

```
Python user code
       │
       │  (.invoke / .stream / .add)
       ▼
PyO3 #[pymethods]            ← GIL held briefly
       │
       │  py.allow_threads { ... }   ← GIL released
       ▼
shared tokio runtime  →  Rust async work  →  Result
       ▲
       │  GIL re-acquired only for return-value conversion
       │
Python receives dict/list/scalar
```

For Python callable nodes (StateGraph) and Python tool functions
(`FunctionTool`) the flow inverts: Rust `Tool::run` re-acquires the GIL via
`Python::with_gil` to call the closure, then drops it before the next await.

## Where to add new pieces

- **New provider**: `crates/litgraph-providers-<name>/` implementing
  `ChatModel`. Add a Python class in `litgraph-py/src/providers.rs` using the
  same `with_cache` / `with_semantic_cache` / `instrument` / `stream` shape;
  register on `extract_chat_model` so `ReactAgent` accepts it.
- **New vector store**: `crates/litgraph-stores-<name>/` implementing
  `VectorStore`. Python class with `as_store()` accessor; add to
  `VectorRetriever::new` extractor.
- **New checkpointer**: `crates/litgraph-checkpoint-<name>/` implementing
  `Checkpointer`. Wrap `block_on` calls if the underlying client is sync.
- **New tool**: prefer `#[tool]` proc-macro from `litgraph-macros`. Function
  signature → JSON Schema via `schemars`.

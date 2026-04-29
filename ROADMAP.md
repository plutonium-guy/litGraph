# litGraph Roadmap

Research-backed prioritization of remaining gaps vs. LangChain + LangGraph,
with a focus on items where Rust gives a real (not cosmetic) advantage:
true parallelism, zero-cost abstractions, no GIL, no Python class-soup.

This is a *living* doc — picked up at the start of each iteration to
choose the next ship target. Numbered items below get crossed off as
they land in `git log` (look for `iter N`).

Snapshot date: 2026-04-29 · last shipped: iter 180 (Studio debug
router for `litgraph-serve`, feature-gated). Tracker of *what's done*
lives in `FEATURES.md` and `CLAUDE_CODE_FEATURES.md`; this doc is
*what's next*.

---

## Guiding philosophy (recap)

1. **Rust does the work, Python is the shim.** Hot paths (HTTP, SSE
   parse, tokenize, embed math, vector search, JSON parse, graph
   scheduling, RRF, MMR) are Rust. Python is `#[pyfunction]` and
   `extract_chat_model`.
2. **Parallelism is a first-class feature.** Tokio for I/O fan-out;
   Rayon for CPU-bound batching; `py.allow_threads` (PyO3 0.22+ →
   `py.detach`) around every blocking call so the GIL is dropped.
3. **No bloat.** Each capability is its own crate behind a feature
   flag. Default features stay tight; users opt in.
4. **Graph-first, no Runnable cathedral.** A `StateGraph` + a few
   functions cover what LangChain spreads across `LCEL`, `Runnable`,
   `Chain`, `Memory`, and N variants of each.

If a gap doesn't pay rent against (1)–(4), it goes under the
"won't do" list at the bottom.

---

## Prioritization rubric

Each candidate scores on five axes (1–5, higher = more compelling):

| Axis              | What it asks                                           |
|-------------------|--------------------------------------------------------|
| **User impact**   | How often does a real agent author hit this?           |
| **Parity weight** | Does its absence stop people picking litGraph?         |
| **Rust win**      | Does Rust make this materially faster/safer than LC?   |
| **Surface area**  | How many crates does it touch? (lower → easier ship)   |
| **Test cost**     | Can it be unit-tested without a paid LLM?              |

Total = sum. Anything ≥18 is fair game for the next ten iters.

---

## Tier 1 — ship in the next ~10 iters

### 1. EnsembleRetriever — weighted RRF ✅ shipped iter 181
- **Status:** ✅. `litgraph_retrieval::EnsembleRetriever` — per-child
  weights, weighted RRF, `tokio::join_all` fan-out. Python:
  `litgraph.retrieval.EnsembleRetriever`.
- **What:** `Vec<(Arc<dyn Retriever>, f32)>` fanned out via
  `tokio::join_all`, fused with weighted reciprocal rank
  fusion: `score(d) = Σ_i w_i / (k_rrf + rank_i(d))`.
- **Why:** LangChain's `EnsembleRetriever` is the canonical recipe
  for combining a sparse (BM25) + dense (vector) retriever where
  the sparse one is noisier and should get less weight. Equal-weight
  RRF is suboptimal; production users tune ratios.
- **Rust win:** all child retrievers run truly concurrent. Python's
  `EnsembleRetriever` runs sequentially under the GIL.
- **Effort:** 1 iter. New file `ensemble.rs`, ~150 LOC + tests.

### 2. `before_tool` / `after_tool` middleware hooks
- **Status:** 🟡. `before/after_model` exists. Tool wrappers
  (`RetryTool`, `TimeoutTool`, `OffloadingTool`) cover
  point use-cases, but a generic chain over the agent loop's tool
  invocations is still missing.
- **What:** Extend `MiddlewareChain` to intercept the tool-execution
  branch of `ReactAgent` / `SupervisorAgent`. Same trait shape as
  the model hooks.
- **Why:** First-class observability + cost control across all
  tools without per-tool wrapping.
- **Rust win:** zero per-call cost when chain is empty (compiles
  out via monomorphization).
- **Effort:** 1–2 iters. Touches agents + middleware crates.

### 3. Vector-indexed semantic search on `Store` ✅ shipped iter 185
- **Status:** ✅ in-memory tier. `litgraph_core::SemanticStore` wraps
  any `Store` with an `Embeddings` provider; `semantic_search` runs
  brute-force Rayon-parallel cosine over a namespace.  Python:
  `litgraph.store.SemanticStore(store, embedder)`.
- **Remaining:** pgvector-backed indexed search on `PostgresStore`
  for the >10k items/namespace tier.

### 4. Postgres `Store` already shipped → wire vector index
- **Status:** ✅ structurally, 🟡 functionally. `PostgresStore`
  exists. Adding a `pgvector` column for the `value` JSON's optional
  embedding completes #3 for distributed deployments.

### 5. Functional API: `@entrypoint` + `@task` (Python)
- **Status:** ❌.
- **What:** Decorator-based alternative to `StateGraph` for simple
  workflows. `@entrypoint` marks the root, `@task` marks a
  sub-routine; the runtime auto-builds the graph.
- **Why:** Lets a 5-line script author skip `add_node` / `add_edge`
  ceremony.
- **Rust win:** thin — Rust core already runs the graph. This is a
  Python-side ergonomics layer.
- **Effort:** 1 iter (py-only).

### 6. `pyo3-stub-gen` auto-generated `.pyi`
- **Status:** 🟡. Hand-written stubs in `litgraph-stubs/` drift.
- **What:** Wire `pyo3-stub-gen` so stubs regenerate on every
  `maturin build`.
- **Why:** Pyright import warnings hurt agent-authored code (and
  Claude Code's own UX when editing user repos that import
  `litgraph`).
- **Effort:** 1 iter.

### 7. Pydantic-coerced state + StreamPart on Python side
- **Status:** ❌.
- **What:** `litgraph.StateModel` (Pydantic v2 base) so the Python
  state dict gets validated/typed; `StreamPart` discriminated
  union for stream events.
- **Why:** IDE autocomplete on stream chunks. Today everything is
  `dict[str, Any]`.
- **Effort:** 1–2 iters.

### 8. Local chat model — candle / mistral.rs
- **Status:** ❌. Embeddings have a local path (fastembed), chat
  doesn't. This is the biggest "air-gapped agent" gap.
- **What:** New crate `litgraph-providers-local` wrapping
  `mistralrs` (or `candle` directly) behind the `ChatModel` trait.
- **Why:** Full offline agent. Strong differentiator from LangChain
  whose local stories all go through `ollama` or `llama.cpp`
  external processes.
- **Rust win:** in-process inference, no IPC; can share GPU
  context with embeddings.
- **Effort:** 3–4 iters. Heavy because of model loading + KV-cache
  + tokenization edge cases.

### 9. Webhook-resume bridge for interrupts
- **Status:** 🟡. Today: user wires their own. Add a thin
  `litgraph-serve --features webhooks` route that accepts
  `POST /threads/:id/resume` with a JSON `Command` body.
- **Effort:** 1 iter (small, mostly axum routing).

### 10. Pregel-style super-step parallel execution audit
- **Status:** ✅ scheduler exists, but worth a perf pass — currently
  Send fan-out runs futures concurrently but always on the same
  Tokio runtime; nodes that are CPU-bound (e.g. running a local
  embedder inline) block the runtime. Audit + add a
  `spawn_blocking` escape hatch for CPU-heavy nodes.
- **Effort:** 1 iter.

---

## Tier 2 — useful but lower priority

- WhatsApp / Telegram **history** loaders (Telegram is push-only;
  no usable history. WhatsApp Business API is feasible but
  paperwork-heavy.)
- WebSocket streaming endpoint on `litgraph-serve` (today: SSE
  covers 95% of use-cases; WS is a nice-to-have for back-channel
  cancellation).
- Video-in modality — needs provider-side support, not framework
  work.
- Sentence/NLTK/SpaCy splitters — recursive char + token splitter
  cover the cases. Skip unless someone files a real ask.
- LanceDB / Pinecone vector stores — heavy deps, low marginal
  value over hnsw + pgvector + qdrant + chroma + weaviate.

## Won't do (deferred indefinitely)

- LangChain `Callbacks` API parity — surface area is enormous and
  `CostTracker + GraphEvent + AgentEvent + tracing` already covers
  every concrete use-case asked for.
- Streaming tool execution — would require a `Tool` trait revamp
  (yielding tokens from a tool). The few real use-cases (long
  shell jobs) are better served by `OffloadingTool` + a follow-up
  read.
- Zapier / N8N tool — userland integration; out of framework scope.

---

## Parallelism showcases — where Rust earns its keep

These are existing primitives that already use real parallelism;
listed here as a reference target ("when adding X, follow these
patterns").

| Primitive                   | Pattern                       | Why this is a Rust win |
|-----------------------------|-------------------------------|------------------------|
| `HybridRetriever`           | `tokio::join_all` over kids   | LangChain runs them sequential under GIL |
| `MultiQueryRetriever`       | `JoinSet` over N variants     | Same — N parallel embed+search |
| `parallel_ingest` (Rayon)   | CPU fan-out on splitters      | Pure CPU, GIL would kill it |
| Eval harness                | bounded `Semaphore` fan-out   | Lets a 1k-case suite run with 50-way concurrency safely |
| Send fan-out scheduler      | per-node `tokio::spawn`       | Pregel-style supersteps |
| `RerankingRetriever` batch  | one model call for N docs     | Tower of 1 vs N |
| `EnsembleRetriever` (iter 181) | weighted RRF + `join_all`  | Per-child weights with parallel fan-out — LangChain serialises |
| `batch_concurrent` (iter 182) | `Semaphore` + `JoinSet`     | Bounded-concurrency batch over any `ChatModel`; order-preserving, per-call `Result` |
| `embed_documents_concurrent` (iter 183) | chunk + `JoinSet` | Splits inputs into chunks, fans chunks across `Semaphore`; aligned output, fail-fast |
| `RaceChatModel` (iter 184) | `JoinSet` + `abort_all` | Hedged-request pattern: invoke N concurrently, first success wins, losers cancelled. Tail-latency reduction across providers/regions |
| `SemanticStore` (iter 185) | Rayon `par_iter` cosine + `par_sort` | Brute-force semantic search on any `Store`; Rayon makes it CPU-saturating across cores. Python's `numpy` would still hit the GIL on the gather step |
| `EnsembleReranker` (iter 186) | `tokio::join_all` over rerankers | Fans N rerankers over the SAME candidates concurrently; weighted RRF on ranks (scale-free across providers). LangChain's reranker chains run sequential |
| `load_concurrent` (iter 187) | `spawn_blocking` + `Semaphore` | Sync `Loader::load()` calls fan out onto Tokio's blocking pool; aligned output, per-loader `Result`. LangChain's loader chains are sequential by convention |
| `MultiVectorRetriever` (iter 188) | composes `embed_documents_concurrent` | Indexing N perspectives per parent uses chunked Tokio fan-out from iter 183. Retrieval dedups by parent_id in linear time |
| `multiplex_chat_streams` (iter 189) | `tokio::mpsc` channel-fan-in | Per-stream task forwards events into a shared channel; outer Stream drains in arrival order. First channel-based primitive — distinct from JoinSet/Semaphore lineage; one slow/failing stream never stalls peers |
| `retrieve_concurrent` (iter 190) | Tokio `JoinSet` + `Semaphore` over `Retriever::retrieve` | Completes the parallel-batch trio (`batch_concurrent`/`embed_documents_concurrent`/this). Same retriever, N caller queries — eval / agentic batch path |
| `tool_dispatch_concurrent` (iter 191) | Tokio `JoinSet` + `Semaphore` over heterogeneous `Tool::run` | Fourth in the parallel-batch family — different topology (HashMap router, heterogeneous tools per call). Standalone helper for Plan-and-Execute and custom orchestrators outside the React loop |
| `RaceEmbeddings` (iter 192) | Tokio `JoinSet` + `abort_all` over `Embeddings` | Embeddings analogue of iter 184 `RaceChatModel`. Hedge OpenAI / Voyage / local fastembed for tail-latency cuts on the embed-query critical path |
| `RaceRetriever` (iter 193) | Tokio `JoinSet` + `abort_all` over `Retriever` | Completes the race trio across the read-side traits (Chat/Embeddings/Retriever). Hedge a fast cache against a slow primary for latency-min retrieval; vs `EnsembleRetriever` which fuses for quality |
| `TimeoutChatModel` + `TimeoutEmbeddings` (iter 194) | `tokio::time::timeout` (concurrent inner-future vs deadline-timer, `select`-style) | Different shape from `JoinSet/abort_all` race patterns: the "competitor" is a deadline timer, not another provider. Drops the inner future on timeout, releasing connection / parse state |
| `broadcast_chat_stream` (iter 195) | `tokio::sync::broadcast` (1→N fan-out) | Inverse of iter 189's `mpsc`-based fan-in. One upstream stream, N concurrent subscribers; lazy-spawn pump to avoid event loss vs late subscribers; `Lagged(n)` per subscriber on capacity overflow so a slow consumer doesn't stall fast ones |
| `ingest_to_stream` (iter 196) | Multi-stage `mpsc` pipeline | First **pipeline-parallelism** primitive: three Tokio tasks (load / split / embed) connected by bounded `mpsc` channels, each stage runs while later stages drain. Composes iter 187 (load_concurrent) + a splitter closure + iter 183 (embed_documents_concurrent) with backpressure between stages |
| `rerank_concurrent` (iter 197) | Tokio `JoinSet` + `Semaphore` over `Reranker::rerank` | Adds a fifth axis to the parallel-batch family (chat/embed/retrieve/tool/rerank). One reranker, N independent `(query, candidates)` pairs — eval / batch-rerank path |
| `Bm25Index::add` parallel build (iter 198) | Rayon `par_iter` on tokenize+count per doc | Pure CPU parallelism for index construction (vs the I/O-bound parallelism of the JoinSet/Semaphore family). Each doc tokenizes independently in a Rayon worker; DF merge happens sequentially under the write lock |

---

## Out-of-band research log

Findings worth keeping around as we plan future iters.

- **LangChain `EnsembleRetriever`** — equal weights default, but
  the usable form takes `weights: list[float]`. Internally
  sequential; each `get_relevant_documents` call serial under GIL.
  We can match the API and beat the perf trivially.
- **LangGraph `Functional API`** (Sept 2024) — `@entrypoint` /
  `@task`. Implemented purely as a graph-building convenience layer
  over `StateGraph`. Their docs explicitly call out: "every
  functional API workflow desugars to a StateGraph." That makes
  this a Python-only port for us.
- **mistralrs vs candle for #8** — `mistralrs-core` ships a
  high-level `Engine` with KV-cache + paged attention + speculative
  decoding. `candle` is closer to the metal but every project ends
  up rebuilding the same engine. Pick `mistralrs` unless we hit a
  licensing wall.
- **pyo3-stub-gen** — works on PyO3 0.22 (we're on 0.28). Need to
  verify compatibility before adopting; otherwise the manual
  `litgraph-stubs/*.pyi` flow stays.
- **Webhook-resume design** — should re-use `Command::resume(value)`
  and `Command::goto(node)` so we don't invent a third execution
  control surface.

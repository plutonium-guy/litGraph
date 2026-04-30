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

### 9. Webhook-resume bridge for interrupts ✅ shipped iter 201 + 202
- **Status:** ✅. `litgraph_core::ResumeRegistry` (201) ships the
  oneshot-backed coordination primitive; `litgraph_serve::resume::
  resume_router` (202) wraps it in axum endpoints — `POST
  /threads/:id/resume`, `DELETE /threads/:id/resume`,
  `GET /resumes/pending`. Mounts alongside the chat / studio routers.

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
| `Progress<T>` (iter 199) | `tokio::sync::watch` latest-value broadcast | Completes the channel-shape trio: mpsc fan-in (189), broadcast fan-out (195), watch latest-value (199). Multiple observers, intermediate values collapse — observers see only the latest snapshot, perfect for progress UIs |
| `ingest_to_stream_with_progress` (iter 200) | Composition of iter 196 + iter 199 | First **composition** of two prior parallelism primitives: the multi-stage ingest pipeline (196) updates an `IngestProgress` watcher (199) at each stage boundary. Demonstrates the compositional payoff of building primitives that snap together |
| `ResumeRegistry` (iter 201) | `tokio::sync::oneshot` per thread id | Fourth channel shape in the lineage: oneshot signal (single-fire, single consumer). Foundation for the LangGraph interrupt-resume pattern — agent parks on `await_resume()`, an HTTP handler / Slack callback fires `resume(thread_id, value)` from anywhere |
| `resume_router` (iter 202) | axum router over `ResumeRegistry` | Wire-protocol completion of iter 201: `POST /threads/:id/resume {value}` delivers, `DELETE /threads/:id/resume` cancels, `GET /resumes/pending` lists. Composes a coordination primitive into a real prod-ready HTTP feature in <100 LOC of glue |
| `mmr_select` parallel (iter 203) | Rayon `par_iter` on per-candidate scoring | The greedy-pick outer loop stays sequential (each pick depends on prior picks); the per-iteration score loop is now Rayon-parallel — independent O(\|picked\|) cosine sims per candidate. Deterministic tie-break on lower index keeps parallel picks bit-identical to a sequential reference |
| `embedding_redundant_filter` parallel (iter 204) | Rayon `par_iter::any` with short-circuit | Sibling pattern to iter 203: outer kept-list loop sequential, inner "is candidate similar to ANY kept doc?" probe parallelizes across cores via `any` (which short-circuits on first hit, just like the sequential `for ... break`). 3 cross-impl tests cover small / larger / threshold-extreme pools |
| `batch_concurrent_with_progress` (iter 205) | Composes iter 182 + iter 199 | Second progress-aware composition (after iter 200's progress-aware ingestion). `BatchProgress { total, completed, errors }` watcher updated as each `ChatModel.invoke` completes — drop-in replacement for `batch_concurrent` when the caller wants live eval-harness counters |
| `embed_documents_concurrent_with_progress` (iter 206) | Composes iter 183 + iter 199 | Third progress-aware composition. `EmbedProgress { total_texts, total_chunks, completed_chunks, completed_texts, errors }` watcher updated as each chunk's embed call returns — drop-in replacement for bulk-indexing flows that want a tqdm-style counter / ETA |
| `retrieve_concurrent_with_progress` (iter 207) | Composes iter 190 + iter 199 | Fourth progress-aware composition. `RetrieveProgress { total, completed, docs_returned, errors }` watcher updated as each query completes — eval-harness pattern with a live counter |
| `tool_dispatch_concurrent_with_progress` (iter 208) | Composes iter 191 + iter 199 | Fifth progress-aware composition. `ToolDispatchProgress { total, completed, errors, unknown_tool_errors }` — unknown-tool errors bucketed separately so dashboards distinguish a routing-bug regression (LLM emitted a name your registry doesn't know) from a runtime tool failure |
| `rerank_concurrent_with_progress` (iter 209) | Composes iter 197 + iter 199 | Closes the progress-aware family across all 6 parallel-batch axes (ingest, chat batch, embed batch, retriever, tool, rerank). `RerankProgress { total, total_candidates, completed, docs_returned, errors }` — `total_candidates` exposed separately so eval reports can compute "X% of candidates reranked" rather than just "X% of pairs" |
| `batch_concurrent_stream` (iter 210) | mpsc-backed streaming variant of iter 182 | Yields `(idx, Result<ChatResponse>)` pairs in completion order as each invoke finishes, instead of buffering the whole `Vec`. Caller can render results live, dispatch downstream work on early completers, or drop the stream to abort remaining in-flight work (the spawned pump calls `set.abort_all()` on receiver-drop). First "stream-out" variant in the parallel-batch family — same primitive, different consumer shape |
| `embed_documents_concurrent_stream` (iter 211) | mpsc-backed streaming variant of iter 183 | Streaming-variant pattern extended to the embeddings axis. Yields `(chunk_idx, Result<Vec<Vec<f32>>>)` as each chunk's embed call returns; bulk indexers upsert chunks into a vector store on early completers without waiting for the slowest. Same abort-on-drop semantics as iter 210 |
| `retrieve_concurrent_stream` (iter 212) | mpsc-backed streaming variant of iter 190 | Streaming-variant pattern extended to the retriever axis. Yields `(query_idx, Result<Vec<Document>>)` as each query completes — eval harnesses can render rows live; drop the stream to abort the rest |
| `tool_dispatch_concurrent_stream` (iter 213) | mpsc-backed streaming variant of iter 191 | Streaming-variant pattern extended to the tool dispatch axis. Yields `(call_idx, Result<Value>)` as each tool call completes — orchestrators can chain follow-ups on fast tools without waiting for the slowest |
| `rerank_concurrent_stream` (iter 214) | mpsc-backed streaming variant of iter 197 | Streaming-variant pattern extended to the rerank axis. Yields `(pair_idx, Result<Vec<Document>>)` as each rerank call completes; closes 5 of 6 streaming-variant axes |
| `load_concurrent_stream` (iter 215) | mpsc-backed streaming variant of iter 187 | Streaming-variant pattern extended to the loader axis. Yields `(loader_idx, LoaderResult<Vec<Document>>)` as each blocking `load()` returns from `spawn_blocking`. Sixth distinct primitive in the streaming family — alongside the buffered-Vec and progress-aware patterns, the parallel-batch toolbox now offers three consumer shapes per axis |
| `batch_concurrent_stream_with_progress` (iter 216) | Composes iter 205 + iter 210 | First combination of two consumer shapes (stream + watcher) in one call. Per-row events drive a UI list view while `BatchProgress { total, completed, errors }` drives a summary progress bar — both backed by the same batch task. Demonstrates the four-quadrant matrix per axis: buffered Vec / progress-aware Vec / streaming / streaming-with-progress |
| `embed_documents_concurrent_stream_with_progress` (iter 217) | Composes iter 206 + iter 211 | Combined consumer shape extended to the embed axis. Both chat-batch and embed-batch axes now ship all four quadrants. Same consistency contract as iter 216 (counter ticks before stream item) so observers and stream consumers see synchronized state |
| `retrieve_concurrent_stream_with_progress` (iter 218) | Composes iter 207 + iter 212 | Combined consumer shape extended to the retriever axis. Three of six axes now ship the full four-quadrant matrix |
| `tool_dispatch_concurrent_stream_with_progress` (iter 219) | Composes iter 208 + iter 213 | Combined consumer shape extended to the tool axis. Four of six axes now ship the full four-quadrant matrix. `unknown_tool_errors` bucketing carries over so observers see routing-vs-runtime breakdown live |
| `rerank_concurrent_stream_with_progress` (iter 220) | Composes iter 209 + iter 214 | Combined consumer shape extended to the rerank axis. Five of six axes now ship the full four-quadrant matrix |
| `load_concurrent_with_progress` + `load_concurrent_stream_with_progress` (iter 221) | Closes the loader axis | Two functions in one iter — the loader axis was missing the progress-aware variant since iter 187, so this iter retroactively ships it AND the combined stream-with-progress sibling. The four-quadrant consumer matrix now closes across all six parallel-batch axes |
| `SemanticStore::bulk_put` (iter 222) | Composes iter 185 + iter 183 | Single-call bulk indexer over `SemanticStore`. Internally embeds the whole batch via `embed_documents_concurrent` (chunk-and-fan-out under Semaphore) and writes results to the underlying `Store` one at a time — far cheaper than calling `put` N times serially. Closes LangGraph's `BaseStore::mset` parity gap |
| `SemanticStore::bulk_delete` (iter 223) | Pair to bulk_put | Aligned `Vec<Result<bool>>` over `Store::delete`. Per-tenant namespace cleanup, retention sweeps, TTL boundary drops. Closes LangGraph's `BaseStore::mdelete` parity gap |
| `SemanticStore::bulk_get` (iter 224) | Third of the bulk trio | Aligned `Vec<Result<Option<(text, value)>>>` for known-key fetches. Distinct from `semantic_search` (which ranks by meaning). Surfaces corrupt-shape errors per key without tanking the whole batch. Closes the full `BaseStore::{mset, mdelete, mget}` parity |
| `ShutdownSignal` (iter 225) | `tokio::sync::Notify` + AtomicBool fired flag | Fifth distinct channel shape: N-waiter EDGE signal. Single `signal()` wakes every current and future waiter (idempotent); late `wait()` resolves instantly via the flag fast-path so no Notify-replay quirk. Channel-shape table now: mpsc / broadcast / watch / oneshot / Notify-edge |
| `until_shutdown` (iter 226) | `tokio::select!` over fut + ShutdownSignal::wait | Composable future combinator: `until_shutdown(model.invoke(...), &shutdown).await` returns `Some(T)` if the future completed, `None` if shutdown won. Inner future is dropped on shutdown so HTTP / DB / sleep resources are released promptly — no orphan in-flight work. Fast-path `is_signaled()` check skips polling the inner entirely if signal already fired |
| `batch_concurrent_with_shutdown` (iter 227) | Composes iter 182 + iter 225 | First parallel-batch ↔ coordination bridge. Distinct from wrapping `batch_concurrent` in `until_shutdown` (which discards everything if shutdown wins): preserves PARTIAL progress. Long eval batch finishes 60% before Ctrl+C → those 60% bank as `Ok`, remaining slots become `Err("cancelled by shutdown")`. Mechanical extension to the other 5 batch axes available |
| `embed_documents_concurrent_with_shutdown` (iter 228) | Composes iter 183 + iter 225 | Partial-progress preservation extended to the embeddings axis. Per-chunk granularity — bulk-indexer can flush only the `Ok` slots into a partial-but-valid embedding result on shutdown |
| `retrieve_concurrent_with_shutdown` (iter 229) | Composes iter 190 + iter 225 | Partial-progress preservation extended to the retriever axis. Three of six axes (chat / embed / retrieve) now bridge to the coordination primitives; tool / rerank / loader remain |
| `tool_dispatch_concurrent_with_shutdown` (iter 230) | Composes iter 191 + iter 225 | Partial-progress preservation extended to the heterogeneous tool axis. Four of six axes now bridge to coordination; rerank and loader remain. Real prod use: a Plan-and-Execute orchestrator with long-running tools banks completed tool results on cancel so agent context stays consistent |
| `rerank_concurrent_with_shutdown` (iter 231) | Composes iter 197 + iter 225 | Partial-progress preservation extended to the rerank axis. Five of six axes now bridge to coordination; only the loader axis remains. Long ensemble rerank runs over many (query, candidates) pairs bank completed `Ok` rerankings on Ctrl+C — eval reports stay computable on the partial result, remaining slots fill with `Err("cancelled by shutdown")` |
| `load_concurrent_with_shutdown` (iter 232) | Composes iter 187 + iter 225 | **Closes the bridge family.** Partial-progress preservation extended to the sixth and final parallel-batch axis (loader). All 6 axes (chat / embed / retriever / tool / rerank / loader) now bridge to `ShutdownSignal`. Real prod use: an ingestion job over hundreds of S3 keys + sitemap URLs banks completed loaders on Ctrl+C — half-loaded corpus stays valid, downstream embed/index stages can ship the partial result. Caveat: `Loader::load()` runs on `spawn_blocking`, so already-blocking OS threads finish their current call naturally — `abort_all()` only cancels the waiting-on-permit + join-await phases. Output slot still resolves to `Err("cancelled by shutdown")` once the await is dropped |
| `batch_concurrent_stream_with_shutdown` (iter 233) | Composes iter 210 + iter 225 | **Opens the second bridge family** (streaming + coordination), distinct from iter 227's Vec + coordination bridge. The streaming family already aborts on consumer-drop; this adds *producer-side* graceful end-of-interest. One central `ShutdownSignal` can stop multiple parallel batch streams without each consumer needing to drop its receiver — useful in orchestrators that own many sub-streams. Consumer sees a partial prefix in completion order, then the stream ends cleanly (no `Err`-filled tail). Mechanical extension to the other 5 stream axes (embed/retriever/tool/rerank/loader) available |
| `embed_documents_concurrent_stream_with_shutdown` (iter 234) | Composes iter 211 + iter 225 | Stream + coordination bridge extended to embed axis. Two of six stream axes (chat / embed) now bridge. Real prod use: a multi-collection bulk-ingestor running 5 parallel embed streams into 5 vector-store collections — a shared `ShutdownSignal` from a Ctrl+C / pod-shutdown handler ends every embed stream cleanly so each collection's already-flushed chunks stay valid |
| `retrieve_concurrent_stream_with_shutdown` (iter 235) | Composes iter 212 + iter 225 | Stream + coordination bridge extended to retriever axis. Three of six stream axes (chat / embed / retriever) now bridge. Real prod use: a long eval-harness streaming retrievals over thousands of queries — Ctrl+C fires the shared `ShutdownSignal`, every parallel retrieval stream terminates cleanly, drained results stay valid for a partial eval report |
| `tool_dispatch_concurrent_stream_with_shutdown` (iter 236) | Composes iter 213 + iter 225 | Stream + coordination bridge extended to tool axis. Four of six stream axes (chat / embed / retriever / tool) now bridge. Real prod use: a Plan-and-Execute orchestrator running a fan of tool calls per plan step — Ctrl+C fires the shared `ShutdownSignal`, every dispatch stream ends cleanly so accumulated tool-result state stays consistent with what's actually been executed (no half-applied side effects from aborted-mid-call tools) |
| `rerank_concurrent_stream_with_shutdown` (iter 237) | Composes iter 214 + iter 225 | Stream + coordination bridge extended to rerank axis. Five of six stream axes (chat / embed / retriever / tool / rerank) now bridge; only loader remains. Real prod use: a long ensemble eval streaming reranks over many `(query, candidates)` pairs — Ctrl+C fires the shared `ShutdownSignal`, every rerank stream terminates cleanly, drained results stay valid for a partial eval report |
| `load_concurrent_stream_with_shutdown` (iter 238) | Composes iter 215 + iter 225 | **Closes the stream + coordination bridge family.** Every one of the six parallel-batch axes (chat / embed / retriever / tool / rerank / loader) now exposes BOTH Vec + shutdown (iters 227-232) AND stream + shutdown (iters 233-238) variants. Real prod use: a multi-source ingestion crawler streaming docs from S3 + sitemap + GitHub + Confluence loaders into a single downstream chunker — Ctrl+C fires the shared `ShutdownSignal`, every loader stream ends cleanly, the partial corpus already drained into the chunker stays valid for a partial index. Same loader-axis caveat as iter 232: blocking `load()` threads finish their current call naturally; `abort_all` only cancels the waiting-on-permit + join-await phases |
| `Barrier` (iter 239) | `tokio::sync::Notify` + AtomicUsize counter + AtomicBool released-flag | **Sixth distinct channel shape**: wait-for-N rendezvous. After mpsc / broadcast / watch / oneshot / Notify-edge, the channel-shape table now closes a coordination gap: synchronized cohort start. `Barrier::new(n)` requires N participants to call `wait()`; the N-th arrival flips a released flag and `notify_waiters` wakes every pending sleeper. Late arrivals past N return instantly. The shutdown-aware `wait_with_shutdown(&shutdown)` returns `Some(())` on release / `None` on shutdown — pending waiters wake instead of parking forever when the orchestrator abandons a synchronized step. Distinct from `tokio::sync::Barrier`: that one provides only a non-shutdown `wait`; this version composes with the iter-225 coordination layer. Real prod use: coordinated multi-agent rounds (lockstep step-by-step execution), warm-up rendezvous (N workers each load model weights then start serving together), phase synchronization (pipeline stage N+1 can't begin until every item of stage N has finished) |
| `CountDownLatch` (iter 240) | `tokio::sync::Notify` + AtomicUsize counter (saturating-decrement) | **Sister to `Barrier` with asymmetric roles**: producers call `count_down()` (no wait) as work finishes, observers call `wait()` (no decrement) for the count to reach zero. The decoupling matters when producers and observers are different roles or modules — observer just needs a clone of the latch, doesn't need `JoinHandle`s or a `JoinSet` to track who's still running. Saturating-decrement via `compare_exchange` so extra `count_down()` calls past zero are no-ops (won't wrap). Same shutdown-aware `wait_with_shutdown(&shutdown)` semantics as `Barrier`: open-state wins over fired-shutdown (the work is done, no point reporting cancellation). Real prod use: fan-out completion gate (spawn N retrievers, await latch to know everyone returned regardless of who spawned them), initialization barrier (5 caches start filling, main task waits for all 5 to report ready before serving traffic), cleanup synchronization (every worker `count_down`s as it drains; supervisor waits for graceful exit) |
| `KeyedMutex<K>` (iter 241) | `parking_lot::Mutex<HashMap<K, Weak<tokio::sync::Mutex<()>>>>` | **Per-key async serialization**: different keys run in parallel; same-key callers queue in arrival order. Common prod pattern not directly in tokio: "only one task per `thread_id` mutating conversation state at a time" with thousands of threads independent. The `Weak` trick gives bounded memory: when no caller holds the lock and none are waiting, the inner `Arc<Mutex>` drops to refcount=0 → `Weak::upgrade` returns None → the next lookup creates a fresh `Mutex` for that key. `cleanup()` drops stale `Weak`s — cheap, safe to call from a periodic task; mostly matters for unbounded-key workloads (ephemeral request IDs / one-shot trace IDs) where stale entries would otherwise grow monotonically at ~16 bytes each. Real prod use: per-thread ReAct serialization, per-user tool-call coupling for state-mutating tools (Notion/GitHub/Jira), per-resource writer exclusivity (one upsert per vector-store collection at a time) |
| `RateLimiter` (iter 242) | `parking_lot::Mutex` over `(tokens: f64, last_refill: Instant)` + lazy-refill on every `acquire`/`try_acquire` | **Reusable async token-bucket** primitive distinct from `RateLimitedChatModel` (which wraps one ChatModel). Any caller — chat, embed, tool, loader — can charge against one shared budget. Useful when a single quota (e.g. one OpenAI key's TPM) covers heterogeneous calls. Lazy-refill design: no background task, each call recomputes the current token count from `last_refill_instant` (correct under arbitrary clock-tick patterns). Burst allowed up to `capacity`; sustained rate capped at `refill_per_sec`. Three modes: `try_acquire(n)` non-blocking; `acquire(n)` blocks until tokens accumulate; `acquire_with_shutdown(n, &shutdown)` cancellable. Critical detail: the shutdown path does NOT deduct tokens — the budget is preserved for surviving callers. Asks larger than `capacity` are clamped (otherwise would block forever, since the bucket can never accumulate more than capacity). Real prod use: shared provider quota across N agents (one OpenAI key serving 5 concurrent agents, every agent calls `limiter.acquire(tokens_estimate)` before its HTTP request), egress traffic shaping (outbound HTTP fan-out limited to N req/sec across all loaders), per-user fairness via the keyed-registry pattern from iter 241 (one limiter per `user_id` enforces "X requests per minute per user") |
| `CircuitBreaker` (iter 243) | `parking_lot::Mutex<InnerState>` (Closed{n_failures} / Open{until: Instant} / HalfOpenProbing) — `compare_exchange`-style state transitions inside the same lock guard | **Three-state resilience primitive** distinct from `RetryingChatModel`. RetryingChatModel retries on individual call errors (right when failures are *transient*); CircuitBreaker stops the bleeding when an upstream is *down* — after N consecutive failures, every subsequent call fails fast for `cooldown`, giving the upstream room to heal. Composable wrap-any-future API: `breaker.call(|| chat.invoke(msgs)).await`. Half-open semantics: exactly one in-flight probe allowed; concurrent callers see `CallError::CircuitOpen`; probe success → Closed (counter reset); probe failure → Open (cooldown reset). Manual `trip(cooldown)` / `reset()` hooks for ops runbooks ("we know provider X is down, open the breaker for 60s while we cut over"). Real prod use: third-party provider outage (after 5 consecutive 503s, breaker opens for 30s — agents fall back via `FallbackChatModel` instead of retrying for 30s each), vector-store quarantine (route reads to HNSW replica while pgvector is sick), tool blast-radius limiter (agent reasons about an unavailable tool rather than waiting on timeouts) |
| `CircuitBreakerChatModel` (iter 244) | `Arc<dyn ChatModel>` + `Arc<CircuitBreaker>`; map `CallError::CircuitOpen → Error::Provider("circuit breaker open")` so the existing fallback chain pattern-matches it as a transient | **Bridges the iter-243 primitive into the existing `ChatModel` resilience family**. Wraps any inner ChatModel; both `invoke` and `stream` admission-gate through the breaker. Streams are wrapped at the handshake (open → fail-fast without invoking inner stream); mid-stream failures stay the consumer's responsibility — matches the breaker's admission-control semantics. Stacks naturally with the existing toolkit; recommended outer-to-inner ordering: `CircuitBreakerChatModel` (fail-fast on persistent outage) → `FallbackChatModel` (switch provider on circuit-open) → `RetryingChatModel` (retry primary on transient errors) → `RateLimitedChatModel` (local rate cap) → real provider. The Error::Provider error message ensures `is_transient` doesn't accidentally retry circuit-open as a 5xx — the message text doesn't match the 5xx patterns |
| `CircuitBreakerEmbeddings` (iter 245) | `Arc<dyn Embeddings>` + `Arc<CircuitBreaker>` (one shared breaker spans both `embed_query` and `embed_documents`) | **Embed-axis mirror of iter 244**. The shared-breaker design matters: `embed_query` and `embed_documents` typically hit the same upstream; one provider going sick should fail-fast both call shapes. A flapping query path opens the breaker for document indexing (verified by test). Same composition pattern as the chat side — stack with `FallbackEmbeddings` for circuit-open → secondary embedder. With this iter the resilience matrix is symmetric across chat/embed: every chat resilience wrapper (Retrying / RateLimited / Fallback / Race / Timeout / CircuitBreaker) has its embedding sibling |
| `SharedRateLimited{ChatModel, Embeddings}` (iter 246) | `Arc<dyn ChatModel>` / `Arc<dyn Embeddings>` + `Arc<RateLimiter>` (iter-242 primitive) | **Bridges iter-242 `RateLimiter` into the resilience family**. Distinct from the existing `RateLimitedChatModel` / `RateLimitedEmbeddings` which own their own bucket per wrapper: this variant takes a shared `Arc<RateLimiter>` so multiple wrappers share ONE budget. The realistic prod pattern: one provider API key has ONE TPM/RPM quota covering several model variants (gpt-4 / gpt-4-turbo / gpt-4o-mini all on one key). With per-wrapper buckets the aggregate exceeds the real budget; with this variant they all charge against one limiter. The shared bucket can also span the chat/embed axes — useful when one OpenAI key covers both completions AND embeddings APIs. Each call charges 1 token at handshake; users wanting token-weighted accounting can call `limiter.acquire(estimate).await` directly upstream of the wrapped model. Verified by tests including a 4-call burst across two distinct wrapped models that must take >= 30ms with shared cap=2/refill=50/sec |
| `KeyedSerializedChatModel` (iter 247) | `Arc<dyn ChatModel>` + `Arc<KeyedMutex<String>>` (iter-241 primitive) + `Arc<dyn Fn(&[Message], &ChatOptions) -> Option<String>>` key extractor | **Bridges iter-241 `KeyedMutex` into the chat family**. The realistic prod use case: a stateful agent's ReAct loop reads-then-writes shared conversation state per `thread_id`. Two concurrent steps for the same thread interleave their reads/writes and corrupt state. A single global mutex fixes correctness but serializes every thread one-at-a-time. This wrapper lifts the per-thread ReAct-step lock into the model layer: same `thread_id` queues, different `thread_id`s run concurrently. Key extractor returns `Option<String>` — `None` means "no serialization for this call" (passes through unlocked), useful for one-shot calls without thread context. Streaming: the lock is held only across the handshake (the inner `stream()` call); it releases when this method returns since the caller drives the stream itself. For per-step exclusion across the whole stream lifetime, hold the lock externally around the `stream` + `drain` cycle |
| `Bulkhead` (iter 248) | `Arc<tokio::sync::Semaphore>` + `AtomicU64` rejected counter | **Concurrent-call cap with REJECTION semantics**, named after the "Release It!" pattern (separate failure domains so one saturated dependency doesn't drown the process). Distinct from a plain Semaphore: where Semaphore queues callers indefinitely, a Bulkhead surfaces the saturation as a *signal* via `try_enter() -> Option<BulkheadGuard>`. Three modes: `try_enter` (non-blocking, instant reject if at cap), `enter` (block until slot opens — Semaphore-equivalent, provided so the wrapper is useful when the caller actually wants queueing), `enter_with_timeout(t)` (block up to `t` then reject). Tracks `rejected_count` and `in_flight()` for telemetry. The real value vs `Arc<Semaphore>` is the *intent contract* — the type name documents that callers are expected to handle rejection rather than wait forever. Real prod use: per-tool concurrent cap (5 in-flight max for flaky API; the 6th caller gets BulkheadFull so the agent picks a different action rather than waiting), vector-store connection budget (cap retrievers below pool size to leave headroom for writes), outbound HTTP fan-out cap (skip the (N+1)th slow request rather than queue forever) |
| `BulkheadChatModel` + `BulkheadEmbeddings` (iter 249) | `Arc<dyn ChatModel>` / `Arc<dyn Embeddings>` + `Arc<Bulkhead>` (iter-248 primitive) + `BulkheadMode::{Reject, WaitUpTo(Duration)}` | **Bridges iter-248 into the resilience family**. Two modes per wrapper (configurable at construction): `Reject` for instant rejection on cap (the typical bulkhead use), `WaitUpTo(t)` for block-then-reject with a deadline. The critical design choice — bulkhead-full surfaces as `Error::RateLimited { retry_after_ms: None }`, NOT a custom error variant. Reason: `is_transient` already matches `RateLimited`, so `RetryingChatModel` retries with backoff (slot may have opened) and `FallbackChatModel` switches provider — both interactions just work, no extra wiring. One `Arc<Bulkhead>` can span multiple wrappers AND across chat+embed axes, enforcing one cap over a shared resource budget (verified by the chat-and-embed-share-one-budget test). With this iter, every base coordination/resilience primitive that makes sense as a single-call wrapper has its ChatModel/Embeddings sibling: CircuitBreaker (244/245), SharedRateLimiter (246), KeyedMutex (247-chat-only), Bulkhead (249). The Barrier and CountDownLatch primitives remain unwrapped — they're cohort-coordination shapes, not single-call patterns |
| `hedged_call(primary, backup, hedge_delay)` (iter 250) | `tokio::pin!` + `tokio::select! { biased; primary; sleep(hedge_delay) }` for phase 1; `tokio::select! { primary; backup }` for phase 2 | **Tail-latency mitigation combinator** distinct from iter-184 `RaceChatModel`. Race issues to *every* inner provider simultaneously (right when "first response wins, cost-no-object"). Hedge issues primary alone for `hedge_delay`; only if primary hasn't finished does backup also run. Fast-path requests pay zero overhead — only the slow tail incurs the second-call cost. Standard pattern from Dean & Barroso 2013 ("The Tail At Scale"). Implementation detail: phase-1 select uses `biased` so primary wins the race-to-flag if both happen to be ready in the same tick (cheap correctness guard against a rare scheduling artifact). Loser future is dropped at end-of-`select!`, releasing whatever HTTP / DB / sleep it held via tokio cancellation. Generic over future output type, so it composes with `Result<_, _>` (the natural shape for hedging fallible operations). Real prod use: LLM provider with 500ms p50 / 30s p99 — set `hedge_delay = 2s`; calls under 2s pay nothing, slow tail covered by backup. Multi-region failover: primary in us-east-1, backup in us-west-2; hedge after 1s |
| `HedgedChatModel` + `HedgedEmbeddings` (iter 251) | Two `Arc<dyn ChatModel>` (or `Embeddings`) + `Duration` hedge delay; delegates to iter-250 `hedged_call` | **Bridges iter-250 into the resilience family**. Distinct from iter-184 `RaceChatModel`: race issues to both providers simultaneously (always doubles cost); hedge only pays the second-call cost when primary is slow. The right trade-off when median latency is fine and you only want to insure against the p99. Embed axis: both `embed_query` and `embed_documents` hedged. Streaming caveat: `stream()` is primary-only — token streams can't be cleanly raced (chunks can't merge or be chosen between mid-stream); callers needing stream tail-latency mitigation should run their own per-chunk timeout + restart-on-different-provider logic. With this iter, every "operate over a single call" coordination/resilience primitive (CircuitBreaker, SharedRateLimiter, KeyedMutex, Bulkhead, Hedged) has its ChatModel + Embeddings sibling — the resilience matrix is fully populated |
| `Singleflight<K, V>` (iter 252) | `parking_lot::Mutex<HashMap<K, broadcast::Sender<V>>>` + leader/follower state machine in `get_or_compute` | **Request-coalescing primitive** — same shape as Go's `golang.org/x/sync/singleflight`. When N concurrent callers ask for the same key, only ONE inner computation runs; followers subscribe to the leader's `tokio::sync::broadcast` channel and receive the cloned result. The map entry is removed AFTER the leader finishes so a fresh `get_or_compute` call starts a new compute (no stale-cache problem). Distinct from `KeyedMutex` (iter 241): KeyedMutex serializes same-key callers — each one runs its own compute one-at-a-time. Singleflight runs the compute *once* and broadcasts. `V: Clone` is required because `broadcast::Sender::send` clones to each receiver; use `Arc<T>` for expensive types so the broadcast is just an Arc-clone. Edge cases: leader cancellation (channel closed without send) → followers fall back to running their own compute; channel-Lagged (shouldn't happen with capacity=1 single-broadcast but handled defensively the same way). Real prod use: cache-miss thundering herd (100 agents want the same embedding → 1 HTTP call, 100 results), idempotent-tool coalescing (10 concurrent stock-price-for-AAPL calls → 1 upstream), lazy model-weights initialization (5 workers want the same weights → 1 load) |
| `SingleflightEmbeddings` (iter 253) | `Arc<dyn Embeddings>` + `Arc<Singleflight<String, Arc<Result<Vec<f32>, String>>>>` | **Bridges iter-252 into embeddings**. N concurrent `embed_query("foo")` calls share ONE inner HTTP call; followers get the leader's result via `tokio::sync::broadcast`. The `Arc<Result<Vec<f32>, String>>` type is forced by Singleflight's `V: Clone` constraint — `litgraph_core::Error` isn't Clone, so error variants are stringified into `Error::Provider(s)` on the caller side. Lossy by design: keeping exact error variants would require `Arc<Error>`, but converting back via `try_unwrap` is racy with multiple receivers (always fails when ≥1 follower is still around) so stringification is the practical choice. Documented in the wrapper's rustdoc; callers needing exact error variants should not coalesce. `embed_documents` passes through unchanged — multi-doc batches rarely repeat exactly, coalescing them would need a hashable key over `Vec<String>` for marginal expected win. With this iter, the singleflight pattern joins the resilience matrix on the embeddings axis (chat axis remains un-coalesced — system prompts are typically distinct per call, and dedup of model invocations changes semantics in a way users shouldn't get for free) |
| `RecordingChatModel` + `ReplayingChatModel` + `Cassette` (iter 254) | `parking_lot::Mutex<Cassette>` shared between recorder and the test scope; cassette serde-JSON; blake3 over canonical JSON of `(messages, opts)` for hash key | **VCR-style record/replay for agent tests** — substantial feature missing from most LangChain alternatives. Pivot from the small-primitive cadence to a real prod-grade testing-infra capability. Use case flow: developer wraps their live model in `RecordingChatModel` during a one-time real-traffic test run; serializes the resulting `Cassette` to disk (a JSON file in `tests/cassettes/`); CI runs use `ReplayingChatModel::new(loaded_cassette, None)` for deterministic, no-API-cost, no-quota-burn replay. Optional `passthrough: Option<Arc<dyn ChatModel>>` for "record-then-fill-gaps" workflows: cassette miss falls through to the live model. Hash determinism guaranteed by canonical-JSON serialization of all serde-derived request fields (no HashMap fields in messages or opts), so test cassettes are stable across rebuilds. Streaming intentionally not recorded — replay of token streams without preserving inter-chunk timing is misleading; stream calls pass through. Required adding `serde`, `serde_json`, `blake3` as runtime deps to litgraph-resilience (previously dev-only). Composes cleanly with the rest of the resilience family — wrap a recorder *outside* a retry/breaker chain to capture the actual user-visible request/response pairs, or *inside* to capture the underlying provider behavior |
| `Cassette::{load,save}_from_file` + `RecordingEmbeddings` + `ReplayingEmbeddings` + `EmbedCassette` + `EmbedExchange::{Query, Documents}` (iter 255) | `serde_json::to_string_pretty` + `std::fs` for IO; `EmbedExchange` is a tagged-union serde enum so one cassette covers both `embed_query` and `embed_documents` recordings | **Closes the record/replay workflow.** File IO: `Cassette::save_to_file(path)` creates parent directories with `create_dir_all` and writes pretty JSON; `Cassette::load_from_file(path)` reads + parses. Errors surface as `Error::Other(...)`. Embeddings axis: same recorder/replayer pattern with `EmbedCassette` (a tagged-union enum so query/documents responses share one cassette file). `embed_query_hash(text)` is blake3 over the text bytes; `embed_documents_hash(texts)` is blake3 over canonical JSON of the `Vec<String>` (order-sensitive — embed_documents returns aligned vectors so reordering changes the request semantically). `ReplayingEmbeddings::dimensions()` defaults to the first exchange's vector length when no passthrough is set, otherwise delegates to passthrough — keeps `dimensions()` correct for downstream code that consults it pre-call. Real prod use: `cargo test` in CI runs the full agent flow (chat + embed + retrieval) against committed cassettes, no API keys required, response stable across runs. With this iter the record/replay story is symmetric across chat/embed and end-to-end usable; the only major missing piece (chunk-timing-aware streaming replay) remains documented as out-of-scope |
| `RecordingTool` + `ReplayingTool` + `ToolCassette` + `tool_args_hash` (iter 256) | Same machinery as iter 254/255 but over `litgraph_core::tool::Tool`. `ToolExchange { request_hash, tool_name, args, response }` with both args and response as `serde_json::Value` for full agent-tool fidelity | **Third record/replay axis** (chat / embed / tool). Hash key: blake3 over canonical JSON of `args`. The `ReplayingTool` schema-handling deserves a note: when no passthrough is configured, the schema is synthesized from a configurable `name` + `description` (`with_name` / `with_description` builders) plus a minimal `{"type": "object"}` parameters object. That's enough to satisfy callers that only need the cassette's responses (typical agent-test setup); when full schema fidelity is needed, the optional passthrough's schema is proxied. Builder pattern keeps the API ergonomic for the common test setup of `ReplayingTool::new(cass, None).with_name("search")`. Real prod use: agent integration tests that exercise external-API tools — record against a staging environment once, commit the cassette to `tests/cassettes/`, replay in CI without any service dependencies (no Slack, no GitHub, no real database hits). With this iter the record/replay matrix is complete across the three primary axes — combined with iter 254/255 file IO, the entire agent loop (LLM calls + embeddings + tool calls) is now CI-deterministic with one consistent VCR-style API |
| `RssAtomLoader` (iter 257) | `quick-xml` pull parser walking a single state machine over both RSS 2.0 and Atom 1.0 inputs; feature-gated `rss` (re-uses the existing `quick-xml` optional dep that was already pulled in by the `docx` feature) | **Pivot from the resilience cadence to a new loader.** RSS/Atom is the canonical way blogs / news / podcast / release-note feeds publish a stream of recent items; the framework already had 24+ loaders but this canonical one was missing. Single parser handles both formats: namespace-prefix-stripped tag matching (`content:encoded` → `encoded`) so the same match arms work for either schema, attribute-based `<link href="…"/>` (Atom) and text-content `<link>https://…</link>` (RSS) both supported, body-resolution priority `content > summary > title` so title-only items still emit a non-empty document. `pubDate` (RFC822) is best-effort normalized to RFC3339. Per-item metadata: `title`, `link`, `published`, `feed_title`, `feed_url`. Configurable: `with_max_items` cap to bound feed-bomb risk, `with_skip_empty` filter, `with_user_agent` / `with_timeout` knobs (matches the convention from `SitemapLoader` etc). The `parse(&self, xml: &str) -> Vec<FeedItem>` method is `pub` so tests can drive the parser directly without reqwest — keeping the unit-test layer cheap and offline. Real prod use: news brief generator (ingest top tech-blog feeds, agent summarizes the last 24h), release-notes RAG (ingest a project's `releases.atom`, agent answers "what changed in v3.x?"), competitive monitoring (cluster competitor blog feeds for theme detection) |
| `HackerNewsLoader` (iter 258) | Public HN Firebase API at `hacker-news.firebaseio.com/v0/` — no auth, no rate-limit headers; HN explicitly invites traffic. Two-step fetch: `<feed>stories.json` returns IDs (~500 max), then `item/<id>.json` per item. Rayon-parallel item fetch since per-item calls are independent | **Second loader pivot in this iter cluster.** Six feed sources via `HnFeed::{Top, New, Best, Ask, Show, Job}`. Each item produces a Document with the title and text combined (for Ask/Show posts where the body content is the actual prose) and metadata `{hn_id, title, url, by, score, time, type, descendants, feed}`. Items lacking both title and text are dropped silently — story-deletion / dead-account markers shouldn't pollute the corpus. The loader is offline-testable: `HnItem::into_document(feed_name)` is public, so tests drive the parser without reqwest. `with_base_url` knob lets integration tests point at a fake-HN local server. Default `max_items=30` to keep the typical fetch bounded; bump for batch-ingestion runs. Real prod use: tech-news brief, title-only zero-shot embedding corpus (cheap to build, useful for testing semantic search), Show-HN trend monitoring over time |
| `PriorityQueue<T>` (iter 259) | `parking_lot::Mutex<BinaryHeap<Entry<T>>>` + `AtomicU64` insertion sequence + `tokio::sync::Notify` for waking poppers. `Entry: Ord` orders by priority desc, then seq asc | **Async work queue with priority-based pop**, distinct from `tokio::sync::mpsc` (FIFO across the whole channel — a high-priority task pushed late waits behind every earlier-pushed task). Within the same priority, FIFO order is preserved via an insertion sequence number. Three API shapes: `try_pop` non-blocking, `pop` blocking with `Notify::notified` race-safe pattern (register before re-check, same as `ShutdownSignal::wait`), `pop_with_shutdown` for cancellable graceful drain. The `Send + 'static` bound on `T` is intentional — items must be safe to move across task boundaries since the queue is typically shared via `Arc<PriorityQueue<T>>`. Builder-free API by design: most prod uses set the priority per-push at the call site rather than via wrapper config. Real prod use: urgent retries first (a graph node that failed and is being re-scheduled jumps ahead of fresh work), hard-cases-first eval harness (most-likely-to-fail rows scored first so cancelled runs surface failures fastest), latency-budget UI requests (`priority=10` jumps the batch). Pairs naturally with iter-248 `Bulkhead` (concurrent-cap on top of priority) and iter-225 `ShutdownSignal` (graceful drain via `pop_with_shutdown`) |

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

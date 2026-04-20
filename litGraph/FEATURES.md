# litGraph — Feature List

Production-grade, slim alternative to LangChain + LangGraph.
Rust core, Python bindings via PyO3 0.28 + maturin.

## Guiding Principles

1. **Rust heavy lifting, Python ergonomics** — every hot path (HTTP, SSE parse, tokenize, embed math, vector search, JSON parse, graph scheduling) runs in Rust. Python is a thin shim.
2. **True parallelism** — Tokio for I/O fan-out, Rayon for CPU-bound batching, GIL released (`py.detach`) around every heavy block. No GIL-bound asyncio overhead.
3. **Shallow call stacks** — ≤2 frames from user code to HTTP. No 6-layer `Runnable.invoke` pipeline.
4. **Split crates, zero default features** — pay only for what you import. No `langchain-community` mega-dep.
5. **OTel-native observability** — not LangSmith-locked. `tracing` + OpenTelemetry exporter from day 1.
6. **Inspectable prompts** — `on_request` hook exposes final HTTP body. Solves 50% of debug pain.
7. **SemVer discipline** — slow deprecation cycle, clear migration paths.
8. **Graph-first** — LangGraph-style StateGraph is the headline primitive. LangChain's class zoo (chains/memory/agent variants) collapses into functions + graph nodes.

---

## Architecture

```
litgraph/                            (Cargo workspace)
├── litgraph-core/                   no PyO3, pure Rust
│   ├── message / content-part types
│   ├── prompt templates
│   ├── Runnable trait (Step<I,O>)
│   ├── ChatModel / Embeddings traits
│   ├── Tool trait + #[tool] macro
│   └── errors (thiserror)
├── litgraph-graph/                  StateGraph executor
│   ├── petgraph::StableGraph backend
│   ├── Kahn scheduler + JoinSet + Semaphore
│   ├── CancellationToken wiring
│   ├── reducers (derive macro)
│   └── checkpointers trait
├── litgraph-providers-openai/       async-openai adapter
├── litgraph-providers-anthropic/
├── litgraph-providers-gemini/
├── litgraph-providers-bedrock/
├── litgraph-providers-ollama/
├── litgraph-stores-usearch/         embedded vector (default)
├── litgraph-stores-qdrant/
├── litgraph-stores-pgvector/
├── litgraph-stores-lancedb/
├── litgraph-loaders/                text/pdf/html/md/json/csv
├── litgraph-splitters/              recursive/token/md/tree-sitter
├── litgraph-checkpoint-sqlite/
├── litgraph-checkpoint-postgres/
├── litgraph-checkpoint-redis/
├── litgraph-tracing-otel/
└── litgraph-py/                     PyO3 bindings (thin shim)
```

**Rule:** zero PyO3 imports outside `litgraph-py`. Core is usable as a pure Rust crate.

---

## v1.0 Must-Have Features

### Models & Embeddings
- `ChatModel` trait: `invoke`, `stream`, `batch`, `ainvoke`
- `Embeddings` trait with batched `embed_documents` / `embed_query`
- Provider adapters: OpenAI, Anthropic, Gemini, Bedrock, Ollama (ships 5)
- Native function/tool calling per provider
- SSE streaming via `eventsource-stream` + bounded mpsc → Python async iterator
- Zero-copy embedding tensors to Python via `rust-numpy`
- Token counting: HF `tokenizers` + `tiktoken-rs`
- Retry with exponential backoff + jitter (`backon`); skip 4xx except 408/429
- `on_request` / `on_response` hook for prompt inspection

### Prompts & Output
- `ChatPromptTemplate`: role-tagged parts, `{var}` interpolation via `minijinja`
- Partial application (bind subset, return new template)
- Structured output via native tool calling + `schemars`
- Streaming JSON parser (incremental) — `struson` or simd-json state machine
- `#[derive(Deserialize, JsonSchema)]` → auto schema → parsed struct
- Retry-on-parse-fail wrapper with repair prompt

### Tools & Agents
- `Tool` trait, typed args via serde + schemars (auto-schema)
- `#[tool]` proc-macro: function → registered tool with JSON schema
- Concurrent tool execution (`JoinSet`) for parallel tool calls
- Built-in tools: HTTP, SQL (`sqlx`), shell (sandbox-gated), calculator (`evalexpr`)
- Tool-calling agent loop (prebuilt, LangGraph-style `create_react_agent`)
- Recursion guards, max-step limits, cost caps

### Retrieval & RAG
- `Retriever` trait: `async fn retrieve(query, k) -> Vec<Document>`
- Dense vector retriever over any `VectorStore`
- BM25 retriever (pure Rust: `bm25` crate or `tantivy`)
- Hybrid (RRF / weighted fusion)
- Reranker hook (Cohere/Jina/local cross-encoder via `ort` + `fastembed-rs`)
- `VectorStore` trait: add/search/delete/search-by-vector + metadata filter
- Embedded default: `usearch` (SIMD, i8/f16 quantization)
- Clients: Qdrant, pgvector, Pinecone, Weaviate, Chroma

### Ingestion (Rayon-Parallel)
- Loaders: text, PDF (`pdfium-render`), HTML (`scraper`+`readability`), Markdown, JSON/JSONL, CSV
- Splitters: recursive char, token-based, Markdown-aware, code-aware (`tree-sitter`)
- Directory loader: glob + `rayon::par_iter` for parallel ingest (Python can't match this under GIL)
- Pipelined stage executor: load → split → embed → upsert via bounded mpsc channels

### StateGraph (headline primitive)
- Typed state via `#[derive(GraphState)]` with per-field reducers (`#[reduce(append)]`, `#[reduce(replace)]`, user-defined)
- Node = `async fn(&State) -> StateUpdate` (partial update, not replace)
- Edges: static, conditional (fn → enum variant), entry, END
- Enum-based node IDs → compile-time name validation
- `petgraph::StableGraph` backend; cycle detection, topo sort
- Kahn scheduler: `JoinSet` + `Semaphore(max_parallel)` + `CancellationToken`
- Subgraphs as nodes (shared or namespaced state slice)
- `Send`-style fan-out API: one node emits N parallel child invocations, reducer collects
- Parallel branches — free because Rust has no GIL

### Persistence & Durability
- `Checkpointer` trait keyed by `thread_id` + step
- Implementations: in-memory, SQLite (`rusqlite`/`sqlx`), Postgres (`sqlx`), Redis
- Serialization: `bincode` or `rmp-serde` for snapshots (not JSON — LangGraph's perf drag)
- Resumable execution after crash
- Per-node retry policy (`backon`)
- Idempotency via step-ID keys

### Human-in-the-Loop
- Interrupt before / after by node name
- `interrupt(payload)` inside node: suspends with serializable payload, resume via `Command { resume: ... }`
- State editing from outside (fork branch, replay)
- Time travel — replay from any checkpoint

### Streaming
- Modes: `values` (full state), `updates` (diffs), `messages` (token-by-token), `custom` (user events)
- Rust: `impl Stream<Item = GraphEvent>`
- Python: async iterator class (`__aiter__`/`__anext__`) via `pyo3-async-runtimes`

### Memory (slim)
- `trim_messages`: token-count window + message-count window
- Summarization helper (free function)
- Vector-backed conversation memory = reuse retriever + checkpointer

### Caching
- Cache trait keyed by (model, prompt, params) hash
- Backends: in-memory (`moka` LRU), SQLite, Redis (`fred`)
- Semantic cache (embed query, cosine-threshold) — v1.1

### Observability
- `tracing` crate with structured spans (one span per node)
- OpenTelemetry exporter (`opentelemetry` + `opentelemetry-otlp`)
- Event bus / callback trait: `on_llm_start/end/token`, `on_tool_start/end`, `on_node_start/end`, errors
- Token & cost accounting hook
- Batched callbacks (handle → `Vec<Event>` per tick) — avoids per-token GIL thrash

### PyO3 Bindings
- Native `async fn` in `#[pyfunction]` for one-shot async; `pyo3-async-runtimes` for long-lived streams
- `Bound<'py, T>` in params, `Py<T>` in stored state
- `py.detach(||)` around every CPU-bound/Rayon block
- `rust-numpy`: zero-copy `PyArray<f32>` ↔ `ndarray` for embeddings
- `anyhow` feature: `anyhow::Error` → `PyRuntimeError` auto
- `thiserror` + `create_exception!` for typed Python exception subclasses
- `pyo3-stub-gen`: `.pyi` files generated at build time → IDE autocomplete
- abi3 wheels (`abi3-py39`) via maturin-action + `--zig` for manylinux cross-compile
- Thread-safe for free-threaded Python 3.13

---

## v1.1+ Nice-to-Have

- Semantic cache, semantic splitter
- LangSmith OTel compatibility shim
- LanceDB, Weaviate, Chroma, Pinecone stores (some gated features)
- Multi-modal content blocks (images → base64 → provider)
- Multi-query retriever, contextual compression, parent-document, self-query
- Supervisor / swarm / hierarchical multi-agent prebuilts
- `Command(goto=agent_b, update=...)` handoff primitive
- Local inference adapter via `mistral.rs` or `candle`
- Few-shot template with example selector
- Cost dashboard helper
- Dead-letter queue hook
- Structured-chat agent (for models without native tool calling)

---

## Explicitly Out of Scope (the bloat cut)

- Legacy `LLM` (completion-only) base class — chat-only
- 200+ community provider integrations → BYO trait, 5 first-class
- 150+ loaders (Notion/Slack/Confluence/etc.) → community plugin crates
- 60+ exotic vector stores (Vald/Tair/Marqo/Vearch/…)
- 100+ tool wrappers (Zapier/Gmail/GitHub/Jira/…) → userland
- 20+ exotic cache backends (MongoDB/Cassandra/Astra/Momento/Couchbase)
- Deprecated memory classes (`ConversationEntityMemory`, `ConversationKGMemory`, `ConversationTokenBufferMemory`, `CombinedMemory`)
- Non-LCEL chain classes (`LLMChain`, `SequentialChain`, `RouterChain`, `MultiPromptChain`, `TransformChain`, `MapReduceChain`, `RefineChain`, `StuffDocumentsChain`) — use graph nodes
- `PipelinePromptTemplate`, `StringPromptTemplate` hierarchy, `FewShotPromptWithTemplates`
- Regex-parsing `ZeroShotReAct`, `SelfAskWithSearch`, `ConversationalAgent`, `PlanAndExecute` as distinct classes
- Auto-GPT / BabyAGI ports
- LLM-as-judge eval framework (separate crate if ever)
- Dual sync/async callback manager (LangChain's confusion source)
- `MergerRetriever`, `TimeWeightedVectorStoreRetriever`, `WebResearchRetriever`, `ZepRetriever`, etc.
- Long-tail `CommaSeparatedListOutputParser`-style trivial parsers

---

## Rust Dependency Plan

**Core (always pulled):**
- `tokio` + `tokio-util` — runtime, CancellationToken, mpsc
- `rayon` — CPU-bound batching
- `petgraph` — DAG (StableGraph)
- `reqwest` + `eventsource-stream` — HTTP + SSE
- `serde`, `serde_json`, `simd-json` — serialization
- `schemars` — JSON Schema from Rust types
- `thiserror` + `anyhow` — errors
- `tracing` + `tracing-subscriber` + `opentelemetry` — observability
- `backon` — retry
- `minijinja` — prompt interpolation
- `bincode` or `rmp-serde` — checkpoint serialization

**Feature-gated:**
- `async-openai` — OpenAI adapter (`features = ["openai"]`)
- `tokenizers` — HF tokenizers
- `tiktoken-rs` — OpenAI token count
- `fastembed` — local embeddings (`features = ["local-embed"]`)
- `ort` — ONNX runtime (`features = ["onnx"]`)
- `usearch` / `hnsw_rs` — vector index (`features = ["vector-embedded"]`)
- `qdrant-client`, `lancedb` — remote vector stores
- `sqlx` — SQL checkpointer / pgvector
- `fred` — Redis
- `pdfium-render`, `scraper`, `tree-sitter` — loaders/splitters
- `bm25` / `tantivy` — BM25
- `evalexpr` — calculator tool
- `mistralrs` — local inference (optional)

**Python layer:**
- `pyo3` 0.28 (abi3-py39)
- `pyo3-async-runtimes` — asyncio bridge
- `rust-numpy` — zero-copy tensors
- `pyo3-stub-gen` — .pyi generation
- `maturin` — build system

---

## Parallelism Design (the wedge vs LangChain)

### Graph Executor — Kahn + JoinSet
```rust
let mut ready: VecDeque<NodeIdx> = roots();
let mut running: JoinSet<(NodeIdx, Result<StateUpdate>)> = JoinSet::new();
let sem = Arc::new(Semaphore::new(max_parallel));
let cancel = CancellationToken::new();

loop {
    while let Some(idx) = ready.pop_front() {
        let permit = sem.clone().acquire_owned().await?;
        let child_cancel = cancel.child_token();
        running.spawn(async move {
            let _permit = permit;
            tokio::select! {
                _ = child_cancel.cancelled() => (idx, Err(Cancelled)),
                r = run_node(idx) => (idx, r),
            }
        });
    }
    let Some(res) = running.join_next().await else { break };
    let (idx, update) = res??;
    apply_update(&mut state, update);
    for succ in successors(idx) {
        if dec_indegree(succ) == 0 { ready.push_back(succ); }
    }
    if running.is_empty() && ready.is_empty() { break; }
}
```

### Rayon for CPU Math
Cosine batches, reranker scoring, chunking, quantization → `rayon::par_iter` inside `py.detach(||)`. **Never** mix rayon into tokio tasks without `spawn_blocking` — deadlock risk.

### Backpressure
Bounded `mpsc::channel(64)` for token streaming. Unbounded = memory leak under slow consumers. Size = 2× node concurrency cap.

### SSE Pipeline
`reqwest::Response::bytes_stream()` → `eventsource-stream` → `BoxStream<Result<ChatEvent>>` → `mpsc::channel` → Python async iterator class.

### What Python/LangChain Can't Match
1. **Parallel ingestion** — 10k docs: Rayon par-iter across cores, no GIL. Python needs multiprocessing (fork overhead, IPC).
2. **Concurrent tool/retriever fan-out** — 100 parallel LLM calls in map-reduce without asyncio↔GIL contention.
3. **Zero-copy embeddings** — f32 arrays to numpy without serialization roundtrip.
4. **Pipelined ingest** — load→split→embed→upsert stages scale independently via channels. LangChain has no native pipelined ingestion.
5. **Lock-free shared state** (`dashmap`, `arc-swap`) across parallel graph branches without global lock.
6. **GIL release everywhere** — multiple Python threads calling into lib get true parallelism.

---

## Benchmark Targets (criterion + E2E)

Must beat LangChain by ≥3× on:
- **Ingest 10k docs** (load → split → embed-batch → vector upsert): target 5×
- **1k agent loops** (tool-calling ReAct, 3 tools, 5 steps avg): target 3× p50, 5× p99
- **Streaming 1k tokens** (per-token callback latency): target 10×
- **Graph execution** (20-node DAG, 8 parallel branches): target 4×
- **Cold start** (import + first request): target 5×
- **RSS at 10k message histories**: target 3× smaller

### Current numbers (iter 6 — criterion on commodity macOS darwin)

```
graph_fanout/1           8.2 µs        122K nodes/s      (single-node scheduler overhead)
graph_fanout/4           8.4 µs        474K nodes/s
graph_fanout/16         21.6 µs        742K nodes/s
graph_fanout/64         90.6 µs        706K nodes/s

bm25_index/1k          4.4 ms          225K docs/s       (indexing)
bm25_index/10k        44.8 ms          223K docs/s
bm25_search/1k        48.6 µs          20.6M elem/s      (per-doc scoring, rayon-parallel)
bm25_search/10k        247 µs          40.5M elem/s
bm25_search/50k       2.13 ms          23.4M elem/s

vector_search/memory/10k     287 µs          34.9M elem/s     (brute-force rayon cosine)
vector_search/hnsw/10k        33 µs           305M elem/s     (HNSW, instant-distance)
vector_search/memory/100k   4.43 ms          22.6M elem/s
vector_search/hnsw/100k       41 µs           2.43G elem/s    ← 107× brute-force
```

Graph scheduler per-node overhead ≈1.3µs. BM25 scales linearly across corpora sizes.
LangGraph's Python scheduler adds GIL + asyncio overhead (~ms per fanout step in
published profiles); this is genuinely the order-of-magnitude win claimed in the
wedge above.

Harness lives in `crates/litgraph-bench/benches/`. Run with:
```bash
cargo bench -p litgraph-bench --bench bm25 -- --quick
cargo bench -p litgraph-bench --bench graph_parallel -- --quick
cargo bench -p litgraph-bench --bench splitters -- --quick
cargo bench -p litgraph-bench --bench cache -- --quick
```

---

## What Makes This Project the Wedge

`rig` is the closest Rust competitor — ergonomic, many providers, but **no state-graph + checkpointer + HITL story**. `swiftide` nails streaming RAG but is linear, not DAG. `llm-chain` is stalled. `langchain-rust` inherits LangChain's abstraction debt. `rs-graph-llm` is conceptually closest but thin.

**Nobody ships LangGraph-quality typed StateGraph + checkpointers + HITL with Python bindings in Rust.** That's the wedge. Build that well, wrap it with slim LangChain equivalents, release GIL everywhere, benchmark brutally, and the value prop sells itself.

---

## Sources

- LangChain: https://python.langchain.com/docs/
- LangGraph: https://langchain-ai.github.io/langgraph/
- LangGraph persistence: https://langchain-ai.github.io/langgraph/concepts/persistence/
- Octomind "Why we no longer use LangChain": https://www.octomind.dev/blog/why-we-no-longer-use-langchain-for-building-our-ai-agents
- Hamel Husain on prompts: https://hamel.dev/blog/posts/prompt/
- PyO3: https://pyo3.rs/v0.28.2/
- PyO3 async: https://pyo3.rs/v0.28.2/async-await
- PyO3 free-threading: https://pyo3.rs/v0.28.2/free-threading
- rig: https://github.com/0xPlaygrounds/rig
- swiftide: https://github.com/bosun-ai/swiftide
- async-openai: https://github.com/64bit/async-openai
- rust-genai: https://github.com/jeremychone/rust-genai
- candle: https://github.com/huggingface/candle
- mistral.rs: https://github.com/EricLBuehler/mistral.rs
- usearch: https://github.com/unum-cloud/usearch
- hnsw_rs: https://github.com/jean-pierreBoth/hnswlib-rs
- fastembed-rs: https://github.com/Anush008/fastembed-rs
- ort: https://github.com/pykeio/ort
- petgraph: https://github.com/petgraph/petgraph
- qdrant-client: https://github.com/qdrant/rust-client
- pyo3-stub-gen: https://github.com/Jij-Inc/pyo3-stub-gen
- rust-numpy: https://github.com/PyO3/rust-numpy
- backon: https://github.com/Xuanwo/backon
- tokio CancellationToken: https://docs.rs/tokio-util/latest/tokio_util/sync/struct.CancellationToken.html
- tokio JoinSet: https://docs.rs/tokio/latest/tokio/task/struct.JoinSet.html

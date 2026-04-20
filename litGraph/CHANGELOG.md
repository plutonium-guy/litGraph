# Changelog

All notable changes to litGraph are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to Semantic Versioning.

## [Unreleased]

### Added
- **Providers**: OpenAI, Anthropic, Google Gemini — all with native tool calling
  and SSE streaming. OpenAI-compatible base URLs cover Ollama / vLLM / Together
  / Groq / Fireworks / DeepSeek / LM Studio.
- **StateGraph executor**: typed state, partial-update reducers (`merge_append`,
  `merge_replace`), conditional edges, dynamic `goto`, parallel fan-out via
  `Send`-style commands, `interrupt_before` / `interrupt_after`, time-travel
  resume from checkpoint.
- **Checkpointers**: in-memory, SQLite (WAL), Postgres (deadpool-pooled,
  upsert-on-conflict), Redis (ZSET-per-thread, O(log n) latest).
- **Vector stores**: in-memory rayon brute-force, embedded HNSW
  (`instant-distance`, pure Rust), Qdrant REST, Postgres + pgvector.
- **Retrieval**: `Retriever` + `Reranker` traits, BM25 (Okapi rayon-parallel),
  hybrid RRF, dense `VectorRetriever`.
- **Loaders**: text, JSONL, Markdown, directory (rayon-parallel glob).
- **Splitters**: recursive character (UTF-8 safe), Markdown header (with
  breadcrumb metadata).
- **Agents**: ReactAgent (tool-calling loop, concurrent tool execution),
  SupervisorAgent (handoff/finish multi-agent routing).
- **Tools**: `Tool` trait, `FnTool` builder, `#[tool]` proc-macro that derives
  JSON Schema from the args type via `schemars`.
- **Observability**: Callback bus with batched drain (avoids per-token GIL
  thrash from Python subscribers), CostTracker (per-model PriceSheet → USD),
  InstrumentedChatModel, OTel exporter (feature-gated).
- **Cache**: `Cache` trait, MemoryCache (moka LRU + TTL), SqliteCache,
  SemanticCache (embedding-cosine lookup), `CachedModel` +
  `SemanticCachedModel` wrappers.
- **Python bindings (`litgraph` package)**: full surface — `litgraph.graph`
  (StateGraph, GraphStream), `litgraph.providers` (OpenAIChat / AnthropicChat
  / GeminiChat with `.invoke`, `.stream`, `.with_cache`, `.with_semantic_cache`,
  `.instrument`), `litgraph.agents` (ReactAgent, SupervisorAgent),
  `litgraph.tools` (FunctionTool), `litgraph.embeddings` (FunctionEmbeddings),
  `litgraph.retrieval` (Bm25Index, MemoryVectorStore, HnswVectorStore,
  QdrantVectorStore, PgVectorStore, VectorRetriever),
  `litgraph.splitters` (RecursiveCharacterSplitter, MarkdownHeaderSplitter),
  `litgraph.loaders` (TextLoader, JsonLinesLoader, MarkdownLoader,
  DirectoryLoader), `litgraph.observability` (CostTracker), `litgraph.cache`
  (MemoryCache, SqliteCache, SemanticCache).
- **Benchmarks** (criterion): graph fanout, BM25, splitters, cache, HNSW vs
  brute-force. Numbers in `FEATURES.md`. Highlight: HNSW search at 100k docs
  is **107× faster** than brute-force cosine.
- README.md with quickstarts. LICENSE (Apache-2.0).

### Architecture
- 22 split crates with zero default features — pay only for what you import.
- Shared tokio runtime in `litgraph-py` (one per process); GIL released around
  every async / Rayon block.
- bincode-serialized state snapshots for checkpoints (compact, fast).
- Submodules registered in `sys.modules` so `from litgraph.X import Y` works.

### Tests
- 44 Rust unit + integration tests across all crates.
- 33 Python E2E tests covering StateGraph, RAG pipeline, streaming, agents,
  cache, observability, multi-agent supervisor.
- Fake HTTP servers used to verify provider SSE streams, cache wiring, and
  cost-instrumentation pipelines without live API calls.

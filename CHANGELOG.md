# Changelog

All notable changes to litGraph are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project adheres
to Semantic Versioning.

## [Unreleased]

### Fixed
- **OpenAI provider — streaming token usage was always zero.** The streaming
  request never sent `stream_options: {"include_usage": true}`, and per the
  OpenAI streaming spec a server only reports usage when the client opts in.
  The `done` event therefore carried `{prompt: 0, completion: 0, total: 0}` on
  every spec-compliant backend (OpenAI, Ollama, vLLM, LM Studio), so
  `CostTracker` accounted every streamed call as free. DeepSeek sends usage
  unprompted, which is why the gap went unnoticed. Intermediate chunks carrying
  an explicit `"usage": null` are now ignored so they cannot clobber the real
  totals.
- **`litgraph-serve` auth layers did not compile for any caller.** Both
  `auth::bearer_layer` and `auth::forwarded_user_layer` were unusable, not
  merely mis-documented: axum places `Request` inside the extractor tuple as
  the final `FromRequest`, and both hand-written type aliases omitted it, so
  neither layer had a `Service` impl and `Router::layer` rejected them. The
  documented drop-in example now compiles unchanged.
- **`agents_extras.BigToolAgent` rejected every native embeddings provider.**
  The constructor called `embeddings.embed(...)`, which no class in
  `litgraph.embeddings` implements — they expose `embed_documents` /
  `embed_query` — so construction always raised `AttributeError`. Both shapes
  are now accepted, with a clear `TypeError` when neither is present.
- **`recipes.serve` misrouted a `CompiledGraph` into the chat-model branch.**
  The check keyed on the absence of `.compile`, which a compiled graph also
  lacks, so callers got a misleading `ValueError` instead of the documented
  `NotImplementedError`. Graphs are now identified positively.
- **`pytest-asyncio` was undeclared**, so five async tests errored on a clean
  environment. Added to `pixi.toml`.

### Changed
- **Live integration suite targets any OpenAI-compatible endpoint.** DeepSeek
  remains the default; `LITGRAPH_TEST_BASE_URL` / `_MODEL` / `_API_KEY` /
  `_TIMEOUT_S` / `_EMBED_MODEL` point it at Ollama, vLLM, LM Studio, or
  Together. Skips are now keyed on endpoint capability
  (`python_tests/integration/_capabilities.py`) rather than a hardcoded
  provider name, which un-blocked five tests that had been skipped as
  permanently unsupported — `LlmJudge` (2), `synthesize_eval_cases` (2), and
  `BigToolAgent`, whose stub had been hiding the defect above.
- **Docs corrected against the code.** `COMPARISON.md` claimed multi-tenant
  auth scaffolding as shipped; the identity extracted by `forwarded_user_layer`
  is consumed by nothing and `studio_router` applies no per-thread ACL, so the
  serve binary is still single-tenant. `MISSING_FEATURES.md` still listed the
  WebSocket endpoint as missing although it ships behind feature `ws`.
  `AGENT_DX.md` still described `recipes.serve` as rendering a command string.

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

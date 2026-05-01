# litGraph — Feature List

Production-grade, slim alternative to LangChain + LangGraph.
Rust core, Python bindings via PyO3 0.28 + maturin.

## Status — 2026-04-29 (iter 162)

39 crates · ~1081 Rust tests · ~948 Python tests · all passing.

**Legend:** ✅ done · ⏳ in flight · ❌ not started · 🚫 deferred indefinitely

### Done — v1.0 must-haves
- ✅ ChatModel + Embeddings traits
- ✅ Providers: OpenAI, OpenAIResponses, Anthropic, Gemini (AI Studio + Vertex), Bedrock (native + Converse), Cohere, OpenAI-compat (Ollama, Groq, Together, Mistral, DeepSeek, xAI, Fireworks)
- ✅ Native function/tool calling per provider
- ✅ SSE streaming → Python async iterator
- ✅ Tokenizers (tiktoken-rs + HF tokenizers)
- ✅ Resilience: RetryingChatModel + RateLimitedChatModel + FallbackChatModel + TokenBudgetChatModel + PiiScrubbingChatModel + PromptCachingChatModel + CostCappedChatModel + **SelfConsistencyChatModel** (iter 140, Wang et al 2022)
- ✅ on_request hook
- ✅ ChatPromptTemplate + FewShot + MessagesPlaceholder + minijinja strict-undefined + SemanticSimilarityExampleSelector (iter 143) + LengthBasedExampleSelector (iter 144) + from_json/to_json/from_dict/to_dict (iter 150) + **extend/+/concat composition** (iter 152, layer base + role + task templates)
- ✅ Structured output via StructuredChatModel + schemars
- ✅ Tool trait + #[tool] proc-macro + JSON schema
- ✅ Built-in tools (~30): Calculator, HttpRequest, ReadFile/WriteFile/ListDirectory, Shell, SqliteQuery, BraveSearch/Tavily/DuckDuckGo, TavilyExtract (iter 141), Mcp, Whisper, Dalle, Tts, CachedTool, PythonRepl, Webhook, GmailSend (iter 148), WebFetch (iter 151), **TimeoutTool + RetryTool** (iter 159, tool resilience wrappers), **CurrentTimeTool** (iter 279, date-aware reasoning), **RegexExtractTool** (iter 280, structured-data extraction from unstructured text), **JsonExtractTool** (iter 281, JSONPath-lite for API responses), **UrlParseTool** (iter 282, URL components + query params), **HashTool** (iter 283, blake3/sha256/sha512/md5 fingerprinting), **Base64Tool** (iter 284, encode/decode w/ standard + url_safe variants), **UuidTool** (iter 285, v4/v7 ID generation w/ hyphenated/simple/urn formats), **TextDiffTool** (iter 286, line-level diff for code-change/audit workflows), **JwtDecodeTool** (iter 295, OAuth/auth-flow JWT inspection — header+payload, no signature verify)
- ✅ Agents: ReactAgent (tool-calling, parallel) + SupervisorAgent + TextReActAgent (text-mode for non-tool-calling models, with streaming) + **PlanAndExecuteAgent** (iter 155, two-phase plan→execute)
- ✅ Vector stores: memory, hnsw, pgvector, chroma, qdrant, weaviate
- ✅ Retrievers: Vector, BM25, Hybrid (RRF), Reranking, ParentDocument, MultiQuery, ContextualCompression, SelfQuery, TimeWeighted, HyDE, **MaxMarginalRelevance** (iter 157, diversity-aware over-fetch+select), **Ensemble** (iter 181, weighted RRF with per-child weights, concurrent fan-out)
- ✅ Document transformers: MMR, EmbeddingRedundantFilter, LongContextReorder
- ✅ Loaders (24): text, markdown, json, jsonl, csv, html, pdf, docx, directory, web, notion, slack, confluence, github-issues, github-files, gmail, gdrive, linear, jira, s3, jupyter (iter 146), gitlab-issues (iter 149), gitlab-files (iter 154), **sitemap** (iter 160, crawl docs sites)
- ✅ Splitters: recursive char (with language separators), markdown header, html header, json, semantic (embedding-based), CodeSplitter (definition-boundary, iter 142), **TokenTextSplitter** (iter 158, exact-token-count via tiktoken/HF)
- ✅ Parallel ingestion (Rayon)
- ✅ StateGraph + reducers macro + Send fan-out + Kahn scheduler
- ✅ Checkpointers: memory, sqlite, postgres, redis
- ✅ Streaming events: values / updates / messages / custom
- ✅ Memory: BufferMemory, TokenBufferMemory, SummaryBufferMemory (iter 137), **VectorStoreMemory** (iter 156, topic-relevance retrieval), summarize_conversation, SqliteChatHistory, **PostgresChatHistory** (iter 162, distributed durable)
- ✅ Caching: memory, sqlite, **redis** (iter 161, distributed cross-process), embedding, semantic
- ✅ Observability: CostTracker + GraphEvent + AgentEvent + tracing spans
- ✅ PyO3 abi3 wheels (cp39+) via maturin
- ✅ Output parsers: JSON (StructuredChatModel), XML (flat + nested), comma-list, numbered-list, markdown-list, boolean, ReAct text-mode, **markdown-table** (iter 153)
- ✅ format_instructions helpers (paired with each parser)
- ✅ String evaluators (10): exact_match, levenshtein_ratio, jaccard_similarity, regex_match, json_validity, embedding_cosine, contains_all/any
- ✅ **Eval harness** (iter 145) — `run_eval(cases, target, scorers, max_parallel)` golden-dataset runner with bounded concurrency, per-case + aggregate report, pluggable scorers (exact_match/jaccard/levenshtein/contains_all/regex/**llm_judge** iter 147)
- ✅ **Synthetic eval-set generation** (iter 163) — `synthesize_eval_cases(model, seeds, target_count, criteria)` expands hand-written seeds into a larger dataset via structured-output LLM call; dedups against seeds (case-insensitive + trim), caps at target, drops empty inputs. Python: `litgraph.evaluators.synthesize_eval_cases(seeds, model, target_count, criteria=None)`.
- ✅ MCP support (client + server; server spec-complete with **resources + prompts** iter 139)
- ✅ Modality matrix complete: text in/out, image in/out, audio in/out
- ✅ Human-in-the-loop: interrupt + Command(resume/goto)

### Left — v1.0 must-haves
- ✅ **OutputFixingParser** (iter 119) — `fix_with_llm`, `parse_with_retry`, Python `parse_json_with_retry`. LangChain parity.
- ✅ **Time travel + state history API** (iter 120) — `state_history`, `rewind_to`, `fork_at`, `clear_thread` on Checkpointer + PyCompiledGraph. Native DELETE paths on sqlite/pg/redis. Scheduler serialization swapped bincode→rmp-serde (fixed pre-existing PyCompiledGraph.resume() bug for Value-state graphs).
- ✅ **OpenTelemetry OTLP exporter** (iter 121) — new crate `litgraph-tracing-otel`. `init_otlp(endpoint, service_name)` / `init_stdout()` / in-memory for tests. Batch span processor. Env var fallbacks. Drop-guard. Python `litgraph.tracing.{init_otlp, init_stdout, shutdown}`.
- ❌ **`pyo3-stub-gen` `.pyi` generation** — line 157. Every Pyright import warning is from missing stubs.
- ✅ **Streaming JSON parser** (iter 122) — `parse_partial_json` + `repair_partial_json`. Auto-closes unclosed braces/quotes/brackets. Monotonic-growth invariant. Powers progressive structured-output UIs.

### Left — local inference (lines 173, 219-220)
- ✅ **fastembed-rs** local embeddings (no-network) — iter 177
- ❌ **candle / mistral.rs** local chat (in-process small models)
- ✅ **ort** ONNX runtime (local cross-encoder rerankers) — iter 179 via `litgraph-rerankers-fastembed`

### Left — v1.1 nice-to-haves
- ✅ LangSmith OTel compat shim (iter 127) — `init_langsmith(api_key, project_name)` + generic `init_otlp_http(endpoint, service_name, headers)`. Traces flow to LangSmith UI with zero re-plumbing.
- ✅ Webhook / generic notifier tool (iter 123) — `WebhookTool` with Slack/Discord/generic presets. URL hard-coded (not agent-controllable). Python `litgraph.tools.WebhookTool(url, preset, ...)`.
- ✅ `LinearIssuesLoader` (iter 124) — GraphQL-backed. First of its kind in the loader stack.
- ✅ `JiraIssuesLoader` (iter 125) — REST v3 + JQL. Cloud (Basic: email+token) or Data Center (Bearer PAT). ADF description→text walker.
- ✅ `S3Loader` (iter 126) — SigV4 (reused from bedrock). list-objects-v2 + get-object. Prefix/ext/exclude/size filters. Works with MinIO/R2/B2 via `base_url`. **Loader matrix complete — 20 sources.**
- 🚫 LanceDB / Pinecone vector stores — heavy deps (arrow-rs, datafusion); deferred per memory
- 🚫 LangChain `Callbacks` API parity — wide surface; CostTracker + events already cover use cases
- 🚫 Streaming tool execution — requires Tool trait extension; deferred

### Iteration log (recent — last 30)
- 107 ReAct parser · 108 format_instructions · 109 TextReActAgent · 110 TextReActAgent.stream() · 111 string evaluators · 112 doc transformers · 113 FallbackChatModel · 114 Whisper · 115 Dalle · 116 Tts · 117 CachedTool · 118 PythonReplTool · 119 OutputFixingParser · 120 time travel + state history · 121 litgraph-tracing-otel · 122 partial JSON · 123 WebhookTool · 124 LinearIssuesLoader · 125 JiraIssuesLoader + ADF walker · 126 S3Loader · 127 LangSmith OTel shim · 128 LlmJudge · 129 PII scrubber · 130 TokenBudgetChatModel · 131 MCP server · 132 HydeRetriever · 133 FallbackEmbeddings · 134 Retrying + RateLimited Embeddings · 135 PiiScrubbingChatModel · 136 PromptCachingChatModel · 137 SummaryBufferMemory · 138 CostCappedChatModel · 139 MCP resources + prompts · 140 SelfConsistencyChatModel · 141 TavilyExtract tool · 142 CodeSplitter · 143 SemanticSimilarityExampleSelector · 144 LengthBasedExampleSelector · 145 EvalHarness · 146 JupyterNotebookLoader · 147 LlmJudgeScorer · 148 GmailSendTool · 149 GitLabIssuesLoader · 150 ChatPromptTemplate file-loading · 151 WebFetchTool · 152 prompt composition · 153 MarkdownTableParser · 154 GitLabFilesLoader · 155 PlanAndExecuteAgent · 156 VectorStoreMemory · 157 MaxMarginalRelevanceRetriever · 158 TokenTextSplitter · 159 TimeoutTool + RetryTool · 160 SitemapLoader · 161 RedisCache · 162 PostgresChatHistory · 163 SyntheticEvalSetGenerator · 164 RedisChatHistory · 165 ArxivLoader + WikipediaLoader · 166 PubMedLoader · 167 OffloadingTool + OffloadBackend · 168 DatasetVersioning + RunStore · 169 PromptHub (Filesystem + Http + Caching) · 170 litgraph-serve (axum REST + SSE) · 171 EpisodicMemory + MemoryExtractor · 172 PostgresStore · 173 AssistantManager (versioned config snapshots) · 174 Table + TableQuery (Pandas-parser parity) · 175 YouTubeTranscriptLoader · 176 DiscordChannelLoader · 177 FastembedEmbeddings (local ONNX) · 178 OutlookMessagesLoader · 179 FastembedReranker (local ONNX cross-encoder) · 180 Studio debug router (litgraph-serve studio feature) · 181 EnsembleRetriever (weighted RRF, concurrent fan-out) · 182 batch_concurrent (bounded-concurrency ChatModel batch, order-preserved) · 183 embed_documents_concurrent (chunked parallel embedder) · 184 RaceChatModel (concurrent invoke, first-success-wins, abort losers) · 185 SemanticStore (Rayon cosine search over any Store) · 186 EnsembleReranker (concurrent reranker fusion via weighted RRF) · 187 load_concurrent (bounded-concurrency multi-loader fan-out via spawn_blocking) · 188 MultiVectorRetriever (N perspectives per parent, indexed via embed_documents_concurrent) · 189 multiplex_chat_streams (mpsc channel-fan-in across N model streams) · 190 retrieve_concurrent (Semaphore-bounded fan-out of one retriever over many queries) · 191 tool_dispatch_concurrent (heterogeneous parallel tool calls outside the React loop) · 192 RaceEmbeddings (concurrent N-provider race, abort losers, latency-min embed) · 193 RaceRetriever (race retrievers for latency-min hits, completes the race trio) · 194 TimeoutChatModel + TimeoutEmbeddings (per-call deadline via tokio::time::timeout) · 195 broadcast_chat_stream (tokio::sync::broadcast 1→N stream fan-out) · 196 ingest_to_stream (multi-stage backpressured load→split→embed pipeline) · 197 rerank_concurrent (one reranker, N (query,candidates) pairs in parallel) · 198 Rayon-parallel BM25 index build (tokenize + count per doc in parallel, merge under lock) · 199 Progress<T> (tokio::sync::watch latest-value observability primitive) · 200 ingest_to_stream_with_progress (composes iter 196 + iter 199 — observable pipeline counters) · 201 ResumeRegistry (tokio::sync::oneshot coordination — interrupt-resume foundation) · 202 axum webhook-resume bridge for litgraph-serve (POST/DELETE/GET resume endpoints) · 203 Rayon-parallel mmr_select (per-candidate scoring loop, deterministic tie-break) · 204 Rayon-parallel embedding_redundant_filter (par_iter::any short-circuits across cores) · 205 batch_concurrent_with_progress (composes iter 182 + iter 199 — live ChatModel batch counters) · 206 embed_documents_concurrent_with_progress (composes iter 183 + iter 199 — live bulk-embedding counters) · 207 retrieve_concurrent_with_progress (composes iter 190 + iter 199 — live multi-query eval counters) · 208 tool_dispatch_concurrent_with_progress (composes iter 191 + iter 199 — unknown-tool errors bucketed) · 209 rerank_concurrent_with_progress (closes the progress-aware family across all 6 axes) · 210 batch_concurrent_stream (mpsc-backed streaming variant — yield (idx, Result) as each completes, abort-on-drop) · 211 embed_documents_concurrent_stream (streaming-variant pattern extended to embeddings axis) · 212 retrieve_concurrent_stream (streaming-variant extended to retriever axis) · 213 tool_dispatch_concurrent_stream (streaming-variant extended to tool axis) · 214 rerank_concurrent_stream (streaming-variant extended to reranker axis — 5 of 6 axes covered) · 215 load_concurrent_stream (streaming variant for the loader axis — 6 distinct primitives now stream) · 216 batch_concurrent_stream_with_progress (first composition: stream items + Progress watcher in one call) · 217 embed_documents_concurrent_stream_with_progress (combined consumer shape extended to embed axis) · 218 retrieve_concurrent_stream_with_progress (combined consumer shape extended to retriever axis) · 219 tool_dispatch_concurrent_stream_with_progress (combined consumer shape extended to tool axis) · 220 rerank_concurrent_stream_with_progress (combined consumer shape extended to rerank axis — 5/6 axes) · 221 load_concurrent_with_progress + load_concurrent_stream_with_progress (loader axis, four-quadrant matrix complete across all 6 axes) · 222 SemanticStore::bulk_put (LangGraph BaseStore::mset parity, composes iter 183) · 223 SemanticStore::bulk_delete (LangGraph BaseStore::mdelete parity, retention sweeps) · 224 SemanticStore::bulk_get (LangGraph BaseStore::mget parity — closes the full bulk trio) · 225 ShutdownSignal (tokio::sync::Notify N-waiter edge signal — fifth channel shape) · 226 until_shutdown future combinator (composable graceful-cancel for any await call) · 227 batch_concurrent_with_shutdown (preserves partial progress on Ctrl+C — first parallel-batch ↔ coordination bridge) · 228 embed_documents_concurrent_with_shutdown (partial-progress preservation extended to embed axis) · 229 retrieve_concurrent_with_shutdown (partial-progress preservation extended to retriever axis — 3/6 axes bridged) · 230 tool_dispatch_concurrent_with_shutdown (extended to tool axis — 4/6 axes bridged) · 231 rerank_concurrent_with_shutdown (partial-progress preservation extended to rerank axis — 5/6 axes bridged; loader axis remains) · 232 load_concurrent_with_shutdown (closes the bridge family — all 6 parallel-batch axes now interop with ShutdownSignal) · 233 batch_concurrent_stream_with_shutdown (opens the second bridge family: streaming + coordination — producer-side graceful end-of-interest, one signal stops many streams) · 234 embed_documents_concurrent_stream_with_shutdown (stream-coordination bridge extended to embed axis — 2/6 stream axes bridged) · 235 retrieve_concurrent_stream_with_shutdown (stream-coordination bridge extended to retriever axis — 3/6 stream axes bridged) · 236 tool_dispatch_concurrent_stream_with_shutdown (stream-coordination bridge extended to tool axis — 4/6 stream axes bridged) · 237 rerank_concurrent_stream_with_shutdown (stream-coordination bridge extended to rerank axis — 5/6 stream axes bridged; only loader remains) · 238 load_concurrent_stream_with_shutdown (closes the stream+coord bridge family — every axis now ships BOTH Vec+shutdown AND stream+shutdown variants) · 239 Barrier (wait-for-N rendezvous primitive — sixth channel shape; shutdown-aware variant returns None if signal fires before threshold) · 240 CountDownLatch (decoupled producer/observer coordination — count_down/wait split by role; pairs with Barrier as the asymmetric-role variant) · 241 KeyedMutex (per-key async serialization — same key serializes, different keys parallel; Weak-based bounded memory, cleanup() for ephemeral keys) · 242 RateLimiter (async token-bucket primitive — lazy refill, burst-up-to-capacity, shutdown-aware acquire that preserves budget on cancel) · 243 CircuitBreaker (three-state resilience primitive — Closed/Open/HalfOpenProbing with consecutive-failure threshold + cooldown; composable wrap-any-future API) · 244 CircuitBreakerChatModel (bridges iter-243 primitive into the resilience family — fail-fast wrap of any ChatModel; streams short-circuit at handshake) · 245 CircuitBreakerEmbeddings (embed-axis mirror — one shared breaker covers embed_query AND embed_documents) · 246 SharedRateLimited{Chat,Embeddings} (bridges iter-242 RateLimiter into resilience family — one budget shared across many wrappers, including across chat/embed axes) · 247 KeyedSerializedChatModel (bridges iter-241 KeyedMutex into chat family — per-thread ReAct step lock, different threads parallel) · 248 Bulkhead (concurrent-call cap with rejection — try_enter / enter / enter_with_timeout, rejected_count telemetry; classic Release-It pattern) · 249 BulkheadChatModel + BulkheadEmbeddings (bridge iter-248 into resilience family; Reject vs WaitUpTo modes; bulkhead-full → Error::RateLimited so retry/fallback chains compose) · 250 hedged_call (tail-latency mitigation combinator — run primary; after hedge_delay also run backup; race them; loser dropped on tokio cancellation) · 251 HedgedChatModel + HedgedEmbeddings (bridge iter-250 into resilience family — backup only fires on slow primary, zero overhead on fast path; streams primary-only) · 252 Singleflight<K,V> (request-coalescing primitive — N concurrent callers for same key share ONE compute; cache-miss thundering-herd mitigation) · 253 SingleflightEmbeddings (bridges iter-252 into embeddings — N identical embed_query calls share ONE HTTP request; embed_documents passes through) · 254 RecordingChatModel + ReplayingChatModel + Cassette (VCR-style record/replay for deterministic agent tests; blake3 over canonical JSON for hash key; serializable cassette format) · 255 Cassette::{load,save}_from_file + RecordingEmbeddings + ReplayingEmbeddings + EmbedCassette (closes the record/replay workflow — file IO + embed-axis parity) · 256 RecordingTool + ReplayingTool + ToolCassette (third record/replay axis — agent integration tests with deterministic tool side effects) · 257 RssAtomLoader (RSS 2.0 / Atom 1.0 unified loader; feature-gated `rss`; quick-xml pull parser) · 258 HackerNewsLoader (public HN Firebase API; 6 feed sources via HnFeed enum; Rayon-parallel item fetch) · 259 PriorityQueue<T> (async priority work queue with FIFO tie-breaking; try_pop/pop/pop_with_shutdown) · 260 MetricsRegistry + Counter/Gauge/Histogram (in-process metrics; atomic hot path; Prometheus text-format export) · 261 MetricsChatModel + MetricsEmbeddings (auto-instrumented wrappers — invocations/errors/in_flight/latency in 4 atomic ops per call) · 262 MetricsTool (third axis of metrics auto-instrumentation; default prefix = sanitized tool name) · 263 SingleflightTool (closes the request-coalescing matrix on the tool axis — N concurrent identical run calls share ONE invocation; idempotent-only) · 264 BitbucketIssuesLoader (third Git provider; Bitbucket Cloud API v2; BBQL filters; optional comment threads) · 265 BitbucketFilesLoader (recursive repo source fetcher; closes the {GitHub,GitLab,Bitbucket} × {Issues,Files} matrix) · 266 RagFusionRetriever (LLM query expansion + reciprocal-rank fusion; Cormack 2009 RRF + Raudaschl 2023) · 267 StepBackRetriever (Zheng et al. 2023 step-back prompting — abstract query for cross-abstraction-level recall) · 268 SubQueryRetriever (decompose compound queries into atomic sub-questions; round-robin interleave + dedup) · 269 MarkdownTableSplitter (preserve GFM markdown tables as atomic chunks; non-table prose via inner splitter) · 270 CsvRowSplitter (row-aware CSV chunking; RFC-4180 quoted-newline handling; header preserved per chunk) · 271 TimeoutRetriever (per-call deadline wrapper for any Retriever; mirrors TimeoutChatModel/TimeoutEmbeddings) · 272 RetryingRetriever (auto-retry transient retrieval errors with jittered exponential backoff; closes the retry-wrapper trio chat/embed/retriever) · 273 MetricsRetriever (closes the metrics-instrumentation matrix across chat/embed/tool/retrieve; tighter default histogram buckets for retrieval-typical latency) · 274 RecordingRetriever + ReplayingRetriever + RetrieverCassette (closes the record/replay matrix across chat/embed/tool/retrieve; full agent loop is now CI-deterministic) · 275 CircuitBreakerRetriever (vector-store breaker for fail-fast on outage; chat/embed/retrieve coverage) · 276 BulkheadRetriever (cap concurrent vector-store calls; Reject / WaitUpTo modes; Error::RateLimited so retry chains compose) · 277 HedgedRetriever (tail-latency mitigation; backup retriever fires only on slow primary; chat/embed/retrieve coverage) · 278 SingleflightRetriever (request-coalescing for hot queries; embed/tool/retrieve coverage) · 279 CurrentTimeTool (date-aware reasoning — ISO8601 timestamp + weekday + Unix + tz-offset; agents need to know "what day is it?") · 280 RegexExtractTool (apply regex to unstructured text; all/first/captures modes; universal data-extraction primitive) · 281 JsonExtractTool (JSONPath-lite for JSON values; $.users[0].name / $.users[*].email / $.results[-1] syntax) · 282 UrlParseTool (parse URLs; query params with repeated-key array collapse; OAuth callback / redirect-allowlist workflows) · 283 HashTool (blake3/sha256/sha512/md5; lowercase-hex digest; dedup / integrity / cache-key fingerprinting) · 284 Base64Tool (encode/decode w/ standard + url_safe variants; JWT-header-decode workflow) · 285 UuidTool (v4/v7 generation; v7 timestamp-ordered default for DB-locality-friendly primary keys) · 286 TextDiffTool (line-level diff; unified + structured output formats; code-change + audit workflows) · 287 SentenceSplitter (rule-based sentence-boundary splitter; abbreviation + acronym + decimal handling; closes the 🟡 sentence-splitter gap) · 288 JsonLinesSplitter (one chunk per NDJSON record; OpenAI-fine-tune-format compatible; pretty + skip-invalid knobs) · 289 detect_drift + DriftReport + CaseDrift (eval drift detector — per-case regressions/improvements/stable-failures + aggregate deltas; CI-gateable via has_regressions()) · 290 mcnemar_test + McNemarResult (statistical significance for paired binary eval outcomes; chi-squared + p-value + CI-gate flag) · 291 SemanticCachedRetriever (semantic-similarity-keyed retrieval cache; FAQ-style "phrased-differently" hits; threshold + LRU + optional TTL) · 292 CachedEmbeddings (exact-match TTL cache for embed_query + embed_documents; LRU cap; composes with SingleflightEmbeddings as cache-outside / dedup-inside) · 293 HtmlSectionSplitter (semantic-block HTML splitter — article/section/main/aside; depth-tracking for nested tags; case-insensitive) · 294 StackExchangeLoader (api.stackexchange.com Q&A loader; site + tags filter; optional accepted-answer concat; Rayon-parallel batch answers fetch) · 295 JwtDecodeTool (decode JWT header+payload via URL_SAFE_NO_PAD; surfaces `signature_present` + `expired` (RFC 7519 §4.1.4); signature verify intentionally out-of-scope — belongs in auth layer not agent; closes JWT inspection one-shot vs prior 3-call chain) · 296 CachedChatModel (live LLM-response cache for prod cost reduction; LRU + optional TTL; reuses cassette `exchange_hash` so cache & cassette agree on request equality; distinct from `RecordingChatModel` (test workflow) and `PromptCachingChatModel` (Anthropic server-side prompt cache) — the three compose) · 297 wilcoxon_signed_rank_test (non-parametric paired-continuous significance test for cosine/BLEU/LLM-judge floats; average-rank ties + tie-corrected σ + continuity correction; pairs with iter-290 McNemar — binary vs continuous closes the eval-significance pair) · 298 paired_effect_size (Cohen's d_z + Hedges' g for paired continuous outcomes; magnitude bands negligible/small/medium/large per Cohen 1988; uniform-shift std=0 surfaces as NaN+"undefined" not zero — matters because d is mean-relative-to-spread; pairs with Wilcoxon — significance + effect-size = full decision-making toolkit) · 299 bootstrap_eval_ci + bootstrap_mean_ci + ConfidenceInterval (non-parametric percentile-bootstrap CI for any scorer mean; xorshift64 inline PRNG — no `rand` dep; seed-based reproducibility; works on any distribution including skewed/bounded eval scores where t-CI fails) · 300 MapReduceSummarizer (chunk → map (parallel per-chunk summary, semaphore-bounded) → reduce (combine); recursive-reduce mode for inputs too large for a single reduce pass; termination guarantee — only recurse when summaries.len() > 2 so grouping actually shrinks input; classic chain pattern, useful for book-scale docs / multi-hour transcripts where stuff-documents would overflow context)

---


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

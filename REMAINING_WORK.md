# What's left in litGraph — single-pane view

**Snapshot:** 2026-08-22 · implementation audit against the current tree.

Items marked shipped below were verified in code during this pass. The
remaining provider-key and service-gated rows are validation work, not local
framework implementation gaps.

This file is the consolidated "what's left" picture as of today. It
points at the deeper docs rather than duplicating them — read the
linked sources for prioritisation rationale.

- `ROADMAP.md` — research-backed tier-1/tier-2 list, *what's next*
- `MISSING_FEATURES.md` — short actionable view of gaps + won't-do
- `INTEGRATION_TESTS.md` — what's tested live, what's blocked, why
- `FEATURES.md` — what's already done

---

## TL;DR — five buckets

1. **Live-test blockers** — generic OpenAI-compatible tests run locally with
   Ollama; only protocol-specific or external-service tests remain gated.
2. **Provider-key gated** — exists in code, can't run without that
   provider's API credential.
3. **Service-gated** — exists in code, needs a running DB / vector
   store / OTel collector / MCP server / etc.
4. **Real feature gaps** — would close parity vs. LangChain/LangGraph
   or unlock new capability. Tier-1 / Tier-2 lists in ROADMAP.md.
5. **Won't do (deliberate)** — recorded so we don't relitigate.

---

## 1. Live-test blockers

The formerly DeepSeek-blocked cases now run against Ollama; see
`INTEGRATION_TESTS.md` for the endpoint and capability flags.

| Skip | Root cause | Unblocks when… |
|---|---|---|
| `LlmJudge` (2 cases) | ✅ Live-tested with Ollama schema mode. | Set `LITGRAPH_TEST_BASE_URL` and `LITGRAPH_TEST_MODEL`. |
| `synthesize_eval_cases` (2 cases) | ✅ Live-tested with Ollama schema mode. | Same Ollama configuration. |
| `BigToolAgent` (1 case) | ✅ Live-tested with Ollama plus `nomic-embed-text-v2-moe`. | Set `LITGRAPH_TEST_EMBED_MODEL`. |
| `NamespacedMemory` (1 case) | ✅ Unblocked: append-only native memories now use a shared, thread-safe per-namespace sidecar while still writing through to the native backend. | Covered offline against native `BufferMemory`; live external backends remain service-gated. |

---

## 2. Provider-key gated (works in code, no live coverage)

Tests live in `python_tests/integration/` but skip cleanly without the
key. Set the env var to enable.

| Provider | Env var | Adds tests for |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `OpenAIChat` direct, `OpenAIEmbeddings`, `OpenAIResponses` API, DALL·E, Whisper, TTS |
| Anthropic | `ANTHROPIC_API_KEY` | `AnthropicChat`, **thinking blocks** (DeepSeek doesn't emit them), **prompt caching** (`cache_control` field is provider-specific) |
| Cohere | `COHERE_API_KEY` | `CohereChat`, `CohereEmbeddings`, `CohereReranker` |
| Voyage | `VOYAGE_API_KEY` | `VoyageEmbeddings`, `VoyageReranker` |
| Jina | `JINA_API_KEY` | `JinaEmbeddings`, `JinaReranker` |
| Tavily | `TAVILY_API_KEY` | `TavilySearchTool`, `TavilyExtractTool` |
| Brave Search | `BRAVE_API_KEY` | `BraveSearchTool` |
| Gemini AI Studio | `GOOGLE_API_KEY` | `GeminiChat`, `GeminiEmbeddings` |
| **Gemini Vertex** | `GOOGLE_APPLICATION_CREDENTIALS` (Service Account JSON) | Vertex auth path — different request shape |
| **AWS Bedrock** | AWS standard chain (env / shared creds / IMDS) | `BedrockChat` (native + Converse API), Bedrock embeddings — SigV4 signing |

Notable functionality that becomes testable with these keys:
- **Embeddings live** (no DeepSeek model exposes embeddings)
- **`evaluators.embedding_cosine`** (needs embeddings)
- **`PairwiseEvaluator`** (uses StructuredChatModel like LlmJudge)
- **Image generation** (DALL·E)
- **Audio I/O** (Whisper transcribe, TTS)
- **OpenAI Responses API** (`/responses` endpoint, not `/chat/completions`)

---

## 3. Service-gated (needs a running backend)

Same shape — code exists; tests skip without the service.

| Subsystem | Needs | Notes |
|---|---|---|
| Vector stores live | running Qdrant / pgvector / Chroma / Weaviate / Milvus / Redis-search / Neo4j | Out of scope for a single-key run; would need a compose-up integration suite |
| Postgres / SQLite checkpointers live | a DB | Mock-state unit tests cover shape; live tests would catch driver/migration regressions |
| MCP server live | a running MCP endpoint | Today covered by an in-process fake server in `python_tests/test_mcp_*.py` |
| `litgraph-serve` HTTP | spawn the binary | Rust integration tests in `crates/litgraph-serve/tests/` cover this; no Python live test |
| OTel exporter live | OTLP collector (e.g. Jaeger, Honeycomb) | Tests today verify span shape via in-process exporter |
| Memory backends extras | Cassandra / DynamoDB / Mongo / Redis live | NamespacedMemory + Cassandra/DynamoDB/Mongo memory classes exist in `memory_extras` |
| Loader extras | Github / Slack / Notion / Confluence / GDrive / Linear / Jira / S3 / Airtable / HF Datasets / IMAP / Outlook / Reddit / Sitemap | Tokens + endpoints required |

---

## 4. Real feature gaps (open work, ranked)

### Tier-1 — ship in the next ~10 iters (per ROADMAP.md)

These were called out *before* iter 376; some have shipped since.
Cross-checked against current state.

| # | Item | Status |
|---|---|---|
| 1 | **EnsembleRetriever** — weighted RRF | ✅ shipped iter 181 |
| 2 | **`before_tool` / `after_tool` middleware** | ✅ shipped (iter 348-350 Rust + iter 376 Python adapter via `HookedTool.to_function_tool`) |
| 3 | **Vector-indexed semantic search on `Store`** | ✅ shipped iter 185 |
| 4 | **Postgres `Store` vector-index wiring** | ✅ |
| 5 | **Functional API: `@entrypoint` + `@task`** | ✅ shipped (live-tested in `test_functional_api.py`, `test_workflow_*.py`) |
| 6 | **`pyo3-stub-gen` auto `.pyi`** | ❌ still hand-rolled `litgraph-stubs/` + drift checker |
| 7 | **Pydantic-coerced state + `StreamPart`** | ✅ shipped iter 378 + 379. `StateGraph(state_schema=Pydantic\|dataclass\|TypedDict)` auto-dumps input and coerces invoke/resume output. Typed dataclass variants and the event-kind enum cover both payload narrowing and stable event names. |
| 8 | **Local chat model — `candle` / `mistral.rs`** | 🚫 deferred — iter 380 ships `MistralRsChat` adapter + `ModelBackend` trait + `MockModelBackend` scaffold (dormant; `engine` feature off by default → zero workspace build cost). Real `mistralrs::Engine` wiring (~3-4 iters: model loader, tokenizer, sampling loop, KV cache, streaming callback) deferred indefinitely. Users wanting local chat today: run `mistralrs-server` (Docker / native) + point `OpenAIChat(base_url=…)` at it — same path as Ollama. |
| 9 | **Webhook-resume bridge for interrupts** | ✅ shipped iter 201/202 |
| 10 | **Pregel-style super-step parallel exec audit** | ✅ partial — iter 377 ships `StateGraph::add_blocking_node` + `add_fallible_blocking_node` (spawn_blocking escape hatch for CPU-bound nodes — local-model forward pass, heavy tokenize, PDF rasterize). Explicit super-step contract (Pregel-style barrier API) still pending. |

**Net Tier-1 remaining:** none actionable. Item 6 (auto-stubs)
skipped — annotation churn vs marginal gain over `check_stubs.py`.
Item 7 fully closed iters 378 + 379. Item 8 (local model)
deferred — see row above. Item 10 partially closed iter 377
(super-step contract still open but escape hatch shipped).

### Tier-2 / Missing-feature gaps (per MISSING_FEATURES.md)

#### Agent / tool ergonomics
- ✅ **Streaming tool execution alternative** — `OffloadingTool` +
  `OffloadBackend` provide the result-poll pattern for long-running jobs
- ✅ **Tool-call budget caps** — `ToolBudgetMiddleware` in
  Rust and PyO3 caps calls per agent turn via `before_tool` denial.

#### Graph
- ✅ Branch fan-in **dedup-by-key reducers** — explicit
  `merge_dedup_by_key` and deterministic reducer factory `dedup_by_key`
- ✅ **Parallel-for shorthands** — fixed-count `add_parallel_for` and
  state-driven `parallel_for`

#### Memory / store
- ✅ **NamespacedMemory** works with metadata-aware and append-only native
  backends

#### Eval & reproducibility
- ✅ **Eval-suite live smoke test** — `LlmJudge` and synthetic eval generation
  passed against Ollama schema mode on 2026-08-22
- ❌ **Trajectory replay CLI** — take an OTel trace ID and replay the
  exact prompt against a chosen model

#### Serve
- ⏳ **Studio UI parity for local graphs** — `studio` feature flag in
  `litgraph-serve` covers cloud API surface only
- ✅ **`recipes.serve` actually spawns the binary** (iter 384) — new
  PyO3 module `litgraph.serve.spawn_chat(model, host, port)` binds
  the axum listener synchronously (port-in-use → `OSError`) then
  spawns the server on the shared tokio runtime. Returns a
  `ServeHandle` (`.address()`, `.url()`, `.model()`, `.shutdown()`
  — idempotent + graceful). `recipes.serve(model)` calls into it;
  `recipes.serve(graph)` still raises with a clear deferral message
  (graph-shaped serving is a separate scope item).

#### Python ergonomics
- ✅ **Implicit Pydantic coercion** on `StateGraph(state_schema=)` (iter 378)
- ✅ **`StreamPart` typed enum** mirroring `ChatStreamEvent` (iter 379)

#### Observability
- ✅ **Trace exemplars** linking OTel span → prompt+completion excerpt
  (iter 385) — `InstrumentedChatModel` opens a `chat.invoke` /
  `chat.stream` span and records `prompt_excerpt` +
  `completion_excerpt` attrs (truncated to 512 B / UTF-8 boundary,
  control chars collapsed, configurable via
  `LITGRAPH_EXEMPLAR_BYTES`). Standalone helpers in
  `litgraph-tracing-otel::exemplars` remain for callers wanting to
  attach exemplars to other spans.
- ❌ **"Turn replay" CLI** from OTel trace ID

### Loaders / splitters / stores nice-to-haves

- ❌ WhatsApp Business loader (paperwork-heavy)
- ❌ IMAP / Outlook loaders (most teams ETL out of inbox)
- ❌ Audio/video transcription loader (route through `WhisperTool`)
- ❌ NLTK / SpaCy sentence splitters (covered by recursive-char +
  token splitters; only if a user files a concrete miss)
- ❌ Regex-driven splitter for log files / structured text
- ❌ LanceDB / Pinecone backends (low marginal value vs. 6 existing)
- ❌ "Blackhole" store for pipeline benchmarking

### CLI / DX

- ✅ `litgraph init <template>` repo scaffold (already shipped — 3 templates: chat-agent, rag, eval-suite — `python/litgraph/_init.py`)
- ✅ `litgraph trace` viewer (OTel JSON → graph timeline in terminal)
  (iter 386) — pure-Python, stdlib-only. Accepts SDK stdout shape
  AND OTLP JSON envelope (`resourceSpans`/`scopeSpans`/`spans`),
  harness JSONL, JSON, or JSONL. Supports normalized `--json` output
  and `--limit`. ANSI-coloured + indented by parent; surfaces
  `prompt_excerpt`, `completion_excerpt`, `model`, `error`. Falls
  back to plain text when stdout isn't a TTY.

### Docs

- ✅ "Migrate from LangChain" guide (iter 387 — `MIGRATION_LANGCHAIN.md`,
  15 idiom side-by-sides covering chat, streaming, structured output,
  embeddings/vector stores, RAG, tool-calling agents, agent events,
  memory, StateGraph, parallel fan-out, conditional edges,
  interrupts, checkpointers, time travel, HTTP serve)
- 🟡 Per-crate READMEs: `litgraph-gateway` now has a canonical quickstart;
  the remaining adapter crates still rely on workspace-level guides

### Performance / build

- ❌ Criterion compare bot in CI (>5% regression flag on
  graph-fanout / BM25 / HNSW micro-benches)
- ❌ Free-threaded Python 3.13t wheel published (build path exists
  via `--features no-gil`; not in the published artifact yet)

---

## 5. Won't do (deliberate, per MISSING_FEATURES.md)

Recorded so we don't relitigate:

- **LangChain Callbacks parity** — surface area is enormous; the
  callback bus + `CostTracker` + `GraphEvent` cover the concrete asks.
- **Zapier / N8N tools** — userland integration; out of framework
  scope.
- **Video-in modality** as framework code — provider-side problem.
- **Per-class deprecated LangChain chains** (`LLMChain`,
  `SequentialChain`, `MultiPromptChain`, …) — anti-thesis of the
  project.

---

## 6. Discovered during integration testing (this session)

Issues surfaced by the live-test pass and either fixed in-place or
documented:

### Fixed
- **iter 354 — `recipes.summarize._content_of`** read `content` but
  native providers return `text`. Now tolerates both keys.
- **iter 360 — `BroadcastHandle.subscribe` panic from sync Python.**
  Wrapped the call in `runtime::rt().enter()` so the lazy
  `tokio::spawn` lands on the bridge runtime. Activated iter 376
  after a `maturin develop` rebuild.
- **iter 366 — `recipes.serve` attr-check mismatch.** Looked for
  `graph_id`/`compile`; `CompiledGraph` has `invoke`/`stream`. Now
  accepts both shapes.
- **iter 373 — `SwarmAgent.invoke` contract bug.** Was passing a
  list-of-messages to `agent.invoke()`, but native `ReactAgent.invoke`
  takes a string. Now detects native agent classes and extracts the
  latest user content.
- **iter 376 — `HookedTool` ↔ ReactAgent adapter.** Native ReactAgent
  rejects Python tool wrappers via `extract_tools`. Added
  `HookedTool.to_function_tool(callable, schema=, description=)`
  which builds a fresh native `FunctionTool` whose body fires the
  Python before/after/budget hooks.

### Documented as gotchas (not bugs — counter-intuitive APIs)
- `OpenAIChat.invoke()` accepts only `temperature`, `max_tokens`,
  `response_format` — no `stop`, `top_p`, `seed`, `presence_penalty`.
- DeepSeek's `response_format=json_object` requires the literal string
  `"json"` in the prompt; `json_schema` is unsupported.
- `MiddlewareChat` does NOT expose `.invoke()` on the Python surface
  — drive through `ReactAgent` / `SupervisorAgent`.
- `StateGraph.compile()` is single-shot per builder; the resulting
  `CompiledGraph` is reusable.
- `streaming.stream_events` only emits lifecycle events
  (`on_chat_model_start`/`_end`); delta tokens stay on the underlying
  `model.stream(...)` iterator.
- `coerce_stream` is async-only; wrap `model.stream(...)` (sync) with
  a one-line `async def to_async(it): for x in it: yield x`.
- `prompt_hub.Prompt.render(**vars)` uses Python `str.format`
  (single-brace `{var}`); use `ChatPromptTemplate` for minijinja
  `{{ var }}`.
- Per-call wrappers expose under `*Chat`, NOT `*ChatModel`
  (`TokenBudgetChat`, `CostCappedChat`, `PiiScrubbingChat`, …).
- `@entrypoint()` requires parens; wraps `async def`.
- `TokenBufferMemory(max_tokens, counter)` requires a counter
  callable (no default).
- `ReactAgent.invoke(user)` takes a string, NOT a messages list.

---

## How to advance an item from this list

1. **Tier-1 / -2 features** — implement the change, add a `iter N`
   line to the commit, cross it off in `ROADMAP.md` /
   `MISSING_FEATURES.md`.
2. **Provider-gated tests** — set the env var, drop the skip-guard,
   add a row to `INTEGRATION_TESTS.md` "Conditionally testable".
3. **Service-gated tests** — start the service in `docker-compose`,
   add a `pytest.skipif(not <reachable>())` guard, document the
   compose snippet in the test docstring.
4. **Won't-do items** — file an issue with (a) a concrete agent
   author hit, (b) the prioritisation-rubric scores from
   `ROADMAP.md`. Without both, it stays where it is.

---

## What this doc is NOT

- It's not a TODO list — items don't have owners or due dates.
- It's not a release checklist — see `RELEASING.md` for that.
- It's not exhaustive — long-tail "we could also" items live in
  per-crate issues; this is the framework-level view.

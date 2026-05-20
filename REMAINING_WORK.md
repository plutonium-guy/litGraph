# What's left in litGraph — single-pane view

**Snapshot:** 2026-05-07 · post-iter-376 · 132 live integration tests
passing / 6 cleanly skipped.

This file is the consolidated "what's left" picture as of today. It
points at the deeper docs rather than duplicating them — read the
linked sources for prioritisation rationale.

- `ROADMAP.md` — research-backed tier-1/tier-2 list, *what's next*
- `MISSING_FEATURES.md` — short actionable view of gaps + won't-do
- `INTEGRATION_TESTS.md` — what's tested live, what's blocked, why
- `FEATURES.md` — what's already done

---

## TL;DR — five buckets

1. **Live-test blockers** — tests we can't run today against DeepSeek
   alone (6 cases). All have documented reasons.
2. **Provider-key gated** — exists in code, can't run without that
   provider's API credential.
3. **Service-gated** — exists in code, needs a running DB / vector
   store / OTel collector / MCP server / etc.
4. **Real feature gaps** — would close parity vs. LangChain/LangGraph
   or unlock new capability. Tier-1 / Tier-2 lists in ROADMAP.md.
5. **Won't do (deliberate)** — recorded so we don't relitigate.

---

## 1. Live-test blockers (6 skipped tests)

All against DeepSeek; documented in `INTEGRATION_TESTS.md` → Blocked.

| Skip | Root cause | Unblocks when… |
|---|---|---|
| `LlmJudge` (2 cases) | DeepSeek rejects `response_format=json_schema` (returned by `StructuredChatModel.with_strict(true)`). | DeepSeek adds schema mode OR `StructuredChatModel` falls back to `json_object` + post-validate. |
| `synthesize_eval_cases` (2 cases) | Same root cause as `LlmJudge`. | Same fix unblocks both. |
| `BigToolAgent` (1 case) | Requires an embeddings provider to score the tool catalogue; DeepSeek has none. | Provide an OpenAI/Cohere/Voyage/Jina/FastEmbed key + provider. |
| `NamespacedMemory` (1 case) | Native litGraph memory classes silently DROP the `metadata` field on append; namespace filter on read returns empty. | Add a metadata-preserving native memory backend, OR refactor `NamespacedMemory` to maintain its own per-namespace deque on top of `append`. |

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
| 7 | **Pydantic-coerced state + `StreamPart`** | ✅ shipped iter 378 + 379. iter 378: `StateGraph(state_schema=Pydantic\|dataclass\|TypedDict)` auto-dumps input + auto-coerces invoke/resume output. iter 379: `StreamPart` typed-enum mirror (`Delta` / `ToolCallDelta` / `Done`) via `parse_stream_part(s)` + async variant — frozen dataclasses, `match`-narrowing, zero new deps. |
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
- ❌ **Streaming tool execution** — `OffloadingTool` + result-poll
  pattern for long-running shell jobs
- ⏳ **Tool-call budget caps** — `ToolBudget` exists in Python
  `tool_hooks`, no Rust-side cost ceiling mirroring
  `CostCappedChatModel`

#### Graph
- ✅ Branch fan-in **dedup-by-key reducer** (iter 382 — `merge_dedup_by_key(current, update, key)`)
- ✅ **`parallel_for` shorthand** for fan-out-N-copies pattern (iter 381 — `StateGraph::add_parallel_for(name, n, worker)`)

#### Memory / store
- ❌ **NamespacedMemory** that works on native backends (see Live-test
  blockers above — needs metadata-preserving native memory)

#### Eval & reproducibility
- ❌ **Eval-suite live smoke test** — would call the model
  recursively; covered today by mock unit tests only
- ❌ **Trajectory replay CLI** — take an OTel trace ID and replay the
  exact prompt against a chosen model

#### Serve
- ⏳ **Studio UI parity for local graphs** — `studio` feature flag in
  `litgraph-serve` covers cloud API surface only
- ❌ **`recipes.serve` actually spawns the binary** — today returns
  the shell-command string only (intentional stub; `iter 366` fixed
  the type-check bug)

#### Python ergonomics
- ✅ **Implicit Pydantic coercion** on `StateGraph(state_schema=)` (iter 378)
- ✅ **`StreamPart` typed enum** mirroring `ChatStreamEvent` (iter 379)

#### Observability
- ❌ **Trace exemplars** linking OTel span → prompt+completion excerpt
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

- ❌ `litgraph init <template>` repo scaffold
- ❌ `litgraph trace` viewer (OTel JSON → graph timeline in terminal)

### Docs

- ❌ "Migrate from LangChain" guide (top-20 idiom side-by-sides)
- ❌ Per-crate README pointing at canonical example

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

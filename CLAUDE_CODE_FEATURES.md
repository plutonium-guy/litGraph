# Claude Code → Prod-Ready Agent Feature Map

Features distilled from **LangChain** + **LangGraph** that matter for an LLM
assistant (Claude Code) to ship a production-ready agent **without writing
framework plumbing**. Each row marks status in `litGraph`.

Audit date: 2026-04-29 · Source of truth: `crates/litgraph-py/src/*.rs`,
`python_tests/*.py`, `FEATURES.md`.

Tracks LangChain **1.0** (Sep 2025 rewrite) + LangGraph **1.1** (2026) feature
surface — middleware, deep agents, functional API, type-safe streaming v2,
long-term Store. Older (v0.3) features rolled in from prior audit.

**Legend:** ✅ done · 🟡 partial · ❌ missing · 🚫 won't do

> Install/run uses **uv** (no venv activate):
> ```bash
> uv pip install litgraph        # wheel install
> uv run python examples/rag_agent.py
> uv run pytest python_tests/
> ```

---

## 1. Models — talk to LLMs

| LangChain/LangGraph feature | Why Claude Code cares | Status |
|---|---|---|
| `ChatModel` unified interface | One API across providers, swap models without rewrites | ✅ `litgraph.providers` |
| OpenAI chat | Default for most prod agents | ✅ `OpenAIChat`, `OpenAIResponses` |
| Anthropic chat (Claude) | First-class Claude support | ✅ `AnthropicChat` |
| Google Gemini (AI Studio + Vertex) | Multi-cloud option | ✅ |
| AWS Bedrock (native + Converse) | Enterprise/SOC2 deployments | ✅ |
| Cohere Command | Alt frontier model | ✅ |
| OpenAI-compat (Ollama, Groq, Together, Mistral, DeepSeek, xAI, Fireworks) | Cheap/local fallback | ✅ |
| Local inference (llama.cpp/candle) | Air-gapped agents | ❌ candle/mistral.rs |
| SSE streaming | Token-stream UX | ✅ async iterator |
| Native tool/function calling | Agents need it | ✅ per provider |
| Structured output (JSON schema) | Typed agent results | ✅ `StructuredChatModel` + schemars |
| Vision (image-in) | Multimodal agents | ✅ |
| Audio I/O (whisper, TTS) | Voice agents | ✅ `WhisperTool`, `TtsTool` |
| Image generation (DALL-E) | Creative agents | ✅ `DalleTool` |
| Async batch API | Bulk evaluation | ✅ `litgraph_core::batch_concurrent` (+ `_fail_fast`) — bounded-concurrency parallel `ChatModel.invoke` over Tokio + Semaphore, order-preserving, per-call `Result`. Python: `litgraph.agents.batch_chat(model, inputs, max_concurrency, fail_fast=False)`. (iter 182) Plus `batch_concurrent_with_progress` (iter 205) — same shape, updates a `Progress<BatchProgress>` watcher with `{total, completed, errors}` for live eval-harness counters. Plus `batch_concurrent_stream` (iter 210) — yields `(idx, Result)` pairs as each call completes (mpsc-backed) so callers can render results live, dispatch downstream work on early-completers, and abort-on-drop to stop in-flight work. Plus `batch_concurrent_stream_with_progress` (iter 216) — first composition combining stream + watcher in one call: per-row events drive a UI list view; counters drive a summary progress bar. Plus `batch_concurrent_with_shutdown` (iter 227) — preserves partial progress on Ctrl+C: completed slots stay `Ok`, in-flight slots become `Err("cancelled by shutdown")`. First parallel-batch primitive bridged to the coordination primitives. Plus `batch_concurrent_stream_with_shutdown` (iter 233) — opens the second bridge family (streaming + coordination): producer-side graceful end-of-interest. One signal source can stop *many* streams without each consumer needing to drop its receiver. Consumer observes a partial prefix in completion order, then the stream ends cleanly. |
| Token counters (tiktoken/HF) | Pre-flight cost/context limits | ✅ `litgraph.tokenizers` |

## 2. Resilience — survive prod

| Feature | Why | Status |
|---|---|---|
| Retry w/ exp backoff + jitter | Flaky upstream | ✅ `RetryingChatModel` |
| Rate limiter | Per-key RPM/TPM caps | ✅ `RateLimitedChatModel` |
| Fallback chain | Model A down → try B | ✅ `FallbackChatModel` |
| Race / hedged requests | Latency-min: A and B in parallel, first wins | ✅ `litgraph_resilience::RaceChatModel` — Tokio `JoinSet` + `abort_all` cancels losers as soon as a winner emerges; aggregates errors only if every inner fails. Python: `litgraph.providers.RaceChat(models)`. (iter 184) |
| Multiplexed live streaming | Render N model token streams side-by-side | ✅ `litgraph_core::multiplex_chat_streams` — Tokio `mpsc` channel-fan-in; per-event `model_label` tag; one slow / failing model never blocks the others. Python: `litgraph.agents.multiplex_chat_streams(models, messages)`. (iter 189) |
| Broadcast streaming (1 → N) | Live UI + audit log + sidecar evaluator on the same stream | ✅ `litgraph_core::broadcast_chat_stream` — `tokio::sync::broadcast` channel; lazy-spawned pump to avoid races against subscribers; per-subscriber `Lagged` notice on capacity overflow. Python: `litgraph.agents.broadcast_chat_stream(model, messages, capacity)` returns a `BroadcastHandle.subscribe()` iterator. Inverse of `multiplex_chat_streams` (which is N → 1 fan-in). (iter 195) |
| Latest-value progress observability | Progress UIs / health probes / agent dashboards | ✅ `litgraph_core::Progress<T>` — `tokio::sync::watch`-backed; multiple observers read current state on demand; rapid intermediate writes collapse to latest. Python: `litgraph.observability.Progress(initial)` with `.set` / `.snapshot` / `.observer()` / `wait_changed`. Completes the channel-shape trio (mpsc 189, broadcast 195, watch 199). (iter 199) |
| Embeddings race / hedged requests | Tail-latency cut on the embed-query critical path | ✅ `litgraph_resilience::RaceEmbeddings` — Tokio `JoinSet` + `abort_all`; first success wins, losers cancelled; dim-mismatch rejected at construction. Python: `litgraph.embeddings.RaceEmbeddings(providers)`. (iter 192) |
| Retriever race / hedged requests | Hedge fast cache vs slow primary | ✅ `litgraph_retrieval::RaceRetriever` — Tokio `JoinSet` + `abort_all` over N retrievers; first success wins, losers cancelled. Use for **latency** (vs `EnsembleRetriever` for **quality**). Python: `litgraph.retrieval.RaceRetriever(children)`. (iter 193) |
| Per-call timeout deadline (chat + embed) | SLA enforcement, circuit-breaker preconditions | ✅ `litgraph_resilience::{TimeoutChatModel, TimeoutEmbeddings}` — `tokio::time::timeout` runs the inner future and a deadline timer concurrently; first to complete wins, inner is dropped on timeout. Composes through `extract_chat_model` / `extract_embeddings`. Python: `litgraph.providers.TimeoutChat(model, timeout_ms)`, `litgraph.embeddings.TimeoutEmbeddings(inner, timeout_ms)`. (iter 194) |
| Circuit breaker (chat) | Stop bleeding load against a sick upstream | ✅ `litgraph_resilience::CircuitBreakerChatModel` — bridges the iter-243 `CircuitBreaker` primitive into the `ChatModel` family. Wraps any inner model. After N consecutive failures, every call fails fast with `Error::Provider("circuit breaker open")` for the cooldown window. After cooldown, exactly one probe is allowed; success closes the breaker, failure re-opens. Stack with `FallbackChatModel` so circuit-open routes to the secondary provider immediately, not after retry timeouts. Streams are wrapped at the handshake (open → fail-fast `stream()` without invoking inner; mid-stream failures are consumer's responsibility). (iter 244) |
| Circuit breaker (embed) | Same blast-radius limit on the embeddings axis | ✅ `litgraph_resilience::CircuitBreakerEmbeddings` — embed-axis mirror of iter 244. Wraps any `Embeddings` with the iter-243 primitive. ONE shared breaker covers both `embed_query` and `embed_documents` so a flapping query path also opens the breaker for document indexing (and vice-versa) — captures the underlying upstream's health state across both call shapes. Stack with `FallbackEmbeddings` for circuit-open → secondary provider routing. (iter 245) |
| Circuit breaker (retrieve) | Same blast-radius limit on the retriever axis | ✅ `litgraph_retrieval::CircuitBreakerRetriever` — retriever-axis mirror. Wraps any `Retriever` with the iter-243 primitive. Real prod use: vector store goes down → after N consecutive failures the breaker opens → agents fail fast and route to a backup retriever (typically via `EnsembleRetriever([primary_with_breaker, backup_bm25])`) instead of hammering the sick service. The "circuit breaker open" error message intentionally doesn't match `is_transient`'s 5xx pattern, so a `RetryingRetriever` outer doesn't re-trigger the open breaker. Lives in litgraph-retrieval (consistent with iter 271-274 retriever-wrapper placement). With this iter the circuit-breaker matrix covers chat/embed/retrieve — the three primary axes where it makes sense (tool axis has different resilience model via iter-159). (iter 275) |
| Shared-budget rate limit (chat + embed) | One quota across many wrapped models | ✅ `litgraph_resilience::{SharedRateLimitedChatModel, SharedRateLimitedEmbeddings}` — bridges the iter-242 `RateLimiter` primitive into the resilience family. Distinct from `RateLimitedChatModel` (which owns its bucket): takes `Arc<RateLimiter>` so multiple wrappers — even across model variants AND across chat/embed axes — share ONE budget. Right when one provider API key has one TPM/RPM quota covering several model variants (gpt-4 / gpt-4-turbo / gpt-4o-mini all on one key). Each call charges 1 token; for token-weighted accounting the limiter exposes lower-level `acquire(n)`. (iter 246) |
| Per-key serialized chat | Per-thread ReAct step lock (no cross-thread serialization) | ✅ `litgraph_resilience::KeyedSerializedChatModel` — bridges the iter-241 `KeyedMutex<String>` primitive into the chat family. Caller supplies `key_fn: Fn(&[Message], &ChatOptions) -> Option<String>` (typically `\|_, opts\| opts.metadata.get("thread_id").cloned()`). Same key serializes; different keys run concurrently; `None` passes through unlocked. Solves the per-thread ReAct correctness problem without imposing a global mutex on every call. Stream handshake also locks; the lock releases after `stream()` returns (the caller drives the stream itself). (iter 247) |
| Bulkhead (chat + embed) | Concurrent-call cap with rejection, not queueing | ✅ `litgraph_resilience::{BulkheadChatModel, BulkheadEmbeddings}` — bridges the iter-248 `Bulkhead` primitive into the resilience family. Two modes via `BulkheadMode`: `Reject` (instant rejection on cap) or `WaitUpTo(Duration)` (block up to t, then reject). Bulkhead-full surfaces as `Error::RateLimited { retry_after_ms: None }` so the existing `is_transient` classifier matches it: `RetryingChatModel` outer retries (slot may have opened), `FallbackChatModel` outer switches provider — no extra wiring. One `Arc<Bulkhead>` can span multiple wrappers AND across chat+embed axes, enforcing one cap over a shared resource budget. (iter 249) |
| Bulkhead (retrieve) | Cap concurrent vector-store calls to protect connection pools | ✅ `litgraph_retrieval::BulkheadRetriever` — retriever-axis sibling to iter-249. Same `Reject` / `WaitUpTo` modes; same `Error::RateLimited` outcome so `RetryingRetriever` retries on bulkhead-full. Real prod use: cap pgvector reads at the connection-pool size minus headroom for writes; cap qdrant CPU usage by limiting concurrent queries; protect in-process HNSW from memory pressure under heavy fan-out. One `Arc<Bulkhead>` can span multiple retrievers — useful for "all retrievers share one budget" patterns where the underlying resource is shared. With this iter the bulkhead matrix covers chat (249) / embed (249) / retrieve (276). (iter 276) |
| Hedged requests (chat + embed) | Tail-latency mitigation — pay the second-call cost only on slow tails | ✅ `litgraph_resilience::{HedgedChatModel, HedgedEmbeddings}` — bridges the iter-250 `hedged_call` combinator into the resilience family. Distinct from `RaceChatModel` (iter 184) which doubles cost by issuing to both providers always: hedge issues primary alone for `hedge_delay`; only if primary is slow does backup also fire. Tests verify backup is NOT invoked when primary is fast (zero overhead on fast path). For the embed axis, both `embed_query` and `embed_documents` are hedged. Streaming is primary-only (token streams can't be cleanly raced — chunks can't merge or be chosen between mid-stream). (iter 251) |
| Hedged requests (retrieve) | Tail-latency mitigation for slow vector stores | ✅ `litgraph_retrieval::HedgedRetriever` — retriever-axis sibling. Distinct from `RaceRetriever` (iter 193) which always doubles cost: hedge only fires the backup if primary exceeds `hedge_delay`. Real prod use: HNSW with cold pages (1ms p50 / 50ms p99) with hedge=5ms and a backup HNSW replica; multi-region setups where us-east-1 is fast normally but us-west-2 is the failover. Loser future is dropped on tokio cancellation, releasing the held HTTP/DB resource. With this iter the hedge matrix covers chat (251) / embed (251) / retrieve (277). (iter 277) |
| Embed query coalescing | Cache-miss thundering-herd mitigation | ✅ `litgraph_resilience::SingleflightEmbeddings` — bridges the iter-252 `Singleflight` primitive into the embeddings family. N concurrent identical `embed_query` calls share ONE upstream HTTP call; followers receive the leader's result via `tokio::sync::broadcast`. `embed_documents` passes through (multi-doc batches rarely repeat exactly; coalescing them would need a `Vec<String>` hash key for low expected win). Errors collapse to `Error::Provider(s)` on the singleflight path (lossy by design — `Error` isn't `Clone`, and running each follower's compute on leader-fail defeats the purpose). Real prod use: 50 agents starting up all embed the same long system prompt → 1 HTTP call; hot search query embedded 100×/sec from different threads → 1 call per dedup window; eval harness scoring same query against many retrievers → query embed computed once. (iter 253) |
| Chat-response cache (exact-match TTL) | Avoid redundant LLM calls across runs — cost reduction | ✅ `litgraph_resilience::CachedChatModel` (iter 296) — exact-match in-memory chat-response cache. Distinct from neighboring primitives: **`RecordingChatModel`/`ReplayingChatModel`** (iter 254) are for *test workflows* (record once, replay forever in CI); this is for *production* (LRU+TTL cache hits skip the provider call to save tokens / latency). **`PromptCachingChatModel`** (iter 136) controls Anthropic's *server-side* prompt cache via cache_control headers; this caches the *entire response client-side* — zero provider call on hit. The two compose: client-side cache hits skip the network roundtrip entirely; misses with prompt caching still get input-token savings upstream. Cache key: `exchange_hash` (the same blake3-over-canonical-JSON function as the cassette infrastructure) so cache and cassette agree on what counts as "the same request." `with_max_entries(n)` LRU cap (default 1000); optional `with_ttl(d)` (default `None`). `cache_len()` for telemetry, `clear()` for test reset. Streaming bypasses (token streams can't replay without inter-chunk timing — and stream callers want streaming UX). Win conditions: eval harness re-runs over the same dataset, agents over a fixed FAQ where users phrase identically (verbatim — fuzzy matches need a semantic layer), demo/dev environments while iterating on downstream UI, multi-stage agent loops where retry/restart re-issues the same call. Anti-pattern: high-temperature creative generation (cache-hit rate near zero), long-tail prompts (memory pressure with no hits). |
| Embed cache (exact-match TTL) | Avoid redundant embedding API calls across runs | ✅ `litgraph_resilience::CachedEmbeddings` (iter 292) — exact-match string-key cache. Distinct from iter-253 `SingleflightEmbeddings` (which dedups *concurrent* identical calls): this caches *across* calls. Once "hello" is embedded, subsequent `embed_query("hello")` calls skip the upstream until the entry expires or evicts. `with_max_entries(n)` LRU cap (default 1000); optional `with_ttl(d)` (default `None` = forever — useful when the model is fixed). `embed_query` and `embed_documents` use separate caches, both keyed exactly (different orderings of the same docs are different cache keys, by design — embedding output IS order-sensitive). Composes with `SingleflightEmbeddings`: stack cache outside, dedup inside. Real prod: popular search queries embedded once per cache window, repeated agent runs over the same query corpus during eval/development, pre-warming cache from a known-popular query list at startup. |
| Tool call coalescing | Idempotent expensive lookups | ✅ `litgraph_resilience::SingleflightTool` — third axis of the request-coalescing family (chat intentionally not coalesced; embed iter 253; tool this iter). N concurrent identical `run(args)` calls share ONE upstream invocation. Hash key: blake3 over canonical JSON of `args` (same function as iter 256 `tool_args_hash`). Real prod use: idempotent expensive lookups (`lookup_user("alice")` called 10× concurrently → 1 DB round-trip), hot search query coalescing, stable function tools where output is a pure function of args. **Critical: must NOT wrap tools with side effects** — coalescing collapses N intent-distinct calls into one execution, deduplicating the side effects too. Documented in the wrapper rustdoc. (iter 263) |
| Retrieval coalescing | Cache-miss thundering herd on hot queries | ✅ `litgraph_retrieval::SingleflightRetriever` — fourth axis of request-coalescing. N concurrent agents asking the same `retrieve(query, k)` → one vector-store hit, all callers get the same `Vec<Document>`. Hash key: blake3 over canonical JSON of `(query, k)` (matches iter-274 retriever-cassette hash so the two compose if used together). Different `k` values coalesce independently. Errors stringified to `Error::Provider` (lossy-by-design, same as iter 253 / iter 263 — `Error` isn't Clone). Coalesce when retrieval is a pure function of `(query, k)`; don't wrap retrievers with telemetry side effects that need per-call recording. (iter 278) |
| Semantic-similarity retrieval cache | FAQ-style queries phrased differently | ✅ `litgraph_retrieval::SemanticCachedRetriever` (iter 291) — caches retrieval results across calls when queries are *semantically similar* (cosine ≥ threshold). Distinct from iter-278 SingleflightRetriever which dedups *concurrent* identical calls; this caches *across* calls. Real prod: FAQ agents where users phrase the same question differently — "How do I reset my password?" / "I forgot my password" / "password recovery steps" all hit the same cache line. Configurable `with_threshold(0.95)` (default), `with_max_entries(1000)` LRU cap, optional `with_ttl(d)` for cache expiry. Different `k` values keep separate cache lines (result-list shapes differ). Linear scan over the cache — tune `max_entries` to your latency budget. |
| Record / replay (VCR-style cassettes) | Deterministic agent tests, no API in CI, regression replay | ✅ `litgraph_resilience::{RecordingChatModel, ReplayingChatModel, Cassette, exchange_hash}` — record real LLM traffic to a serializable `Cassette { version, exchanges }`, replay in tests. Hash key: blake3 over canonical JSON of `(messages, opts)` so identical requests deterministically match. ReplayingChatModel takes optional `passthrough` (typically the live model) for record-then-fill-gaps workflows; without passthrough, miss returns `Error::Provider("no recorded response for hash <…>")`. JSON round-trip verified — cassettes live next to test files. Streams pass through unrecorded (chunk-timing replay isn't useful). Single iter; substantial feature missing from most LangChain alternatives. (iter 254) Plus `Cassette::{load_from_file, save_to_file}` + `RecordingEmbeddings` + `ReplayingEmbeddings` + `EmbedCassette` + `embed_query_hash` + `embed_documents_hash` (iter 255) — file IO closes the workflow gap (record once, save to `tests/cassettes/foo.json`, load in CI), and the embeddings axis gets the same machinery (`EmbedExchange::{Query, Documents}` enum so a single cassette covers both methods). `save_to_file` auto-creates parent directories; `load_from_file` errors surface as `Error::Other`. With this iter the record/replay story is end-to-end: deterministic CI for both chat and embed paths, no API cost, cassettes versioned in source control. Plus `RecordingTool` + `ReplayingTool` + `ToolCassette` + `tool_args_hash` (iter 256) — third record/replay axis (chat / embed / tool). Hash key: blake3 over canonical JSON of `args`. `ReplayingTool` synthesizes a minimal schema when no passthrough is set (with `with_name` / `with_description` builders) or proxies the passthrough's schema. Real prod use: agent integration tests that hit external APIs via tools — record against staging, replay in CI without service dependencies. Plus `RecordingRetriever` + `ReplayingRetriever` + `RetrieverCassette` + `retrieve_hash` (iter 274) — fourth and final record/replay axis. Hash key: blake3 over canonical JSON of `(query, k)` (so different `k` values record/replay independently). Lives in litgraph-retrieval to avoid the circular dep. With this iter the record/replay matrix covers all four primary axes: chat / embed / tool / retrieve. The full agent loop — LLM call + embedding + tool call + retrieval — is now CI-deterministic with one consistent VCR-style API. |
| RSS / Atom feed loader | News & blog ingestion pipelines | ✅ `litgraph_loaders::RssAtomLoader` (feature-gated `rss`) — fetches RSS 2.0 / Atom 1.0 feeds and parses each `<item>` / `<entry>` into a `Document`. Single quick-xml pull-parser walks both formats; per-item fields collected into `FeedItem { title, link, summary, content, published, guid, feed_title }`. Body-resolution priority: content > summary > title (so title-only items still produce a non-empty doc). Namespace-prefix-stripped tag matching so `content:encoded` is read regardless of XML namespace. RFC822 → RFC3339 normalization on `pubDate`. Attribute-based `<link href="…"/>` (Atom) AND text-content `<link>https://…</link>` (RSS) both handled. `with_max_items(n)` cap, `with_skip_empty(true)` filter, `with_user_agent` / `with_timeout` knobs. (iter 257) |
| HackerNews loader | Tech news / research corpora | ✅ `litgraph_loaders::HackerNewsLoader` (no feature gate) — fetches stories from the public HN Firebase API at `hacker-news.firebaseio.com/v0/`. Six feed sources via `HnFeed::{Top, New, Best, Ask, Show, Job}`; per-item metadata `{hn_id, title, url, by, score, time, type, descendants, feed}`. Rayon-parallel item fetch (the per-item HTTP calls are independent and HN explicitly invites traffic). Title+text combined for `Ask`/`Show` posts so the embeddable content is the actual prose, not just the title. `with_base_url` for testing against a local fake server. Default `max_items=30` to bound a default fetch at a reasonable size. (iter 258) |
| Bitbucket issues loader | Third Git provider parity (after GitHub iter 41-42 and GitLab iter 149/154) | ✅ `litgraph_loaders::BitbucketIssuesLoader` — Bitbucket Cloud REST API v2 at `api.bitbucket.org/2.0/`. Both auth modes: app-password Basic auth (`with_oauth(false)`, default) or OAuth bearer (`with_oauth(true)`). BBQL filter syntax for `state` (`open`/`resolved`/`closed`/`all`; aliases `opened` translated) + `kind` (`bug`/`enhancement`/`task`/`proposal`/`all`). Cursor-based pagination via Bitbucket's `next` URL field. Per-issue metadata `{workspace, repo, issue_id, state, kind, priority, assignee, reporter, created_on, updated_on, votes, watches, link}`. `with_include_comments(true)` appends the comment thread to the issue document so a single doc covers the full discussion. Offline-testable via `pub fn issue_to_document(&self, &Value, &[Value])`. (iter 264) |
| Bitbucket files loader | Round out Bitbucket parity with the GitHub/GitLab files variants | ✅ `litgraph_loaders::BitbucketFilesLoader` — recursive source-file fetcher via Bitbucket's `/src/{ref}/{path}` endpoint. Behavior depends on path target: directory → JSON listing with `commit_directory` / `commit_file` entries; file → raw text body. BFS walk with `with_max_depth` cap (default 32). Same auth + filter knobs as the issues loader; standard `with_extensions` allowlist (suffix match, case-insensitive, dot-prefixed normalization), `with_exclude_paths` substring-match denylist (default skips `node_modules/`, `target/`, `.git/`, lockfiles, minified bundles), `with_max_files` total cap (default 500), `with_max_file_size_bytes` per-file size cap (default 1 MiB to skip blobs masquerading as text). Per-file metadata `{workspace, repo, ref, path}`. With this iter the Git-host loader matrix is fully populated: `{GitHub, GitLab, Bitbucket} × {Issues, Files}`. (iter 265) |
| Stack Exchange loader | Community Q&A corpus for technical agents | ✅ `litgraph_loaders::StackExchangeLoader` (iter 294) — fetches questions + (optional) accepted answers from any Stack Exchange site (stackoverflow / serverfault / superuser / unix / dba / math / physics / 100+ others) via `api.stackexchange.com/2.3/`. `with_site(...)`, `with_tags(["rust","tokio"])` filter (any-of, semicolon-joined), `with_max_questions(n)` cap (default 30), `with_include_answers(true)` to concat the highest-voted answer body into each question Document. API key optional (`with_key(...)` for ~10k req/day quota vs ~300 unkeyed). Rayon-parallel batch fetch of answer bodies. Per-question metadata `{site, question_id, title, tags, score, view_count, answer_count, is_answered, link, owner}`. Real prod: technical-agent grounding on community Q&A, trend monitoring of tag activity, doc supplementation. (iter 294) |
| Token budget guard | Stop runaway prompts | ✅ `TokenBudgetChatModel` |
| Cost cap | Hard $ ceiling per run | ✅ `CostCappedChatModel` |
| PII scrubber (input/output) | Compliance | ✅ `PiiScrubbingChatModel` |
| Prompt cache wrapper | Cut Anthropic/OpenAI cost | ✅ `PromptCachingChatModel` |
| Self-consistency voting | Boost reasoning accuracy | ✅ `SelfConsistencyChatModel` |
| Tool-level timeout | Stuck shell/HTTP | ✅ `TimeoutTool` |
| Tool-level retry | Transient tool errors | ✅ `RetryTool` |
| Embedding retries/fallbacks | RAG ingest reliability | ✅ |

## 3. Prompts — templates that don't rot

| Feature | Why | Status |
|---|---|---|
| `ChatPromptTemplate` (role-tagged) | Compose system/user/assistant | ✅ |
| `MessagesPlaceholder` | Insert chat history slot | ✅ |
| Jinja interpolation, strict-undefined | Catch missing vars at compile | ✅ minijinja |
| FewShotPromptTemplate | In-context examples | ✅ |
| SemanticSimilarityExampleSelector | Pick relevant examples | ✅ |
| LengthBasedExampleSelector | Fit examples to budget | ✅ |
| Partial application (bind vars) | Curry templates | ✅ |
| from/to JSON · from/to dict | Save/load prompts | ✅ |
| Composition (extend/+/concat) | Layer base+role+task | ✅ |
| Hub pull (`langchain hub`) | Share prompts | ✅ `litgraph_core::{PromptHub, FilesystemPromptHub, CachingPromptHub}` + `litgraph_loaders::HttpPromptHub` — versioned `name@v2` refs, JSON-on-disk or HTTP fetch, bearer/header auth, traversal-hardened, list/push/pull, process-local cache wrapper |

## 4. Output Parsers — turn text into structs

| Feature | Why | Status |
|---|---|---|
| JSON / Pydantic struct out | Typed results | ✅ `StructuredChatModel` |
| XML parser (flat + nested) | Some models prefer XML | ✅ |
| Comma-list / numbered / markdown-list | Quick lists | ✅ |
| Boolean parser | yes/no agents | ✅ |
| ReAct text-mode parser | Models without tool calling | ✅ |
| Markdown-table parser | Tabular extraction | ✅ |
| `format_instructions` helpers | Auto-tell LLM the format | ✅ |
| OutputFixingParser (retry-on-parse-fail) | Self-heal | ✅ |
| Streaming partial JSON | Live structured UIs | ✅ `parse_partial_json` |
| Pandas DataFrame parser | Data agents | ✅ `litgraph_core::{Table, TableQuery, parse_table_json, parse_table_csv, table_format_instructions}` — three ingest formats (`{columns,rows}` / records / CSV with quote+CRLF handling), query lang `column:`/`row:`/`<col>:<row>`/`mean:`/`sum:`/`min:`/`max:`/`count:`/`unique:`, type-checked numeric ops, null-skipping count |

## 5. Tools — let agent do things

| Feature | Why | Status |
|---|---|---|
| Tool trait + auto JSON schema | Agent self-describes capabilities | ✅ `#[tool]` macro |
| Concurrent tool fan-out | Multi-tool calls in parallel | ✅ JoinSet |
| HTTP request | API calls | ✅ `HttpRequest` |
| Shell exec (sandboxed) | Coding agents | ✅ `Shell` |
| File read/write/list | File ops | ✅ |
| SQL query (sqlite/pg) | DB agents | ✅ `SqliteQuery` |
| Calculator | Math grounding | ✅ |
| Python REPL | Exec generated code | ✅ `PythonRepl` |
| Web search (Brave, Tavily, DDG) | Research agents | ✅ |
| Web fetch | Pull URL → text | ✅ `WebFetchTool` |
| Web extract (Tavily) | Clean article text | ✅ `TavilyExtract` |
| Cached tool wrapper | Skip dup calls | ✅ `CachedTool` |
| Webhook / Slack / Discord notify | Agent → human notify | ✅ `WebhookTool` |
| Gmail send | Email agents | ✅ `GmailSendTool` |
| MCP tool client | Connect external MCP servers | ✅ `Mcp` |
| MCP server (expose own tools) | Be a tool provider | ✅ resources + prompts |
| Date/time grounding | Date-aware reasoning ("today", "next Tuesday") | ✅ `litgraph_tools_utils::CurrentTimeTool` (iter 279) — returns `iso8601`, `unix`, `weekday`, `date`, `time`, `tz` for the current moment. Optional `tz_offset_hours` arg (fractional, e.g. 5.5 for IST, -7 for PDT). Without this, LLMs have no reliable way to ground a "today" / "next Tuesday" / "two weeks from now" reasoning step — their training cutoff is in the past. Pure-Rust via `chrono` workspace dep (no `chrono-tz` so IANA-name resolution is not supported; explicit-offset is sufficient for most prod cases). |
| Regex extraction | Pull structured fields from unstructured text | ✅ `litgraph_tools_utils::RegexExtractTool` (iter 280) — apply a regex to a string and return matches/captures. Three modes: `all` (every match string), `first` (first match or null), `captures` (each match + numbered capture groups). Universal: any agent that fetches unstructured text (web pages, API responses, log lines) can extract specific fields without writing a custom parsing tool. Uses the `regex` workspace dep (no new third-party). |
| JSON path extraction | Pull specific fields from JSON API responses | ✅ `litgraph_tools_utils::JsonExtractTool` (iter 281) — sibling to `RegexExtractTool` for JSON. Tiny JSONPath subset (deliberate — easy for LLMs to generate without confusion): `$` for root, `.field` for keys, `[N]` for index (negative N counts from end), `[*]` for all array elements. Examples: `$.users[0].name`, `$.users[*].email`, `$.results[-1]`. The `json` arg accepts both a JSON value AND a string (auto-parsed). `[*]` skips elements where the inner path resolves to null, so `$.users[*].email` returns only the users that have an email field. Pairs with `HttpRequestTool` for "fetch API → drill into specific field" agent workflows. |
| URL parsing | Validate redirect hosts, extract OAuth callback params | ✅ `litgraph_tools_utils::UrlParseTool` (iter 282) — parse URLs into `{scheme, host, port, path, query, query_params, fragment, username, password}`. Repeated query params (`?tag=a&tag=b&tag=c`) collapse into a JSON array; single-value params stay strings. Default ports (443/https, 80/http) return null in the `port` field — explicit-only. Percent-encoded query values are decoded. Real prod use: agent receives an OAuth callback URL and pulls the `code` / `state` params; agent validates that a user-supplied URL's `host` is on an allowlist before redirecting. Uses the `url` crate (already a transitive dep via reqwest). |
| Hash digest | Content fingerprinting / dedup / cache keys | ✅ `litgraph_tools_utils::HashTool` (iter 283) — compute lowercase-hex digest via blake3 (default), sha256, sha512, or md5. Tested against known fixed-input vectors for every algorithm. Real prod use: dedup detection, content-integrity checks, agent-generated cache keys. Algorithm-choice guidance baked into the rustdoc: blake3 unless interop forces something else; md5 only for non-security legacy fingerprinting. Reuses sha2 / md-5 / hex crates that were already transitive deps. |
| Base64 encode/decode | JWT headers, base64-encoded image URLs, API payloads | ✅ `litgraph_tools_utils::Base64Tool` (iter 284) — encode or decode base64 in either standard (RFC 4648 §4, `+/=` alphabet) or URL-safe (RFC 4648 §5, `-_` alphabet, no padding) variant. Decoded output is UTF-8; non-UTF-8 binary decode surfaces as `Error::InvalidInput` (callers needing binary should decode in code, not via this tool). Real prod scenarios: JWT header inspection (`url_safe` variant), decoding base64 image URLs, encoding payloads for APIs that require base64 in request bodies. Verified by a JWT-header-decode test against the standard `{"alg":"HS256","typ":"JWT"}` vector. |
| UUID generation | Mint IDs, idempotency keys, trace IDs | ✅ `litgraph_tools_utils::UuidTool` (iter 285) — generate v4 (random) or v7 (timestamp-ordered, default) UUIDs. Three formats: `hyphenated` (default), `simple` (no-hyphen 32-char hex), `urn` (`urn:uuid:...`). Returns an array even for `count=1` so callers don't special-case length. Default `v7` because timestamp-ordered IDs preserve B-tree page locality for DB primary keys — better default than v4 for most agent workflows that mint IDs for storage. `v4` for opaque tokens where ID ordering would leak information. `count` is capped at 100 (sanity guard against LLM looping). Verified by a sortable-by-creation-time test that generates 10 v7 IDs serially and verifies sort-order matches insertion-order. |
| Text diff | Code-change agents, document review, audit trails | ✅ `litgraph_tools_utils::TextDiffTool` (iter 286) — line-level diff between two strings. Two output formats: `unified` (traditional `+`/`-`/`@@` diff, compact for LLM summarization) or `structured` (JSON with separate `additions` / `deletions` arrays + count summary, better for programmatic filtering). `context_lines` knob for unified mode (default 3). Built on the `similar` crate. Real prod: agent reading two versions of a config / spec / doc emits a structured changelog; code-change agents show "what did the refactor change?" before applying or reporting; test-failure analysis diffs actual vs expected output. |
| JWT decode (header+payload, no signature verify) | OAuth flow debugging, agent inspection of bearer tokens, claim extraction | ✅ `litgraph_tools_utils::JwtDecodeTool` (iter 295) — decode a JWT (`header.payload.signature`) into its parsed header + payload objects. Returns `{header, payload, signature_present, expired}` where `expired` is `null` when no `exp` claim is present (RFC 7519 §4.1.4). Uses `URL_SAFE_NO_PAD` base64 (the JWT-mandated variant). **Does NOT verify the signature** — verification needs the issuer's signing key and belongs in the auth layer (server-side middleware, not agent), and the docstring loudly warns about this. Real prod: agent extracting `sub`/`user_id` from a request token to scope its tool calls; debugging "why did this OAuth callback 401?" by inspecting `iss`/`aud`/`exp`/`scope`; checking expiry before retry storms. Verified by 10 tests including the canonical RFC 7519 §A.1 test vector. Closes the JWT-inspection one-shot vs the prior 3-call chain (string.split + Base64Tool + json_extract). |
| Eval drift detection | Per-case regression / improvement vs baseline | ✅ `litgraph_core::detect_drift(&baseline, &current, threshold) -> DriftReport` (iter 289) — compare two `EvalReport`s, surface per-case `regressions` (passed in baseline, failed in current), `improvements` (failed in baseline, passed in current), `stable_failures` (still failing in both — likely dataset issues), per-scorer `aggregate_deltas` for the headline summary, plus `missing_in_current` / `new_in_current` for case-set drift. `DriftReport::has_regressions()` is the natural CI gate: `if drift.has_regressions() { fail_build() }`. Threshold tunable: 0.5 default for binary pass/fail scorers, smaller (e.g. 0.1) for continuous scorers. Aggregate score deltas hide case-level story; this primitive surfaces what actually changed when a model upgrade or prompt tweak ships. |
| Eval statistical significance (binary) | Is the pass-rate change real or noise? | ✅ `litgraph_core::mcnemar_test(&baseline, &current) -> Vec<McNemarResult>` (iter 290) — McNemar's chi-squared test for paired binary outcomes. Per-scorer 2×2 contingency table `{a: pass-pass, b: pass-fail (regression), c: fail-pass (improvement), d: fail-fail}`. Continuity-corrected statistic `(|b-c|-1)²/(b+c)`. Two-tailed p-value via the χ²(1) ↔ Z² identity + Abramowitz & Stegun erf approximation (max error ~1.5e-7). `significant_at_05` boolean for CI gates. `small_sample` flag when `b+c < 25` — the chi-squared approximation is unreliable in that regime; users are advised to prefer an exact binomial test (out-of-scope here, flagged so callers know). Pairs with iter-289 `detect_drift`: drift surfaces the cases that moved; mcnemar_test answers "is the movement statistically meaningful?". |
| Eval statistical significance (continuous) | Is the score-shift real or noise? | ✅ `litgraph_core::wilcoxon_signed_rank_test(&baseline, &current) -> Vec<WilcoxonResult>` (iter 297) — Wilcoxon signed-rank test for paired *continuous* outcomes (cosine similarity, BLEU, raw LLM-judge floats, embedding-recall@k). Non-parametric counterpart to the paired t-test — makes no normality assumption, only that diffs are symmetric around the median under H₀. Per-scorer: pair (baseline, current) cases by input, compute `diff = current - baseline`, drop zero diffs (Wilcoxon convention), rank `|diff|` from 1..n with **average ranks for ties** (e.g. two values tied for ranks 3+4 each get 3.5), sum positive ranks (`w_plus`) and negative ranks (`w_minus`), test statistic `w = min(w_plus, w_minus)`. Normal approximation under H₀: μ = n(n+1)/4, σ² = n(n+1)(2n+1)/24 − Σ(t³−t)/48 (with **tie correction** in σ); continuity correction +0.5 toward μ. Two-tailed p via the same Abramowitz & Stegun erf approximation as McNemar. `significant_at_05` boolean; `small_sample` flag when `n < 20`. `mean_diff` reported across ALL paired cases (zeros included) so callers get the directional summary regardless of significance. Closes the eval-significance gap McNemar's docstring already pointed at: "for continuous scorers, use a paired t-test or Wilcoxon" — Wilcoxon is now built-in. Skewed-diff distributions (most cases unchanged, a handful regressed sharply) keep Wilcoxon well-calibrated where a paired t-test would inflate false positives. |
| Streaming tool execution | Long-running tool stream | 🚫 deferred |
| Zapier / N8N tool | Citizen-dev integrations | ❌ userland |

## 6. Agents — orchestration patterns

| Feature | Why | Status |
|---|---|---|
| ReAct tool-calling agent | Default agent loop | ✅ `ReactAgent` |
| ReAct text-mode | Non-tool-calling models | ✅ `TextReActAgent` |
| Plan-and-Execute | Two-phase reasoning | ✅ `PlanAndExecuteAgent` |
| Supervisor multi-agent | Router over specialists | ✅ `SupervisorAgent` |
| Swarm/handoff (`Command(goto=)`) | Agent-to-agent jump | ✅ Command primitive |
| Dynamic subagent spawn (tool-style) | Delegate w/ isolated context | ✅ `SubagentTool` |
| Parallel ReAct tool calls | Speed | ✅ |
| Recursion / max-step guard | Avoid infinite loops | ✅ |
| Agent event stream | UI progress | ✅ `AgentEventStream` |
| Pre-built `create_react_agent` factory | One-liner agents | ✅ `ReactAgent.new()` |
| `create_deep_agent` one-call factory | Loads AGENTS.md+skills, injects PlanningTool+VFS | ✅ `litgraph.deep_agent.create_deep_agent(model, tools=…, agents_md_path=…, skills_dir=…)` |

## 7. StateGraph — LangGraph headline

| Feature | Why | Status |
|---|---|---|
| Typed state + reducers | Safe parallel writes | ✅ derive macro |
| Static + conditional edges | Branch logic | ✅ |
| Entry / END markers | Graph boundaries | ✅ |
| Subgraphs | Compose graphs | ✅ |
| `Send` fan-out (map-reduce) | N parallel children | ✅ |
| Kahn parallel scheduler | True parallelism | ✅ Rust JoinSet |
| Cycle detection | Catch bad graphs | ✅ |
| Cancellation token | Abort runs | ✅ |
| Streaming modes (values/updates/messages/custom) | Live UI | ✅ |
| Visualize graph (Mermaid `graph TD`) | Debug | ✅ `StateGraph.to_mermaid()` / `.to_ascii()` (also on `CompiledGraph`); conditional edges shown as `{?}` diamond |

## 8. Persistence + Time Travel

| Feature | Why | Status |
|---|---|---|
| Checkpointer trait | Resumable agents | ✅ |
| Memory checkpointer | Tests | ✅ |
| SQLite checkpointer | Single-host prod | ✅ |
| Postgres checkpointer | Multi-host prod | ✅ |
| Redis checkpointer | Hot-state ephemeral | ✅ |
| State history (list versions) | Debug/replay | ✅ `state_history` |
| Rewind to checkpoint | Undo | ✅ `rewind_to` |
| Fork branch from checkpoint | What-if exploration | ✅ `fork_at` |
| Clear thread | GDPR delete | ✅ `clear_thread` |

## 9. Human-in-the-Loop

| Feature | Why | Status |
|---|---|---|
| `interrupt(payload)` inside node | Pause for approval | ✅ |
| Resume via `Command(resume=...)` | Continue with human input | ✅ |
| `goto` redirect after resume | Reroute mid-graph | ✅ |
| State edit before resume | Correct agent | ✅ via fork |
| Interrupt before/after by node name | Static breakpoints | ✅ |

## 10. Memory / Chat History

| Feature | Why | Status |
|---|---|---|
| `BufferMemory` | Last-N turns | ✅ |
| `TokenBufferMemory` | Trim by token count | ✅ |
| `SummaryBufferMemory` | Summarize old turns | ✅ |
| `VectorStoreMemory` | Topic-relevant recall | ✅ |
| `summarize_conversation` helper | One-shot summary | ✅ |
| SQLite chat history | Single-host durable | ✅ |
| Postgres chat history | Distributed durable | ✅ |
| Redis chat history | Hot ephemeral | ✅ `litgraph-memory-redis::RedisChatHistory` — LIST per session + STRING pin + sessions SET; per-session TTL with `with_ttl`/`set_ttl`; auto-reconnect via `ConnectionManager`; symmetric API to `PostgresChatHistory` / `SqliteChatHistory` |
| Entity memory / KG memory | Deprecated in LC | 🚫 |

## 11. RAG — Retrieval

| Feature | Why | Status |
|---|---|---|
| Vector retriever | Baseline RAG | ✅ |
| BM25 retriever (lexical) | Keyword grounding | ✅ — `Bm25Index::add` runs Rayon-parallel tokenization + per-doc term-counting (iter 198), then merges DF under the write lock. Linear-with-cores indexing throughput on large corpora; search was already Rayon-parallel. |
| Hybrid (RRF) retriever | Best of both | ✅ |
| Reranking retriever (Cohere/Jina/Voyage) | Quality lift | ✅ |
| EnsembleReranker (concurrent reranker fusion) | Reduce per-model bias | ✅ `litgraph_retrieval::EnsembleReranker` — fans N rerankers over the same candidates concurrently via `tokio::join_all`, fuses orderings with weighted RRF (rank-based, scale-free across providers). Python: `litgraph.retrieval.EnsembleReranker(children, weights, rrf_k)`; composes as `RerankingRetriever(base, ensemble)`. (iter 186) |
| Local ONNX reranker (no API key) | Air-gap quality lift | ✅ `litgraph-rerankers-fastembed::FastembedReranker` — ONNX cross-encoder via fastembed; `BGERerankerBase` default (English), `BGERerankerV2M3`/`JINARerankerV2BaseMultilingual` for multilingual; CPU-bound calls in `spawn_blocking`; live-verified rerank picks correct top-1 |
| MaxMarginalRelevance | Diversity | ✅ — `mmr_select` runs Rayon-parallel candidate scoring (iter 203). Each candidate's per-iteration score is independent so the inner loop scales linear-with-cores; deterministic tie-break on lower index keeps picks bit-identical to a sequential reference. |
| ParentDocumentRetriever | Small-chunk match, big-chunk return | ✅ |
| MultiVectorRetriever | N caller-supplied perspectives per parent | ✅ `litgraph_retrieval::MultiVectorRetriever` — caller supplies summaries / hypothetical Qs / chunks per parent; indexing fans out via `embed_documents_concurrent` (iter 183), retrieval dedups by parent_id and returns the parent. Python: `litgraph.retrieval.MultiVectorRetriever(vector_store, embeddings, parent_store)`. (iter 188) |
| MultiQueryRetriever | Query rewriting | ✅ |
| RagFusionRetriever | Query expansion + reciprocal-rank fusion | ✅ `litgraph_retrieval::RagFusionRetriever` — same paraphrase-generation pattern as `MultiQueryRetriever` but fuses via RRF instead of dedup-with-first-wins. Doc that ranks #2 in query A and #1 in query B beats a doc that ranks #1 in query A but isn't found in B. Cormack 2009 RRF (k=60 default), Raudaschl 2023 popularization for retrieval-augmented generation. `with_branch_k(n)` controls per-query depth (default 10) so RRF has more candidates to score across. Use over MultiQuery when cross-paraphrase consistency matters more than the original query's exact ranking. (iter 266) |
| StepBackRetriever | Recall docs at a different abstraction level | ✅ `litgraph_retrieval::StepBackRetriever` — Zheng et al. 2023 step-back prompting adapted for retrieval. LLM generates a more abstract / higher-level "step-back" question; retrieves with both original AND step-back, unions and dedups by id. Distinct from MultiQuery (paraphrases at SAME abstraction level), RagFusion (paraphrases + RRF), and HyDE (hypothetical document, not abstract query). Useful for highly specific factual queries that don't match any document title verbatim but live inside more general docs. Built-in few-shot examples from the paper (`with_include_examples(false)` to omit). Skips the step-back branch if the LLM echoes the original query (degenerate case). (iter 267) |
| SubQueryRetriever | Decompose compound queries into atomic parts | ✅ `litgraph_retrieval::SubQueryRetriever` — splits compound user questions ("Compare X and Y", "What's the relationship between A, B, C") into atomic sub-questions, retrieves for each in parallel, round-robin merges + dedups. Distinct from the rest of the query-expansion family (MultiQuery / RagFusion / StepBack / HyDE) which all paraphrase or abstract a single intent: SubQuery instead splits multi-intent. Round-robin interleaving in the merge step ensures no single sub-query dominates the head of the result list. `with_include_original(true)` to also retrieve for the unsplit query. `with_max_sub_queries(n)` cap (default 4). Falls back to literal query if decomposition is empty. (iter 268) |
| TimeoutRetriever | Per-call deadline on slow vector stores | ✅ `litgraph_retrieval::TimeoutRetriever` — `tokio::time::timeout` wrapping any inner `Retriever`. Mirrors `TimeoutChatModel` / `TimeoutEmbeddings` (iter 194); inner future is dropped on timeout, releasing whatever HTTP/DB connection it held. Returns `Error::Timeout`, which the existing `RetryingChatModel` / `is_transient` classifier treats as retryable — so wrapping a slow retriever inside an outer retry loop just works. Composes through any retriever combinator (Ensemble, RagFusion, RerankingRetriever) — pass it where any `Arc<dyn Retriever>` is accepted. (iter 271) |
| RetryingRetriever | Auto-retry transient retrieval errors | ✅ `litgraph_retrieval::RetryingRetriever` — closes the retry-wrapper trio (chat/embed/retriever). `backon` exponential backoff with jitter; retries `RateLimited`, `Timeout`, and 5xx-pattern provider errors. Stack with `TimeoutRetriever` for "retry on per-call deadline" — `Retrying(Timeout(inner))` is the standard prod pattern for slow vector stores. Default config: 5 attempts, 200ms→10s exponential with 2× factor + jitter — same defaults as `RetryingChatModel` for consistency. Lives in litgraph-retrieval (not -resilience) to avoid a circular dep. (iter 272) |
| ContextualCompressionRetriever | Chunk filtering | ✅ |
| SelfQueryRetriever | LLM extracts metadata filter | ✅ |
| TimeWeightedRetriever | Recent docs first | ✅ |
| HyDE retriever | Hypothetical doc embed | ✅ |
| EnsembleRetriever | Weighted fusion | ✅ `litgraph_retrieval::EnsembleRetriever` — per-child weights, weighted RRF, `tokio::join_all` fan-out. Python: `litgraph.retrieval.EnsembleRetriever`. (iter 181) |
| Doc transformers (MMR, redundant filter, long-context reorder) | Pre-LLM cleanup | ✅ — `mmr_select` (iter 203) and `embedding_redundant_filter` (iter 204) both run Rayon-parallel on their per-candidate inner loops. The greedy outer loops stay sequential (algorithmically required); the parallel inner work scales linear-with-cores for big over-fetch pools. Both verified bit-identical to sequential reference. |

## 12. Vector Stores

| Feature | Why | Status |
|---|---|---|
| In-memory store | Tests/demos | ✅ |
| HNSW (embedded) | Single-host fast | ✅ |
| pgvector | Postgres deployments | ✅ |
| Qdrant | Managed prod | ✅ |
| Chroma | Local dev | ✅ |
| Weaviate | Hybrid features | ✅ |
| Pinecone | SaaS | 🚫 deferred |
| LanceDB | Embedded analytics | 🚫 deferred |
| Metadata filter on search | Multi-tenant RAG | ✅ |

## 13. Embeddings

| Feature | Why | Status |
|---|---|---|
| OpenAI embeddings | Default | ✅ |
| Anthropic embeddings | n/a (Anthropic ships none) | 🚫 |
| Voyage embeddings | Best-in-class | ✅ |
| Cohere embeddings | Multilingual | ✅ |
| Gemini embeddings | Vertex stack | ✅ |
| Bedrock embeddings (Titan) | AWS | ✅ |
| Jina embeddings | OSS option | ✅ |
| fastembed-rs (local, no network) | Air-gapped | ✅ `litgraph-embeddings-fastembed::FastembedEmbeddings` — ONNX-backed, default `bge-small-en-v1.5` 384-dim, batch `embed_documents`, all `EmbeddingModel` variants (BGE/E5/MiniLM/multilingual) selectable via `with_model`; CPU-bound calls run in `spawn_blocking` so async runtime stays free; rustls TLS so no openssl dep |
| Embedding retry/fallback | Prod | ✅ |
| Bounded-concurrency embed batch | Bulk ingestion | ✅ `litgraph_core::embed_documents_concurrent` — chunk-and-fan-out over Tokio + Semaphore, order-preserving, fail-fast. Python: `litgraph.embeddings.embed_documents_concurrent(emb, texts, chunk_size, max_concurrency)`. (iter 183) Plus `embed_documents_concurrent_with_progress` (iter 206) — same shape, updates an `EmbedProgress { total_texts, total_chunks, completed_chunks, completed_texts, errors }` watcher for live bulk-indexing counters. Plus `embed_documents_concurrent_stream` (iter 211) — yields `(chunk_idx, Result<Vec<Vec<f32>>>)` as each chunk completes; bulk indexers can write to vector store on early completers, abort-on-drop kills tail-latency runaway. Plus `embed_documents_concurrent_stream_with_progress` (iter 217) — combined consumer shape (stream + watcher in one call). The chat-batch and embed-batch axes both have all four consumer shapes now (buffered Vec / progress Vec / stream / stream-with-progress). Plus `embed_documents_concurrent_with_shutdown` (iter 228) — partial-progress preservation on Ctrl+C: completed chunks bank as `Ok` so they can flush to the vector store before exit. Plus `embed_documents_concurrent_stream_with_shutdown` (iter 234) — second stream-coordination bridge (after iter 233 chat). One signal stops every parallel embed stream a multi-collection bulk-ingestor owns; per-chunk consumers see a clean prefix + early end. |
| Bounded-concurrency retrieval batch | Eval / agentic many-query flows | ✅ `litgraph_retrieval::retrieve_concurrent` (+ `_fail_fast`) — Tokio Semaphore-bounded fan-out of `Retriever::retrieve` over N caller queries against ONE retriever; aligned output, per-query `Result`. Python: `litgraph.retrieval.retrieve_concurrent(retriever, queries, k, max_concurrency, fail_fast=False)`. (iter 190) Plus `retrieve_concurrent_with_progress` (iter 207) — same shape, updates a `RetrieveProgress { total, completed, docs_returned, errors }` watcher for live multi-query eval counters. Plus `retrieve_concurrent_stream` (iter 212) — yields `(query_idx, Result<Vec<Document>>)` as each query completes; abort-on-drop kills in-flight queries. Plus `retrieve_concurrent_stream_with_progress` (iter 218) — combined consumer shape (stream + watcher in one call). All four consumer shapes shipped per axis. Plus `retrieve_concurrent_with_shutdown` (iter 229) — partial-progress preservation on Ctrl+C: completed queries bank as `Ok`, in-flight queries become `Err("cancelled by shutdown")`. Plus `retrieve_concurrent_stream_with_shutdown` (iter 235) — stream-coordination bridge extended to retriever axis (3/6 stream axes bridged). One signal stops every parallel retriever stream a multi-query eval driver owns; consumer sees a prefix + clean end. |
| Bounded-concurrency tool dispatch | Plan-and-Execute / orchestrators outside React loop | ✅ `litgraph_core::tool_dispatch_concurrent` (+ `_fail_fast`) — heterogeneous `(tool, args)` calls fan out under Semaphore; aligned output, per-call `Result`, unknown-tool errors isolated. Python: `litgraph.agents.tool_dispatch_concurrent(tools, calls, max_concurrency, fail_fast=False)`. (iter 191) Plus `tool_dispatch_concurrent_with_progress` (iter 208) — `ToolDispatchProgress { total, completed, errors, unknown_tool_errors }` watcher; unknown-tool failures bucketed separately so dashboards can distinguish a routing-bug regression from runtime tool errors. Plus `tool_dispatch_concurrent_stream` (iter 213) — yields `(call_idx, Result<Value>)` as each call completes; abort-on-drop kills in-flight tool calls. Plus `tool_dispatch_concurrent_stream_with_progress` (iter 219) — combined consumer shape (stream + watcher in one call). All four consumer shapes shipped per axis. Plus `tool_dispatch_concurrent_with_shutdown` (iter 230) — partial-progress preservation on Ctrl+C: completed tools bank as `Ok` so agent context stays consistent with what's actually been done. Plus `tool_dispatch_concurrent_stream_with_shutdown` (iter 236) — stream-coordination bridge extended to tool axis (4/6 stream axes bridged). One signal stops every parallel tool-dispatch stream a Plan-and-Execute orchestrator owns; consumer sees a clean prefix + early end. |
| Bounded-concurrency rerank batch | Eval / batch-rerank flows | ✅ `litgraph_retrieval::rerank_concurrent` (+ `_fail_fast`) — Tokio Semaphore-bounded fan-out of `Reranker::rerank` over N `(query, candidates)` pairs against ONE reranker; aligned output, per-pair `Result`. Adds a fifth axis to the parallel-batch family (chat/embed/retrieve/tool/rerank). Python: `litgraph.retrieval.rerank_concurrent(reranker, pairs, top_k, max_concurrency, fail_fast=False)`. (iter 197) Plus `rerank_concurrent_with_progress` (iter 209) — closes the progress-aware family across all six axes. `RerankProgress { total, total_candidates, completed, docs_returned, errors }`. Plus `rerank_concurrent_stream` (iter 214) — yields `(pair_idx, Result<Vec<Document>>)` as each pair completes; abort-on-drop kills in-flight rerank calls. Plus `rerank_concurrent_stream_with_progress` (iter 220) — combined consumer shape (stream + watcher in one call). All four consumer shapes shipped per axis. Plus `rerank_concurrent_with_shutdown` (iter 231) — partial-progress preservation on Ctrl+C. Plus `rerank_concurrent_stream_with_shutdown` (iter 237) — stream-coordination bridge extended to rerank axis (5/6 stream axes bridged; loader axis remains). One signal stops every parallel rerank stream a multi-stage eval driver owns; consumer sees a prefix + clean end. |
| Zero-copy numpy interop | Speed | ✅ rust-numpy |

## 14. Document Loaders

24 loaders shipped — covers the high-value LangChain set.

| Loader | Status |
|---|---|
| Text / Markdown / JSON / JSONL / CSV / HTML / PDF / DOCX | ✅ |
| Directory (parallel, Rayon) | ✅ |
| Web / Sitemap | ✅ |
| Notion · Slack · Confluence | ✅ |
| GitHub issues + files · GitLab issues + files | ✅ |
| Linear · Jira · Gmail · GDrive | ✅ |
| S3 / R2 / B2 / MinIO | ✅ |
| Jupyter notebook | ✅ |
| Discord / Telegram / WhatsApp | 🟡 `litgraph_loaders::DiscordChannelLoader` (REST `/messages` paginated via `before` cursor; bot/bearer auth; per-message author/timestamp/attachments/mentions metadata; oldest-first delivery; capped fetch). Telegram bot API is push-only — no usable history loader. WhatsApp still pending. |
| YouTube transcript / Vimeo | 🟡 `litgraph_loaders::YouTubeTranscriptLoader` (timedtext endpoint, no auth/key; full transcript → content, per-cue start_ms/dur_ms in metadata; URL/short/embed/shorts/live/bare-id all extract). Vimeo still pending. |
| arXiv / Wikipedia / PubMed | ✅ `litgraph_loaders::ArxivLoader` (Atom), `WikipediaLoader` (MediaWiki Action API), `PubMedLoader` (NCBI E-utilities — esearch+efetch, structured-abstract section labels preserved, MeSH terms, DOI/PMCID, normalised pub_date, API-key support) |
| Office365 / Outlook | ✅ `litgraph_loaders::OutlookMessagesLoader` — Microsoft Graph `/me/messages`, bearer auth, folder/search/filter narrowing, `@odata.nextLink` pagination, `Prefer: outlook.body-content-type="text"` header so embedders see prose not HTML, subject+body concatenated for content, full sender/recipient/conversation metadata |
| Concurrent multi-loader fan-out | Parallel ingestion across many sources | ✅ `litgraph_loaders::load_concurrent` (+ `_flat`) — bounded-concurrency `Loader::load()` fan-out via Tokio `spawn_blocking` + `Semaphore`; aligned output, per-loader `Result`. Python: `litgraph.loaders.load_concurrent(loaders, max_concurrency, fail_fast=False)`. (iter 187) Plus `load_concurrent_stream` (iter 215) — yields `(loader_idx, LoaderResult<Vec<Document>>)` as each loader finishes; ingest dashboards can render each source as it lands; abort-on-drop kills in-flight `spawn_blocking` work. Plus `load_concurrent_with_progress` + `load_concurrent_stream_with_progress` (iter 221) — close the four-quadrant consumer matrix for the loader axis. `LoadProgress { total, completed, docs_loaded, errors }` for live ingest counters. Plus `load_concurrent_with_shutdown` (iter 232) — sixth and final coordination bridge: partial-progress preservation extended to the loader axis. All 6 parallel-batch axes now bridge to `ShutdownSignal` (chat/embed/retriever/tool/rerank/loader). Plus `load_concurrent_stream_with_shutdown` (iter 238) — **closes the stream+coord bridge family**: every parallel-batch axis now exposes BOTH Vec+shutdown AND stream+shutdown variants. Eight consumer shapes per axis ship across the matrix (buffered Vec / progress Vec / Vec+shutdown / stream / stream+progress / stream+shutdown). |
| Backpressured ingestion pipeline | One call: load → split → embed → stream | ✅ `litgraph_loaders::ingest_to_stream` — three-stage Tokio pipeline (loaders, splitter closure, embedder) connected by bounded `mpsc` channels. Each stage runs concurrently — while loaders pull later sources, the splitter is already chopping earlier ones, and the embedder is batching the first chunks. Per-stage failures surface as `Err` items on the output stream without short-circuiting. (iter 196) |
| Pipeline progress observability | UI bar / dashboard / stuck-stage detection | ✅ `litgraph_loaders::ingest_to_stream_with_progress(...)` — composes iter 196 (pipeline) + iter 199 (`Progress<T>`). Pipeline updates an `IngestProgress` struct (`loaders_done`, `docs_loaded`, `chunks_split`, `chunks_embedded`, `batches_emitted`, error counts) that any number of observers can snapshot mid-flight. (iter 200) |

## 15. Splitters

| Feature | Status |
|---|---|
| RecursiveCharacterTextSplitter (lang-aware) | ✅ |
| MarkdownHeaderTextSplitter | ✅ |
| HTMLHeaderTextSplitter | ✅ |
| JSONSplitter | ✅ |
| SemanticChunker (embedding-based) | ✅ |
| CodeSplitter (definition-boundary) | ✅ |
| TokenTextSplitter (exact tokens) | ✅ |
| MarkdownTableSplitter (preserve GFM tables) | ✅ `litgraph_splitters::MarkdownTableSplitter` (iter 269) — walks the markdown line-by-line, detects GFM table blocks (`| header |` row + `| --- |` separator + N data rows), emits each table as a SINGLE chunk regardless of size. Non-table prose passes through an inner splitter (default `RecursiveCharacterSplitter` with chunk_size 1000, overlap 200). Solves the embedding-comprehension problem where naive splitters fragment tables mid-row. Optional `with_max_table_chars(Some(n))` enables row-wise fragmentation for tables larger than `n` chars while preserving the header + separator on every fragment — graceful degradation for tables that exceed the embedding context window. Recognizes alignment separators (`:---`, `---:`, `:---:`). Borderless / non-GFM tables (without the `\|---\|` row) intentionally NOT recognized — that ambiguous case is left to `MarkdownHeaderSplitter`. |
| CsvRowSplitter (row-aware CSV chunks) | ✅ `litgraph_splitters::CsvRowSplitter` (iter 270) — splits CSV text into chunks of `rows_per_chunk` data rows (default 100), preserving the header row on every chunk so columns stay interpretable downstream. RFC-4180 quoted-field handling: cells wrapped in `"..."` containing literal newlines stay intact (the row-walker tracks an in-quote flag). CRLF line endings normalized. `with_header(false)` for headerless CSV (treats every line as a data row). Solves the same problem as `MarkdownTableSplitter` for CSV inputs: existing char/token splitters break rows mid-cell, destroying the column-position semantics. |
| Sentence/NLTK/SpaCy splitters | ✅ `litgraph_splitters::SentenceSplitter` (iter 287) — rule-based sentence-boundary splitter; no ML model dependency. Handles common false positives that defeat naive `split(".")`: titles (Dr., Mr., Prof.), Latin abbreviations (e.g., i.e., etc.), business suffixes (Inc., Ltd., Co.), acronyms (U.S., U.S.A., U.K.), decimal numbers (3.14), ellipses (...), trailing close-quote/paren attached to prior sentence. ~95% coverage of clean prose; for adversarial input (clinical notes, legal docs) use a Punkt-style ML splitter as a separate dep. `with_min_length(n)` glues short fragments onto the prior sentence so callers don't get `.` or `1.` as standalone "sentences." |
| JSONL splitter | One chunk per JSONL record (logs, OpenAI fine-tune format) | ✅ `litgraph_splitters::JsonLinesSplitter` (iter 288) — split NDJSON / JSONL inputs into one chunk per record. Distinct from `JsonSplitter` (which navigates a single JSON tree). Real prod use: structured-log streams (Datadog/Loki/ELK NDJSON exports), OpenAI fine-tune `{"messages": [...]}` format, HuggingFace streaming datasets, replay traces. `with_pretty(true)` re-formats each line as multi-line JSON for readability; `with_skip_invalid(true)` silently drops lines that don't parse (default keeps them verbatim so issues surface to the caller). Handles CRLF line endings. Skips blank lines. Round-tripped against an OpenAI fine-tune fixture. |
| HTML section splitter | Semantic-block splitting (`<article>`, `<section>`, etc) | ✅ `litgraph_splitters::HtmlSectionSplitter` (iter 293) — splits HTML by HTML5 semantic-block tags. Default tag set: `article`, `section`, `main`, `aside`, `nav`, `header`, `footer`. Distinct from `HtmlHeaderSplitter` (which splits by `<h1>`/`<h2>` hierarchy): use this when content uses semantic-tag conventions instead of heading-based hierarchy (modern blog templates, docs sites, articles published with HTML5 structure). Handles nested same-tag pairs by depth-tracking — only the OUTER section emits as one chunk; nested sections become part of the outer's content. `with_tags(["article", "div.post"])` overrides the default set; `with_drop_outside(true)` excludes content not inside any matched section. Strips `<script>` / `<style>` / comments before processing. Case-insensitive tag matching. `split_to_documents()` returns Documents with `tag` and `section_index` metadata. |

## 16. Caching

| Feature | Why | Status |
|---|---|---|
| In-memory LLM cache | Dev | ✅ |
| SQLite cache | Single-host | ✅ |
| Redis cache | Distributed cross-process | ✅ |
| Embedding cache | Skip dup embeds | ✅ |
| SQLite embedding cache | Persistent | ✅ |
| Semantic cache (cosine threshold) | Reuse near-dupes | ✅ |

## 17. Evaluation

| Feature | Why | Status |
|---|---|---|
| Eval harness (`run_eval`) | Golden-set runner | ✅ |
| Bounded parallel eval | Speed | ✅ `max_parallel` |
| String evaluators (10) | Cheap auto-grading | ✅ exact_match, levenshtein, jaccard, regex, json_validity, contains_all/any, embedding_cosine |
| LLM-as-judge | Quality grading | ✅ `LlmJudge`, `LlmJudgeScorer` |
| Trajectory evaluators | Agent path grading | ✅ `litgraph.evaluators.evaluate_trajectory(actual, expected, policy)`; policies: `contains_all`, `exact_order`, `subsequence` (LCS), `levenshtein` |
| Pairwise comparison | A/B model | ✅ `litgraph.evaluators.PairwiseEvaluator(model, criteria=None)` — returns `{winner, confidence, reason}`; deterministic order randomization for position-bias mitigation |
| Synthetic data generation | Bootstrap eval set | ✅ `litgraph.evaluators.synthesize_eval_cases(seeds, model, target_count, criteria=None)` — LLM-driven structured-output expansion of seed cases; dedups against seeds, caps at `target_count`, drops empty inputs |
| Dataset versioning | Track regressions | ✅ `litgraph_core::{DatasetManifest, RunRecord, RunStore, regression_check, record_and_check}` — BLAKE3 fingerprint over canonicalised cases (order-stable, metadata-ignored), `InMemoryRunStore` + `JsonlRunStore` (append-atomic JSONL, restart-survival), per-scorer regression alerts with tolerance, fingerprint-mismatch suppresses noise on dataset edits |

## 18. Observability

| Feature | Why | Status |
|---|---|---|
| `tracing` spans per node/tool/llm | Structured logs | ✅ |
| OpenTelemetry OTLP exporter | APM integration | ✅ `litgraph-tracing-otel` |
| `init_stdout` for dev | Local debug | ✅ |
| LangSmith OTel shim | LC users migration | ✅ `init_langsmith` |
| Cost tracker | $ accounting | ✅ |
| Graph events / agent events | UI progress | ✅ |
| `on_request` / `on_response` hooks | Inspect HTTP body | ✅ |
| Token usage events | Per-call accounting | ✅ |
| Datadog / NewRelic native | OTLP covers it | ✅ via OTLP |
| Phoenix (Arize) integration | Trace UI | ✅ via OTLP |
| In-process metrics aggregation | Cheap counters/gauges/histograms for `/metrics` endpoint | ✅ `litgraph_core::MetricsRegistry` (iter 260) — atomic Counter / Gauge / Histogram primitives keyed by name. Lock-free hot-path updates (AtomicU64 / AtomicI64 / per-bucket atomics + f64-bits CAS for histogram sum). Get-or-create lookup so multiple call sites share one underlying instrument. `to_prometheus()` renders text exposition format suitable for direct response from a Prometheus scrape endpoint. Distinct from the tracing/OTel layer (per-request span propagation): metrics is for "rate of X across the whole process right now" aggregates that don't need per-request correlation. Real prod use: agent loop counters (`tool_calls_total{name="…"}`), in-flight gauges via wrapper inc-on-entry/dec-on-exit, latency histograms with configurable buckets. |
| Auto-instrumented chat + embed | Drop-in `/metrics` for any wrapped model | ✅ `litgraph_resilience::{MetricsChatModel, MetricsEmbeddings}` — bridges the iter-260 `MetricsRegistry` into the resilience family. Auto-bumps four metrics on every call: `<prefix>_invocations_total` (counter), `<prefix>_errors_total` (counter), `<prefix>_in_flight` (gauge with RAII guard so it stays correct under cancellation/panic), `<prefix>_latency_seconds` (histogram, default `[0.005..30]` geometric buckets). Default prefix `chat` / `embed`; `with_prefix` for per-model labeling (`openai_gpt4_invocations_total` etc); `with_buckets` for custom histogram buckets. Metric handles pre-resolved at construction so the hot path is pure atomic ops — no HashMap lookup per call. Streaming: same metrics recorded at handshake (per-token timing remains the consumer's responsibility). (iter 261) |
| Auto-instrumented tools | Per-tool metrics for agent debugging | ✅ `litgraph_resilience::MetricsTool` — third axis of metrics auto-instrumentation. Same four-metric shape as chat+embed siblings. Default prefix is the tool's own name (sanitized for Prometheus naming rules — `http.get` → `http_get`); `with_prefix` overrides for explicit labels. Per-tool metrics are especially valuable for agent debugging since agents make many tool calls per session — knowing which tools fail / are slow / are hot is the first thing you want from a `/metrics` dashboard. (iter 262) |
| Auto-instrumented retrievers | RAG observability — vector-store latency / error tracking | ✅ `litgraph_retrieval::MetricsRetriever` — fourth and final axis of metrics auto-instrumentation. Same four-metric shape (invocations / errors / in_flight / latency) as siblings. Default prefix `retrieve`; `with_prefix("pgvector")` / `with_prefix("qdrant")` etc for per-store labeling so multi-store deployments produce distinct metric series. Histogram defaults are tighter than chat/embed (`[0.001, 0.0025, ..., 10.0]` seconds) since retrieval is usually faster than LLM calls. Lives in litgraph-retrieval (not -resilience) to avoid a circular dep. With this iter the metrics-instrumentation matrix covers all four primary axes: chat (261), embed (261), tool (262), retrieve (273). (iter 273) |

## 19. Deployment / Serve

| Feature | Why | Status |
|---|---|---|
| LangServe REST endpoints | Quick HTTP API | ✅ `litgraph-serve::serve_chat(model, addr)` — axum-backed; `/invoke`, `/stream` (SSE + `[DONE]` sentinel), `/batch`, `/health`, `/info`. `router_for(model)` returns the bare `Router` for tower middleware (CORS/auth/rate-limit). |
| LangGraph Cloud / Platform | Hosted runtime | 🚫 out of scope |
| FastAPI integration example | DIY serve | 🟡 native `litgraph-serve` covers the use case; standalone FastAPI example deferred |
| WebSocket streaming | Live UI | 🟡 user wires it |
| MCP server | Expose agent as MCP | ✅ |

## 20. Multi-modal

| Feature | Status |
|---|---|
| Text in/out | ✅ |
| Image in (vision) | ✅ |
| Image out (DALL-E) | ✅ |
| Audio in (Whisper STT) | ✅ |
| Audio out (TTS) | ✅ |
| Video in | ❌ |

## 21. Free-threaded Python 3.13

| Feature | Why | Status |
|---|---|---|
| `py.detach` around heavy work | Real parallelism | ✅ everywhere |
| abi3 wheels (cp39+) | Wide compat | ✅ maturin |
| `.pyi` stubs (pyo3-stub-gen) | IDE autocomplete | 🟡 hand-written stubs in `litgraph-stubs/`, no auto-gen |
| Free-threaded build tested | 3.13t support | ✅ FREE_THREADING.md |

---

# LangChain 1.0 / LangGraph 1.1 — 2025-2026 surface

LangChain shipped a full rewrite (Sep 2025, "v1.0", skipped 0.3) + LangGraph
1.1 (2026). New primitives below — mapped to litGraph status.

## 22. Middleware (LC 1.0 — context engineering)

LangChain 1.0 reframes the agent loop around middleware (Express-style hooks
before/during/after model calls). Powers prompt caching, conversation
compression, tool-result offload, context quarantine.

| Feature | Why | Status |
|---|---|---|
| `before_model` hook | Mutate messages pre-call | ✅ `AgentMiddleware::before_model` |
| `after_model` hook | Mutate response post-call | ✅ `AgentMiddleware::after_model` |
| `before_tool` / `after_tool` | Wrap tool calls | 🟡 `RetryTool`/`TimeoutTool` cover specific cases |
| Composable middleware chain | Stack hooks declaratively | ✅ `litgraph.middleware.MiddlewareChain` (onion order: before in-order, after reversed) |
| `MiddlewareChat` adapter | Plug chain into any `ChatModel` | ✅ accepted by `ReactAgent`/`SupervisorAgent`/etc. |
| Prompt caching middleware | Auto-mark cache breakpoints | ✅ `PromptCachingChatModel` (wrapper, not yet ported to chain) |
| Conversation compression middleware | Trim long context | ✅ `SummaryBufferMemory` + `MessageWindowMiddleware` |
| `SystemPromptMiddleware` | Idempotent system prompt injection | ✅ |
| `LoggingMiddleware` | `tracing` events around every call | ✅ |
| Tool-result offload middleware | Push large outputs to filesystem/store | ✅ `litgraph_core::OffloadingTool` wraps any `Tool`; oversized results go to `OffloadBackend` (in-memory or filesystem; pluggable trait). Returns a `{_offloaded, handle, size_bytes, preview, tool}` marker so the model still has context. `resolve_handle()` + `is_offloaded_marker()` for fetch-back; default 8 KiB threshold tunable per tool. |
| Context quarantine (subagent) | Isolate sub-task context | 🟡 `SupervisorAgent` provides isolation |
| Dynamic system prompt assembly | Per-call system-prompt builder | 🟡 `ChatPromptTemplate.compose` |

## 23. Deep Agents harness (LC 1.0)

`deepagents` package: harness layered on agent + LangGraph runtime. Adds
planning tool, virtual filesystem, subagent spawning, AGENTS.md memory file,
skills directory, prompt-caching middleware.

| Feature | Why | Status |
|---|---|---|
| Planning tool (todo write/read) | Agent self-organizes long tasks | ✅ `litgraph.tools.PlanningTool` (list/add/set_status/update/clear; status: pending/in_progress/done/cancelled) |
| Virtual filesystem backend | Sandboxed scratch space across turns | ✅ `litgraph.tools.VirtualFilesystemTool` (read/write/append/list/delete/exists; size cap; `..` rejected) |
| Subagent spawn primitive | Delegate to scoped sub-agent | ✅ `litgraph.tools.SubagentTool(name, desc, react_agent)` — parent gets a tool that runs the inner ReactAgent in isolated context per call |
| AGENTS.md / memory files loader | Persistent system-prompt context | ✅ `litgraph.prompts.load_agents_md(path)` |
| Skills directory loader | Domain-specific prompt packs | ✅ `litgraph.prompts.load_skills_dir(dir)` (YAML frontmatter for `name`/`description`, sorted, hidden + non-`.md` skipped) |
| `SystemPromptBuilder` | Assemble base + AGENTS.md + skills into system prompt | ✅ `litgraph.prompts.SystemPromptBuilder` |
| Anthropic prompt-caching middleware | Cost cut on long contexts | ✅ |
| Async subagents | Concurrent sub-tasks | ✅ Rust JoinSet (Supervisor) |
| Multi-modal subagent inputs | Image/audio in subagent | ✅ |
| Human-in-the-loop middleware | Approval gate | ✅ `interrupt` + `Command(resume)` |

## 24. Functional API (`@entrypoint` + `@task`)

LangGraph alternative to StateGraph DSL — plain Python control flow with
checkpointer + interrupt support.

| Feature | Why | Status |
|---|---|---|
| `@entrypoint` decorator (workflow root) | Skip graph DSL | ❌ |
| `@task` decorator (async work unit) | Future-like subtask | ❌ |
| `previous` thread-state access | Resume across calls | ✅ via Checkpointer + thread_id |
| Checkpointer compatibility | Resume mid-flow | ✅ wrap user fn would work, no decorator sugar |
| Sync + async function support | Unified API | ✅ at graph level |

## 25. Type-safe streaming v2 (LangGraph 1.1)

| Feature | Why | Status |
|---|---|---|
| Self-describing `StreamPart` (type discriminator) | Type narrow per chunk | 🟡 events have `kind`, not Pydantic-coerced |
| Pydantic / dataclass coercion of state chunks | IDE types in stream | ❌ Python state stays `dict` |
| Dedicated `interrupts` field on values stream | Clean state, no `__interrupt__` pollution | ✅ events carry interrupts separately |
| `stream_version="v2"` opt-in | Backwards compat | n/a — single stream API |
| Cleaner interrupt access | `for part in stream:` | ✅ |

## 26. Long-term memory Store (LangGraph)

Distinct from short-term checkpointer — JSON document store keyed by
`(namespace, key)` for cross-thread / cross-session memory.

| Feature | Why | Status |
|---|---|---|
| `BaseStore` namespace+key API | Multi-tenant long-term mem | ✅ `litgraph_core::store::Store` trait |
| `InMemoryStore` (dev) | Local prototyping | ✅ `litgraph.store.InMemoryStore` |
| `PostgresStore` (prod) | Durable distributed | ✅ `litgraph-store-postgres::PostgresStore` — `TEXT[]` namespace + GIN index, JSONB values, per-item TTL with lazy + manual `evict_expired()` sweep, SQL-side `query_text` ILIKE + JSON-path `#>` matches (up to 8 clauses, falls back to client-side beyond), shared deadpool with checkpointer |
| Vector-indexed semantic search on Store | Memory recall by meaning | ✅ `litgraph_core::SemanticStore` — wraps any `Store` with an `Embeddings` provider; `put(ns, key, text, value)` embeds + stores, `semantic_search(ns, query, k)` ranks Rayon-parallel cosine. Python: `litgraph.store.SemanticStore(store, embedder)`. (iter 185) Plus `bulk_put` (iter 222), `bulk_delete` (iter 223), `bulk_get` (iter 224) — closes the full LangGraph `BaseStore::{mset, mdelete, mget}` parity trio. Bulk indexer embeds via iter 183's chunked Tokio fan-out; bulk fetches surface per-key shape errors distinctly so corrupt rows don't tank the whole batch. |
| `LangMem` SDK (episodic memory) | Auto-extract memories | ✅ `litgraph_core::{EpisodicMemory, MemoryExtractor, Memory}` — LLM extraction via structured output, kind/importance/source_thread metadata, importance threshold filtering, namespaced storage on any `Store` impl, `recall(query, k)` + `recall_as_system_message` ready-to-prepend |
| TTL on memory entries | Auto-expire stale | ✅ `ttl_ms` per put, lazy eviction on read/search |
| Per-user namespace isolation | GDPR / multi-tenant | ✅ namespace tuple + prefix search |
| `put` / `get` / `delete` / `search` / `list_namespaces` ops | CRUD on long-term mem | ✅ |
| `pop(ns, key)` convenience | Atomic get+delete | ✅ Python-only |
| JSON-pointer match filter | Field-eq filtering on search | ✅ `matches=[("/role", "admin")]` |
| `query_text` substring filter | Cheap full-text scan | ✅ case-insensitive |

## 27. Pydantic 2 / typed state

| Feature | Why | Status |
|---|---|---|
| Pydantic 2 internals | Faster, stricter validation | n/a Rust-side uses serde + schemars |
| Pydantic state schema in graph | Typed reads/writes | 🟡 Python state via `dict`, Rust via `#[derive(GraphState)]` |
| Auto Pydantic→JSON-Schema for tools | Tool args validation | ✅ schemars side |
| Zero compatibility shims | Direct user-Pydantic | ✅ Pydantic 2 only target |

## 28. Agent-as-API patterns (deployment)

| Feature | Why | Status |
|---|---|---|
| LangServe drop-in REST | One-line serve | ✅ `litgraph-serve::serve_chat` |
| LangGraph Server | Hosted runtime + UI | 🚫 out of scope |
| LangGraph Studio (visual debugger) | Step-debug graph | 🟡 `litgraph-serve --features studio` ships REST debug endpoints over any `Checkpointer` — `/threads/:id/state`, `/history`, `/checkpoints/:step` (base64 state), `/rewind`, `DELETE /threads/:id`. Drop-in for any UI; bring-your-own front-end. |
| Assistants API (LangGraph) | Per-graph config snapshots | ✅ `litgraph_core::{AssistantManager, Assistant, AssistantPatch}` — CRUD + monotonic version bumping, immutable `<id>@v<n>` archives for audit history, `get_version` lookup, scoped per `graph_id`, backed by any `Store` impl (InMemory / PostgresStore) |
| Webhook resume after interrupt | External system → resume | ✅ `litgraph_core::ResumeRegistry` (iter 201) + `litgraph_serve::resume::resume_router` (iter 202) — `tokio::sync::oneshot` coordination underneath, axum router on top. `POST /threads/:id/resume {value}` delivers, `DELETE /threads/:id/resume` cancels, `GET /resumes/pending` lists. Python: `litgraph.observability.ResumeRegistry`. |
| Graceful shutdown coordination | Wake N worker tasks on Ctrl+C / drain | ✅ `litgraph_core::ShutdownSignal` (iter 225) — `tokio::sync::Notify`-backed N-waiter edge signal with a "fired" flag so late waiters resolve instantly after signal. Distinct from `oneshot` (one waiter, one value) and `broadcast` (queued events). Python: `litgraph.observability.ShutdownSignal`. Plus `until_shutdown(fut, &shutdown)` (iter 226) — composable future combinator that races any future against the signal; drops the inner on shutdown so HTTP / DB / sleep resources held inside get released promptly. Plus `Barrier` (iter 239) — sixth distinct channel shape: wait-for-N rendezvous. `Barrier::new(n)` requires N participants to call `wait()`; the N-th arrival unblocks every pending waiter simultaneously. Late arrivals past N return instantly. Shutdown-aware variant `wait_with_shutdown(&shutdown)` returns `Some(())` on release, `None` if the signal fires first — pending waiters wake instead of parking forever when the orchestrator abandons a synchronized step. Real prod use: coordinated agent rounds (5 agents finish their step in parallel, all unblock together for next round), warm-up rendezvous (N workers each load model weights, then start serving in lockstep), phase synchronization (pipeline stage N+1 can't begin until every item of stage N has finished). Plus `CountDownLatch` (iter 240) — sister primitive to Barrier with **decoupled signaling**: producers call `count_down()` (no wait) when work finishes, observers call `wait()` (no decrement) for the count to hit zero. Right when producers and observers are different roles. Spawn N background workers, hold a clone of the latch, await `wait()` to know everyone returned — no `JoinHandle` tracking required. Shutdown-aware variant included. Plus `KeyedMutex<K>` (iter 241) — per-key async serialization. Different keys run in parallel; same-key callers queue. Uses `Weak` references so entries clean themselves up when no caller holds the lock and no one is waiting; `cleanup()` drops stale `Weak`s for unbounded-key workloads (ephemeral request IDs). Real prod use: per-thread agent serialization (ReAct step for `thread_id=X` finishes before next step for `X` runs; thousands of threads independent), per-user rate-coupling, per-resource exclusivity (one writer per vector-store collection / shard). Plus `RateLimiter` (iter 242) — async token-bucket primitive distinct from `RateLimitedChatModel` (which wraps a single `ChatModel`). Reusable: any caller — chat / embed / tool / loader — can charge against one shared budget. Lazy-refill (no background task), bursting allowed up to `capacity`, sustained rate capped at `refill_per_sec`. `try_acquire` non-blocking; `acquire` blocks until tokens accumulate; `acquire_with_shutdown` cancellable (tokens NOT deducted on shutdown so budget is preserved for surviving callers). Asks larger than capacity are clamped (otherwise would block forever). Real prod use: shared OpenAI quota across 5 agents, egress traffic shaping, paired with `KeyedMutex` for per-user fairness. Plus `CircuitBreaker` (iter 243) — three-state resilience primitive (Closed / Open / HalfOpenProbing) with consecutive-failure threshold + cooldown. Distinct from `RetryingChatModel` which retries on individual call errors: the breaker stops retrying when an upstream is *down*, giving it room to heal. Composable: wrap any future-returning closure in `breaker.call(|| f()).await`. Half-open probe semantics: exactly one probe allowed at a time; concurrent callers see `CircuitOpen` until probe completes; probe success closes the breaker, failure re-opens with fresh cooldown. Manual `trip(cooldown)` / `reset()` for ops runbooks. Real prod use: provider outage isolation, vector-store quarantine, tool blast-radius limiter. Plus `Bulkhead` (iter 248) — concurrent-call cap with **rejection** semantics (named after the "Release It!" pattern: separate failure domains so one saturated dependency doesn't drown the process). Distinct from a plain Semaphore: at-cap callers can `try_enter` for instant rejection (signal "this dependency is saturated, shed load") rather than queueing forever. Three modes: `try_enter` non-blocking, `enter` blocking (Semaphore-equivalent), `enter_with_timeout(t)` block up to t then reject. Tracks `rejected_count` for telemetry. Real prod use: per-tool concurrent cap (5 in-flight max for flaky API; 6th gets BulkheadFull so agent picks a different action), vector-store connection budget, outbound HTTP fan-out cap. Plus `hedged_call(primary, backup, hedge_delay)` (iter 250) — tail-latency mitigation combinator. Run primary alone for `hedge_delay`; if not done, ALSO start backup and race them. Distinct from `RaceChatModel` (iter 184) which issues to all simultaneously: hedge only pays the second-call cost on slow tail requests. Standard pattern from "The Tail At Scale" (Dean & Barroso 2013). Loser future is dropped, releasing held resources via tokio cancellation. Real prod use: LLM with 500ms p50 / 30s p99 — set hedge_delay = 2s; calls under 2s pay zero overhead, slow tail covered by backup. Plus `Singleflight<K, V>` (iter 252) — request-coalescing primitive (Go's `golang.org/x/sync/singleflight`). When N concurrent callers ask for the same key, only ONE inner computation runs; the others await the leader's result via `tokio::sync::broadcast`. `V: Clone` constraint (use `Arc<T>` for expensive types so broadcast clones cheaply). Solves cache-miss thundering-herd: 100 agent requests for the same embedding → 1 HTTP call, 100 results. Real prod use: embedding cache priming, idempotent tool-result coalescing, lazy initialization of shared resources. Plus `PriorityQueue<T>` (iter 259) — async work queue with `u64`-priority pop. Distinct from `tokio::sync::mpsc` (FIFO across the whole channel): a high-priority task pushed late jumps ahead of earlier-pushed work. Within the same priority, FIFO order is preserved via an insertion sequence number. Three modes: `try_pop` non-blocking, `pop` blocking, `pop_with_shutdown` cancellable. Real prod use: urgent retries first (a graph node that failed and is being re-scheduled jumps ahead of fresh work), hard cases first (eval harness scores most-likely-to-fail rows first so cancelled runs surface failures fastest), latency-budget UI requests (`priority=10` jumps the batch). |

---

## What's left for "Claude Code can ship a prod agent without coding plumbing"

Top gaps to close, ranked by user-impact for a no-code-glue path:

1. ✅ **Long-term memory `Store`** — core trait + `InMemoryStore` + `PostgresStore` shipped; `SemanticStore` (iter 185) adds Rayon-parallel cosine semantic-search recall on top of any `Store`.
2. 🟡 **Middleware chain primitive** — `before/after_model` chain shipped (`litgraph.middleware`, 7 Py + 6 Rust tests). Built-ins: Logging, MessageWindow, SystemPrompt. `before/after_tool` hooks + tool-result offload still pending.
3. ✅ **Deep Agents harness** — `PlanningTool` + `VirtualFilesystemTool` + `load_agents_md` + `load_skills_dir` + `SystemPromptBuilder` + `SubagentTool` + one-call `litgraph.deep_agent.create_deep_agent(...)` factory all shipped (43 Rust + 41 Py tests across the seven).
4. ❌ **Functional API** (`@entrypoint` + `@task`) — Python decorator alternative to graph DSL. Trims LOC for simple workflows.
5. ❌ **Pydantic-coerced state in Python** — type-safe stream chunks, IDE-narrow types. (Rust side already typed.)
6. ❌ **`pyo3-stub-gen` auto-stubs** — manual stubs go stale. Pyright import warnings hurt agent-authored code.
7. ✅ **fastembed-rs local embeddings** — `litgraph-embeddings-fastembed::FastembedEmbeddings` ships ONNX-backed local embeddings; default `bge-small-en-v1.5`, all fastembed models selectable.
8. ❌ **candle / mistral.rs local chat** — full offline agent.
9. ✅ **LangServe-style HTTP serve crate** — `litgraph-serve::serve_chat(model, addr)` ships REST + SSE in one call. (CLI wrapper still pending.)
10. ✅ **Graph visualizer (Mermaid)** — `to_mermaid()` + `to_ascii()` on StateGraph + CompiledGraph (8 Rust + 9 Py tests). PNG render still pending (out-of-process via `mmdc` or `kroki`).
11. ✅ **Eval coverage** — trajectory evaluator, `PairwiseEvaluator`, and `synthesize_eval_cases` shipped. Eval suite covers golden-set runs, trajectory grading, A/B judging, and seed-based dataset synthesis.
12. 🟡 **Discord/YouTube loaders** — long-tail integrations remaining. (Redis chat history shipped iter 164. arXiv + Wikipedia loaders shipped iter 165. PubMed loader shipped iter 166.)

## Quick prod-ready agent recipe (uv, no venv)

```bash
uv pip install litgraph
```

```python
# agent.py — Claude Code can write this end-to-end
from litgraph.providers import AnthropicChat
from litgraph.tools import HttpRequest, BraveSearch, Calculator
from litgraph.agents import ReactAgent
from litgraph.observability import CostTracker
from litgraph.tracing import init_otlp

init_otlp("http://otel-collector:4317", service_name="my-agent")
tracker = CostTracker()

llm = AnthropicChat(model="claude-opus-4-7", on_request=tracker.hook)
agent = ReactAgent(llm=llm, tools=[HttpRequest(), BraveSearch(), Calculator()],
                   max_steps=10)
print(agent.invoke("What's the GDP of France in EUR per capita?"))
```

Run:
```bash
uv run python agent.py
```

Everything in this snippet is ✅ today.

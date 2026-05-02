# Missing & Nice-to-have Features

Snapshot: 2026-05-02 (iter 325). For full long-form prioritisation,
see `ROADMAP.md`. This file is the short, actionable view: each item
is either *missing* (a real gap) or *nice-to-have* (would help, not
load-bearing).

Items are **descriptive only** — no code in this PR. Track shipped
work with `iter N` in the commit log.

---

## Missing — true gaps vs. parity

### Agent / tool ergonomics
- `before_tool` / `after_tool` middleware hooks. Today the closest
  things are `on_request` (HTTP body) and the callback bus. Tool-
  granularity hooks would let users wrap retries, redaction, and
  audit logging without touching every tool.
- Streaming tool execution. Real use-cases are rare (long shell
  jobs). A `OffloadingTool` + result-poll pattern probably covers it
  without a `Tool` trait revamp.
- Tool-call budget caps mirroring `CostCappedChatModel` — limit *N*
  tool invocations per turn, not just dollars.

### Graph
- Pregel-style super-step parallel execution audit. The Kahn
  scheduler already runs independent nodes concurrently; an explicit
  super-step contract (with barrier semantics) would make
  reproducibility and step-by-step debug strictly defined.
- Branch fan-in deduplication (currently fan-in merges aggregated
  state by user-supplied reducer; an opt-in dedup-by-key reducer
  would avoid hand-rolling it).
- A `parallel_for` shorthand for the common "fan out N copies of the
  same node" pattern.

### Local / on-device models
- Local chat model via `candle` or `mistral.rs`. Cuts the OpenAI
  dependency for local dev + privacy-sensitive workloads. Adds a
  `LocalChatModel` adapter with the same `ChatModel` trait.
- Local embedding model parity for FastEmbed (already shipped) +
  GGUF/llama.cpp option for users who want a single binary.

### Provider coverage
- Mistral *native* tool-call format (today routed via OpenAI-compat
  shim — fine but loses Mistral-specific features like
  prompt-mode).
- Anthropic computer-use / web-fetch tool surfaces (the new
  Anthropic-managed tools — currently passthrough only).
- OpenAI Realtime API for voice agents.

### Memory / store
- Vector-indexed semantic search on the `Store` trait beyond the
  postgres + sqlite implementations (e.g., a Redis-vector backend
  for ops shops already running Redis).
- Hierarchical / namespaced memory (per-thread, per-org, per-app)
  with a single store. Today the user composes namespacing via
  `SearchFilter`; a first-class `Namespace` would tighten this.

### Eval & reproducibility
- Eval cache keyed on `(case_hash, model_hash, params_hash)` so re-runs
  skip already-scored items. Today re-running an eval re-spends.
- Confidence intervals on aggregate metrics (bootstrap resampling
  exists in `litgraph-core/src/eval/stats.rs` but isn't surfaced in
  the Harness report by default).

### Serve
- WebSocket streaming endpoint on `litgraph-serve`. SSE covers the
  forward path; WS would unlock client-driven cancellation +
  back-channel control without a second connection.
- Multi-tenant auth scaffolding (`X-Forwarded-User`, JWT validator,
  per-thread ACL). Today the serve binary is single-tenant.

### Python ergonomics
- `pyo3-stub-gen` auto-generated `.pyi` to replace hand-rolled
  `litgraph-stubs` so they can never drift from PyO3 signatures.
- Pydantic-coerced state on the Python side: today the user opts in
  via `coerce_one`/`coerce_stream`; making this implicit on
  `StateGraph(state_schema=BaseModel)` would match LangGraph
  ergonomics.
- `StreamPart` Python type that mirrors the Rust `ChatStreamEvent`
  enum more closely (today a duck-typed dict).

### Observability
- Trace exemplars: link an OTel span to its prompt + completion
  excerpt without dumping the full body. Helps debug high-cardinality
  flows.
- A "turn replay" CLI that takes an OTel trace ID and replays the
  exact prompt against a chosen model — closes the loop on prod
  debug.

---

## Nice to have — would polish, not block

### Loaders
- WhatsApp Business API loader (paperwork-heavy; Telegram is
  push-only and not feasible).
- Mailbox loaders beyond Gmail (IMAP, Outlook). Most teams already
  ETL email out of the inbox layer.
- Audio/video transcription loader (route through `WhisperTool`,
  cache transcripts).

### Splitters
- NLTK/SpaCy sentence splitter parity. Recursive-char + token
  splitters cover the cases; only worth adding if a user files a
  concrete miss.
- A regex-driven splitter for log files / structured text.

### Vector stores
- LanceDB and Pinecone backends. Low marginal value over the
  existing 6 backends and they pull heavy deps.
- A "blackhole" store (drop on add, empty on search) for benchmarking
  the rest of the pipeline without store noise.

### CLI / DX
- `litgraph init <template>` to scaffold a minimal repo (one graph,
  one provider, one tool, a test).
- A `litgraph trace` viewer that ingests OTel JSON and renders a
  graph timeline in the terminal.
- Studio UI parity for *local* graphs (today the `studio` feature
  flag in `litgraph-serve` covers the cloud API surface only).

### Docs
- A "migrate from LangChain" guide with side-by-side translations
  for the top 20 LangChain idioms.
- Per-crate README pointing at the canonical example for that
  subsystem.

### Performance
- A criterion compare bot in CI that flags >5% regression on a
  fixed set of micro-benches (graph fanout, BM25, HNSW).
- Free-threaded Python 3.13 wheel published with `--features
  no-gil` so users can opt in to the no-GIL build path.

---

## Won't do (recorded so we don't relitigate)

- **LangChain Callbacks parity.** Surface area is enormous; the
  callback bus + `CostTracker` + `GraphEvent` cover concrete asks.
- **Zapier / N8N tools.** Userland integration; out of framework
  scope.
- **Video-in modality** as framework code. Provider-side problem.
- **Per-class deprecated chains** (`LLMChain`, `SequentialChain`,
  `MultiPromptChain`, …). Anti-thesis of the project.

---

If you want a feature on this list moved up, file an issue with a
concrete use-case and the prioritisation-rubric scores from
`ROADMAP.md` filled in. Without (1) impact on a real agent author and
(2) a non-trivial Rust win, it stays where it is.

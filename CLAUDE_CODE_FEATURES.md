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
| Async batch API | Bulk evaluation | 🟡 `batch` per provider |
| Token counters (tiktoken/HF) | Pre-flight cost/context limits | ✅ `litgraph.tokenizers` |

## 2. Resilience — survive prod

| Feature | Why | Status |
|---|---|---|
| Retry w/ exp backoff + jitter | Flaky upstream | ✅ `RetryingChatModel` |
| Rate limiter | Per-key RPM/TPM caps | ✅ `RateLimitedChatModel` |
| Fallback chain | Model A down → try B | ✅ `FallbackChatModel` |
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
| Hub pull (`langchain hub`) | Share prompts | ❌ no hub |

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
| Pandas DataFrame parser | Data agents | ❌ |

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
| BM25 retriever (lexical) | Keyword grounding | ✅ |
| Hybrid (RRF) retriever | Best of both | ✅ |
| Reranking retriever (Cohere/Jina/Voyage) | Quality lift | ✅ |
| MaxMarginalRelevance | Diversity | ✅ |
| ParentDocumentRetriever | Small-chunk match, big-chunk return | ✅ |
| MultiQueryRetriever | Query rewriting | ✅ |
| ContextualCompressionRetriever | Chunk filtering | ✅ |
| SelfQueryRetriever | LLM extracts metadata filter | ✅ |
| TimeWeightedRetriever | Recent docs first | ✅ |
| HyDE retriever | Hypothetical doc embed | ✅ |
| EnsembleRetriever | Weighted fusion | 🟡 hybrid covers |
| Doc transformers (MMR, redundant filter, long-context reorder) | Pre-LLM cleanup | ✅ |

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
| fastembed-rs (local, no network) | Air-gapped | ❌ |
| Embedding retry/fallback | Prod | ✅ |
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
| Discord / Telegram / WhatsApp | ❌ |
| YouTube transcript / Vimeo | ❌ |
| arXiv / Wikipedia / PubMed | ✅ `litgraph_loaders::ArxivLoader` (Atom), `WikipediaLoader` (MediaWiki Action API), `PubMedLoader` (NCBI E-utilities — esearch+efetch, structured-abstract section labels preserved, MeSH terms, DOI/PMCID, normalised pub_date, API-key support) |
| Office365 / Outlook | ❌ |

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
| Sentence/NLTK/SpaCy splitters | 🟡 recursive covers |

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
| Dataset versioning | Track regressions | ❌ no built-in |

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

## 19. Deployment / Serve

| Feature | Why | Status |
|---|---|---|
| LangServe REST endpoints | Quick HTTP API | ❌ no serve crate |
| LangGraph Cloud / Platform | Hosted runtime | 🚫 out of scope |
| FastAPI integration example | DIY serve | ❌ no example shipped |
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
| Tool-result offload middleware | Push large outputs to filesystem/store | ❌ |
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
| `PostgresStore` (prod) | Durable distributed | ❌ pending crate |
| Vector-indexed semantic search on Store | Memory recall by meaning | 🟡 `VectorStoreMemory` separate; not yet bolted onto Store |
| `LangMem` SDK (episodic memory) | Auto-extract memories | ❌ |
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
| LangServe drop-in REST | One-line serve | ❌ |
| LangGraph Server | Hosted runtime + UI | 🚫 out of scope |
| LangGraph Studio (visual debugger) | Step-debug graph | ❌ no native UI |
| Assistants API (LangGraph) | Per-graph config snapshots | ❌ |
| Webhook resume after interrupt | External system → resume | 🟡 user wires via `Command` |

---

## What's left for "Claude Code can ship a prod agent without coding plumbing"

Top gaps to close, ranked by user-impact for a no-code-glue path:

1. 🟡 **Long-term memory `Store`** — core trait + `InMemoryStore` shipped (`litgraph.store`, 17 Py tests). Postgres backend + vector-indexed semantic search on Store still pending.
2. 🟡 **Middleware chain primitive** — `before/after_model` chain shipped (`litgraph.middleware`, 7 Py + 6 Rust tests). Built-ins: Logging, MessageWindow, SystemPrompt. `before/after_tool` hooks + tool-result offload still pending.
3. ✅ **Deep Agents harness** — `PlanningTool` + `VirtualFilesystemTool` + `load_agents_md` + `load_skills_dir` + `SystemPromptBuilder` + `SubagentTool` + one-call `litgraph.deep_agent.create_deep_agent(...)` factory all shipped (43 Rust + 41 Py tests across the seven).
4. ❌ **Functional API** (`@entrypoint` + `@task`) — Python decorator alternative to graph DSL. Trims LOC for simple workflows.
5. ❌ **Pydantic-coerced state in Python** — type-safe stream chunks, IDE-narrow types. (Rust side already typed.)
6. ❌ **`pyo3-stub-gen` auto-stubs** — manual stubs go stale. Pyright import warnings hurt agent-authored code.
7. ❌ **fastembed-rs local embeddings** — air-gap RAG without OpenAI key.
8. ❌ **candle / mistral.rs local chat** — full offline agent.
9. ❌ **LangServe-style HTTP serve crate** — `litgraph serve graph.py` → REST + SSE one command.
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

# litGraph vs LangChain vs LangGraph — Feature Comparison

Three-way side-by-side. Where litGraph wins, where LangChain / LangGraph
win, where they're equivalent. Snapshot date: 2026-05-03
(litGraph v0.1.2 · LangChain 0.3.x · LangGraph 0.4.x).

**Legend:** ✅ shipped · ⏳ partial · ❌ missing · 🚫 won't do · 💰 paid /
hosted only · 📦 via LangChain (LangGraph delegates) · 📦📦 via
LangChain Community.

LangGraph is the *orchestration* layer; LangChain Core is the
*provider/abstraction* layer; LangChain Community is the *integrations*
catalogue. The three-column layout makes the dependency split explicit:
when LangGraph says "yes, via LangChain", that's a real cost — extra
deps, extra Python frames, separate version-pinning surface.

---

## TL;DR

| Question | Answer |
|---|---|
| **Want managed deployment + visual studio + huge integration list?** | **LangGraph + LangChain** — LangSmith / Cloud / Studio + 200+ provider integrations. |
| **Want sub-microsecond per-node scheduling, true parallelism, no GIL contention, slim deps?** | **litGraph** — Rust core, ~107× faster vector search, ~90× faster scheduler, one wheel. |
| **Want a graph DSL that drops in?** | **LangGraph or litGraph** — `StateGraph`/`add_node`/`add_edge` translate one-to-one. (LangChain doesn't have a graph DSL — it has LCEL.) |
| **Want LCEL chain composition (`|` pipes)?** | **LangChain only.** litGraph + LangGraph favour explicit graphs. |
| **Production-ready today?** | **LangChain + LangGraph** for hosted/managed; **litGraph** for self-hosted Python apps that need throughput. |

If your bottleneck is the LLM call, the framework hardly matters. If
your bottleneck is the framework (long sessions, big retrieval, fan-out
agents), Rust wins.

---

## 1. Headline primitive

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Headline abstraction | `StateGraph` (typed, compiled) | `Runnable` + LCEL `|` pipes | `StateGraph` (compiled) |
| Compile step before run | ✅ | ⏳ (lazy, via `RunnableLambda` chain) | ✅ |
| Static cycle / shape check | ✅ | ❌ | ✅ |
| Drop-in `RunnableInterface` | ❌ (different shape on purpose) | ✅ | ✅ (StateGraph IS Runnable) |
| Functional API (`@entrypoint` / `@task`) | ✅ | ❌ | ✅ |

LangChain is chain-shaped; litGraph + LangGraph are graph-shaped. If
you've already invested in LCEL pipes, LangGraph migration is cheaper
than litGraph migration.

---

## 2. Core graph primitives

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `StateGraph` typed state | ✅ | ❌ (LCEL only) | ✅ |
| Add/remove nodes + edges | ✅ | ❌ | ✅ |
| Conditional edges | ✅ | ⏳ (`RunnableBranch`) | ✅ |
| Reducers (`Annotated[..., add]`) | ✅ | ❌ | ✅ |
| Dynamic fan-out (`Send` API) | ✅ (`add_send`) | ⏳ (`RunnableParallel` static) | ✅ |
| Subgraphs | ✅ | ⏳ (chain-of-chains) | ✅ |
| START / END sentinels | ✅ | n/a | ✅ |
| `compile().invoke / .stream / .batch` | ✅ | ✅ (via Runnable) | ✅ |
| Visualize (Mermaid / Graphviz) | ✅ | ⏳ | ✅ |
| Super-step parallel execution | ✅ (Kahn scheduler) | ❌ | ✅ |

**Verdict:** LangGraph and litGraph functionally equal. LangChain doesn't
play in this column — its parallelism comes from `RunnableParallel`,
which is static fan-out only.

---

## 3. Chain / pipeline DSL

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| LCEL `|` operator chaining | ⏳ (compat shim `litgraph.lcel.Pipe`; explicit graphs are still primary) | ✅ | ❌ (use graph nodes) |
| `Runnable` interface | ❌ (design call) | ✅ | ✅ (StateGraph IS one) |
| `RunnableParallel` | ⏳ (`litgraph.compat.RunnableParallel` shim; graph fan-out preferred) | ✅ | ✅ |
| `RunnableBranch` | ⏳ (`litgraph.compat.RunnableBranch` shim; conditional edges preferred) | ✅ | ✅ |
| `RunnableLambda` | ⏳ (`litgraph.compat.RunnableLambda` shim; nodes ARE functions) | ✅ | ✅ |
| `with_fallbacks` chain | ✅ (`FallbackChatModel`) | ✅ | ✅ |
| Streaming chains | ✅ | ✅ | ✅ |

If LCEL pipes are how your team thinks, LangChain is unique here. The
litGraph design call is "explicit graph > implicit chain" — a node IS
already a function.

---

## 4. Streaming

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Token-level streaming | ✅ | ✅ | ✅ |
| `astream_events` (typed event taxonomy) | ✅ (`litgraph.streaming.astream_events` shim) | ✅ | ✅ |
| Stream modes: `values`, `updates`, `messages`, `debug` | ⏳ (`values`+`updates`; `messages` via callback bus) | n/a | ✅ |
| Multi-consumer broadcast | ✅ (`broadcast(stream, n)`) | ❌ | ❌ |
| Race / first-wins between streams | ✅ (`race(streams)`) | ❌ | ❌ |
| Multiplex streams with origin tags | ✅ (`multiplex(streams)`) | ❌ | ❌ |
| Sub-millisecond per-event overhead | ✅ (Rust SSE parse, ~12 µs/16 KB) | ❌ (~ms-class Python) | ❌ (~ms-class Python) |

**Verdict:** litGraph wins on raw streaming primitives + perf. LangChain
defines the typed event vocabulary that LangGraph inherits.

---

## 5. Checkpointing + time-travel

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| In-memory checkpointer | ✅ | ❌ | ✅ |
| SQLite checkpointer | ✅ (WAL, durable) | ❌ | ✅ |
| Postgres checkpointer | ✅ (deadpool-pooled) | ❌ | ✅ |
| Redis checkpointer | ✅ (ZSET, O(log n) latest) | ❌ | ⏳ (community) |
| Time-travel (resume from any checkpoint id) | ✅ | ❌ | ✅ |
| Branch (fork a checkpoint) | ✅ (`Checkpointer::fork_at`) | ❌ | ✅ (`branch()`) |
| Per-thread state via `thread_id` | ✅ | ❌ | ✅ |
| Resume registry across process restarts | ✅ | ❌ | ⏳ (cloud) |

LangChain has no checkpointing — that's why LangGraph exists.

---

## 6. Agents

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Modern ReAct agent | ✅ (`ReactAgent`) | ⏳ (`AgentExecutor` legacy) | ✅ (`create_react_agent`) |
| Native tool-call format per provider | ✅ | ✅ | ✅ |
| Text-mode ReAct (T/A/O transcript) | ✅ (`TextReactAgent`) | ✅ (legacy) | ⏳ |
| Plan-and-Execute | ✅ (`PlanExecuteAgent`) | ⏳ (deprecated) | ⏳ (recipe) |
| Supervisor (multi-agent routing) | ✅ (`SupervisorAgent`) | ❌ | ✅ (`langgraph-supervisor`) |
| Debate (multi-agent + judge) | ✅ (`DebateAgent`) | ❌ | ❌ |
| Critique-Revise (self-improvement) | ✅ (`CritiqueReviseAgent`) | ❌ | ❌ |
| Self-Consistency (sample N → vote) | ✅ (`SelfConsistencyChatModel`) | ❌ | ❌ |
| Subagent tool (delegate to another agent) | ✅ (`SubagentTool`) | ⏳ | ⏳ |
| Swarm (handoff topology) | ✅ (`litgraph.agents_extras.SwarmAgent`) | ❌ | ✅ (`langgraph-swarm`) |
| BigTool (large-scale tool selection) | ✅ (`litgraph.agents_extras.BigToolAgent`) | ❌ | ✅ (`langgraph-bigtool`) |
| Deep agent factory (one-call wiring) | ✅ (`litgraph.agents.deep`) | ❌ | ✅ (`deepagents` pkg) |
| Legacy `AgentExecutor` API | ⏳ (compat shim `litgraph.compat.AgentExecutor` for porting; modern `ReactAgent` preferred) | ✅ | ❌ |

**Verdict:** litGraph has more research-backed patterns out of the box
(debate, critique-revise, self-consistency); LangGraph has the swarm +
BigTool addons; LangChain still ships the legacy `AgentExecutor` for
back-compat.

---

## 7. Tools

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `@tool` decorator | ✅ (`#[tool]` macro + `FunctionTool`) | ✅ | 📦 (LangChain) |
| JSON Schema autoderivation | ✅ (schemars) | ✅ (Pydantic) | 📦 |
| Built-in: shell / Python REPL / filesystem | ✅ | ⏳ (community) | 📦 |
| Built-in: web fetch / Tavily / DuckDuckGo / webhook | ✅ | ✅ | 📦 |
| Built-in: SQLite / virtual-fs / JSON-Patch / slugify | ✅ | ⏳ | 📦 |
| Built-in: Whisper / TTS / DALL·E / Gmail send | ✅ | ⏳ (community) | 📦 |
| `before_tool` / `after_tool` hooks | ✅ (`litgraph.tool_hooks.{Before,After}ToolHook`) | ✅ (callbacks) | ✅ |
| Streaming tool execution | 🚫 (offload pattern preferred) | ⏳ | ⏳ |
| Tool call budget cap | ✅ (`litgraph.tool_hooks.ToolBudget`) | ❌ | ❌ |
| MCP tool adapter | ✅ | ⏳ | ✅ |
| Total stock tool count | ~ 35 | 200+ (Community) | inherits LangChain |

**Verdict:** litGraph: fattest *built-in* tool set (35+ first-class).
LangChain Community: largest *catalogue* but spread across N packages.

---

## 8. Memory + chat history

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Token-buffer memory | ✅ | ✅ (`ConversationBufferMemory`) | ⏳ (via state) |
| Summary-buffer memory | ✅ | ✅ | ⏳ (recipe) |
| LangMem-style fact extractor | ✅ (`langmem` module) | ⏳ | ✅ (`langmem` pkg) |
| Backend: in-process | ✅ | ✅ | ✅ |
| Backend: SQLite | ✅ | ✅ | ⏳ |
| Backend: Postgres | ✅ | ✅ | ✅ |
| Backend: Redis | ✅ | ✅ | ⏳ |
| Backend: DynamoDB / MongoDB / Cassandra / … | ✅ (`litgraph.memory_extras.{DynamoDB,Mongo,Cassandra}ChatMemory`) | ✅ (Community) | 📦📦 |
| Hierarchical / namespaced memory | ✅ (`litgraph.memory_extras.NamespacedMemory`) | ⏳ | ✅ |
| Vector-backed long-term memory | ✅ (postgres + sqlite) | ✅ | ✅ |

**Verdict:** LangChain's memory backend catalogue is unmatched.
litGraph covers the four production-relevant backends.

---

## 9. Retrieval / RAG

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `Retriever` / `VectorStore` traits | ✅ (native Rust) | ✅ (Python) | 📦 |
| BM25 retriever | ✅ (rayon, 23 M elem/s) | ✅ (Python `rank-bm25`) | 📦 |
| RRF fusion | ✅ (parallel, 6 M docs/s) | ⏳ (manual) | 📦 |
| MMR | ✅ | ✅ | 📦 |
| HyDE | ✅ | ✅ | 📦 |
| Multi-query | ✅ | ✅ | 📦 |
| Self-query | ✅ | ✅ | 📦 |
| Parent-document | ✅ | ✅ | 📦 |
| Multi-vector | ✅ | ✅ | 📦 |
| Time-weighted | ✅ | ✅ | 📦 |
| Ensemble retriever (weighted) | ✅ | ✅ | 📦 |
| Race retriever (first-wins) | ✅ | ❌ | ❌ |
| Step-back / sub-query decomposition | ✅ | ⏳ | 📦 |
| Contextual compression | ✅ | ✅ | 📦 |
| Semantic dedup at ingest | ✅ | ❌ | ❌ |
| Rerankers: Cohere / Voyage / Jina / FastEmbed | ✅ (native) | ✅ (Community) | 📦 |
| **HNSW search throughput / 100 K vecs** | **~ 41 µs (2.4 G elem/s)** | ~ 4 ms (faiss-py) | 📦 (LangChain) |
| **BM25 search / 50 K docs** | **~ 2.1 ms (23.4 M elem/s)** | ~ 200 ms | 📦 |

**Verdict:** litGraph wins decisively on retrieval. Native Rust BM25 +
HNSW + RRF + MMR all on the hot path, no GIL. LangChain has the
broadest catalogue of *integrations* (every cloud retrieval API);
LangGraph inherits LangChain's wrappers wholesale.

---

## 10. Vector stores

| Store | litGraph | LangChain | LangGraph |
|---|---|---|---|
| In-memory | ✅ | ✅ | 📦 |
| HNSW (embedded) | ✅ (instant-distance, Rust) | ✅ (faiss / hnswlib via Community) | 📦 |
| FAISS | ✅ (`litgraph.stores_extras.FaissVectorStore`, lazy-imports `faiss-cpu`) | ✅ | 📦 |
| Qdrant | ✅ (REST, no gRPC) | ✅ | 📦 |
| pgvector | ✅ | ✅ | 📦 |
| Chroma | ✅ | ✅ | 📦 |
| Weaviate | ✅ | ✅ | 📦 |
| LanceDB | ⏳ (`litgraph.stores_extras.LanceDBVectorStore` courtesy adapter — natives preferred) | ✅ | 📦 |
| Pinecone | ⏳ (`litgraph.stores_extras.PineconeVectorStore` courtesy adapter — natives preferred) | ✅ | 📦 |
| Milvus | ✅ (`litgraph.stores_extras.MilvusVectorStore`) | ✅ | 📦 |
| Redis-search | ✅ (`litgraph.stores_extras.RedisSearchVectorStore`) | ✅ | 📦 |
| Neo4j (vector) | ✅ (`litgraph.stores_extras.Neo4jVectorStore`) | ✅ | 📦 |
| MongoDB Atlas Vector | ✅ (`litgraph.stores_extras.MongoAtlasVectorStore`) | ✅ | 📦 |
| Total | 6 native + 8 via `stores_extras` (FAISS / Milvus / Redis-search / Neo4j / Mongo Atlas / Pinecone / LanceDB / Cassandra) | 80+ | inherits |

LangChain wins coverage; litGraph covers the production-relevant 6.

---

## 11. Document loaders

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Text · JSONL · MD · Dir · CSV | ✅ (rayon-parallel) | ✅ | 📦 |
| PDF · DOCX | ✅ | ✅ | 📦 |
| HTML · sitemap · web | ✅ | ✅ | 📦 |
| S3 · Google Drive | ✅ | ✅ | 📦 |
| Confluence · Jira · Linear · Notion · Slack | ✅ | ✅ | 📦 |
| GitHub (files + issues) · GitLab · Discord · Wikipedia | ✅ | ✅ | 📦 |
| Gmail · Jupyter | ✅ | ✅ | 📦 |
| Outlook · IMAP · WhatsApp · YouTube · Hugging Face · Airtable · Reddit · Twitter · … | ✅ (`litgraph.loaders_extras`: IMAP / Outlook / YouTube / Reddit / Airtable / Twitter / HF Datasets / WhatsApp Cloud — all 8 named sources covered) | ✅ (Community, ~150 total) | 📦 |
| **Parallel ingest** | ✅ (rayon, all loaders) | ⏳ (depends on loader) | 📦 |
| Total stock loaders | ~ 25 | 150+ | inherits |

**Verdict:** LangChain wins on breadth (~6× more loaders). litGraph
covers the mainstream business loaders and adds parallel ingest as a
default, not a per-loader feature.

---

## 12. Splitters

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Recursive character | ✅ | ✅ | 📦 |
| Markdown header | ✅ | ✅ | 📦 |
| HTML header | ✅ | ✅ | 📦 |
| JSON splitter | ✅ | ✅ | 📦 |
| Token splitter (tiktoken / HF) | ✅ | ✅ | 📦 |
| Semantic chunker | ✅ | ✅ | 📦 |
| Code-aware (tree-sitter, definition-level) | ✅ | ⏳ (language-specific) | 📦 |
| NLTK / SpaCy sentence splitter | ✅ (`litgraph.splitters_extras.{Nltk,Spacy}SentenceSplitter`) | ✅ | 📦 |

**Verdict:** Near-parity, with LangChain edging out on NLP-toolkit
splitters.

---

## 13. Output parsers / structured output

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `with_structured_output` | ✅ | ✅ | ✅ (via LangChain) |
| Pydantic v2 | ✅ | ✅ | ✅ |
| Dataclass / TypedDict | ✅ | ⏳ | ✅ |
| Raw JSON Schema | ✅ | ✅ | ✅ |
| Stream coercion (`coerce_one`/`coerce_stream`) | ✅ | ❌ | ❌ |
| Partial-JSON repair | ✅ (Rust, 904 MB/s) | ⏳ (`json-repair` py) | ⏳ |
| Output fixer (LLM-on-error retry) | ✅ | ✅ | ⏳ |
| Boolean / list / numbered-list parsers | ✅ | ✅ | ✅ |
| Markdown table parser | ✅ | ⏳ | ⏳ |
| ReAct step parser | ✅ | ✅ | ⏳ |
| XML tag parser | ✅ | ✅ | ⏳ |
| Custom regex parser | ✅ | ✅ | ⏳ |

litGraph wins on stream coercion + repair perf. LangChain has the
deepest catalogue of legacy parsers.

---

## 14. Prompt templates

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `ChatPromptTemplate` | ✅ | ✅ | ✅ (LangChain) |
| `MessagesPlaceholder` | ✅ | ✅ | ✅ |
| Few-shot prompt template | ✅ | ✅ | ✅ |
| Semantic-similarity example selector | ✅ | ✅ | ✅ |
| Length-based example selector | ✅ | ✅ | ✅ |
| Strict-undefined Jinja (catch typos) | ✅ (minijinja) | ⏳ | ⏳ |
| from/to JSON · from/to dict | ✅ | ✅ | ✅ |
| Compose: extend / `+` / concat | ✅ | ✅ | ✅ |
| Hub (community-shared prompts) | ⏳ (`litgraph.prompt_hub` registry + `prompts/` folder seed) | ✅ (LangChain Hub) | ✅ |

LangChain wins on the Hub ecosystem; litGraph wins on strict-Jinja
catch-typos-at-render-time.

---

## 15. Resilience wrappers

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Retry (exponential backoff + jitter) | ✅ | ✅ | 📦 |
| Fallback (provider failover) | ✅ | ✅ (`with_fallbacks`) | 📦 |
| Rate limiter | ✅ | ⏳ (`InMemoryRateLimiter`) | 📦 |
| Token budget cap | ✅ | ❌ | ❌ |
| Cost cap (USD ceiling) | ✅ | ❌ | ❌ |
| PII scrubbing pre-call | ✅ | ⏳ (community) | ⏳ |
| Prompt-cache middleware | ✅ | ⏳ | ⏳ |
| Timeout wrapper | ✅ | ✅ (asyncio) | ✅ |
| Composes freely (decorator stack) | ✅ | ⏳ | ⏳ |

litGraph wins. Resilience is a first-class subsystem; in LangChain it's
scattered across Core + Community; LangGraph delegates.

---

## 16. Caching

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| In-memory cache | ✅ | ✅ | 📦 |
| SQLite cache | ✅ | ✅ | 📦 |
| Redis cache | ✅ (`RedisCache`, `litgraph-cache::redis_backend`) | ✅ | 📦 |
| Semantic cache (embedding similarity) | ✅ | ✅ | 📦 |
| GPTCache adapter | ✅ (`litgraph.cache_extras.GPTCacheAdapter`) | ✅ | 📦 |
| Embedding cache (separate from model cache) | ✅ | ⏳ | ⏳ |
| Composes (semantic → identity → model) | ✅ | ⏳ | ⏳ |

LangChain's cache backend catalogue is wider; litGraph composes cleaner.

---

## 17. Observability + tracing

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Callback handler interface | ✅ | ✅ | 📦 |
| Cost tracker | ✅ (`CostTracker`) | ⏳ (LangSmith) | ⏳ (LangSmith) |
| `on_request` raw HTTP hook | ✅ | ❌ | ❌ |
| OTel exporter (OTLP gRPC + HTTP) | ✅ | ⏳ (community) | ⏳ |
| LangSmith integration | ⏳ (shim) | ✅ (first-class) | ✅ (first-class) |
| Trace exemplars (link span ↔ prompt) | ✅ (`litgraph_tracing_otel::exemplars::attach_*`) | ✅ (LangSmith) | ✅ |
| GraphEvent / NodeEvent stream | ✅ | n/a | ✅ |
| Stdout / file logger out of the box | ✅ | ✅ | ✅ |

litGraph wins on OTel + raw-HTTP debugging; LangChain/LangGraph win on
LangSmith depth (it's the same product).

---

## 18. Human-in-the-loop (HITL)

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| `interrupt_before` / `interrupt_after` | ✅ | ❌ | ✅ |
| Dynamic `interrupt(payload)` from a node | ✅ | ❌ | ✅ |
| Resume with edited state | ✅ (`compiled.resume(...)`) | ❌ | ✅ (`Command(resume=...)`) |
| Webhook-resume bridge (HTTP → resume) | ✅ | ❌ | 💰 (Cloud) |
| Pending interrupt inspection | ✅ | ❌ | ✅ |

LangChain has no HITL — that's why LangGraph exists.

---

## 19. Provider coverage — chat models

| Provider | litGraph | LangChain | LangGraph |
|---|---|---|---|
| OpenAI (Chat Completions + Responses) | ✅ | ✅ | 📦 |
| Anthropic (+ thinking blocks + prompt caching) | ✅ | ✅ | 📦 |
| Google Gemini (AI Studio + Vertex) | ✅ | ✅ | 📦 |
| AWS Bedrock (native + Converse, no AWS SDK) | ✅ | ✅ (depends on boto3) | 📦 |
| Cohere | ✅ | ✅ | 📦 |
| Mistral | ⏳ (via OpenAI-compat) | ✅ (native) | 📦 |
| Ollama / vLLM / Together / Groq / Fireworks / DeepSeek / xAI / LM Studio | ✅ (via OpenAI-compat) | ✅ (each native) | 📦 |
| HuggingFace TGI · IBM watsonx · Databricks · Snowflake Cortex · Replicate · NVIDIA NIM | ✅ (`litgraph.providers_extras` one-liners) | ✅ (Community) | 📦 |
| Local model via candle / mistral.rs | ✅ via OpenAI-compat (`providers_extras.{mistralrs,vllm,llamacpp}_chat`); native candle Rust crate still planned | ❌ | ❌ |
| Total provider count | ~ 6 native + 8 OpenAI-compat | 50+ | inherits |

LangChain wins on native-integration breadth; litGraph hits the
top-tier providers natively + everything OpenAI-compatible for free.

---

## 20. Embedding providers

| Provider | litGraph | LangChain | LangGraph |
|---|---|---|---|
| OpenAI · Cohere · Voyage · Jina | ✅ | ✅ | 📦 |
| Bedrock · Gemini | ✅ | ✅ | 📦 |
| FastEmbed (local ONNX) | ✅ | ✅ | 📦 |
| HuggingFace Inference / Sentence-Transformers / Instructor / E5 / NVIDIA NIM | ✅ (`litgraph.embeddings_extras` — all 5 adapters) | ✅ | 📦 |

LangChain wins on integration breadth.

---

## 21. Evaluation

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Eval driver / harness | ✅ (`EvalHarness`) | ⏳ (`langchain.evaluation`) | ⏳ (LangSmith Eval) |
| BLEU (multi-ref) / ROUGE-N / ROUGE-L | ✅ | ⏳ (string distance only) | ❌ |
| chrF / chrF++ | ✅ | ❌ | ❌ |
| METEOR-lite | ✅ | ❌ | ❌ |
| BERTScore-lite | ✅ | ⏳ (via deps) | ❌ |
| WER / CER (+ sub/ins/del breakdown) | ✅ | ❌ | ❌ |
| TER (with shifts) | ✅ | ❌ | ❌ |
| Relaxed Word Mover Distance | ✅ | ❌ | ❌ |
| Pearson / Spearman / Kendall's tau-b | ✅ | ❌ | ❌ |
| Paired permutation test | ✅ | ❌ | ❌ |
| LLM-as-judge | ✅ | ✅ | ✅ (LangSmith) |
| Pairwise evaluator | ✅ | ✅ | ✅ |
| Trajectory eval | ✅ | ✅ | ✅ |
| Dataset versioning + regression alerts | ✅ | ❌ | 💰 (LangSmith) |

**Verdict:** litGraph has a stand-alone, self-hosted eval suite.
LangChain ships some string-distance scorers; LangSmith owns the rest
behind a paywall.

---

## 22. MCP (Model Context Protocol)

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| MCP client (stdio + HTTP/SSE) | ✅ | ⏳ (community) | ✅ |
| MCP server (expose own tools) | ✅ | ❌ | ⏳ |
| MCP tool adapter (drop-in to agent) | ✅ | ⏳ | ✅ |
| Resource + prompt support | ✅ | ⏳ | ⏳ |

litGraph + LangGraph both ship first-class MCP; LangChain Core hasn't.

---

## 23. Deployment

| Feature | litGraph | LangChain | LangGraph |
|---|---|---|---|
| HTTP serve binary (REST + SSE) | ✅ (`litgraph-serve`) | ⏳ (`LangServe`) | ✅ (`langgraph-cli serve`) |
| LangGraph Cloud-API compatible | ✅ (Studio router behind feature flag) | ❌ | ✅ (native) |
| Managed cloud hosting | ❌ (self-host only) | 💰 (LangSmith deploy) | 💰 (LangGraph Cloud) |
| Studio UI (visual debugger) | ⏳ (cloud-API surface; no local UI) | ⏳ (LangServe Playground) | ✅ |
| Multi-tenant auth scaffolding | ✅ (`litgraph_serve::auth::{bearer_layer,forwarded_user_layer}`) | ❌ | 💰 |
| WebSocket endpoint | ✅ (`litgraph_serve::ws`, feature `ws`) | ⏳ (LangServe) | ⏳ |
| Single binary deploy (no Python at edge) | ✅ (Rust binary) | ❌ | ❌ |

LangGraph wins for managed deploy; litGraph wins for self-hosted
single-binary deploy.

---

## 24. Performance

Apples-to-apples micro-benches (criterion on macOS arm64). LangChain /
LangGraph numbers approximated from public `pytest-benchmark` runs of
equivalent operations. Numbers shift case-by-case but the shape doesn't
— every primitive on the litGraph hot path is a Rust call; LangChain
hits Python; LangGraph hits LangChain.

| Operation | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Graph fanout / 64 nodes | ~ 90 µs | n/a | ~ 8 ms (~ 90× slower) |
| BM25 search / 50 K docs | ~ 2.1 ms | ~ 200 ms | ~ 200 ms (uses LangChain) |
| HNSW search / 100 K vecs | ~ 41 µs | ~ 4 ms | ~ 4 ms (uses LangChain) |
| SSE parse / 16 KB chunk | ~ 12 µs | ~ 2 ms | ~ 2 ms |
| JSON repair / 256 B | ~ 280 ns | ~ 50 µs | ~ 50 µs |
| RRF fuse / 4 × 100 lists | ~ 65 µs | ~ 5 ms | ~ 5 ms |
| Cold-import time | < 50 ms | ~ 500 ms – 2 s | ~ 1–3 s |

If your throughput SLO is dominated by framework overhead (fan-out
agents, retrieval-heavy pipelines, long sessions), the gap shows. If
it's dominated by the LLM call, the framework hardly matters.

---

## 25. Runtime / dependencies

| Dimension | litGraph | LangChain | LangGraph |
|---|---|---|---|
| Core language | Rust | Python | Python |
| Python binding | abi3-py39 (one wheel covers 3.9–3.13+) | n/a | n/a |
| GIL behaviour | dropped (`py.detach()`) on every blocking call | held | held |
| Free-threaded Python 3.13t | ✅ supported | ⏳ | ⏳ |
| Default wheel size | ~ 13 MB (one native .so) | ~ 5 MB (Core) + 50–500 MB (Community) | ~ 5 MB + LangChain |
| Required runtime deps | none (Python stdlib) | `langchain-core` + Pydantic + jsonpatch + … | `langchain-core` + `langgraph-core` + … |
| Optional integrations | Cargo features (zero defaults) | pip extras + langchain-* sub-packages | pip extras |
| Cold-import time | < 50 ms | ~ 500 ms – 2 s | ~ 1–3 s |
| Type checker friendliness | ✅ (PEP 561 stubs) | ⏳ (improved in 0.3.x) | ⏳ |

LangChain itself is now split into ~ 30 sub-packages (`langchain-core`,
`langchain-openai`, `langchain-anthropic`, `langchain-aws`, …) to manage
dep weight. Even so, a typical LangChain agent install is ~ 200 MB of
transitive deps; litGraph is ~ 13 MB.

---

## 26. Ecosystem / community

| Dimension | litGraph | LangChain | LangGraph |
|---|---|---|---|
| GitHub stars | ~ 100s (early) | 90 K+ | 12 K+ |
| Production users | early adopters | thousands | hundreds |
| Hub (shared prompts/agents) | ⏳ (`prompts/` folder + `litgraph.prompt_hub`) | ✅ | ⏳ |
| Tutorials / blog posts | small | enormous | growing |
| Stack Overflow tag | ❌ | ✅ | ⏳ |
| Hiring market familiarity | low | high | medium |

LangChain wins decisively on ecosystem — that's its biggest moat.

---

## 27. When LangChain is the better choice

- You want **LCEL pipes** (`prompt | model | parser`) as your composition model.
- You need an **integration that only exists in LangChain Community** (a niche loader, vector store, tool, or memory backend).
- You want **LangChain Hub** for shared prompts / agents.
- Your team already knows LangChain idioms and you're optimising for hiring/onboarding speed.
- You don't need durable orchestration (no checkpointing, no HITL).

## 28. When LangGraph is the better choice

- You want a **graph DSL** + **LangChain provider catalogue** + **LangSmith / Cloud / Studio**.
- You want **managed cloud hosting** with autoscaling, persistence, RBAC.
- You're already on **LangSmith** for tracing, eval, prompt management.
- You want **branch / time-travel** ergonomics with the polish of an established product.
- You're a Python shop with no Rust toolchain.

## 29. When litGraph is the better choice

- **Throughput / latency matters.** Long agent sessions, retrieval-heavy
  workloads, fan-out across many tools — Rust gives 50–100× headroom on
  framework overhead.
- **You self-host and pay for compute.** A Rust binary uses ~ 10× less
  CPU than equivalent Python for the same orchestration → bills shrink.
- **You need true parallelism.** Free-threaded Python 3.13t works, but
  most stacks are still GIL-bound. litGraph drops the GIL.
- **You want slim, auditable deps.** 1 wheel (~ 13 MB) vs. 200 MB of
  Python transitive deps. Easier supply-chain review.
- **You want a stand-alone eval suite.** BLEU, ROUGE, chrF, METEOR,
  BERTScore-lite, WER, TER, RWMD, statistical tests — all native, no
  LangSmith required.
- **You want resilience built in, not assembled.** Retry, fallback,
  rate-limit, budget, cost-cap, PII scrubbing, prompt-cache — all
  composable as decorators in one crate.
- **You want a single-binary edge deploy.** No Python runtime needed at
  the edge.

---

## 30. Migration cheat sheet

### From LangChain (chains)

| LangChain | litGraph |
|---|---|
| `prompt | model | parser` (LCEL) | three nodes in a `StateGraph`, or a `@task` chain |
| `RunnableParallel({...})` | parallel branches in `StateGraph` |
| `RunnableBranch((cond, x), default)` | `add_conditional_edges` |
| `RunnableLambda(f)` | `g.add_node("f", f)` |
| `with_fallbacks([m1, m2])` | `FallbackChatModel([m1, m2])` |
| `AgentExecutor(agent, tools)` | `ReactAgent(model, tools)` |
| `ConversationBufferMemory()` | `TokenBufferMemory(model_name=...)` |
| `RetrievalQA.from_chain_type(...)` | compose `Retriever` + an agent or graph node |
| `OutputParser` | `with_structured_output(Schema)` |
| `LangChainTracer()` | `tracing.init_otlp(...)` or `CostTracker` |

### From LangGraph

| LangGraph | litGraph |
|---|---|
| `from langgraph.graph import StateGraph, END` | `from litgraph.graph import StateGraph, END` |
| `from langgraph.checkpoint.memory import MemorySaver` | (default; pass `thread_id` to `.invoke()`) |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | `from litgraph.checkpoint import SqliteSaver` |
| `from langgraph.prebuilt import create_react_agent` | `from litgraph.agents import ReactAgent` |
| `Command(goto=..., update=...)` | return `NodeOutput.goto(...)` from a node |
| `interrupt(payload)` | `g.interrupt_before("node")` + `compiled.resume(...)` |
| `compiled.stream(state, stream_mode="values")` | `for ev in compiled.stream(state):` |
| `langgraph.func.entrypoint` / `task` | `litgraph.functional.entrypoint` / `task` |
| `MessagesState` | `add_messages` reducer on a state channel |

---

## 31. Honesty notes

This doc is written by a litGraph maintainer, so:

- **Benchmarks** are micro, not end-to-end. End-to-end timings are
  dominated by the LLM call, where all three are equal.
- **LangChain / LangGraph numbers** are approximated from public
  `pytest-benchmark` runs and may have shifted in newer versions.
- **Feature checkmarks** for LangChain reflect the 0.3.x lineage with
  Community packages; LangGraph reflects the 0.4.x lineage. Both move
  fast; check `pip show langchain langgraph` if a row looks stale.
- **litGraph "✅"** sometimes means "shipped, not yet at LangChain's
  scale of polish" — see [MISSING_FEATURES.md](MISSING_FEATURES.md) for
  the gaps we ourselves track.
- **LangChain Community wraps** count as ✅ for LangChain and 📦 for
  LangGraph, since LangGraph imports LangChain. That accurately reflects
  the dep cost.

If a row here looks unfair to LangChain or LangGraph, file an issue
and we'll fix it.

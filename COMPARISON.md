# litGraph vs LangGraph — Feature Comparison

Honest, side-by-side. Where litGraph wins, where LangGraph wins, where
they're equivalent. Snapshot date: 2026-05-02 (litGraph v0.1.1 ·
LangGraph 0.4.x lineage).

**Legend:** ✅ shipped · ⏳ partial · ❌ missing · 🚫 won't do · 💰 paid /
hosted only.

---

## TL;DR

| Question | Answer |
|---|---|
| **Want managed deployment + visual studio + huge integration list?** | **LangGraph** — LangSmith / LangGraph Cloud / Studio are mature. |
| **Want sub-microsecond per-node scheduling, true parallelism, no GIL contention, slim deps?** | **litGraph** — Rust core, ~107× faster vector search, ~90× faster scheduler. |
| **Want a graph DSL that feels the same in either?** | **Both** — `StateGraph`/`add_node`/`add_edge`/`compile` translate one-to-one. |
| **Production-ready today?** | **LangGraph** for hosted; **litGraph** for self-hosted Python apps that need throughput. |

If your bottleneck is the LLM call, the framework hardly matters. If your
bottleneck is the framework (long sessions, big retrieval, fan-out
agents), Rust wins.

---

## 1. Core graph primitives

| Feature | litGraph | LangGraph |
|---|---|---|
| `StateGraph` with typed state | ✅ | ✅ |
| Add/remove nodes + edges | ✅ | ✅ |
| Conditional edges | ✅ | ✅ |
| Reducers (LangGraph `Annotated[..., add]`) | ✅ (state-channel reducer) | ✅ |
| Dynamic fan-out (`Send` API) | ✅ (`add_send`) | ✅ |
| Subgraphs (compose graphs) | ✅ | ✅ |
| START / END sentinels | ✅ | ✅ |
| `compile().invoke / .stream / .batch` | ✅ | ✅ |
| Visualize (Mermaid / Graphviz) | ✅ | ✅ |
| Super-step parallel execution | ✅ (Kahn scheduler) | ✅ |
| Cycle detection at compile time | ✅ | ✅ |

**Verdict:** Functional parity. The DSL was deliberately designed to be a one-to-one drop-in.

---

## 2. Streaming

| Feature | litGraph | LangGraph |
|---|---|---|
| Token-level streaming from chat models | ✅ | ✅ |
| Streaming `StateGraph.stream()` | ✅ | ✅ |
| Stream modes: `values`, `updates`, `messages`, `debug` | ⏳ (`values`, `updates`; `messages` via callback bus; `debug` via OTel) | ✅ |
| Multiple-consumer broadcast | ✅ (`broadcast(stream, n)`) | ❌ (manual `tee`) |
| Race / first-wins between streams | ✅ (`race(streams)`) | ❌ |
| Multiplex streams with origin tags | ✅ (`multiplex(streams)`) | ❌ |
| `astream_events` (LangChain-shaped) | ❌ (use callback bus) | ✅ |
| Sub-millisecond per-event overhead | ✅ (Rust SSE parse, ~12 µs/16 KB) | ❌ (~ms-class Python overhead) |

**Verdict:** litGraph wins on raw streaming primitives + perf. LangGraph
wins on the LangChain-typed event taxonomy (`astream_events`).

---

## 3. Checkpointing + time-travel

| Feature | litGraph | LangGraph |
|---|---|---|
| In-memory checkpointer (default) | ✅ | ✅ |
| SQLite checkpointer | ✅ (WAL mode, durable) | ✅ |
| Postgres checkpointer | ✅ (deadpool-pooled) | ✅ |
| Redis checkpointer | ✅ (ZSET, O(log n) latest) | ⏳ (community) |
| Time-travel — resume from any checkpoint id | ✅ | ✅ |
| Branch — fork a checkpoint into two threads | ⏳ (manual: copy the thread) | ✅ (`branch()`) |
| Per-thread state isolation via `thread_id` | ✅ | ✅ |
| Resume registry across process restarts | ✅ | ⏳ (cloud-only) |

**Verdict:** Backend parity + Redis edge to litGraph; LangGraph has the
nicer `branch()` ergonomics.

---

## 4. Agents

| Feature | litGraph | LangGraph |
|---|---|---|
| Prebuilt ReAct agent | ✅ (`ReactAgent`) | ✅ (`create_react_agent`) |
| Native tool-call format per provider | ✅ | ✅ |
| Text-mode ReAct (thought/action/obs transcript) | ✅ (`TextReactAgent`) | ⏳ |
| Plan-and-Execute | ✅ (`PlanExecuteAgent`) | ⏳ (recipe) |
| Supervisor (multi-agent routing) | ✅ (`SupervisorAgent`) | ✅ (`langgraph-supervisor`) |
| Debate (multi-agent + judge) | ✅ (`DebateAgent`) | ❌ |
| Critique-Revise (self-improvement loop) | ✅ (`CritiqueReviseAgent`) | ❌ |
| Self-Consistency (sample N → vote) | ✅ (`SelfConsistencyChatModel`) | ❌ |
| Subagent tool (delegate to another agent) | ✅ (`SubagentTool`) | ⏳ |
| Swarm (handoff topology) | ❌ | ✅ (`langgraph-swarm`) |
| BigTool (large-scale tool selection) | ⏳ (RAG over tool descriptions, manual) | ✅ (`langgraph-bigtool`) |
| Deep agent factory (one-call wiring) | ✅ (`litgraph.agents.deep`) | ✅ (`deepagents`) |

**Verdict:** litGraph has more research-backed agent patterns out of
the box; LangGraph has the swarm + BigTool addons.

---

## 5. Tools

| Feature | litGraph | LangGraph |
|---|---|---|
| Function tool decorator | ✅ (`@tool` macro in Rust + `FunctionTool` in Python) | ✅ |
| JSON-schema autoderivation | ✅ (schemars) | ✅ (Pydantic) |
| Built-in: shell / Python REPL / filesystem | ✅ | ⏳ (community) |
| Built-in: web fetch / Tavily / DuckDuckGo / webhook | ✅ | ✅ |
| Built-in: SQLite / virtual-fs / JSON-Patch / slugify | ✅ | ❌ |
| Built-in: Whisper / TTS / DALL·E / Gmail send | ✅ | ❌ |
| `before_tool` / `after_tool` middleware hooks | ❌ (planned) | ✅ |
| Streaming tool execution | 🚫 (offload pattern preferred) | ⏳ |
| Tool call budget cap | ⏳ (cost-cap covers $; not call-count) | ❌ |
| MCP tool adapter | ✅ (`McpToolAdapter`) | ✅ |

**Verdict:** litGraph has a fatter battery of stock tools; LangGraph has
the cleaner middleware model.

---

## 6. Memory

| Feature | litGraph | LangGraph |
|---|---|---|
| Token-buffer memory | ✅ | ✅ (via state) |
| Summary-buffer memory | ✅ | ⏳ (recipe) |
| LangMem-style fact extractor | ✅ (`langmem` module) | ✅ (`langmem` package) |
| Backend: in-process | ✅ | ✅ |
| Backend: SQLite | ✅ | ⏳ |
| Backend: Postgres | ✅ | ✅ |
| Backend: Redis | ✅ | ⏳ |
| Hierarchical / namespaced memory | ⏳ (filter-based) | ✅ |

**Verdict:** litGraph has more first-class backends; LangGraph has cleaner namespacing.

---

## 7. Retrieval / RAG

| Feature | litGraph | LangGraph |
|---|---|---|
| `Retriever` / `VectorStore` traits | ✅ | ❌ (uses LangChain) |
| BM25 retriever (in-process, parallel) | ✅ (rayon, 23M elem/s) | ⏳ (LangChain wrapper) |
| RRF fusion (hybrid retrieval) | ✅ (parallel) | ⏳ (LangChain wrapper) |
| MMR | ✅ | ✅ |
| HyDE | ✅ | ⏳ (recipe) |
| Multi-query | ✅ | ✅ |
| Self-query | ✅ | ✅ |
| Parent-document | ✅ | ✅ |
| Multi-vector | ✅ | ✅ |
| Time-weighted | ✅ | ✅ |
| Ensemble retriever (weighted) | ✅ | ✅ |
| Race retriever (first-wins) | ✅ | ❌ |
| Step-back / sub-query decomposition | ✅ | ⏳ |
| Contextual compression | ✅ | ✅ |
| Semantic dedup (ingestion-time) | ✅ | ❌ |
| Rerankers: Cohere / Voyage / Jina / FastEmbed | ✅ | ⏳ (LangChain) |
| Vector stores: HNSW / Qdrant / pgvector / Chroma / Weaviate | ✅ | ⏳ (LangChain) |
| **HNSW search throughput** | **2.4 G elem/s** | LangChain-dep, ~10–100× slower |

**Verdict:** litGraph wins decisively. Retrieval is the litGraph killer
feature: native Rust BM25 + HNSW + RRF + MMR all on the hot path,
no GIL.

---

## 8. Observability

| Feature | litGraph | LangGraph |
|---|---|---|
| Callback bus | ✅ | ✅ |
| Cost tracker | ✅ (`CostTracker`) | ⏳ (LangSmith) |
| `on_request` hook (raw HTTP body) | ✅ | ❌ |
| OTel exporter (OTLP gRPC + HTTP) | ✅ | ⏳ |
| LangSmith integration | ⏳ (shim) | ✅ (first-class) |
| Trace exemplars (link span ↔ prompt excerpt) | ❌ (planned) | ✅ |
| GraphEvent / NodeEvent stream | ✅ | ✅ |

**Verdict:** litGraph wins on OTel + raw-HTTP debugging; LangGraph wins
on LangSmith depth.

---

## 9. Human-in-the-loop (HITL)

| Feature | litGraph | LangGraph |
|---|---|---|
| `interrupt_before` / `interrupt_after` | ✅ | ✅ |
| Dynamic `interrupt(payload)` from a node | ✅ | ✅ |
| Resume with edited state | ✅ (`compiled.resume(...)`) | ✅ (`Command(resume=...)`) |
| Webhook-resume bridge (HTTP → resume) | ✅ | 💰 (LangGraph Cloud) |
| Pending interrupt inspection | ✅ | ✅ |

**Verdict:** Parity for the primitives; litGraph ships the webhook bridge
self-host, LangGraph gates it behind Cloud.

---

## 10. Functional API

| Feature | litGraph | LangGraph |
|---|---|---|
| `@entrypoint` decorator | ✅ | ✅ |
| `@task` decorator | ✅ | ✅ |
| Auto-parallel task fan-out | ✅ (tokio JoinSet) | ✅ |
| Same checkpointing/streaming as `StateGraph` | ✅ | ✅ |

**Verdict:** Parity.

---

## 11. Structured output

| Feature | litGraph | LangGraph |
|---|---|---|
| `with_structured_output` | ✅ | ✅ |
| Pydantic v2 | ✅ | ✅ |
| Dataclass / TypedDict | ✅ | ⏳ |
| Raw JSON Schema | ✅ | ✅ |
| Stream coercion (`coerce_one`/`coerce_stream`) | ✅ | ❌ |
| Partial-JSON repair | ✅ (Rust, 904 MB/s) | ⏳ (json-repair Python) |
| Output fixer (LLM-on-error retry) | ✅ | ✅ |

**Verdict:** litGraph wins on stream coercion + repair perf.

---

## 12. Resilience wrappers

| Feature | litGraph | LangGraph |
|---|---|---|
| Retry (with backoff + jitter) | ✅ | ⏳ (LangChain) |
| Fallback (provider failover) | ✅ | ⏳ |
| Rate limiter | ✅ | ⏳ |
| Token budget cap | ✅ | ❌ |
| Cost cap (USD ceiling) | ✅ | ❌ |
| PII scrubbing pre-call | ✅ | ❌ |
| Prompt caching middleware | ✅ | ⏳ |
| Timeout wrapper | ✅ | ✅ (asyncio) |
| Composes freely (decorator stack) | ✅ | ⏳ |

**Verdict:** litGraph wins. Resilience is a first-class subsystem; in
LangGraph you assemble it from LangChain bits.

---

## 13. Evaluation

| Feature | litGraph | LangGraph |
|---|---|---|
| EvalHarness driver | ✅ | ⏳ (LangSmith Eval) |
| BLEU (multi-ref) / ROUGE-N / ROUGE-L | ✅ | ❌ (Python deps) |
| chrF / chrF++ | ✅ | ❌ |
| METEOR-lite | ✅ | ❌ |
| BERTScore-lite | ✅ | ❌ |
| WER / CER (+ sub/ins/del breakdown) | ✅ | ❌ |
| TER (with shifts) | ✅ | ❌ |
| Relaxed Word Mover Distance | ✅ | ❌ |
| Pearson / Spearman / Kendall's tau-b | ✅ | ❌ |
| Paired permutation test | ✅ | ❌ |
| LLM-as-judge | ✅ | ✅ (LangSmith) |
| Pairwise evaluator | ✅ | ✅ |
| Trajectory eval | ✅ | ✅ |
| Dataset versioning + regression alerts | ✅ | 💰 (LangSmith) |

**Verdict:** litGraph has a stand-alone, self-hosted eval suite; LangGraph
defers to LangSmith (managed, paid).

---

## 14. Deployment

| Feature | litGraph | LangGraph |
|---|---|---|
| HTTP serve binary (REST + SSE) | ✅ (`litgraph-serve`) | ✅ (`langgraph-cli serve`) |
| LangGraph Cloud-API compatible endpoints | ✅ (Studio router behind feature flag) | ✅ (native) |
| Managed cloud hosting | ❌ (self-host only) | 💰 (LangGraph Cloud) |
| Studio UI (visual debugger) | ⏳ (cloud-API surface only; no local UI) | ✅ |
| Multi-tenant auth scaffolding | ❌ (planned) | 💰 |
| WebSocket endpoint | ❌ (SSE covers it) | ⏳ |

**Verdict:** LangGraph wins for managed deploys; litGraph wins for
self-host (single Rust binary, no Python runtime needed at the edge).

---

## 15. Provider coverage

### Chat

| Provider | litGraph | LangGraph |
|---|---|---|
| OpenAI (Chat Completions + Responses) | ✅ | ✅ |
| Anthropic | ✅ (+thinking blocks +prompt caching) | ✅ |
| Google Gemini (AI Studio + Vertex) | ✅ | ✅ |
| AWS Bedrock (native + Converse) | ✅ (no AWS SDK dep — pure Rust SigV4) | ✅ |
| Cohere | ✅ | ✅ |
| Mistral | ⏳ (via OpenAI-compat) | ✅ (native) |
| OpenAI-compat: Ollama / vLLM / Together / Groq / Fireworks / DeepSeek / xAI / LM Studio | ✅ | ✅ |
| Local model via candle / mistral.rs | ❌ (planned) | ❌ |

### Embeddings

OpenAI · Cohere · Voyage · Jina · Bedrock · Gemini · FastEmbed (local ONNX). litGraph has all native; LangGraph delegates to LangChain wrappers.

### Vector stores

| Store | litGraph | LangGraph |
|---|---|---|
| In-memory | ✅ | ✅ (LangChain) |
| HNSW (embedded) | ✅ (instant-distance, pure Rust) | ⏳ (faiss/hnswlib via LangChain) |
| Qdrant | ✅ (REST, no gRPC) | ✅ |
| pgvector | ✅ | ✅ |
| Chroma | ✅ | ✅ |
| Weaviate | ✅ | ✅ |
| LanceDB | 🚫 | ✅ |
| Pinecone | 🚫 | ✅ |

### Loaders

litGraph: text, JSONL, MD, dir, CSV, PDF, DOCX, Jupyter, HTML, sitemap, S3, GDrive, Confluence, Jira, Linear, Notion, Slack, Gmail, GitHub (files + issues), GitLab, Discord, Wikipedia. All rayon-parallel.

LangGraph: ~150+ via LangChain ecosystem. Wider coverage; most pull heavy Python deps.

---

## 16. Performance

Apples-to-apples micro-benches (criterion on macOS arm64). Reproduce with `cargo bench -p litgraph-bench`. LangGraph numbers approximated from `pytest-benchmark` runs of equivalent operations on the same hardware.

| Operation | litGraph | LangGraph (approx) | Speed-up |
|---|---|---|---|
| Graph fanout/64 nodes | ~ 90 µs (706 K nodes/s) | ~ 8 ms | ~ 90× |
| BM25 search / 50 K docs | ~ 2.1 ms (23.4 M elem/s) | ~ 200 ms (LangChain-dep) | ~ 100× |
| HNSW search / 100 K vecs | ~ 41 µs (2.4 G elem/s) | ~ 4 ms (faiss-py) | ~ 100× |
| SSE parse / 16 KB chunk | ~ 12 µs (1.3 GB/s) | ~ 2 ms (Python sse-starlette) | ~ 165× |
| JSON repair / 256 B | ~ 280 ns (904 MB/s) | ~ 50 µs (json-repair) | ~ 175× |
| RRF fuse / 4 × 100 lists | ~ 65 µs (6.1 M docs/s) | ~ 5 ms (LangChain) | ~ 75× |

The numbers shift case-by-case but the shape doesn't: every primitive on
the litGraph hot path is a Rust call; LangGraph hits Python.

---

## 17. Runtime / dependencies

| Dimension | litGraph | LangGraph |
|---|---|---|
| Core language | Rust | Python |
| Python binding | abi3-py39 (one wheel covers 3.9–3.13+) | n/a (it IS Python) |
| GIL behaviour | dropped (`py.detach()`) around every blocking call | held during call paths |
| Free-threaded Python 3.13t | ✅ supported | ⏳ (depends on stack) |
| Default wheel size | ~13 MB (one native .so) | ~5–50 MB (varies w/ deps) |
| Required runtime deps | none (Python stdlib only) | `langchain-core` + transitive deps |
| Optional integrations | Cargo features (zero default features) | pip extras |
| Cold-import time | < 50 ms | ~ 500 ms – 2 s |

---

## 18. When LangGraph is the better choice

- You need a **managed cloud** with autoscaling + persistence + RBAC out of the box.
- You want **LangGraph Studio** (visual graph debugger) — litGraph implements the cloud API surface but no local UI yet.
- You're **already on LangSmith** for tracing, eval, prompt management.
- You need a **niche loader / vector store** that only exists in the LangChain ecosystem (LanceDB, Pinecone, etc.).
- You're a Python shop with no Rust toolchain and don't want the build-from-source friction (PyPI wheels exist for both, but litGraph's optional crate features require a Rust compile).

---

## 19. When litGraph is the better choice

- **Throughput / latency matters.** Long agent sessions, retrieval-heavy
  workloads, fan-out across many tools — Rust gives 50–100× headroom on
  framework overhead.
- **You self-host and pay for compute.** A single Rust binary uses
  ~10× less CPU than equivalent Python for the same orchestration, so
  bills shrink.
- **You need true parallelism.** Free-threaded Python 3.13t works, but
  most stacks are still GIL-bound. litGraph drops the GIL.
- **You want slim, auditable deps.** 1 wheel (~13 MB) vs. 200+
  transitive Python deps. Easier supply-chain review.
- **You want a stand-alone eval suite.** BLEU, ROUGE, chrF, METEOR,
  BERTScore-lite, WER, TER, RWMD, statistical tests — all in-process,
  no LangSmith required.
- **You want resilience built in, not assembled.** Retry, fallback,
  rate-limit, budget, cost-cap, PII scrubbing — all native, all
  composable as decorators.

---

## 20. Migration cheat sheet

| LangGraph | litGraph |
|---|---|
| `from langgraph.graph import StateGraph, END` | `from litgraph.graph import StateGraph, END` |
| `from langgraph.checkpoint.memory import MemorySaver` | (default; pass `thread_id` to `.invoke()`) |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | `from litgraph.checkpoint import SqliteSaver` |
| `from langgraph.prebuilt import create_react_agent` | `from litgraph.agents import ReactAgent` |
| `Command(goto=..., update=...)` | return `NodeOutput.goto(...)` from a node |
| `interrupt(payload)` | `g.interrupt_before("node")` + `compiled.resume(...)` |
| `compiled.stream(state, stream_mode="values")` | `for ev in compiled.stream(state):` |
| `from langchain.prompts import ChatPromptTemplate` | `from litgraph.prompts import ChatPromptTemplate` |
| `from langchain.tools import tool` | `from litgraph.tools import FunctionTool` (or `#[tool]` macro in Rust) |
| `RunnableParallel({...})` | parallel branches in `StateGraph` (built-in) |
| `OutputParser` | `with_structured_output(Schema)` |
| `RetrievalQA` | compose `Retriever` + an agent or graph node |
| `langgraph.func.entrypoint` / `task` | `litgraph.functional.entrypoint` / `task` |
| `MessagesState` | `add_messages` reducer on a state channel |

---

## 21. Honesty notes

This doc is written by a litGraph maintainer, so:

- **Benchmarks** are micro, not end-to-end. End-to-end timings are dominated
  by the LLM call, where both frameworks are equal.
- **LangGraph numbers** are approximated from public `pytest-benchmark`
  runs and may have shifted in newer versions.
- **Feature checkmarks** for LangGraph reflect its public API as of the
  0.4.x lineage; LangChain ecosystem (LangChain Core / LangChain
  Community) supplies many of the non-graph features.
- **litGraph's "✅"** sometimes hides "shipped, not yet at LangGraph's
  scale of polish" — see [MISSING_FEATURES.md](MISSING_FEATURES.md) for
  the gaps we ourselves track.

If a row here looks unfair to LangGraph, file an issue and we'll fix it.

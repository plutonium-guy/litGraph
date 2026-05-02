# litGraph

**A production-grade, slim alternative to LangChain + LangGraph. Rust core, Python bindings.**

Hot paths (HTTP, SSE parsing, tokenization, vector math, graph scheduling, JSON repair, RRF, MMR) live in Rust. Python is a thin shim that drops the GIL around every blocking call. Result: shallow stacks, true parallelism, no class-zoo, no 200+ transitive dependencies.

[![status](https://img.shields.io/badge/iter-332-blue)](#)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![rust](https://img.shields.io/badge/rust-1.75%2B-orange)](Cargo.toml)
[![python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![tests](https://img.shields.io/badge/rust%20tests-2536%20passing-brightgreen)](#)
[![tests](https://img.shields.io/badge/python%20tests-1222%20passing-brightgreen)](#)

| Doc | What it answers |
|-----|-----------------|
| [USAGE.md](USAGE.md) | how to use every subsystem |
| [COMPARISON.md](COMPARISON.md) | litGraph vs LangChain vs LangGraph — feature-by-feature |
| [AGENTS.md](AGENTS.md) | rules for AI coding agents working in this repo |
| [AGENT_DX.md](AGENT_DX.md) | features that make Claude Code / Cursor / Cline build with litGraph easily |
| [FEATURES.md](FEATURES.md) | what's done, with status legend |
| [MISSING_FEATURES.md](MISSING_FEATURES.md) | gaps + nice-to-haves (next ~10 iters) |
| [ROADMAP.md](ROADMAP.md) | long-form prioritisation rubric |
| [ARCHITECTURE.md](ARCHITECTURE.md) | crate boundaries + design constraints |
| [KNOWLEDGE.md](KNOWLEDGE.md) | internals + trade-offs |
| [FREE_THREADING.md](FREE_THREADING.md) | Python 3.13 no-GIL notes |
| [CONTRIBUTING.md](CONTRIBUTING.md) | repo conventions |
| [RELEASING.md](RELEASING.md) | release / versioning flow |
| [CHANGELOG.md](CHANGELOG.md) | what shipped per minor |

---

## Table of contents

1. [Why litGraph](#why-litgraph)
2. [Headline benchmarks](#headline-benchmarks)
3. [Install](#install)
4. [60-second tour](#60-second-tour)
5. [Quickstart by subsystem](#quickstart-by-subsystem)
   - [Chat models & providers](#chat-models--providers)
   - [Streaming](#streaming)
   - [Structured output](#structured-output)
   - [Tools + ReAct agents](#tools--react-agents)
   - [StateGraph + parallel branches](#stategraph--parallel-branches)
   - [Functional API (`@entrypoint` + `@task`)](#functional-api)
   - [RAG with HNSW](#rag-with-hnsw)
   - [Memory + chat history](#memory--chat-history)
   - [Caching (model, embedding, semantic)](#caching)
   - [Resilience wrappers](#resilience-wrappers)
   - [Observability + OTel](#observability--otel)
   - [Checkpointing + time travel](#checkpointing--time-travel)
   - [Evaluation harness](#evaluation-harness)
   - [MCP client + server](#mcp-client--server)
   - [HTTP serve](#http-serve)
6. [Workspace layout](#workspace-layout)
7. [Provider / store / loader matrix](#supported-providers-stores-loaders)
8. [Design principles](#design-principles)
9. [Migrating from LangChain / LangGraph](#migrating-from-langchain--langgraph)
10. [FAQ](#faq)
11. [Local models](#local-models)
12. [Development workflow](#development-workflow)
13. [Versioning + releases](#versioning--releases)
14. [License](#license)

---

## Why litGraph

LangChain and LangGraph are mid-migration every quarter, pull hundreds of
optional dependencies, and burn measurable wall-clock time on Python /
asyncio overhead. For production workloads — high throughput, long-running
sessions, durable execution, real concurrency — they are not the right tool.

litGraph keeps the parts that matter:

- **Typed StateGraph** with a deterministic Kahn scheduler.
- **Tool-calling agents** built on the StateGraph (ReAct, Supervisor, Plan-Execute, Debate, Critique-Revise, Self-Consistency).
- **Streaming → Python async iterators** with sub-millisecond per-event overhead.
- **Retrieval primitives**: vector + BM25 + RRF + MMR + HyDE + multi-query + parent-document + self-query + time-weighted + ensemble.
- **Checkpointers** (SQLite / Postgres / Redis) for HITL pauses and time-travel replay.
- **Observability** via callback bus + first-class OpenTelemetry.
- **Structured output** with Pydantic / dataclass / TypedDict / JSON Schema, including stream coercion.

…and drops the rest: deprecated chain classes, 150+ niche loaders, per-feature class hierarchies that fight modern Python typing.

---

## Headline benchmarks

`criterion` micro-benches on macOS arm64 (M-series). Reproduce with `cargo bench -p litgraph-bench`.

```
graph_fanout/64       ~  90 µs       706K nodes/s   (Kahn scheduler ~1.3 µs/node)
bm25_search/50k       ~ 2.1 ms       23.4M elem/s   (rayon-parallel scoring)
hnsw_search/100k      ~  41 µs       2.4G elem/s    (107× over brute-force cosine)
sse_parse/16KB        ~  12 µs       1.3 GB/s       (zero-copy event extraction)
json_repair/256B      ~ 280 ns       904 MB/s       (partial-JSON streaming)
rrf_fuse/4×100        ~  65 µs       6.1M docs/s    (parallel rank-merge)
```

Numbers will drift; the shape is the point — every primitive is a Rust call, no Python interpreter on the hot path.

---

## Install

Wheels for PyPI are pending v1. For now: build from source.

```bash
# Prereqs (one-time):
#   Rust toolchain (stable, ≥ 1.75)  →  https://rustup.rs
#   Python 3.9+ (CPython or PyPy)
#   uv  or  pip + venv               (any modern Python installer)

git clone https://github.com/amiyamandal/litGraph.git
cd litGraph

# Build the native extension into a project venv:
uv venv
uv pip install maturin
maturin develop --release

# Optional: PEP-561 type stubs for IDE autocomplete + Pyright/mypy.
pip install ./litgraph-stubs
```

For pure-Rust use:

```toml
# Cargo.toml — depend only on the crates you need.
[dependencies]
litgraph-core = { path = "crates/litgraph-core" }
litgraph-providers-openai = { path = "crates/litgraph-providers-openai" }
litgraph-stores-hnsw = { path = "crates/litgraph-stores-hnsw" }
```

API keys come from environment by default: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, plus the AWS standard credential chain for Bedrock. Pass `api_key=...` to constructors to bypass.

---

## 60-second tour

```python
from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool

def add(args):
    return {"sum": args["a"] + args["b"]}

agent = ReactAgent(
    OpenAIChat(model="gpt-5"),
    [FunctionTool(
        "add", "Add two integers.",
        {"type": "object",
         "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
         "required": ["a", "b"]},
        add,
    )],
    system_prompt="Be terse.",
)

print(agent.invoke("What is 17 + 25?")["messages"][-1]["content"])
```

That's the whole hello-world. Below: each subsystem in 10–20 lines.

---

## Quickstart by subsystem

### Chat models & providers

```python
from litgraph.providers import (
    OpenAIChat, AnthropicChat, GeminiChat, BedrockChat, CohereChat
)

m = OpenAIChat(model="gpt-5")                 # or "gpt-4o", "gpt-4o-mini"
out = m.invoke([{"role": "user", "content": "Say hi."}])
print(out["content"])
```

OpenAI-compatible servers (Ollama, Groq, Together, Mistral, DeepSeek, xAI, Fireworks, vLLM, LM Studio, Anthropic-OpenAI-compat) reuse `OpenAIChat`:

```python
m = OpenAIChat(model="llama3", api_key="ollama",
               base_url="http://localhost:11434/v1")
```

Native multimodal:

```python
m = OpenAIChat(model="gpt-4o")
out = m.invoke([
    {"role": "user", "content": [
        {"type": "text",  "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://…/cat.png"}},
    ]},
])
```

### Streaming

```python
async for ev in m.stream([{"role": "user", "content": "Count to 5."}]):
    if ev.kind == "text":
        print(ev.text, end="", flush=True)
```

Event kinds: `text`, `tool_call_delta`, `tool_call_complete`, `thinking` (Anthropic), `usage`, `finish`. Helpers in `litgraph.agents`:

- `stream_tokens(...)` — text-only iterator for simple UIs.
- `broadcast(stream, n)` — fan a single stream out to N consumers.
- `race(streams)` — first-event-wins; cancel losers.
- `multiplex(streams)` — interleave with origin tags.
- `batch_chat(model, [msgs1, msgs2, …], max_concurrency=8)` — bounded parallel `.invoke`.

### Structured output

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float

structured = m.with_structured_output(Verdict)
v = structured.invoke([{"role": "user", "content": "Is the sky blue?"}])
print(v.answer, v.confidence)        # both typed
```

Backends: Pydantic v2, dataclasses, TypedDict, raw JSON Schema. Stream coercion via `coerce_one(...)` and `coerce_stream(...)` in `litgraph.coerce`.

### Tools + ReAct agents

```python
from litgraph.tools import FunctionTool, ShellTool, PythonReplTool, FilesystemTools
from litgraph.agents import ReactAgent

agent = ReactAgent(
    OpenAIChat(model="gpt-5"),
    [ShellTool(allow=["ls", "cat", "grep"]),
     PythonReplTool(),
     *FilesystemTools(root="./workspace")],
    system_prompt="You are a careful assistant. Inspect before acting.",
)
state = agent.invoke("Summarise the headline of every README in ./workspace.")
```

Built-in tool families: shell / python REPL / filesystem / virtual-fs / SQLite / JSON-Patch, web fetch / Tavily / DuckDuckGo / webhook, OpenAI media (Whisper / TTS / DALL·E), Gmail send.

Agent shapes: `ReactAgent` (text + native tool-call), `TextReactAgent` (full transcript), `PlanExecuteAgent`, `SupervisorAgent`, `DebateAgent`, `CritiqueReviseAgent`, `SelfConsistencyChatModel`. The `litgraph.agents.deep` factory wires retries, budgets, and tracing in one call.

### StateGraph + parallel branches

```python
from litgraph.graph import StateGraph, START, END

g = StateGraph()
g.add_node("fetch_a", lambda s: {"items": [1]})
g.add_node("fetch_b", lambda s: {"items": [2]})
g.add_node("fetch_c", lambda s: {"items": [3]})
g.add_node("merge",   lambda s: {})
g.add_edge(START, "fetch_a")
g.add_edge(START, "fetch_b")
g.add_edge(START, "fetch_c")
g.add_edge("fetch_a", "merge")
g.add_edge("fetch_b", "merge")
g.add_edge("fetch_c", "merge")
g.add_edge("merge", END)

print(g.compile().invoke({"items": []}))   # items = [1, 2, 3]
```

The three `fetch_*` branches run concurrently on the tokio worker pool — GIL released for the duration. Plain Python asyncio can't get this on a per-node basis without ceremony.

Helpers: `add_conditional_edges`, `add_send` (LangGraph-style dynamic fan-out), `subgraph(...)` (compose), `visualize()` (Mermaid / Graphviz).

### Functional API

For graphs that are mostly straight-line plus occasional fan-out:

```python
from litgraph.functional import entrypoint, task

@task
def fetch(url): ...

@task
def summarise(text): ...

@entrypoint
def pipeline(urls: list[str]):
    pages = [fetch(u) for u in urls]      # auto-parallel
    return [summarise(p) for p in pages]
```

Same checkpoint + streaming model as `StateGraph` — no boilerplate.

### RAG with HNSW

```python
from litgraph.embeddings import OpenAIEmbeddings
from litgraph.retrieval import HnswVectorStore, VectorRetriever, MMRRetriever
from litgraph.splitters import RecursiveCharacterSplitter
from litgraph.loaders import DirectoryLoader

# 1. Ingest (rayon-parallel across files).
docs = DirectoryLoader("./corpus", "**/*.md").load()

# 2. Split.
chunks = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200) \
    .split_documents(docs)

# 3. Embed + store.
emb = OpenAIEmbeddings(model="text-embedding-3-large")
store = HnswVectorStore(dim=3072)
store.add(chunks, emb.embed([c["content"] for c in chunks]))

# 4. Retrieve.
hits = MMRRetriever(VectorRetriever(emb, store), fetch_k=20, k=5,
                    lambda_mult=0.5).retrieve("What do reducers do?")
```

Retrievers in stock: `VectorRetriever`, `BM25Retriever`, `HybridRetriever` (RRF), `EnsembleRetriever`, `MMRRetriever`, `ParentDocumentRetriever`, `MultiVectorRetriever`, `MultiQueryRetriever`, `HyDERetriever`, `SelfQueryRetriever`, `TimeWeightedRetriever`, `RaceRetriever`. Rerankers: Cohere, Voyage, Jina, FastEmbed cross-encoder, `EnsembleReranker`.

### Memory + chat history

```python
from litgraph.memory import TokenBufferMemory, SummaryBufferMemory

mem = TokenBufferMemory(max_tokens=4_000, model_name="gpt-5")
mem.set_system("You are concise.")
mem.add_user("hi")
mem.add_ai("hello")
print(mem.messages())            # exact list to feed back to .invoke()
```

Backends: in-process, SQLite, Postgres, Redis. `SummaryBufferMemory` runs an LLM summary when the buffer fills. `LangMem`-style fact extractor (`litgraph_core::langmem`) distils long-term facts to a `Store`.

### Caching

```python
from litgraph.cache import CachedChatModel, MemoryCache, SqliteCache, SemanticCache

m_cached = CachedChatModel(m, cache=SqliteCache("./cache.db"))
sem = SemanticCache(emb, threshold=0.92)        # embedding-similarity
m_sem = CachedChatModel(m, cache=sem)
```

Cache layers stack: identity-cache → semantic-cache → model. Embedding-side: `CachedEmbeddings(emb, cache=...)` with SQLite-backed spool.

### Resilience wrappers

Each wrapper is a `ChatModel` decorator. They compose freely.

```python
from litgraph.resilience import (
    RetryingChatModel, FallbackChatModel, RateLimitedChatModel,
    TokenBudgetChatModel, CostCappedChatModel, PiiScrubbingChatModel,
    PromptCachingChatModel, TimeoutChatModel,
)

m = RetryingChatModel(m, max_attempts=3, base_delay_ms=200)
m = FallbackChatModel([m, OpenAIChat(model="gpt-4o-mini")])
m = RateLimitedChatModel(m, requests_per_minute=60)
m = TokenBudgetChatModel(m, max_input_tokens=20_000)
m = CostCappedChatModel(m, ceiling_usd=10.0)
m = PiiScrubbingChatModel(m)
```

Equivalents exist for embeddings, retrievers, tools, and rerankers.

### Observability + OTel

```python
from litgraph.observability import on_request, CostTracker

@on_request
def log(body):
    print("HTTP body:", body)            # solves 50% of "why is the model doing that?"

cost = CostTracker({"gpt-5": (2.5, 10.0)})    # ($/Mtok prompt, $/Mtok completion)
m = cost.wrap(m)
m.invoke([{"role": "user", "content": "hi"}])
print(cost.usd())
```

OpenTelemetry (`litgraph.tracing`):

```python
from litgraph import tracing
tracing.init_otlp(endpoint="http://localhost:4317", service_name="my-app")
# every provider call, tool call, retriever, graph step → an OTel span
```

### Checkpointing + time travel

```python
from litgraph.checkpoint import SqliteSaver

saver = SqliteSaver("./graph.db")
app = g.compile(checkpointer=saver)
state = app.invoke({"q": "..."}, config={"thread_id": "t1"})

# Resume from any prior checkpoint:
state2 = app.invoke(None, config={"thread_id": "t1",
                                  "checkpoint_id": "..." })
```

Backends: SQLite (WAL), Postgres (deadpool-pooled), Redis (ZSET-backed, O(log n) latest). The resume registry tracks running graphs across process restarts.

### Evaluation harness

```python
from litgraph.eval import EvalHarness, ExactMatch, BLEU, ROUGE, BERTScoreLite

harness = EvalHarness(
    cases=[{"input": "...", "expected": "..."}],
    target=lambda case: m.invoke([{"role":"user","content":case["input"]}])["content"],
    metrics=[ExactMatch(), BLEU(), ROUGE.l(), BERTScoreLite()],
)
report = harness.run()
print(report.summary())
```

Stock NLG metrics: BLEU (multi-ref), ROUGE-N, ROUGE-L, chrF / chrF++, METEOR-lite, BERTScore-lite, WER / CER (+ sub/ins/del breakdown), TER (with shifts), Relaxed Word Mover Distance. Statistical tests: paired permutation, Pearson / Spearman / Kendall's tau-b. LLM-judge variants via `LlmJudgeScorer`, `PairwiseEvaluator`, `TrajectoryEval`.

### MCP client + server

```python
from litgraph.mcp import McpClient

cli = await McpClient.connect_http(
    "https://mcp.example.com/mcp",
    headers=[("Authorization", "Bearer x")],
)
print(await cli.list_tools())
print(await cli.call_tool("echo", {"text": "hi"}))
```

Server: expose your `litgraph` tools as an MCP server in one call. `McpToolAdapter` lets any MCP tool drop into a litGraph agent's tool list as if native.

### HTTP serve

`litgraph-serve` exposes a compiled graph as a REST + SSE endpoint compatible with the LangGraph cloud API surface:

```bash
cargo run -p litgraph-serve --release -- --graph my.app:graph --port 8080
```

The Studio debug router (Cargo feature `studio`) adds `/runs`, `/threads`, `/checkpoints/<id>` endpoints used by the LangGraph Studio UI.

---

## Workspace layout

```
crates/
├── litgraph-core              traits + types + errors  (zero PyO3)
├── litgraph-graph             StateGraph + Kahn scheduler + checkpointers
├── litgraph-agents            ReactAgent / Supervisor / Plan-Execute / Debate / …
├── litgraph-retrieval         Retriever + BM25 + RRF + MMR + reranker traits
├── litgraph-splitters         Recursive char + Markdown + HTML + JSON + Token + Semantic
├── litgraph-loaders           Text / JSONL / MD / Dir / CSV / PDF / DOCX / Web / S3 / GDrive / …
├── litgraph-observability     Callback bus + CostTracker + GraphEvent
├── litgraph-tracing-otel      OpenTelemetry exporter (feature-gated)
├── litgraph-cache             Memory / SQLite / Semantic caches + CachedModel
├── litgraph-resilience        Retry / Fallback / RateLimit / Budget / PII / Caching
├── litgraph-tokenizers        tiktoken-rs + HF tokenizers (parallel batch)
├── litgraph-macros            #[tool] proc-macro (schemars JSON Schema derive)
│
├── litgraph-providers-openai      OpenAI + OpenAIResponses + OpenAI-compat (Ollama, Groq, …)
├── litgraph-providers-anthropic   Anthropic Messages API + thinking blocks
├── litgraph-providers-gemini      Gemini AI Studio + Vertex
├── litgraph-providers-bedrock     AWS Bedrock (native + Converse) with SigV4
├── litgraph-providers-cohere      Cohere chat + embeddings
├── litgraph-providers-voyage      Voyage embeddings
├── litgraph-providers-jina        Jina embeddings + reranker
│
├── litgraph-embeddings-fastembed  FastEmbed (ONNX, local)
├── litgraph-rerankers-fastembed   FastEmbed cross-encoder
├── litgraph-rerankers-cohere      Cohere reranker
├── litgraph-rerankers-voyage      Voyage reranker
├── litgraph-rerankers-jina        Jina reranker
│
├── litgraph-stores-memory         In-proc brute-force cosine (rayon)
├── litgraph-stores-hnsw           Embedded HNSW (instant-distance)
├── litgraph-stores-qdrant         Qdrant REST
├── litgraph-stores-pgvector       Postgres + pgvector
├── litgraph-stores-chroma         Chroma v1 HTTP
├── litgraph-stores-weaviate       Weaviate REST + GraphQL
│
├── litgraph-memory-sqlite         Chat history → SQLite
├── litgraph-memory-postgres       Chat history → Postgres
├── litgraph-memory-redis          Chat history → Redis
├── litgraph-store-postgres        General Store trait → Postgres (vector-indexed)
│
├── litgraph-checkpoint-sqlite     HITL checkpoints, WAL mode
├── litgraph-checkpoint-postgres   deadpool-pooled
├── litgraph-checkpoint-redis      ZSET-backed, O(log n) latest
│
├── litgraph-tools-search          Tavily / DuckDuckGo / WebFetch / Webhook
├── litgraph-tools-utils           Slugify / VirtualFs / SQLiteQuery / JsonPatch / …
│
├── litgraph-mcp                   MCP client + server (stdio + HTTP/SSE)
├── litgraph-serve                 axum-based REST + SSE server (LangGraph-cloud-compat)
├── litgraph-bench                 criterion micro-benches
└── litgraph-py                    PyO3 bindings (abi3-py39, shared tokio runtime)
```

43 crates total; default-features stay tight, you opt in.

---

## Supported providers, stores, loaders

### Chat models

| Provider | Crate | Notes |
|---|---|---|
| OpenAI | `litgraph-providers-openai` | Chat Completions + Responses, structured output, streaming, native tools |
| OpenAI-compat | same | Ollama, Groq, Together, Mistral, DeepSeek, xAI, Fireworks, vLLM, LM Studio, Anthropic-compat |
| Anthropic | `litgraph-providers-anthropic` | Messages API + thinking blocks + prompt caching |
| Google Gemini | `litgraph-providers-gemini` | AI Studio + Vertex; native tool-call |
| AWS Bedrock | `litgraph-providers-bedrock` | Native + Converse, SigV4 (no AWS SDK dep) |
| Cohere | `litgraph-providers-cohere` | Chat + embeddings + reranker |

### Embedding providers

OpenAI · Anthropic (via OpenAI-compat) · Gemini · Bedrock · Cohere · Voyage · Jina · FastEmbed (local ONNX).

### Vector stores

In-memory · HNSW (embedded) · Qdrant · pgvector · Chroma · Weaviate.

### Loaders (range)

Text · JSONL · Markdown · Directory · CSV · PDF · DOCX · Jupyter · HTML · Sitemap · S3 · Google Drive · Confluence · Jira · Linear · Notion · Slack · Gmail · GitHub (files + issues) · GitLab (files + issues) · Discord · Wikipedia. All ingest in parallel via rayon.

### Splitters

Recursive character · Markdown header · HTML header · JSON · Token (tiktoken / HF) · Semantic chunker · Code-aware splitters (tree-sitter) with definition-level boundaries.

---

## Design principles

1. **Rust does the work, Python is the shim.** Hot paths (HTTP, SSE parse, tokenize, embed math, vector search, JSON parse, graph scheduling, RRF, MMR) live in Rust. Python is `#[pyfunction]` and `extract_chat_model`.
2. **Parallelism is a first-class feature.** `tokio::JoinSet` for I/O fan-out; `rayon::par_iter` for CPU-bound batching; `py.detach()` (PyO3 0.22+) around every blocking call so the GIL drops.
3. **Shallow stacks.** ≤ 2 frames from user code to HTTP. No `Runnable | Runnable | Runnable` magic.
4. **Split crates, zero default features.** Each capability is its own crate behind a feature flag.
5. **Graph-first, no Runnable cathedral.** A `StateGraph` + a few functions cover what LangChain spreads across `LCEL`, `Runnable`, `Chain`, `Memory`, and N variants of each.
6. **Inspectable.** `on_request` hook exposes the final HTTP body. Solves 50% of debug pain.
7. **OTel-native.** Tracing spans + OTLP exporter; LangSmith shim for compat, not lock-in.
8. **SemVer discipline.** Slow deprecation cycle; no breaking changes every minor version.

---

## Migrating from LangChain / LangGraph

The headline primitive has the same shape:

| LangChain / LangGraph | litGraph |
|---|---|
| `from langgraph.graph import StateGraph, END` | `from litgraph.graph import StateGraph, END` |
| `from langgraph.checkpoint.memory import MemorySaver` | (default; pass `thread_id` to `.invoke()`) |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | `from litgraph.checkpoint import SqliteSaver` |
| `from langgraph.prebuilt import create_react_agent` | `from litgraph.agents import ReactAgent` |
| `Command(goto=..., update=...)` | return `NodeOutput.goto(...)` from a node |
| `interrupt(payload)` | `g.interrupt_before("node")`, then `compiled.resume(...)` |
| `compiled.stream(state, stream_mode="values")` | `for ev in compiled.stream(state):` (yields dicts) |
| `from langchain.prompts import ChatPromptTemplate` | `from litgraph.prompts import ChatPromptTemplate` |
| `from langchain.tools import tool` | `from litgraph.tools import FunctionTool` (or `@tool` macro in Rust) |
| `RunnableParallel({...})` | parallel branches in `StateGraph` (built-in) |
| `OutputParser` | `with_structured_output(Schema)` |
| `RetrievalQA` | compose `Retriever` + an agent or graph node |

The graph executor honours LangGraph's super-step semantics: parallel branches in one super-step apply via the reducer; conditional edges return next-node names.

---

## FAQ

**Is this production-ready?**
Used internally for high-traffic agent stacks. The Rust core has 2,500+ unit tests, the Python shim has 1,200+ pytest cases, all passing. Wheels are not yet on PyPI — `maturin develop` from source for now.

**How does it compare with LangChain on speed?**
On graph orchestration (the headline number): ~1.3 µs/node Kahn scheduling vs LangGraph's ~120 µs/node Python orchestration. Per-call overhead in chat invocation is dominated by the model itself; litGraph removes ~3-5 ms of Python-side glue per call.

**Does it work with my self-hosted LLM?**
If it speaks the OpenAI Chat Completions API (Ollama, vLLM, Together, Groq, Fireworks, LM Studio, …), yes — pass `base_url=...` to `OpenAIChat`.

**Does it work with Pydantic?**
Yes — `with_structured_output(MyPydanticModel)` returns instances. Stream variants in `litgraph.coerce`.

**Where do callbacks / middleware live?**
The callback bus + `on_request` hook + `before_node`/`after_node` graph events cover everything LangChain's `Callbacks` API does without the surface-area sprawl. See `litgraph-observability` crate.

**Free-threaded Python (3.13t)?**
Supported. `py.detach()` is called around every blocking section, so a no-GIL Python build can use the full tokio runtime for free. See [FREE_THREADING.md](FREE_THREADING.md).

**What's missing?**
See [MISSING_FEATURES.md](MISSING_FEATURES.md) for the short, actionable list. Headline gaps: `before_tool` / `after_tool` middleware hooks, local chat-model adapter via `candle`, WebSocket endpoint on `litgraph-serve`, vector-indexed search on more `Store` backends, auto-generated stubs via `pyo3-stub-gen`.

---

## Local models

Any OpenAI-compatible server works through `OpenAIChat(base_url=...)`. For Ollama specifically, there's a one-liner:

```python
from litgraph.providers import ollama_chat

m = ollama_chat("llama3.2")              # http://localhost:11434/v1
# Remote:
m = ollama_chat("llama3.2", base_url="http://10.0.0.5:11434/v1")
```

`vLLM`, `Together`, `Groq`, `Fireworks`, `DeepSeek`, `LM Studio`, and `Anthropic`'s OpenAI-compat endpoint all work the same way — `OpenAIChat(api_key=..., model=..., base_url=...)`.

Local embeddings via `litgraph-embeddings-fastembed` (FastEmbed ONNX, no network) and reranking via `litgraph-rerankers-fastembed`.

Roadmap item: a first-class `LocalChatModel` adapter on `candle` / `mistral.rs` so you don't even need the local HTTP server. See [MISSING_FEATURES.md](MISSING_FEATURES.md).

---

## Development workflow

```bash
# Run the full Rust suite (workspace).
cargo test --workspace --lib

# Build the Python wheel into the venv (release-optimised).
source .venv/bin/activate
maturin develop --release

# Run the Python suite (needs `pytest` and the optional deps).
pytest python_tests/

# Lint + type-check Python:
pyright python/

# Lint Rust:
cargo clippy --workspace --all-targets

# Stub-drift checker (catches new bindings missing from .pyi).
python tools/check_stubs.py
```

Conventions:

- **Commit messages**: `Add <feature> — <one-liner> (iter N)` for additive iters; `Fix <subsystem> — <bug> (iter N)` for fixes. Squash to a single semantic commit per iter.
- **No PyO3 in non-py crates.** `litgraph-py` is the single shim; every other crate is usable as a pure Rust dep.
- **Never hold the GIL across blocking I/O.** Wrap with `py.detach()`.
- **One test file per surface in `python_tests/`.** Mirrors the public API.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full set.

---

## Versioning + releases

- Crate-level `version = "0.1.0"` until the public API stabilises.
- Python wheel version mirrors the crate version.
- Pre-1.0: minor bumps may break API; pin to a specific minor in production.
- Post-1.0: SemVer applies (no breaking change in a minor).

Release flow: `maturin build --release --strip` per Python ABI tag, then a wheels-only PyPI publish. See [RELEASING.md](RELEASING.md).

---

## License

[Apache-2.0](LICENSE).

If litGraph saves your team a sprint, consider opening an issue with what you built — that's the only "marketing" the project does.

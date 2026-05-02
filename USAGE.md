# litGraph — Usage Guide

How to use the library, organised by subsystem. For *what's done* see
`FEATURES.md`; for *what's next* see `ROADMAP.md`. For deeper internals
see `ARCHITECTURE.md` and `KNOWLEDGE.md`.

This guide covers Python first (the typical entry point) with the Rust
equivalent next to it. All Rust APIs are usable standalone — `litgraph-py`
is the only crate that imports `pyo3`.

---

## Table of contents

1. [Install](#install)
2. [Chat models & providers](#chat-models)
3. [Streaming](#streaming)
4. [Structured output](#structured-output)
5. [Tools & tool-calling agents](#tools)
6. [StateGraph](#stategraph)
7. [Functional API (`@entrypoint`, `@task`)](#functional)
8. [Embeddings, vector stores & retrieval](#retrieval)
9. [RAG patterns (MMR, HyDE, multi-query, contextual compression)](#rag)
10. [Document loaders & splitters](#loaders)
11. [Memory & chat history](#memory)
12. [Caching (model, embedding, semantic)](#caching)
13. [Resilience wrappers (retry, fallback, rate-limit, budgets)](#resilience)
14. [Observability (callbacks, OTel, cost tracking)](#observability)
15. [Checkpointing & time-travel](#checkpointing)
16. [Evaluation harness](#eval)
17. [MCP client + server](#mcp)
18. [HTTP serve (`litgraph-serve`)](#serve)

---

## <a id="install"></a>1. Install

```bash
# Python (dev install from source — needs Rust toolchain + maturin):
pip install maturin
maturin develop --release

# IDE-friendly stubs (PEP 561):
pip install ./litgraph-stubs

# Rust crate use (workspace member):
[dependencies]
litgraph-core = { path = "crates/litgraph-core" }
litgraph-providers-openai = { path = "crates/litgraph-providers-openai" }
```

PyPI wheels: pending v1 release.

Required env when calling providers: `OPENAI_API_KEY`,
`ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`,
`VOYAGE_API_KEY`, AWS standard chain for Bedrock. Pass `api_key=...` to
constructors to bypass env.

---

## <a id="chat-models"></a>2. Chat models & providers

```python
from litgraph.providers import OpenAIChat, AnthropicChat, GeminiChat, BedrockChat, CohereChat

m = OpenAIChat(model="gpt-5", api_key="sk-...")
res = m.invoke([{"role": "user", "content": "Say hi"}])
print(res["content"])
```

OpenAI-compat servers (Ollama, Groq, Together, Mistral, DeepSeek, xAI,
Fireworks) all reuse `OpenAIChat`:

```python
m = OpenAIChat(model="llama3", api_key="ollama", base_url="http://localhost:11434/v1")
```

Rust:

```rust
use litgraph_providers_openai::OpenAIChat;
use litgraph_core::{ChatModel, ChatOptions, Message, Role};

let m = OpenAIChat::new("sk-...", "gpt-5");
let out = m.chat(&[Message::user("hi")], ChatOptions::default()).await?;
```

---

## <a id="streaming"></a>3. Streaming

```python
async for ev in m.stream([{"role": "user", "content": "Count to 5"}]):
    if ev.kind == "text":
        print(ev.text, end="", flush=True)
```

Event kinds: `text`, `tool_call_delta`, `tool_call_complete`,
`thinking` (Anthropic), `usage`, `finish`. See
`python/litgraph/streaming.py` and `crates/litgraph-core/src/model.rs::ChatStreamEvent`.

`stream_tokens(...)` returns text-only iterator for simple UIs.
`broadcast(stream, n)` fans the same stream out to N consumers.
`race(streams)` returns whichever stream emits first (cancels losers).
`multiplex(streams)` interleaves multiple streams with origin tags.

---

## <a id="structured-output"></a>4. Structured output

Use `with_structured_output` to enforce a Pydantic / dataclass / JSON
Schema shape:

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float

structured = m.with_structured_output(Verdict)
v: Verdict = structured.invoke([{"role": "user", "content": "Is the sky blue?"}])
```

Supports: Pydantic v2, dataclasses, TypedDict, raw JSON Schema,
plus stream coercion via `coerce_one(...)` / `coerce_stream(...)`.
See `python_tests/test_with_structured_output.py` and
`python_tests/test_coerce.py`.

---

## <a id="tools"></a>5. Tools & tool-calling agents

```python
from litgraph.tools import FunctionTool
from litgraph.agents import ReactAgent

def add(args):
    return {"sum": args["a"] + args["b"]}

add_tool = FunctionTool(
    "add", "Add two integers.",
    {"type": "object",
     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
     "required": ["a", "b"]},
    add,
)

agent = ReactAgent(m, [add_tool], system_prompt="Be terse.")
state = agent.invoke("17 + 25?")
print(state["messages"][-1]["content"])
```

Built-in tool families:
- File / shell: `ShellTool`, `PythonReplTool`, `FilesystemTools`,
  `VirtualFsTool`, `SQLiteTool`, `JsonPatchTool`
- Search / web: `TavilyTool`, `DuckDuckGoTool`, `WebFetchTool`,
  `WebhookTool`
- Cloud media: `WhisperTool`, `TtsTool`, `DalleTool`, `GmailSendTool`
- Loaders/utility: see `litgraph-tools-utils` (slugify, etc.) and
  `litgraph-tools-search`.

Agent shapes: `ReactAgent` (text-mode + native tool-call), `TextReactAgent`
(thought/action/observation transcript), `PlanExecuteAgent`,
`SupervisorAgent`, `DebateAgent`, `CritiqueReviseAgent`,
`SelfConsistencyChatModel`, `SubagentTool`. The Deep-Agent factory
(`litgraph.agents.deep`) wires retries, budgets, and tracing in one call.

Rust:

```rust
use litgraph_agents::ReactAgent;
let agent = ReactAgent::new(model, tools, "Be terse.");
let state = agent.invoke("17 + 25?").await?;
```

---

## <a id="stategraph"></a>6. StateGraph

Typed state, deterministic Kahn scheduler, parallel branches via fan-out:

```python
from litgraph.graph import StateGraph

g = StateGraph(state_schema=dict)

def planner(state):
    return {"plan": ["search", "summarise"]}

def search(state):
    return {"hits": ["a", "b"]}

def summarise(state):
    return {"summary": ", ".join(state["hits"])}

g.add_node("plan", planner)
g.add_node("search", search)
g.add_node("sum", summarise)
g.add_edge("plan", "search")
g.add_edge("search", "sum")
g.set_entry_point("plan")
g.set_finish_point("sum")

result = g.compile().invoke({})
```

Helpers: `add_conditional_edges`, `add_send` (dynamic fan-out à la
LangGraph `Send`), `subgraph(...)` (compose into a parent graph),
`visualize()` (Mermaid / Graphviz).

Parallel branches execute on a tokio JoinSet; CPU-bound nodes can use
`rayon` directly inside the closure — no GIL contention.

---

## <a id="functional"></a>7. Functional API

For graphs that are mostly straight-line plus occasional fan-out,
`@entrypoint` + `@task` removes the StateGraph boilerplate:

```python
from litgraph.functional import entrypoint, task

@task
def fetch(url): ...

@task
def summarise(text): ...

@entrypoint
def pipeline(urls: list[str]):
    pages = [fetch(u) for u in urls]   # auto-parallel
    return [summarise(p) for p in pages]
```

Same checkpoint + streaming model as `StateGraph`.

---

## <a id="retrieval"></a>8. Embeddings, vector stores & retrieval

```python
from litgraph.embeddings import OpenAIEmbeddings, FastEmbedEmbeddings
from litgraph.stores import HnswStore, QdrantStore, PgVectorStore, ChromaStore, WeaviateStore, MemoryStore

emb = OpenAIEmbeddings(model="text-embedding-3-large")
store = HnswStore(dim=3072)
store.add_documents([{"page_content": "...", "metadata": {}}], embeddings=emb)
hits = store.similarity_search("query", k=5)
```

Stores: in-memory, HNSW (rust-hnsw), Qdrant, pgvector, Chroma,
Weaviate. All implement `VectorStore`.

Retrievers: `BM25Retriever`, `HybridRetriever` (RRF fusion),
`EnsembleRetriever`, `MMRRetriever`, `ParentDocumentRetriever`,
`MultiVectorRetriever`, `MultiQueryRetriever`, `HyDERetriever`,
`SelfQueryRetriever`, `TimeWeightedRetriever`, `RaceRetriever`.

Rerankers: Cohere, Voyage, Jina, FastEmbed (cross-encoder), `EnsembleReranker`.

---

## <a id="rag"></a>9. RAG patterns

```python
from litgraph.retrieval import ContextualCompressionRetriever, MMRRetriever, HyDERetriever, MultiQueryRetriever

# MMR for diversity
mmr = MMRRetriever(base=store, fetch_k=20, k=5, lambda_mult=0.5)

# Compression with an LLM filter:
ctx = ContextualCompressionRetriever(base_retriever=mmr, llm=m)

# HyDE: write a hypothetical answer, embed *that*:
hyde = HyDERetriever(llm=m, base=store, k=5)
```

Composable: any `Retriever` plugs into any `Compressor`; both feed
into a RAG agent or a `StateGraph` node.

---

## <a id="loaders"></a>10. Document loaders & splitters

Loaders: text, JSONL, MD, directory, CSV, PDF, DOCX, Jupyter,
HTML/sitemap, S3, GDrive, Confluence, Jira, Linear, Notion, Slack,
GitHub (files + issues), GitLab (files + issues), Gmail. All ingest in
parallel via rayon.

Splitters: `RecursiveCharacterSplitter`, `MarkdownHeaderSplitter`,
`HtmlHeaderSplitter`, `JsonSplitter`, `TokenTextSplitter`,
`SemanticChunker`, code-aware splitters with tree-sitter definitions.

```python
from litgraph.splitters import RecursiveCharacterSplitter
from litgraph.loaders import DirectoryLoader

docs = DirectoryLoader("./corpus", glob="**/*.md").load()
chunks = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=100).split_documents(docs)
```

---

## <a id="memory"></a>11. Memory & chat history

```python
from litgraph.memory import TokenBufferMemory, SummaryBufferMemory

mem = TokenBufferMemory(max_tokens=4000, model_name="gpt-5")
mem.set_system("You are concise.")
mem.add_user("hi")
mem.add_ai("hello")
msgs = mem.messages()
```

Backends: in-process, SQLite, Postgres, Redis. `SummaryBufferMemory`
runs an LLM summary when the buffer fills. `LangMem`-style fact
extractor (`litgraph-core/src/langmem.rs`) distils long-term facts to a
`Store`.

---

## <a id="caching"></a>12. Caching

```python
from litgraph.cache import CachedChatModel, MemoryCache, SqliteCache, SemanticCache

cached = CachedChatModel(m, cache=SqliteCache("./cache.db"))
```

`SemanticCache` matches on embedding similarity (configurable
threshold). Embedding cache lives behind `CachedEmbeddings(emb,
cache=...)` and a SQLite-backed variant for spool persistence.

---

## <a id="resilience"></a>13. Resilience wrappers

All compose freely — they're just `ChatModel` decorators.

```python
from litgraph.resilience import (
    RetryingChatModel, FallbackChatModel, RateLimitedChatModel,
    TokenBudgetChatModel, CostCappedChatModel, PiiScrubbingChatModel,
    PromptCachingChatModel, TimeoutChatModel,
)

m2 = RetryingChatModel(m, max_attempts=3, base_delay_ms=200)
m2 = FallbackChatModel([m2, OpenAIChat(model="gpt-4o-mini")])
m2 = RateLimitedChatModel(m2, requests_per_minute=60)
m2 = TokenBudgetChatModel(m2, max_input_tokens=20_000)
m2 = CostCappedChatModel(m2, ceiling_usd=10.0)
m2 = PiiScrubbingChatModel(m2)
```

Equivalents for embeddings, retrievers, tools, and rerankers live in
the same crates.

---

## <a id="observability"></a>14. Observability

```python
from litgraph.observability import on_request, CostTracker

@on_request
def log(body):
    print("HTTP body:", body)

tracker = CostTracker()
m = tracker.wrap(m)
print(tracker.totals())
```

OpenTelemetry (`litgraph-tracing-otel`): set
`LITGRAPH_OTLP_ENDPOINT=http://localhost:4317` and emit spans for every
provider call, tool call, retriever, and graph step. The SDK installs
an OTLP exporter behind a Cargo feature flag (default-off to keep
binary size honest).

---

## <a id="checkpointing"></a>15. Checkpointing & time-travel

```python
from litgraph.checkpoint import SqliteSaver

saver = SqliteSaver("./graph.db")
app = g.compile(checkpointer=saver)
state1 = app.invoke({"q": "..."}, config={"thread_id": "t1"})

# Replay from any prior state:
state_replay = app.invoke(None, config={"thread_id": "t1", "checkpoint_id": "..."} )
```

Backends: SQLite, Postgres, Redis. Resume registry tracks running graphs
across process restarts.

---

## <a id="eval"></a>16. Evaluation harness

```python
from litgraph.eval import EvalHarness, ExactMatch, BLEU, ROUGE, BERTScoreLite, ChrF, Meteor

harness = EvalHarness(
    cases=[{"input": "...", "expected": "..."}],
    target=lambda case: m.invoke([{"role":"user","content":case["input"]}])["content"],
    metrics=[ExactMatch(), BLEU(), ROUGE.l(), BERTScoreLite()],
)
report = harness.run()
```

NLG metrics include BLEU (multi-ref), ROUGE-N, ROUGE-L, chrF / chrF++,
METEOR-lite, BERTScore-lite, WER/CER (+ sub/ins/del breakdown), TER
(with shifts), Relaxed Word Mover Distance. Statistical tests:
paired permutation, Pearson/Spearman, Kendall's tau-b. LLM-judge
(`LlmJudgeScorer`, `PairwiseEvaluator`, `TrajectoryEval`) plug into the
same harness.

---

## <a id="mcp"></a>17. MCP

Client (talks to any MCP server, stdio or HTTP/SSE):

```python
from litgraph.mcp import McpClient

cli = await McpClient.connect_http("https://mcp.example.com/mcp", headers=[("Authorization", "Bearer x")])
tools = await cli.list_tools()
result = await cli.call_tool("echo", {"text": "hi"})
```

Server: expose your `litgraph` tools as an MCP server in one call.
`McpToolAdapter` lets any MCP tool drop into a litGraph agent's tool
list as if it were native.

---

## <a id="serve"></a>18. HTTP serve

`litgraph-serve` exposes a compiled graph as a REST + SSE endpoint
compatible with the LangGraph cloud API surface:

```bash
cargo run -p litgraph-serve -- --graph my.app:graph --port 8080
```

The Studio debug router (feature `studio`) adds `/runs`, `/threads`,
and `/checkpoints/<id>` endpoints used by the LangGraph Studio UI.

---

## Where to look in the source

| Concern | File / crate |
|---|---|
| Trait definitions | `crates/litgraph-core/src/{model,embeddings,tool,store,retriever}.rs` |
| Streaming events | `crates/litgraph-core/src/model.rs` (`ChatStreamEvent`) |
| Tool macro | `crates/litgraph-macros/src/lib.rs` |
| Graph executor | `crates/litgraph-graph/src/executor.rs` |
| ReAct loop | `crates/litgraph-agents/src/react.rs`, `text_react.rs` |
| Provider HTTP | `crates/litgraph-providers-*/src/lib.rs` |
| Vector store impls | `crates/litgraph-stores-*/src/lib.rs` |
| PyO3 bindings | `crates/litgraph-py/src/*.rs` |
| Python wrappers | `python/litgraph/**/*.py` |
| Examples | `examples/*.py` |
| Per-feature tests | `python_tests/test_*.py` (one file per surface) |

---

## See also

- `README.md` — pitch + headline numbers
- `FEATURES.md` — exhaustive feature checklist
- `MISSING_FEATURES.md` — gaps + nice-to-haves (next ten iters)
- `ROADMAP.md` — long-form prioritisation rubric
- `ARCHITECTURE.md` — workspace shape, design constraints
- `KNOWLEDGE.md` — internals and trade-offs
- `FREE_THREADING.md` — Python 3.13 free-threaded notes
- `CONTRIBUTING.md` — repo conventions
- `RELEASING.md` — release/versioning flow

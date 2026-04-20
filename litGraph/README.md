# litGraph

**Production-grade, slim alternative to LangChain + LangGraph. Rust core, Python bindings.**

Everything heavy (HTTP, SSE parsing, tokenization, vector math, graph scheduling) runs in Rust. Python sees a thin shim. No GIL contention, no 6-layer call stacks, no 200+ transitive dependencies.

See [FEATURES.md](./FEATURES.md) for the full design doc and benchmark numbers.

## Why

LangChain and LangGraph are mid-migration every quarter, pull hundreds of optional deps, and burn measurable wall-clock time on Python/asyncio overhead. For production — high throughput, long sessions, durable execution — they are not the right tool.

litGraph keeps the parts that matter (typed StateGraph, checkpointers, tool-calling agents, streaming, retrieval, observability) and drops the rest (a zoo of deprecated chain classes, 150+ loader integrations, per-feature class hierarchies).

## Headline numbers (criterion, macOS arm64)

```
graph_fanout/64      90 µs       706K nodes/s    (Kahn scheduler, ~1.3µs/node)
bm25_search/50k     2.1 ms       23.4M elem/s   (rayon-parallel scoring)
hnsw_search/100k     41 µs       2.4G elem/s    ← 107× vs brute-force cosine
```

## Install (Python)

```bash
# Dev install from source — requires Rust + maturin
maturin develop --release

# Optional: install PEP 561 type stubs for IDE autocomplete + Pyright/mypy.
pip install ./litgraph-stubs

# PyPI wheels coming soon.
```

## Quickstart: tool-calling agent

```python
from litgraph.providers import OpenAIChat
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool

def add(args):
    return {"sum": args["a"] + args["b"]}

add_tool = FunctionTool(
    "add", "Add two integers.",
    {"type": "object",
     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
     "required": ["a", "b"]},
    add,
)

model = OpenAIChat(api_key="sk-...", model="gpt-5")
agent = ReactAgent(model, [add_tool], system_prompt="Be terse.")
state = agent.invoke("What is 17 + 25?")
print(state["messages"][-1]["content"])
```

## Quickstart: StateGraph with parallel branches

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

compiled = g.compile()
print(compiled.invoke({"items": []}))  # items = [1, 2, 3] (reducer merges arrays)
```

All three `fetch_*` branches run concurrently on the tokio worker pool — GIL released for the duration. Python asyncio can't do this natively.

## Quickstart: streaming token deltas

```python
model = OpenAIChat(api_key="sk-...", model="gpt-5")
for ev in model.stream([{"role": "user", "content": "Tell me a story."}]):
    if ev["type"] == "delta":
        print(ev["text"], end="", flush=True)
```

## Quickstart: RAG with HNSW

```python
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import HnswVectorStore, VectorRetriever
from litgraph.splitters import RecursiveCharacterSplitter
from litgraph.loaders import DirectoryLoader

# 1. Ingest — Rayon-parallel across files.
docs = DirectoryLoader("./corpus", "**/*.md").load()

# 2. Split.
sp = RecursiveCharacterSplitter(chunk_size=1000, chunk_overlap=200)
chunks = sp.split_documents(docs)

# 3. Embed — bring your own provider.
def my_embedder(texts):
    return call_your_embedding_api(texts)  # returns list[list[float]]

e = FunctionEmbeddings(my_embedder, dimensions=1536)
embeddings = my_embedder([c["content"] for c in chunks])

# 4. Store.
store = HnswVectorStore()
store.add(chunks, embeddings)

# 5. Retrieve.
retriever = VectorRetriever(e, store)
hits = retriever.retrieve("what does the docstring say about reducers?", k=5)
```

## Observability + caching (zero setup)

```python
from litgraph.observability import CostTracker
from litgraph.cache import MemoryCache

cost = CostTracker({"gpt-5": (2.5, 10.0)})   # ($/Mtok prompt, $/Mtok completion)
cache = MemoryCache(max_capacity=1000)

model = OpenAIChat(api_key="sk-...", model="gpt-5")
model.instrument(cost)
model.with_cache(cache)

# Every call flows through cache hash → cost-accounting callback bus.
print(cost.usd())
```

## What's in the workspace

```
litgraph-core                   traits, types, errors (no PyO3)
litgraph-graph                  StateGraph + parallel Kahn scheduler + checkpointers
litgraph-agents                 Prebuilt ReactAgent
litgraph-retrieval              Retriever + BM25 + hybrid RRF + reranker trait
litgraph-splitters              Recursive + Markdown header splitters (rayon batch)
litgraph-loaders                Text / JSONL / Markdown / Directory (rayon parallel)
litgraph-observability          Callback bus + CostTracker + OTel (feature-gated)
litgraph-cache                  Memory / SQLite / Semantic caches + CachedModel
litgraph-macros                 #[tool] proc-macro (schemars JSON Schema derivation)

litgraph-providers-openai       OpenAI + OpenAI-compatible (Ollama, vLLM, Together, Groq, …)
litgraph-providers-anthropic    Anthropic Messages API
litgraph-providers-gemini       Google Gemini generateContent / streamGenerateContent

litgraph-stores-memory          In-proc brute-force cosine (rayon)
litgraph-stores-hnsw            Embedded HNSW via instant-distance (pure Rust)
litgraph-stores-qdrant          Qdrant (REST, no gRPC dep)
litgraph-stores-pgvector        Postgres + pgvector extension

litgraph-checkpoint-sqlite      Durable HITL checkpoints (WAL mode)
litgraph-checkpoint-postgres    deadpool-pooled Postgres checkpointer
litgraph-checkpoint-redis       Redis ZSET-backed checkpointer (O(log n) latest)

litgraph-bench                  criterion micro-benches
litgraph-py                     PyO3 bindings (abi3-py39, shared tokio runtime)
```

## Design principles

1. **Rust heavy lifting, Python ergonomics.** Every hot path runs in Rust. GIL released (`py.detach`) around every blocking call.
2. **Shallow call stacks.** ≤2 frames from user code to HTTP.
3. **Split crates, zero default features.** Pay only for what you import.
4. **OTel-native.** No LangSmith lock-in.
5. **Inspectable.** `on_request` hook exposes the final HTTP body — debugs 50% of "why is the model doing that?".
6. **SemVer discipline.** Slow deprecation cycle; no breaking changes every minor version.

## Migrating from LangGraph

The headline primitive has the same shape:

| LangGraph                                          | litGraph                                           |
| -------------------------------------------------- | -------------------------------------------------- |
| `from langgraph.graph import StateGraph, END`      | `from litgraph.graph import StateGraph, END`       |
| `from langgraph.checkpoint.memory import MemorySaver` | (default; pass `thread_id` to `.invoke()`)      |
| `from langgraph.checkpoint.sqlite import SqliteSaver` | Rust crate `litgraph-checkpoint-sqlite`        |
| `from langgraph.prebuilt import create_react_agent` | `from litgraph.agents import ReactAgent`         |
| `Command(goto=..., update=...)`                    | (return `NodeOutput::goto(...)` from a node)       |
| `interrupt(payload)`                               | `g.interrupt_before("node")`, `compiled.resume()`  |
| `compiled.stream(state, stream_mode="values")`     | `for ev in compiled.stream(state):` (returns dicts)|

The graph executor honors LangGraph's super-step semantics: parallel branches
in one super-step apply via the reducer; conditional edges return next-node
names.

## Local models (Ollama / vLLM / Together / Groq / …)

Any OpenAI-compatible server works through `OpenAIChat` with a `base_url`. For
Ollama specifically, there's a one-liner:

```python
from litgraph.providers import ollama_chat

model = ollama_chat("llama3.2")              # http://localhost:11434/v1
# Or remote:
model = ollama_chat("llama3.2", base_url="http://10.0.0.5:11434/v1")
```

`vLLM`, `Together`, `Groq`, `Fireworks`, `DeepSeek`, `LM Studio`, and
`Anthropic`'s OpenAI-compat endpoint all work through `OpenAIChat(api_key=...,
model=..., base_url=...)`.

## License

Apache-2.0. See LICENSE.

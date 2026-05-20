# Migrating from LangChain / LangGraph → litGraph

Top-15 side-by-side recipes. For each idiom: the LangChain code on
the left, the litGraph equivalent on the right, plus a one-line note
on the meaningful behaviour delta where one exists.

**Conventions:**
- LangChain examples use 0.3.x / LangGraph 0.4.x APIs.
- `lc.` prefix = LangChain / LangGraph; `lg.` prefix = litGraph.
- Imports are explicit so the diff line-counts are honest.

Use `litgraph init chat-agent ./my-app` for a working starter project.
See `COMPARISON.md` for the bigger picture (perf / architecture / who
wins where).

---

## 1. Chat model — bare invoke

```python
# LangChain
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-5")
resp = llm.invoke("hi")
print(resp.content)
```

```python
# litGraph
from litgraph.providers import OpenAIChat
llm = OpenAIChat(model="gpt-5")
resp = llm.invoke([{"role": "user", "content": "hi"}])
print(resp["text"])
```

**Delta:** litGraph's `invoke` takes a `list[Message]` (uniform across
providers — no per-provider quirks); the response is a `dict` with
`text`, `tool_calls`, `usage`, `model`. Single-string prompts wrap
explicitly so the caller sees what the provider gets.

---

## 2. Streaming tokens

```python
# LangChain
for chunk in llm.stream("hi"):
    print(chunk.content, end="", flush=True)
```

```python
# litGraph
for ev in llm.stream([{"role": "user", "content": "hi"}]):
    if ev["type"] == "delta":
        print(ev["text"], end="", flush=True)
```

For IDE-narrow typed events:

```python
from litgraph import parse_stream_parts, Delta, Done
for part in parse_stream_parts(llm.stream(msgs)):
    match part:
        case Delta(text=t): print(t, end="")
        case Done(usage=u): print("\n", u)
```

---

## 3. Structured output

```python
# LangChain
from pydantic import BaseModel
class Plan(BaseModel):
    steps: list[str]
structured = llm.with_structured_output(Plan)
plan: Plan = structured.invoke("draft a plan")
```

```python
# litGraph
from litgraph.providers import StructuredChatModel
schema = {"type": "object", "properties": {"steps": {"type": "array",
         "items": {"type": "string"}}}, "required": ["steps"]}
structured = StructuredChatModel(llm, schema=schema, strict=True)
plan = structured.invoke([{"role": "user", "content": "draft a plan"}])
```

**Delta:** litGraph takes JSON Schema directly. For Pydantic shape
wrap with `coerce_one(plan, Plan)` on the way out — works for Pydantic
v1/v2, `@dataclass`, and `TypedDict` via one helper.

---

## 4. Embeddings + vector store

```python
# LangChain
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
emb = OpenAIEmbeddings()
store = Chroma.from_texts(texts, emb, collection_name="docs")
docs = store.similarity_search("query", k=4)
```

```python
# litGraph
from litgraph.embeddings import OpenAIEmbeddings
from litgraph.retrieval import ChromaStore
emb = OpenAIEmbeddings()
store = ChromaStore(emb, collection_name="docs")
store.add_documents([{"page_content": t} for t in texts])
docs = store.similarity_search("query", k=4)
```

**Delta:** litGraph's `Embeddings.embed_documents` is Rayon-parallel
under `py.detach()` — no GIL, batches scale linearly with cores.

---

## 5. RAG — one-call

```python
# LangChain
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-5"),
    retriever=store.as_retriever(search_kwargs={"k": 5}),
)
ans = chain.invoke({"query": "what's X?"})
```

```python
# litGraph
from litgraph.recipes import rag
from litgraph.providers import OpenAIChat
from litgraph.embeddings import OpenAIEmbeddings
agent = rag(documents=docs, model=OpenAIChat(model="gpt-5"),
            embeddings=OpenAIEmbeddings())
ans = agent.invoke("what's X?")
```

---

## 6. Tool-calling agent

```python
# LangChain
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool

@tool
def calc(a: int, b: int) -> int:
    """Add two ints."""
    return a + b

executor = AgentExecutor(
    agent=create_tool_calling_agent(llm, [calc], prompt=...),
    tools=[calc],
)
out = executor.invoke({"input": "what's 2+3?"})
```

```python
# litGraph
from litgraph.agents import ReactAgent
from litgraph.tools import FunctionTool

calc = FunctionTool.from_callable(
    "calc", "Add two ints.",
    {"type": "object",
     "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
     "required": ["a", "b"]},
    lambda a, b: {"sum": a + b},
)
agent = ReactAgent(llm, tools=[calc])
out = agent.invoke("what's 2+3?")
```

**Delta:** litGraph runs every emitted tool-call in parallel via
`JoinSet` (LangChain's executor serialises). No prompt template
required — native provider tool-calling.

---

## 7. Streaming agent events

```python
# LangChain
async for ev in executor.astream_events({"input": "..."}, version="v2"):
    if ev["event"] == "on_chat_model_stream":
        print(ev["data"]["chunk"].content, end="")
```

```python
# litGraph
for ev in agent.stream("..."):
    if ev["type"] == "token_delta":
        print(ev["text"], end="")
```

**Delta:** litGraph emits tagged events (`iteration_start`,
`token_delta`, `tool_call_start`, `tool_call_result`, `final`,
`max_iterations_reached`). For LangChain-shape envelopes use
`litgraph.streaming.astream_events(stream)` — translates 1:1.

---

## 8. Memory — message history

```python
# LangChain
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import SQLiteChatMessageHistory
chain = RunnableWithMessageHistory(
    llm,
    lambda session_id: SQLiteChatMessageHistory(session_id, "db.sqlite"),
    input_messages_key="messages",
)
```

```python
# litGraph
from litgraph.memory import SqliteChatHistory, BufferMemory
history = SqliteChatHistory("db.sqlite", session_id="user-1")
mem = BufferMemory(history=history)
mem.add_user_message("hi")
resp = llm.invoke(mem.messages())
mem.add_assistant_message(resp["text"])
```

For summary-style compaction: `SummaryBufferMemory(llm, max_tokens=...)`.
For topic-aware recall: `VectorStoreMemory(store)`.

---

## 9. StateGraph (LangGraph) — direct translation

```python
# LangGraph
from langgraph.graph import StateGraph, END
from typing import TypedDict

class State(TypedDict):
    n: int

def inc(s: State) -> dict:
    return {"n": s["n"] + 1}

g = StateGraph(State)
g.add_node("inc", inc)
g.add_edge("__start__", "inc")
g.add_edge("inc", END)
compiled = g.compile()
final = compiled.invoke({"n": 0})
```

```python
# litGraph
from litgraph import StateGraph
from litgraph.graph import START, END
from typing import TypedDict

class State(TypedDict):
    n: int

g = StateGraph(state_schema=State)   # typed in + out — iter 378
g.add_node("inc", lambda s: {"n": s["n"] + 1})
g.add_edge(START, "inc")
g.add_edge("inc", END)
final = g.compile().invoke({"n": 0})
```

**Delta:** litGraph's `state_schema=` auto-coerces invoke input + output
across Pydantic / dataclass / TypedDict. Otherwise the surface is 1:1.

---

## 10. Parallel fan-out

```python
# LangGraph — manual Send
from langgraph.types import Send

def fanout(state):
    return [Send("worker", {"idx": i}) for i in range(4)]

g.add_node("fanout", fanout)
g.add_node("worker", worker_fn)
g.add_edge(START, "fanout")
```

```python
# litGraph — add_parallel_for shorthand (iter 381)
g.add_parallel_for("work", 4, lambda s: {"results": [...]})
g.add_edge(START, "work_fanout")
g.add_edge("work_worker", END)
```

---

## 11. Conditional edges

```python
# LangGraph
def route(state):
    return "tools" if state["needs_tool"] else END

g.add_conditional_edges("model", route)
```

```python
# litGraph
g.add_conditional_edges(
    "model",
    lambda s: "tools" if s["needs_tool"] else END,
)
```

---

## 12. Human-in-the-loop / interrupts

```python
# LangGraph
g.compile(interrupt_before=["tools"])
# … run until interrupt fires …
compiled.update_state(config, {"approved": True})
out = compiled.invoke(None, config)
```

```python
# litGraph
g.interrupt_before("tools")
compiled = g.compile()
try:
    compiled.invoke(initial_state, thread_id="t1")
except Exception:
    pass  # interrupt fires as exception
out = compiled.resume("t1", update={"approved": True})
```

For webhook-style resume (no Python loop): mount
`litgraph_serve::resume_router` — `POST /threads/:id/resume {value}`.

---

## 13. Checkpointers

```python
# LangGraph
from langgraph.checkpoint.postgres import PostgresSaver
compiled = g.compile(checkpointer=PostgresSaver.from_conn_string(...))
```

```python
# litGraph
from litgraph.graph import PostgresCheckpointer
compiled = g.compile().with_checkpointer(
    PostgresCheckpointer(dsn="postgres://..."),
)
```

Available: `MemoryCheckpointer`, `SqliteCheckpointer`,
`PostgresCheckpointer`, `RedisCheckpointer`.

---

## 14. Time travel

```python
# LangGraph
states = compiled.get_state_history(config)
compiled.update_state({"configurable": {"thread_id": "t1",
                                         "checkpoint_id": "..."}}, ...)
```

```python
# litGraph
hist = compiled.state_history("t1")     # list of checkpoint dicts
compiled.rewind_to("t1", step=3)        # drop later checkpoints
compiled.fork_at("t1", step=3, new_thread_id="t1-fork")
```

---

## 15. Serve as HTTP

```python
# LangChain
from langserve import add_routes
from fastapi import FastAPI
app = FastAPI()
add_routes(app, llm, path="/llm")
# uvicorn app:app
```

```python
# litGraph (iter 384)
from litgraph.recipes import serve
handle = serve(llm, port=8080)
# blocks; handle.shutdown() to stop
```

Endpoints: `POST /invoke`, `POST /stream`, `POST /batch`,
`GET /health`, `GET /info`. Streaming is SSE (`text/event-stream`).

---

## Where litGraph behaves *differently* (read before porting)

- **No LCEL `|` pipes.** litGraph composes via `StateGraph` /
  `Workflow` (functional API `@entrypoint` + `@task`). Use
  `litgraph.lcel` shim if a port truly needs `|` semantics; native
  callers should switch to graphs.
- **`invoke` always takes a `list[Message]`** (dicts with `role` +
  `content`). LangChain accepts strings too. The explicit form
  removes per-call string-parsing branches.
- **Per-call tool execution is concurrent.** If your LangChain tools
  rely on global state or file-locking, audit before porting — the
  litGraph agent will run them in parallel via `JoinSet`.
- **Errors don't crash agent loops.** Tool failures, middleware
  denials, and unknown tools all surface as `Role::Tool` error
  messages the agent reacts to on the next turn (LangChain raises).
- **OTel-native.** litGraph emits `tracing::info_span!` for every
  provider / agent / graph node and ships an OTLP exporter
  (`litgraph_tracing_otel::init_otlp`). Use `litgraph trace <file>`
  to render the JSON dump as a terminal timeline.

---

## Idioms with no litGraph equivalent (yet)

- **`Runnable.with_retry()` chained on arbitrary chains** — use
  per-axis wrappers: `RetryingChatModel`, `RetryingEmbeddings`,
  `RetryingRetriever`.
- **Hub prompts (`hub.pull("rlm/rag-prompt")`)** — use
  `litgraph.prompt_hub.{Filesystem,Http,Caching}PromptHub`. Public
  registry endpoint isn't shipped.
- **LangSmith UI** — OTel + `init_langsmith()` shim sends traces to
  the LangSmith collector; the eval-suite UI is LangChain-only.

---

## Performance notes worth knowing

Numbers from `litgraph-bench` on an M-series Mac (Rust + criterion).

| Workload | LangChain (Py) | litGraph (Rust) | Speedup |
|---|---|---|---|
| BM25 build (10k docs) | 4.1s | 0.18s | 23× |
| HNSW query (100k vecs) | 320ms | 3.0ms | 107× |
| Graph fan-out (8 nodes) | 92ms | 1.1ms | 84× |

Speedups come from: rayon-parallel index build, in-process vector
math, `tokio::JoinSet` scheduling, no GIL contention on the hot
path. LLM-bound work doesn't show these deltas — the wall clock is
dominated by the model.

---

Send issues / corrections at <https://github.com/plutonium-guy/litGraph/issues>.

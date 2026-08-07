---
layout: default
title: Graphs and workflows
description: Model deterministic, typed, parallel, and durable agent workflows with StateGraph or the functional API.
eyebrow: Orchestrate
---

# Graphs and workflows

Use `StateGraph` when execution has named stages, branches, cycles, checkpoints, or human approval. Use `@entrypoint` and `@task` when the same work reads more naturally as ordinary Python.

## Build a typed graph

```python
from pydantic import BaseModel, Field
from litgraph import StateGraph
from litgraph.graph import END

class ResearchState(BaseModel):
    question: str
    queries: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    answer: str = ""

graph = StateGraph(state_schema=ResearchState)

def plan(state: ResearchState):
    return {"queries": [state.question, f"counterarguments to {state.question}"]}

def search(state: ResearchState):
    return {"evidence": [f"result for {query}" for query in state.queries]}

def write(state: ResearchState):
    return {"answer": "\n".join(state.evidence)}

graph.add_node("plan", plan)
graph.add_node("search", search)
graph.add_node("write", write)
graph.set_entry("plan")
graph.add_edge("plan", "search")
graph.add_edge("search", "write")
graph.add_edge("write", END)

app = graph.compile()
result = app.invoke({"question": "How should we evaluate agents?"})
print(result.answer)
```

With a Pydantic state schema, input and output are coerced at graph boundaries and before Python nodes or routers. Omit `state_schema`, or pass `dict`, for the zero-coercion dictionary API.

## Parallel fan-out

Edges from the same frontier run concurrently in one super-step:

```python
from litgraph.graph import StateGraph, START, END

graph = StateGraph()
graph.add_node("search_docs", search_docs)
graph.add_node("search_code", search_code)
graph.add_node("search_issues", search_issues)
graph.add_node("synthesize", synthesize)

for node in ("search_docs", "search_code", "search_issues"):
    graph.add_edge(START, node)
    graph.add_edge(node, "synthesize")
graph.add_edge("synthesize", END)

result = graph.compile().invoke({"evidence": []})
```

Partial updates are folded through the graph reducer as branches complete. A concurrency semaphore enforces `max_parallel`, and cancellation propagates to child work.

## Conditional routing

Route from state when the next node cannot be expressed as a static edge:

```python
def route(state):
    return "review" if state["confidence"] < 0.8 else "publish"

graph.add_conditional_edges(
    "draft",
    route,
    {"review": "review", "publish": "publish"},
)
```

Use dynamic `Send` values when one node needs to create a variable number of parallel sub-invocations, such as one worker per search query or document shard.

## Checkpoint and resume

```python
from litgraph.checkpoint import SqliteSaver

saver = SqliteSaver("./state/graph.db")
app = graph.compile(checkpointer=saver)

state = app.invoke(
    {"question": "..."},
    config={"thread_id": "research-42"},
)
```

Checkpoints are persisted after each super-step. Use a stable `thread_id` to address an execution across calls or process restarts. Backends include SQLite, Postgres, and Redis.

Replay a prior state by checkpoint identifier:

```python
replayed = app.invoke(
    None,
    config={
        "thread_id": "research-42",
        "checkpoint_id": "checkpoint-id",
    },
)
```

Checkpoints use a compact native representation. Model messages remain JSON-compatible at provider boundaries.

## Human-in-the-loop interrupts

Compile with interrupts before or after sensitive nodes. When an interrupt is reached, litGraph persists the current checkpoint before returning control. Resume uses the same thread and skips the already-observed `interrupt_before` once so execution can advance.

Typical approval flow:

<div class="flow"><span>draft action</span><i>→</i><span>checkpoint</span><i>→</i><span>human review</span><i>→</i><span>resume</span><i>→</i><span>execute</span></div>

Place the interrupt immediately before irreversible work—sending a message, changing production data, or spending beyond a threshold—not before pure planning stages.

## Compose subgraphs

`add_subgraph(...)` lets a parent graph treat a compiled workflow as a node. Use subgraphs to isolate state contracts, reuse a domain workflow, or give a supervisor a set of bounded specialist agents.

Keep subgraph inputs and outputs small and explicit. This makes checkpoints legible and prevents an internal state shape from leaking through the whole system.

## Functional API

For predominantly linear code, use tasks:

```python
from litgraph.functional import entrypoint, task

@task
def fetch(url: str) -> str:
    return client.get(url).text

@task
def summarize(text: str) -> str:
    return model.invoke([{"role": "user", "content": text}])["content"]

@entrypoint
def pipeline(urls: list[str]):
    pages = [fetch(url) for url in urls]
    return [summarize(page) for page in pages]
```

Tasks created in a collection can execute in parallel while preserving the graph’s checkpoint and streaming model.

## Inspect before running

Use `to_ascii()` for terminal output and `to_mermaid()` when embedding a graph in Markdown or another Mermaid-aware surface. Graph visualization is generated from the same node and edge definition that will execute.

## Scheduler model

Each super-step follows the same lifecycle:

1. Deduplicate the current frontier.
2. Check `interrupt_before` and persist if needed.
3. Spawn frontier nodes on a Tokio `JoinSet`, bounded by `max_parallel`.
4. Fold each partial update through the reducer.
5. Resolve static, conditional, explicit `goto`, or dynamic `Send` successors.
6. Check `interrupt_after`, persist the completed super-step, and continue.

The graph stops when the frontier is empty or the recursion limit is reached. This deterministic structure is the foundation for concurrency, cancellation, checkpoint replay, and observability.

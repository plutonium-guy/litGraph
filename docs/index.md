---
layout: default
title: Rust-speed agents, Python-simple
description: litGraph is a production agent framework with a Rust runtime, Python APIs, and a batteries-included development harness.
eyebrow: Production agent framework
---

<div class="hero">
  <h1>Build agents without building the framework first.</h1>
  <p class="lede">litGraph combines an explicit, easy-to-test Python agent harness with a deterministic Rust runtime for graphs, tools, retrieval, streaming, memory, and durable execution.</p>
  <div class="hero-actions">
    <a class="button primary" href="getting-started/">Build your first agent →</a>
    <a class="button" href="architecture/">Explore the architecture</a>
  </div>
  <div class="stat-row">
    <div class="stat"><strong>1 wheel</strong><span>CPython 3.9–3.13+ via abi3</span></div>
    <div class="stat"><strong>45 crates</strong><span>Pay only for the Rust pieces you use</span></div>
    <div class="stat"><strong>0 hard deps</strong><span>Python standard library by default</span></div>
  </div>
</div>

```python
from litgraph import create_agent
from litgraph.providers import OpenAIChat
from litgraph.tools import CalculatorTool

harness = create_agent(
    OpenAIChat(model="gpt-5"),
    tools=[CalculatorTool()],
    instructions="Solve the task and verify the result.",
    trace_path=".litgraph/traces.jsonl",
)

result = harness.run("What is 17 + 25?")
print(result.output)
```

The model is always explicit. Planning, virtual scratch space, streaming, event hooks, JSONL traces, and evaluation are ready when you need them.

## Pick the right level

<div class="grid">
  <a class="card" href="agent-harness/"><strong>AgentHarness</strong><span>The shortest path from a model and tools to a traced, evaluated agent.</span></a>
  <a class="card" href="graphs/"><strong>StateGraph</strong><span>Typed state, conditional routes, parallel branches, checkpoints, interrupts, and replay.</span></a>
  <a class="card" href="models-tools/"><strong>Composable primitives</strong><span>Providers, tools, middleware, streaming, structured output, and resilience wrappers.</span></a>
  <a class="card" href="python-rust/"><strong>Rust crates</strong><span>Use the core traits and specialized crates directly without depending on Python.</span></a>
  <a class="card" href="deployment/"><strong>LLM gateway</strong><span>Front Ollama or hosted OpenAI-compatible deployments with virtual keys, budgets, routing, failover, and SSE.</span></a>
</div>

## What is included

<span class="tag">ReAct</span>
<span class="tag">Plan-execute</span>
<span class="tag">Supervisor</span>
<span class="tag">Typed graphs</span>
<span class="tag">Token streaming</span>
<span class="tag">Structured output</span>
<span class="tag">HNSW</span>
<span class="tag">BM25 + RRF + MMR</span>
<span class="tag">SQLite / Postgres / Redis</span>
<span class="tag">OpenTelemetry</span>
<span class="tag">MCP</span>
<span class="tag">HTTP + SSE</span>
<span class="tag">LLM gateway</span>

The framework includes provider adapters, deterministic testing models, document ingestion, vector stores, memory, caching, retry and budget controls, evaluation metrics, checkpointing, human-in-the-loop interrupts, and HTTP serving. The [reference map](/litGraph/reference/) links every subsystem to its canonical guide, tests, and source.

## A runtime designed for agents

<div class="flow" aria-label="litGraph execution flow">
  <span>Python API</span><i>→</i><span>PyO3 boundary</span><i>→</i><span>shared Tokio runtime</span><i>→</i><span>Rust scheduler</span><i>→</i><span>provider / store</span>
</div>

Hot paths—HTTP, SSE parsing, tokenization, vector math, graph scheduling, JSON repair, RRF, and MMR—live in Rust. Python remains the orchestration surface, with the interpreter released around blocking native work. This produces shallow stacks, real parallel graph branches, and one native wheel across supported CPython versions.

## Deliberately explicit

litGraph avoids global state, import-time monkey-patching, hidden environment discovery, per-tool configuration files, and a hierarchy for every feature. Provider credentials can follow their conventional environment variables, but the model, tools, graph, storage, and policies remain visible in your code.

> **Choose litGraph** when you need predictable execution, a small runtime surface, Rust throughput, and Python ergonomics. Choose a larger ecosystem when connector breadth matters more than runtime control. See the full [framework comparison](https://github.com/plutonium-guy/litGraph/blob/main/COMPARISON.md).

## Continue

Start with the [Pixi setup](/litGraph/getting-started/), build through the [agent harness](/litGraph/agent-harness/), then add [graphs](/litGraph/graphs/) or [retrieval and memory](/litGraph/retrieval-memory/) as the solution grows.

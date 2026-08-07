---
layout: default
title: Architecture
description: Understand litGraph's crate boundaries, deterministic graph scheduler, callback bus, caches, stores, and extension points.
eyebrow: System design
---

# Architecture

litGraph is a Rust workspace with a deliberately thin Python boundary. Core traits remain independent of PyO3, specialized adapters remain optional, and the Python package adds ergonomic composition without moving hot loops back into the interpreter.

## Workspace layers

<div class="flow"><span>Application</span><i>→</i><span>Python wrappers</span><i>→</i><span>litgraph-py</span><i>→</i><span>core traits</span><i>→</i><span>specialized crates</span></div>

| Layer | Responsibilities |
|---|---|
| `litgraph-core` | Messages, model/tool/embedding/store/retriever traits, errors, and shared types. No PyO3. |
| `litgraph-graph` | StateGraph, compiled execution, deterministic scheduling, checkpoints, interrupts, and replay. |
| `litgraph-agents` | ReAct, supervisor, plan-execute, debate, critique/revise, and agent middleware. |
| `litgraph-retrieval` | Retriever and vector-store composition, BM25, RRF, MMR, reranking primitives. |
| `litgraph-loaders` / `splitters` | Parallel document ingestion and chunking. |
| `litgraph-observability` | Event callbacks, batching, instrumentation, and cost tracking. |
| `litgraph-cache` | Exact, SQLite, embedding, and semantic cache composition. |
| Provider/store/checkpoint crates | One external integration per focused crate. |
| `litgraph-macros` | Rust procedural macros such as schema-producing tool declarations. |
| `litgraph-py` | The only PyO3 crate; converts Python values and delegates work to Rust. |
| `python/litgraph` | Thin Python wrappers, decorators, harness, recipes, compatibility, and optional extras. |

This layout lets a Rust service use the scheduler or retrieval system without linking Python, while Python users receive one cohesive package.

## Design constraints

1. **PyO3 exists only in `litgraph-py`.** A Python-specific type must not leak into a core trait or adapter crate.
2. **Blocking native work releases the Python interpreter.** Parallel graph branches and provider I/O must not serialize behind the GIL.
3. **One Tokio runtime is shared per process.** Reusing it avoids runtime startup overhead and keeps spawned tasks in a consistent execution context.
4. **Message boundaries remain interoperable.** Provider messages use JSON-compatible values; internal checkpoints use a compact native representation.
5. **Adapters stay optional.** Providers, stores, and checkpointers are separate crates instead of default features on a monolith.
6. **Behavior remains explicit.** No import-time global registration, invisible model discovery, or environment-driven agent construction.

## StateGraph execution

The scheduler uses Kahn-style super-steps:

<ol class="steps">
  <li><strong>Create the frontier.</strong> Resolve nodes reachable from <code>START</code> or the previous super-step.</li>
  <li><strong>Check pre-node interrupts.</strong> Persist state before returning control.</li>
  <li><strong>Run ready nodes.</strong> Spawn them on a Tokio <code>JoinSet</code>, bounded by a semaphore.</li>
  <li><strong>Fold updates.</strong> Apply each partial result through the configured reducer.</li>
  <li><strong>Resolve successors.</strong> Combine static edges, conditional routes, explicit <code>goto</code>, and dynamic <code>Send</code> work.</li>
  <li><strong>Persist.</strong> Save the completed super-step, honor post-node interrupts, and continue.</li>
</ol>

A cancellation token is inherited by child work. Resume skips the already-hit pre-node interrupt once, preventing an immediate pause loop. The recursion limit bounds cycles and agent loops.

## Python node execution

Most graph scheduling and state folding happens in Rust. A Python node or tool requires the inverse boundary crossing: the native runtime acquires Python to invoke the callable, converts the result, then releases it before continuing native async work.

Keep Python callbacks focused on application logic. Move repeat-heavy parsing, scoring, scheduling, or vector operations into the appropriate Rust crate when profiling shows the boundary dominates.

## Observability pipeline

Producers emit typed events into a native channel. A drain task groups events by batch size or flush interval, then dispatches the batch to subscribers. This converts high-frequency token and graph events into far fewer Python boundary crossings.

`InstrumentedChatModel` wraps a model and emits start/end/error events. `CostTracker` consumes usage events and applies a price sheet. OpenTelemetry adds distributed spans behind an optional feature.

## Cache model

Two model-cache strategies solve different problems:

- **Exact cache:** hashes model, messages, and options. It is deterministic and appropriate when identical requests can reuse identical completed responses.
- **Semantic cache:** embeds the latest user request and applies cosine similarity. It is useful for tolerant FAQ-like traffic but unsafe for tool calls or precise, context-dependent work.

Both bypass token streaming because a partial event sequence is not equivalent to a completed response.

## Vector-store dispatch

Stores implement one async `VectorStore` trait. A `VectorRetriever` composes any store with any embeddings implementation. The PyO3 wrapper extracts supported Python store classes into a native trait object, preserving a single Python constructor while keeping each backend in its own Rust crate.

## Dependency direction

Core types sit at the center; adapters depend inward on them. The binding crate may depend on all exposed pieces, but those pieces never depend back on the binding.

```text
providers ─┐
stores ────┼──> litgraph-core <── litgraph-graph <── litgraph-agents
retrieval ─┘            ^
                        └──────── litgraph-py ──────> Python package
```

When a proposed feature would add Python knowledge to a Rust trait, place conversion in `litgraph-py` instead. When an adapter would add a heavy dependency to unrelated users, give it a focused crate or optional Python extras module.

## Extension points

### Add a provider

Implement `ChatModel` in `crates/litgraph-providers-<name>`. Add a Python wrapper with the established invoke/stream/cache/instrument shape, register extraction so agents accept it, and add one Python test file for the public surface.

### Add a vector store

Implement `VectorStore` in a focused store crate. Add a binding with an `as_store()` conversion and register it with the generic retriever constructor.

### Add a checkpointer

Implement the checkpointer trait in its own crate. Define the concurrency and durability semantics explicitly, then expose a thin Python wrapper.

### Add a tool

For Rust, prefer the `#[tool]` procedural macro so a typed signature produces JSON Schema through `schemars`. For Python, prefer the `@tool` decorator or `FunctionTool` with an explicit schema.

The [Python and Rust guide](/litGraph/python-rust/) covers the complete binding workflow and safety checks.

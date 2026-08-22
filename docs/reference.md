---
layout: default
title: Reference map
description: Find litGraph's canonical guides, public modules, source crates, examples, tests, and project operations documents.
eyebrow: Find anything
---

# Reference map

This site explains the main development paths. The repository documents are the exhaustive, versioned reference for feature matrices, migration details, and maintainer procedures.

## Canonical guides

| Document | Use it for |
|---|---|
| [README](https://github.com/plutonium-guy/litGraph/blob/main/README.md) | Complete project tour, benchmark snapshot, and quickstarts. |
| [USAGE](https://github.com/plutonium-guy/litGraph/blob/main/USAGE.md) | Concise examples for every public subsystem. |
| [COMPARISON](https://github.com/plutonium-guy/litGraph/blob/main/COMPARISON.md) | Feature-by-feature comparison with LangChain and LangGraph. |
| [ARCHITECTURE](https://github.com/plutonium-guy/litGraph/blob/main/ARCHITECTURE.md) | Crate boundaries and runtime constraints. |
| [FEATURES](https://github.com/plutonium-guy/litGraph/blob/main/FEATURES.md) | Implemented feature inventory and design targets. |
| [MIDDLEWARE](https://github.com/plutonium-guy/litGraph/blob/main/MIDDLEWARE.md) | Rust and Python tool-middleware surfaces. |
| [AGENT_DX](https://github.com/plutonium-guy/litGraph/blob/main/AGENT_DX.md) | Agent-builder ergonomics, CLI, recipes, and anti-features. |
| [MIGRATION_LANGCHAIN](https://github.com/plutonium-guy/litGraph/blob/main/MIGRATION_LANGCHAIN.md) | Migration patterns and compatibility helpers. |
| [FREE_THREADING](https://github.com/plutonium-guy/litGraph/blob/main/FREE_THREADING.md) | Python free-threading audit and constraints. |
| [INTEGRATION_TESTS](https://github.com/plutonium-guy/litGraph/blob/main/INTEGRATION_TESTS.md) | Live backend test setup and credential requirements. |
| [Gateway README](https://github.com/plutonium-guy/litGraph/blob/main/crates/litgraph-gateway/README.md) | Virtual keys, Ollama configuration, routing policy, streaming, and benchmarks. |

## Python surfaces

| Module | Main responsibility |
|---|---|
| `litgraph` | `create_agent`, `AgentHarness`, typed StateGraph wrapper, tasks, stream parts. |
| `litgraph.providers` | Native chat-provider adapters. |
| `litgraph.tools` | Tool contracts, `FunctionTool`, decorator, and built-ins. |
| `litgraph.agents` | ReAct and multi-agent orchestration classes. |
| `litgraph.graph` | Native graph primitives, `START`, `END`, routes, sends, and compiled execution. |
| `litgraph.retrieval` | Retrievers, fusion, MMR, compression, and reranking. |
| `litgraph.embeddings` | Native embedding adapters. |
| `litgraph.loaders` / `splitters` | Ingestion and chunking. |
| `litgraph.memory` | Conversation buffers and durable backends. |
| `litgraph.cache` | Exact, SQLite, embedding, and semantic caches. |
| `litgraph.observability` / `tracing` | Callbacks, cost tracking, and OpenTelemetry. |
| `litgraph.middleware` / `tool_hooks` | Native and Python tool policies. |
| `litgraph.mcp` | MCP client, server, and tool adapter. |
| `litgraph.serve` | Native serving bindings. |
| `litgraph.testing` | Deterministic provider and agent test doubles. |
| `litgraph.recipes` | High-level RAG, evaluation, and application recipes. |

Modules ending in `_extras` contain adapters backed by optional Python packages. Their imports remain cheap; third-party libraries load only when the adapter is constructed.

## Rust crate map

| Concern | Crate or location |
|---|---|
| Shared traits and values | `crates/litgraph-core` |
| Graph execution | `crates/litgraph-graph` |
| Agent loops and middleware | `crates/litgraph-agents` |
| Retrieval contracts and algorithms | `crates/litgraph-retrieval` |
| Document loading and splitting | `crates/litgraph-loaders`, `crates/litgraph-splitters` |
| Provider implementations | `crates/litgraph-providers-*` |
| Vector stores | `crates/litgraph-stores-*` |
| Checkpoint backends | `crates/litgraph-checkpoint-*` |
| Observability and cache | `crates/litgraph-observability`, `crates/litgraph-cache` |
| Python bindings | `crates/litgraph-py` |
| OpenAI-compatible LLM gateway | `crates/litgraph-gateway` |
| Benchmarks | `crates/litgraph-bench` |

Trait definitions live under `crates/litgraph-core/src`, streaming events in `model.rs`, graph execution under `crates/litgraph-graph/src`, and PyO3 wrappers under `crates/litgraph-py/src`.

## Learn from examples

The [examples directory](https://github.com/plutonium-guy/litGraph/tree/main/examples) contains runnable patterns. Good entry points include:

- `scripted_agent.py` for a deterministic tool-calling harness;
- `parallel_graph.py` for native fan-out and state merging;
- `rag_agent.py` for retrieval-grounded generation;
- `checkpoint_resume.py` for durable execution;
- `eval_harness.py` for quality measurement;
- provider examples for hosted and OpenAI-compatible models.

Run an example through the managed environment:

```bash
pixi run python examples/scripted_agent.py
```

## Tests as executable reference

Each public Python surface has a matching `python_tests/test_<feature>.py` file. These tests are often the fastest source for exact constructor parameters, error behavior, and result shapes. Rust unit tests live alongside each crate.

Use `python tools/check_stubs.py` after adding a native binding; it detects public classes and functions missing from the `.pyi` package.

## Project operations

| Document | Audience |
|---|---|
| [CONTRIBUTING](https://github.com/plutonium-guy/litGraph/blob/main/CONTRIBUTING.md) | Contributors adding providers, stores, tests, or benchmarks. |
| [RELEASING](https://github.com/plutonium-guy/litGraph/blob/main/RELEASING.md) | Maintainers publishing workspace versions and wheels. |
| [CHANGELOG](https://github.com/plutonium-guy/litGraph/blob/main/CHANGELOG.md) | Users checking shipped changes by version. |
| [MISSING_FEATURES](https://github.com/plutonium-guy/litGraph/blob/main/MISSING_FEATURES.md) | Contributors checking intentional gaps before building. |
| [AGENTS](https://github.com/plutonium-guy/litGraph/blob/main/AGENTS.md) | Coding agents working safely in the repository. |

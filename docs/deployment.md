---
layout: default
title: Serve and deploy
description: Package litGraph, expose graphs through HTTP and SSE, configure runtime state, and ship safely.
eyebrow: Production
---

# Serve and deploy

Production deployment has two independent choices: how the application is packaged and where durable state lives. The agent or graph definition should remain ordinary code in both local and hosted environments.

## Serve a compiled graph

The `litgraph-serve` crate exposes a graph as REST plus server-sent events:

```bash
cargo run -p litgraph-serve -- \
  --graph my.app:graph \
  --port 8080
```

The service surface is compatible with the core LangGraph cloud run model. The optional Studio router adds endpoints for runs, threads, and checkpoints used during debugging.

For an application-specific API, invoke the Python graph or `AgentHarness` from your existing ASGI/WSGI framework and translate native stream events into SSE or WebSocket messages.

## Package Python applications

Published wheels use PyO3’s stable ABI, allowing one platform wheel to cover CPython 3.9–3.13+. For source development:

```bash
pixi run develop
```

For a release wheel:

```bash
maturin build --release
```

The native extension is installed as `litgraph.litgraph`; `python/litgraph` provides the Python-first wrappers and re-exports. Build on a target compatible with the environment where the wheel will run.

## Package Rust services

Rust applications can depend on only the crates they need:

```toml
[dependencies]
litgraph-core = { version = "..." }
litgraph-graph = { version = "..." }
litgraph-providers-openai = { version = "..." }
litgraph-checkpoint-postgres = { version = "..." }
```

The core crates never require Python or PyO3. Provider, store, and checkpointer adapters are separate crates to keep compile time and binary size under control.

## Choose durable state

| State | Development | Production candidates |
|---|---|---|
| Graph checkpoints | SQLite | Postgres, Redis, managed persistent volume |
| Conversation history | Memory, SQLite | Postgres, Redis |
| Vector index | Memory, HNSW | HNSW on durable disk, Qdrant, pgvector, Weaviate |
| Model cache | Memory, SQLite | SQLite on durable disk or a shared backend where supported |
| Traces | JSONL | OpenTelemetry collector and observability backend |

SQLite is an excellent single-process default. Do not mount the same SQLite database for unconstrained concurrent writers across several hosts; move shared coordination to a service designed for it.

## Runtime configuration

Keep secrets in the deployment platform’s secret store. Provider constructors can read conventional environment variables, while non-secret application settings stay in code or `pyproject.toml`.

Common production variables include provider credentials, `LITGRAPH_OTLP_ENDPOINT`, database URLs, and service-specific endpoints. Validate required settings at startup with `litgraph doctor` or application checks rather than discovering failures on the first user request.

## Concurrency and cancellation

The Python binding shares one Tokio runtime per process. Graph branches execute on a bounded `JoinSet`; configure `max_parallel` based on provider limits, tool load, and memory—not only CPU count.

Propagate request cancellation into graph execution and external tools. Pair framework concurrency with provider rate limits and database connection pools so a fan-out cannot exceed downstream capacity.

## Production safety checklist

- Pin a compatible pre-1.0 minor release; minor versions may change APIs.
- Rebuild the native module whenever Rust sources change.
- Validate tool arguments and authorization at the execution boundary.
- Configure timeouts, retries with backoff, rate limits, token budgets, and cost ceilings.
- Place human approval before irreversible or high-impact tools.
- Use persistent checkpoints for workflows that must survive restarts.
- Redact secrets and personal data from logs and traces.
- Export health, latency, error, token, cost, and queue-depth signals.
- Test graceful shutdown and cancellation with work in flight.
- Keep a rollback artifact and checkpoint schema compatibility plan.

## Releases

The workspace version in `Cargo.toml` is the source for wheel metadata. Tags use `vX.Y.Z`; pushing a release tag triggers the trusted-publishing workflow. Because a tag can publish to PyPI, version updates, tests, changelog entries, and release notes should be complete before the tag is created.

See the repository’s [release guide](https://github.com/plutonium-guy/litGraph/blob/main/RELEASING.md) for the exact checklist and platform wheel matrix.

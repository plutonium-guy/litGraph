# Contributing to litGraph

## Quickstart

```bash
git clone https://github.com/amiyamandal/litGraph.git
cd litGraph

# Rust side
cargo build --workspace
cargo test --workspace

# Python side
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop --release
for f in python_tests/test_*.py; do python "$f"; done
```

If you're on a system where `VIRTUAL_ENV` points at a non-existent venv,
override it for cargo: `VIRTUAL_ENV=/path/to/.venv PYO3_PYTHON=/path/to/.venv/bin/python cargo test`.

## Repository conventions

- **Read [ARCHITECTURE.md](./ARCHITECTURE.md) first.** The five
  non-negotiables (no PyO3 outside `litgraph-py`, GIL released on every Rust
  call, one shared tokio runtime, bincode for checkpoints, zero default
  features) catch nearly every "this won't work" PR.
- **Workspace deps live in the root `Cargo.toml`.** Add new shared deps under
  `[workspace.dependencies]`, then reference with `dep.workspace = true` in
  member crates. This keeps versions synced across crates.
- **Edition 2021.** No `let-else && let` chains, no edition-2024 features.
- **No emojis in code or docs** (per CLAUDE.md preference).

## Pull request checklist

- [ ] `cargo fmt --all`
- [ ] `cargo clippy --workspace --all-targets -- -D warnings`
- [ ] `cargo test --workspace` passes
- [ ] If you touched `litgraph-py`: `maturin develop --release` then run
      every file in `python_tests/`.
- [ ] If you added a public API: a test that proves it works end-to-end (not
      just compiles).
- [ ] If you added a feature flag or env var: documented in README or
      ARCHITECTURE.
- [ ] No new dependencies without a one-line justification in the PR
      description (we ship slim).

## Adding a new provider

1. Create `crates/litgraph-providers-<name>/` with `Cargo.toml` referencing
   `litgraph-core.workspace = true`.
2. Implement `ChatModel`. Reuse `eventsource-stream` for SSE; reuse
   `async-stream::try_stream!` for the `stream()` impl shape.
3. Map provider stop reasons to `FinishReason` exactly — agents depend on
   `FinishReason::ToolCalls` to know when to dispatch tools.
4. Add a Python wrapper in `litgraph-py/src/providers.rs` mirroring
   `OpenAIChat`: `invoke` / `stream` / `with_cache` / `with_semantic_cache` /
   `instrument`.
5. Add the new class to `extract_chat_model` in `litgraph-py/src/agents.rs`
   so `ReactAgent` and `SupervisorAgent` accept it.

## Adding a new vector store

1. Implement `litgraph_retrieval::store::VectorStore` (async).
2. If the backend is blocking (rusqlite, sync HTTP), wrap calls in
   `tokio::task::spawn_blocking`.
3. Add a `PyXxxVectorStore` class with an `as_store() -> Arc<dyn
   VectorStore>` accessor, then extend `PyVectorRetriever::new` to accept it.

## Bench discipline

We publish bench numbers in `FEATURES.md` and `README.md`. If you add a code
path that touches one of those benches (graph scheduler, BM25, splitters,
HNSW, cache), re-run the relevant bench with `cargo bench -p litgraph-bench
--bench <name> -- --quick` and update the numbers if they shift > 10%.

## Filing bugs

Include:
- Rust version (`rustc --version`)
- Python version (`python3 --version`)
- Workspace SHA (`git rev-parse HEAD`)
- Minimal reproducer — ideally a single Python file or `cargo test` case.
- For perf bugs: criterion output before/after, not a stopwatch reading.

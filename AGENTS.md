# Agent rules for this repo

Read this first. It tells you (a coding agent — Claude Code, Cursor,
Cline, Aider, etc.) what's where, how to build, how to test, and what
not to do. Skip the discovery phase.

## What this repo is

litGraph: a production-grade slim alternative to LangChain + LangGraph.
Rust core (43 crates) + Python bindings via PyO3 0.22 + maturin. One
abi3 wheel covers Python 3.9–3.13+. Live on PyPI as `litgraph`.

Doc map: see [README.md](README.md) for the top-level index.
Subsystem how-to: [USAGE.md](USAGE.md). Comparison vs LangChain /
LangGraph: [COMPARISON.md](COMPARISON.md). What's missing:
[MISSING_FEATURES.md](MISSING_FEATURES.md). Agent-builder DX
priorities: [AGENT_DX.md](AGENT_DX.md).

## Build

```bash
# Rust workspace check / build / test:
cargo check --workspace
cargo test --workspace --lib
cargo clippy --workspace --all-targets

# Native Python wheel (rebuild after Rust changes):
source .venv/bin/activate
maturin develop --release

# Python tests (against the freshly built native module):
pytest python_tests/

# Stub-drift check (catches new bindings missing from .pyi):
python tools/check_stubs.py
```

The project lives on an external drive on macOS, which spawns
AppleDouble `._*` files. They're gitignored; don't commit them.

## Repo layout

```
crates/                    Rust workspace (43 crates)
├── litgraph-core          traits + types + errors  (zero PyO3)
├── litgraph-graph         StateGraph + Kahn scheduler
├── litgraph-agents        ReactAgent / Supervisor / etc.
├── litgraph-retrieval     Retriever / VectorStore traits + BM25 + RRF + MMR
├── litgraph-providers-*   one crate per LLM provider
├── litgraph-stores-*      one crate per vector store
├── litgraph-checkpoint-*  one crate per checkpointer backend
├── litgraph-py            ← the ONLY crate that imports pyo3
└── …                      see Cargo.toml for the full list

python/litgraph/           Python package (thin shim over native)
python_tests/              one test_<surface>.py per public API
litgraph-stubs/            PEP 561 type stubs (pip install ./litgraph-stubs)
examples/                  runnable hello-world per pattern
tools/                     check_stubs.py + future scaffolding tools
.github/workflows/         workflow.yml — PyPI Trusted Publishing
```

Trait definitions: `crates/litgraph-core/src/{model,embeddings,tool,store,retriever}.rs`.
Streaming events: `crates/litgraph-core/src/model.rs::ChatStreamEvent`.
Graph executor: `crates/litgraph-graph/src/executor.rs`.
PyO3 bindings: `crates/litgraph-py/src/*.rs`.

## Conventions

**Code:**
- **No PyO3 in non-`litgraph-py` crates.** Every other crate is usable
  as a pure Rust dep. Violating this couples the entire workspace to
  Python.
- **Always `py.detach()` around blocking I/O** in PyO3 bindings. Free-
  threaded Python 3.13t depends on it; even GIL Python wins.
- **Default to writing no comments.** Names already say *what*. A
  comment must say *why* — a hidden constraint, a workaround for a
  specific bug, surprising behaviour. No `// removed X` markers.
- **Use TaskCreate for multi-step work.** Don't batch task completion.
- **One test file per public surface in `python_tests/`.** Mirrors the
  API.
- **`#[allow(unused_imports)]`** is the canonical fix for imports used
  only in test mod (`Role`, `Arc` patterns).

**Commits:**
- Format: `<verb> <subject> (iter N)` for additive iters;
  `Fix <subsystem> — <bug> (iter N)` for fixes. Squash to a single
  semantic commit per iter.
- Body: explain the *why*, not the *what* — diff shows what.
- Footer: `Co-Authored-By: Claude Opus 4.7 …` when the tool drove the
  change.

**Dependencies:**
- Don't add a new dep without a one-line justification in the PR.
- New Cargo deps go in workspace `Cargo.toml`'s `[workspace.dependencies]`
  with a fixed minor; crates reference via `<dep>.workspace = true`.
- New Python deps go in `pyproject.toml`'s `[project.optional-dependencies]`
  group, never as a hard requirement (the project depends only on the
  Python stdlib).

**Versioning:**
- Pre-1.0: minor bumps may break the API. Pin to a specific minor in
  prod.
- Tag = `vX.Y.Z`; the workflow.yml at `.github/workflows/` builds +
  publishes via PyPI Trusted Publishing on the tag push.
- Bump `[workspace.package].version` in `Cargo.toml` *before* tagging
  (the wheel's metadata is read from there).

## When you (the agent) get stuck

In rough order:

1. **Read [MISSING_FEATURES.md](MISSING_FEATURES.md).** The thing might
   be intentionally not shipped.
2. **Read [USAGE.md](USAGE.md).** Subsystem how-to with code per
   section.
3. **`cargo clippy --workspace --all-targets`** flags real bugs, not
   just style.
4. **`python tools/check_stubs.py`** flags binding ↔ .pyi drift.
5. **`grep -rn "<symbol>" crates/litgraph-core/src/`** before walking
   into provider crates — most things are defined in core.
6. **Look at `python_tests/test_<feature>.py`** — every public API has
   one and the tests show idiomatic usage.

## Common gotchas

- **External drive on macOS** spawns `._*` AppleDouble files in `.git/`
  and stub dirs. They're harmless (`gitignore`d) but break tools that
  glob `*.pyi`. Solution: filter `if pyi.name.startswith("._")`.
- **Two Python interpreters.** Project venv at `.venv/bin/python` is
  what `maturin develop` updates. Homebrew `/opt/homebrew/bin/python3`
  has a *separate* litgraph install that won't reflect rebuilds.
  Always run tests via `source .venv/bin/activate` first.
- **PyPI canonicalises names.** Project is `litGraph` on the PyPI page
  (display) but the canonical name + `pip install` target is
  `litgraph` (lowercase). The trusted-publisher binding uses the
  canonical form.
- **`manylinux: auto` ≠ `manylinux: 2_28`.** The default rejects ring's
  ARMv8 assembly under qemu-aarch64. workflow.yml uses `2_28`. Don't
  downgrade.
- **macos-13 (Intel) GH-hosted runners are queue-starved.** Wheel
  matrix is aarch64-only; Intel-Mac users build from sdist.

## What NOT to do

- 🚫 Add per-feature `AutoX` magic that guesses config from the
  environment. Magic helps humans and hurts agents.
- 🚫 Hide global state (singletons, monkey-patching on import).
- 🚫 Put PyO3 anywhere outside `crates/litgraph-py/`.
- 🚫 Hold the GIL across blocking I/O.
- 🚫 Add per-tool config files (`.litgraphrc`). pyproject.toml only.
- 🚫 Add LangChain as a dep. The whole point of this project is to
  *not* depend on LangChain.
- 🚫 Remove deprecation warnings without naming the replacement in the
  warning text.

## When in doubt

Ask before:
- Pushing tags (publishes to PyPI — irreversible).
- Force-pushing to main.
- Deleting a checkpointer backend.
- Bumping the rust-version floor (currently 1.75).
- Adding a feature flag whose default would change existing behaviour.

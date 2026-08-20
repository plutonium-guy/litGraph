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

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **litGraph** (17528 symbols, 42150 relationships, 267 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> Index stale? Run `node .gitnexus/run.cjs analyze` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? `npx gitnexus analyze` (npm 11 crash → `npm i -g gitnexus`; #1939).

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows. For regression review, compare against the default branch: `detect_changes({scope: "compare", base_ref: "main"})`.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `query({search_query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `context({name: "symbolName"})`.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method without first running `impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit changes without running `detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/litGraph/context` | Codebase overview, check index freshness |
| `gitnexus://repo/litGraph/clusters` | All functional areas |
| `gitnexus://repo/litGraph/processes` | All execution flows |
| `gitnexus://repo/litGraph/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

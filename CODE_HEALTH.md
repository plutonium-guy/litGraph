# Code health — where the structure hurts and what to do

**Snapshot:** 2026-07-31 · commit `e8163aa` + working-tree refactor ·
128,880 lines of Rust across 43 crates · 2,573 workspace lib tests
passing.

Derived from a GitNexus knowledge-graph index of the workspace
(16,804 nodes / 40,559 edges) rather than from reading files in
isolation, so the claims here are about *relationships* — what depends
on what, what nothing depends on, what is copied where.

Companion docs: `ROADMAP.md` (what to build next), `MISSING_FEATURES.md`
(capability gaps), `REMAINING_WORK.md` (what's blocked). This file is
about the shape of the code we already have.

Every number below is reproducible with the command shown beside it.

---

## TL;DR — five problems, ranked

| # | Problem | Evidence | Effort |
|---|---|---|---|
| 1 | `litgraph-core` is a 29.6k-line monolith that 39 crates depend on | charter files = 779 lines, crate = 29,642 | L |
| 2 | ~3.6k lines in core have no consumer in the repo | 4 modules, zero inbound edges | S |
| 3 | Loader HTTP/builder boilerplate copied up to 18× | `with_timeout` in 18 files | M |
| 4 | The executor everyone runs through is the thinnest-tested crate | `litgraph-graph`: 1 test / 53 lines | M |
| 5 | 232 callers construct a stringly-typed catch-all error | `Error::other` fan-in = 232 | M |

Highest value per unit of risk: **#3**, then **#1**. **#4** is the only
one that pays off in bugs caught rather than lines deleted.

---

## 1. `litgraph-core` has outgrown its charter

`AGENTS.md` describes `litgraph-core` as *"traits + types + errors
(zero PyO3)"*. Those files total **779 lines**:

```
crates/litgraph-core/src/{model,tool,store,error,message}.rs
```

The crate is **29,642 lines across 59 files** — 23% of all Rust in the
workspace. It is also the dependency hub: **39 of 43 crates** list it,
and it absorbs **674+ incoming cross-crate references, roughly 8× the
next crate**. Any edit to any of those 59 files invalidates almost the
whole workspace's incremental build, and every downstream consumer of
the published crate compiles all of it.

The eval subsystem is the cleanest thing to lift out:

| File | Lines |
|---|---|
| `evaluators.rs` | 2,673 |
| `eval_significance.rs` | 1,143 |
| `eval_harness.rs` | 750 |
| `dataset_version.rs` | 684 |
| `eval_correlation.rs` | 654 |
| `eval_bertscore.rs` | 642 |
| `eval_effect_size.rs` | 409 |
| `eval_drift.rs` | 364 |
| `eval_bootstrap.rs` | 349 |
| `eval_synth.rs` | 334 |
| **total** | **8,002** |

That is **27% of the crate**, and **no Rust crate uses any of it**. The
single consumer is `litgraph-py`. Verified two ways — graph edges and
grep:

```bash
grep -rlE --include='*.rs' 'litgraph_core::(evaluators|eval_)' crates/ \
  | grep -v litgraph-core
# -> crates/litgraph-py only
```

### Do this

1. Create `litgraph-eval` with the ten files above. It depends on
   `litgraph-core` for `Error`/`Message`; nothing depends on it except
   `litgraph-py`.
2. Re-export from `litgraph-core` behind a deprecated `pub use` for one
   release so the published API doesn't break, then drop it.
3. Repeat the same test for the next tier once eval is out:
   `batch.rs` (1,371 — 3 consumers, keep), `tool_dispatch.rs` (1,196 —
   `litgraph-py` only), `semantic_store.rs` (758 — `litgraph-py` only).
   A `litgraph-py`-only module in the hub crate is a smell every time.

The goal state is a core that matches its own docs: traits, types,
errors, and the handful of things genuinely shared by 39 crates.

---

## 2. Four modules nothing references

| Module | Lines | Consumers outside core |
|---|---|---|
| `embed_batch.rs` | 1,217 | none |
| `eval_significance.rs` | 1,143 | none |
| `dataset_version.rs` | 684 | none |
| `tool_offload.rs` | 594 | none |
| **total** | **3,638** | |

Not used by other crates, not used by the Python bindings, not used by
examples. Their apparent internal fan-in (`probe` 25 callers,
`make_report` 25) is their own test modules calling themselves.

These overlap with #1: `eval_significance.rs` and `dataset_version.rs`
(1,827 lines) travel with the `litgraph-eval` extraction. Only
`embed_batch.rs` and `tool_offload.rs` (1,811 lines) need a separate
decision. The two counts are not additive.

**This is not proof of dead code.** `litgraph-core` is published, so
`pub` items are API for downstream users we cannot see from here. But a
module with no binding, no internal caller, and no example is either an
unfinished feature or an accidental export.

### Do this

For each of the four, pick one and record the decision here:

- **Ship it** — add the PyO3 binding + a `python_tests/` file + an
  entry in `FEATURES.md`. It becomes a real feature.
- **Gate it** — move behind a non-default Cargo feature so the other
  42 crates stop compiling it.
- **Drop it** — delete, with a `CHANGELOG.md` note.

Doing nothing is the current state and costs everyone build time.

---

## 3. Loader boilerplate is copy-pasted up to 18 ways

Inside `litgraph-loaders` (33 files), the same builder and HTTP plumbing
is reimplemented per loader:

| Function | Files defining it |
|---|---|
| `with_timeout` | 18 |
| `with_base_url` | 15 |
| `with_user_agent` | 12 |
| `client` | 11 |
| `authed` | 9 |
| `urlencode` | 6 |
| `decode_entities` | 5 |
| `issue_to_document` | 5 |
| `fetch_page` / `message_to_document` | 4 each |

```bash
gitnexus cypher "MATCH (f:Function) RETURN f.name AS fn, \
  count(DISTINCT f.filePath) AS files ORDER BY files DESC" -l 40
```

Every new loader re-types the same twelve methods, and a fix to one
(retry semantics, a header, a timeout default) reaches exactly one of
the eighteen.

### Do this

Introduce a shared `HttpLoaderBase` — a struct holding
`client`/`base_url`/`timeout`/`user_agent`/`auth`, plus a
`#[derive(HttpLoaderBuilder)]` or a small declarative macro that
generates the `with_*` chain. Each loader then supplies only:

1. its endpoint/query construction,
2. its response → `Document` mapping.

Deletes roughly a thousand lines, makes a new loader a config struct
plus one parse function, and gives one place to fix transport bugs.
The crate already has 400 passing tests, so the refactor is verifiable
step by step — convert loaders one at a time, not in a big bang.

---

## 4. Test density is inverted against risk

Tests per line, by crate:

| Crate | Lines | Tests | Lines/test |
|---|---|---|---|
| `litgraph-tools-utils` | 10,522 | 324 | 32 |
| `litgraph-core` | 29,642 | 888 | 33 |
| `litgraph-retrieval` | 12,137 | 261 | 46 |
| `litgraph-loaders` | 19,527 | 403 | 48 |
| **`litgraph-graph`** | **1,552** | **29** | **53** |
| `litgraph-agents` | 5,283 | 86 | 61 |

`litgraph-graph` holds `StateGraph` and the Kahn scheduler — the code
path *every* workflow executes. It is the thinnest-covered crate of the
group, and `graph.rs` (the 234-line builder) has no unit test module at
all, only integration coverage in `tests/graph.rs`.

`litgraph-py` (16,085 lines) has zero Rust tests by design — it is
covered by 293 files in `python_tests/`. That is a defensible strategy
for a binding layer, but it does mean the largest non-core crate cannot
be tested without `maturin develop --release` first, making it the
slowest feedback loop in the repo.

### Do this

`Scheduler::execute` was just decomposed into seven helpers with clean
boundaries. That decomposition is what makes the following testable —
they were unreachable inside a 218-line block:

- `normalize_frontier` — pure function. Property-test it: END stripping,
  `Normal` dedup, `Forked` entries never deduped, order preserved.
- `check_interrupt_before` — assert the `skip_interrupt_before_once`
  latch fires exactly once across a resume.
- `fold_node_success` — assert `interrupt_after` cancels siblings and
  the checkpoint lands *before* the `Interrupted` error propagates.

Target: `litgraph-graph` at or below 35 lines/test, matching core.

---

## 5. A third of error construction is stringly-typed

| Constructor | Callers |
|---|---|
| `Error::other` | 232 |
| `Error::invalid` | 88 |
| `Error::provider` | 74 |

`Error::other(String)` is the single most-called function in the
workspace. A caller receiving one cannot match on *why* something
failed — only re-parse the message.

This already has a load-bearing consequence:
`litgraph-resilience::is_transient` decides what to retry by
**string-matching 5xx patterns inside `Error::Provider(s)`**. Retry
correctness currently depends on providers formatting their error
strings consistently.

### Do this

1. Add typed variants for the cases resilience and callers actually
   branch on: `RateLimited { retry_after_ms }` (exists), `Timeout`
   (exists), plus `Http { status: u16, body: String }` and
   `Serde { .. }`.
2. Convert the ~74 `Error::provider` sites in the provider crates
   first — that is where `is_transient` reads.
3. Rewrite `is_transient` to match variants instead of substrings.
4. Leave `Error::other` for genuinely unclassifiable cases; the target
   is not zero, it is "not the most-called function in the codebase".

---

## Already addressed (2026-07-31)

Landed in the working tree, verified at 2,573 tests passing and clippy
unchanged from baseline:

| Before | After |
|---|---|
| `litgraph-resilience/src/lib.rs` — 1 file, 8,363 lines | 16 modules, largest 1,086 |
| `Scheduler::execute` — 218 lines, 57% of its file | 24-line orchestrator + 7 helpers |
| bedrock `stream` ×2 — 258 + 172 lines | 45 + 35 lines |
| `html_to_markdown` — 157 lines | 13 lines |
| s3 `load` — 148 lines | ~55 lines |

Known follow-up: the resilience split duplicated shared test fixtures
(`AlwaysOkModel`, `FlakyEmbed`, `DelayChatModel` and friends) into each
module's test mod, adding ~1,100 lines. Collapse them into a single
`#[cfg(test)] mod testutil` inside the crate.

---

## Deliberately not doing

**The two "circular imports" a graph scan reports** —
`litgraph-graph/src/{graph,scheduler}.rs` and
`litgraph-py/src/{agents,middleware}.rs`. Both are intra-crate Rust
module cycles, which the compiler resolves and which are idiomatic.
The finding is an artifact of language-agnostic file-import analysis.
Breaking them would mean inventing a third module to satisfy a linter.
Recorded here so it doesn't get re-raised on the next scan.

---

## Re-running this analysis

```bash
npx -y gitnexus analyze .          # ~22s, writes .gitnexus/ (git-excluded)
gitnexus cypher "<query>"          # ad-hoc graph queries
gitnexus check --cycles            # structural checks
```

The index is per-commit; `gitnexus status` reports staleness. Note that
cluster and process labels are unenriched (no embedding endpoint was
configured at index time), so `query` does symbol lookup rather than
concept search — the structural layer used throughout this document is
unaffected.

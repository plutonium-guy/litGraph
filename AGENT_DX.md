# Agent-Builder DX — Features for Claude Code, Cursor, Cline, etc.

What would make litGraph the easiest framework for an **AI coding
assistant** to build production apps with? This doc lists concrete,
shippable features ordered by impact-per-effort.

The constraint that drives this doc: a coding agent has limited context,
no human intuition, and pays for every wrong guess. Every feature here
either (a) makes the right code obvious from one file or one search, or
(b) makes the wrong code surface a fast, fixable error.

**Symbols:** ✅ already shipped · 🚧 planned · 💡 proposal (file an
issue if you want it sooner). Effort: **S** ≤ 1 day · **M** ≤ 1 week ·
**L** > 1 week.

---

## Table of contents

1. [Discoverability](#discoverability)
2. [Self-contained docs (where the agent reads)](#self-contained-docs)
3. [Sane defaults + zero-config quickstarts](#sane-defaults)
4. [Error messages that fix themselves](#error-messages)
5. [Type stubs + IDE introspection](#type-stubs)
6. [CLI scaffolding](#cli-scaffolding)
7. [`litgraph doctor` — env diagnostic](#doctor)
8. [`AGENTS.md` / `CLAUDE.md` shipped in the repo](#agents-md)
9. [MCP server for litGraph itself](#mcp-server)
10. [Recipe library + idiom examples](#recipes)
11. [Schema-first contracts](#schema-first)
12. [Trace replay + debug-by-trace-id](#trace-replay)
13. [One-call factories](#one-call-factories)
14. [Doc search CLI](#doc-search)
15. [Skill plugin for Claude Code](#skill-plugin)
16. [Test fixtures the agent can copy-paste](#test-fixtures)
17. [Versioning ergonomics](#versioning)

---

## <a id="discoverability"></a>1. Discoverability

The agent finds the right symbol in one query or it loses ~5 minutes of
context guessing. Optimise the import surface for grep.

| Feature | Status | Effort |
|---|---|---|
| Top-level re-exports of every public type from `litgraph` | ✅ partial | S |
| `litgraph.__all__` exhaustive (so `dir(litgraph)` is complete) | ✅ (iter 339) | S |
| Single import path per concept — no `litgraph.foo.bar.Foo` aliases | ✅ | — |
| `litgraph.<subsystem>.__all__` with one-liner module docstring | 🚧 | S |
| README has a 60-second copy-pasteable hello-world | ✅ (iter 333) | — |
| **`litgraph.recipes` namespace** with one-call patterns (eval, serve; rag/multi_agent planned) — same shape as scikit-learn's `datasets` | ✅ partial (iter 339) | M |

**Why Claude Code cares:** when the agent grep-searches "ReactAgent",
it should find one canonical import path. Multiple aliases
(`from litgraph.agents.react import ReactAgent`,
`from litgraph import ReactAgent`, …) double the failure modes when
the agent picks the wrong one.

---

## <a id="self-contained-docs"></a>2. Self-contained docs (where the agent reads)

Coding agents read three places: docstrings (via introspection),
README, and one or two top-result examples. Every public API needs at
least *one* of these to contain a runnable snippet.

| Feature | Status | Effort |
|---|---|---|
| Every `pyfunction` has a docstring with example | ⏳ | M |
| Every `pyclass` has a docstring with one-line example | ⏳ | M |
| Module docstrings list "common patterns" with code | 🚧 | M |
| README ships per-subsystem 10–20-line snippets | ✅ (iter 333) | — |
| USAGE.md as the "long-form quickstart" doc | ✅ | — |
| **💡 `litgraph examples list`** — CLI that prints the example index | 💡 | S |
| **💡 Examples are tested in CI** (every example file is a pytest case) | 💡 | M |

**Why Claude Code cares:** when the agent calls `help(ReactAgent)`, the
docstring should be enough to write a working call without consulting
external docs. CI-tested examples mean copy-paste won't ship broken
code.

---

## <a id="sane-defaults"></a>3. Sane defaults + zero-config quickstarts

The agent guesses defaults wrong ~30% of the time. Defaults should be
"usable in dev, opinionated for prod."

| Feature | Status | Effort |
|---|---|---|
| `OpenAIChat()` works with `OPENAI_API_KEY` env | ✅ | — |
| `model="gpt-5"` default if user doesn't specify | ⏳ (errors instead) | S |
| `RetryingChatModel(m)` with sane retry policy by default | ✅ | — |
| `StateGraph()` accepts no args for "I don't care about typed state" | ⏳ | S |
| `ReactAgent(m, tools)` works without `system_prompt` | ✅ | — |
| **💡 `quickstart()` factory** — pick reasonable model/embeddings/store/tools from env | 💡 | M |
| **💡 `from_env()` constructors** on every provider | 💡 | S |

**Why Claude Code cares:** when the agent writes `OpenAIChat(model=...)`
and forgets `api_key=...`, it should still work because env was set.
Demanding every arg amplifies the agent's wrong-guess rate.

---

## <a id="error-messages"></a>4. Error messages that fix themselves

Every error message should answer "what should I have done?". This is
the highest-leverage DX investment for AI coders.

| Feature | Status | Effort |
|---|---|---|
| `OPENAI_API_KEY` missing → "set OPENAI_API_KEY env or pass api_key=..." | ✅ | — |
| Wrong message shape → show the expected schema | ⏳ | M |
| `with_structured_output(BadType)` → list valid types | ⏳ | S |
| Tool schema mismatch → diff vs expected | ⏳ | M |
| Missing `thread_id` for checkpointer → suggest it | ⏳ | S |
| Provider 4xx → echo the body in the error (provider gave the answer) | ✅ | — |
| **💡 "Did you mean?" hints** for typo'd kwargs | 💡 | M |
| **💡 Pretty multi-line panic** with file:line + suggested fix | 💡 | M |

**Why Claude Code cares:** the agent reads the error, plans a fix, and
re-runs. If the error doesn't name the fix, the agent chains 3+ wrong
fixes before stumbling on the right one — and burns context every
iteration.

---

## <a id="type-stubs"></a>5. Type stubs + IDE introspection

| Feature | Status | Effort |
|---|---|---|
| PEP 561 `litgraph-stubs` package on PyPI | ⏳ (built, not yet published) | S |
| Every native binding has a `.pyi` entry | ✅ (verified by `tools/check_stubs.py`) | — |
| Stub-drift CI gate | ⏳ | S |
| Pydantic / TypedDict on every state schema | ✅ | — |
| `pyo3-stub-gen` auto-generation (replace hand-rolled stubs) | 🚧 (roadmap) | L |
| Generic types preserved across decorators (`with_structured_output[T]`) | ⏳ | M |

**Why Claude Code cares:** tools like Pyright run inside Claude Code and
flag type errors before the agent ships code. Accurate stubs cut the
"agent ships type-error code" rate ~80%.

---

## <a id="cli-scaffolding"></a>6. CLI scaffolding

A coding agent that needs to bootstrap a new project is most efficient
when there's a `--template` flag.

| Feature | Status | Effort |
|---|---|---|
| `litgraph init <template>` scaffolds a minimal repo | 🚧 (planned in MISSING_FEATURES) | M |
| Templates: `chat-agent`, `rag`, `react-agent`, `multi-agent`, `eval-suite`, `serve` | 💡 | M |
| Scaffolded repo includes: `pyproject.toml`, `.env.example`, `tests/`, `README.md`, `Makefile` | 💡 | S |
| Scaffolded repo runs `pytest` green out of the box (with mock provider) | 💡 | M |
| `litgraph add-tool <name>` generates a stub tool with schema + test | 💡 | M |
| `litgraph add-node <name>` adds a graph node + wiring | 💡 | M |

**Why Claude Code cares:** "create a new RAG agent" → `litgraph init rag
my-app && cd my-app && pytest` — the agent's first run is green, then
it iterates on real concerns. No 30-minute "set up the project"
detour.

---

## <a id="doctor"></a>7. `litgraph doctor` — env diagnostic

Half of "it doesn't work" issues are env-related. A `doctor` subcommand
saves the agent from chasing them.

| Feature | Status | Effort |
|---|---|---|
| Native module built + version matches | ✅ (iter 339) | S |
| API key env vars set per provider | ✅ (iter 339) | S |
| Connectivity test to each configured provider | 🚧 | S |
| Tokenizer files downloaded (HF cache) | 🚧 | S |
| Postgres / Redis / SQLite reachable (if configured) | 🚧 | S |
| Free-threaded vs GIL Python build | ✅ (iter 339) | S |
| `litgraph doctor --json` for agent consumption | ✅ (iter 339) | S |

**Why Claude Code cares:** "why doesn't this work?" → `litgraph doctor`
prints a 6-line diagnostic. The agent gets exact-error→exact-fix
mapping in one shell call.

---

## <a id="agents-md"></a>8. `AGENTS.md` / `CLAUDE.md` in the repo

Convention shared across Claude Code, Cursor, Cline, Aider. A
top-level `AGENTS.md` that codifies "how to work in this repo" gives
every coding agent the right context on first read.

| Feature | Status | Effort |
|---|---|---|
| Top-level `AGENTS.md` with: build cmds, test cmds, lint, repo layout, conventions | ✅ (iter 339) | S |
| Per-subsystem `AGENTS.md` (e.g. `crates/litgraph-py/AGENTS.md`) | 💡 | M |
| `AGENTS.md` shipped in the wheel so `pip install litgraph` includes it | 💡 | S |
| Pointer from README to AGENTS.md | ✅ (iter 339) | S |

**What `AGENTS.md` should contain:**

```markdown
# Agent rules for this repo

## Build
- `maturin develop --release` rebuilds the native module.
- `cargo test --workspace --lib` for Rust tests.
- `pytest python_tests/` for Python tests (uses the project venv).

## Conventions
- No PyO3 in non-`litgraph-py` crates.
- Always `py.detach()` around blocking I/O.
- Commit messages: `<verb> <subject> (iter N)`.
- Don't add new dependencies without a justification line in the PR.

## Where things live
- Rust traits: crates/litgraph-core/src/{model,embeddings,tool,store,retriever}.rs
- Python wrappers: crates/litgraph-py/src/*.rs
- Examples: examples/*.py (each is a runnable hello-world)
- Tests: python_tests/test_<feature>.py (one file per surface)

## When you (the agent) get stuck
- Check `MISSING_FEATURES.md` — it might be intentional.
- `cargo clippy --workspace --all-targets` flags real bugs.
- `python tools/check_stubs.py` flags binding ↔ .pyi drift.
```

**Why Claude Code cares:** every coding agent's first action in a new
repo is "read README, README points to docs, agent burns context."
`AGENTS.md` is the canonical "skip the discovery phase, here are the
rules" file.

---

## <a id="mcp-server"></a>9. MCP server for litGraph itself

Most impactful single feature. Ship an MCP server that exposes:

- Full-text search over docs + examples
- API surface introspection (`list_classes`, `list_functions`, `get_signature`, `get_docstring`)
- Working code generation for canonical patterns
- Live `cargo check` / `pytest` invocation against the user's repo

| Feature | Status | Effort |
|---|---|---|
| `litgraph mcp serve` → MCP server over stdio | 💡 | M |
| Tool: `search_docs(query)` → relevant doc chunks | 💡 | M |
| Tool: `get_signature(symbol)` → typed signature + docstring | 💡 | S |
| Tool: `get_example(pattern)` → runnable snippet | 💡 | S |
| Tool: `validate_code(snippet)` → returns errors | 💡 | M |
| Tool: `run_tests(filter)` → pytest output | 💡 | S |
| Bundled MCP config that Claude Code can `claude mcp add` in one line | 💡 | S |

**Why Claude Code cares:** instead of grep-walking the source tree,
Claude calls `search_docs("how do I add memory to a ReactAgent?")` and
gets a typed answer. Cuts agent latency 10× on lookups.

---

## <a id="recipes"></a>10. Recipe library

Not the same as `examples/` — *recipes* are 30–80-line single-file
patterns that the agent treats as a starting point.

| Feature | Status | Effort |
|---|---|---|
| `examples/rag_agent.py` | ✅ | — |
| `examples/streaming_chat.py` | ✅ | — |
| `examples/parallel_graph.py` | ✅ | — |
| `examples/langgraph_compat.py` | ✅ | — |
| **💡 `examples/multi_agent_supervisor.py`** | 💡 | S |
| **💡 `examples/eval_harness.py`** | 💡 | S |
| **💡 `examples/checkpoint_resume.py`** | 💡 | S |
| **💡 `examples/mcp_client_server.py`** | 💡 | S |
| **💡 `examples/observability_otel.py`** | 💡 | S |
| **💡 `examples/structured_output.py`** | 💡 | S |
| **💡 `examples/local_ollama.py`** | 💡 | S |
| **💡 `examples/web_research.py`** (Tavily + RAG + ReAct) | 💡 | M |
| **💡 `examples/code_review_bot.py`** (CritiqueRevise) | 💡 | M |
| **💡 `examples/sql_agent.py`** (SQLite tool + structured output) | 💡 | M |
| Each example is `uv run python examples/X.py` and exits 0 with no API call required (uses scripted mock model) | 💡 | M |
| Examples live in CI so they don't bit-rot | 💡 | M |

**Why Claude Code cares:** the agent's first move on "build me a SQL
agent" is to look for an existing example. If `examples/sql_agent.py`
exists and runs, the agent diffs from there instead of writing from
scratch — much lower error rate.

---

## <a id="schema-first"></a>11. Schema-first contracts

Coding agents are great at filling in JSON Schema. Lean into it.

| Feature | Status | Effort |
|---|---|---|
| Tools take JSON Schema | ✅ | — |
| `with_structured_output(Schema)` | ✅ | — |
| Graph state declared as `Pydantic.BaseModel` works | ✅ | — |
| **💡 `Tool.from_schema(json_schema)` constructor** | 💡 | S |
| **💡 `agent.expected_state_schema()` returns a JSON Schema** | 💡 | M |
| **💡 OpenAPI export** of `litgraph-serve` endpoints | 💡 | M |
| **💡 Schema-validated streaming events** (so the agent knows the shape per stream mode) | 💡 | M |

**Why Claude Code cares:** the agent can introspect a graph and
generate code that matches the state shape. Less guessing means fewer
type errors.

---

## <a id="trace-replay"></a>12. Trace replay + debug-by-trace-id

When the agent's code fails in production, the agent's next action is
"reproduce locally." A trace-replay tool makes that one command.

| Feature | Status | Effort |
|---|---|---|
| OTel trace export | ✅ | — |
| **💡 `litgraph replay <trace-id>`** — fetch span + replay agent locally | 💡 | L |
| **💡 `litgraph trace inspect <id>`** — pretty-print a trace timeline | 💡 | M |
| **💡 Persistent prompt+response capture** (pluggable callback writing JSONL) | 💡 | S |
| **💡 `litgraph diff-traces a b`** — what changed between two runs | 💡 | M |

**Why Claude Code cares:** prod debug is the worst-context environment
for an agent. Reducing "fetch trace, find span, reproduce, fix" to one
command cuts diagnose-time 5–10×.

---

## <a id="one-call-factories"></a>13. One-call factories

Lower the barrier from "intent" to "working agent."

| Feature | Status | Effort |
|---|---|---|
| `ReactAgent(m, tools, system_prompt=...)` | ✅ | — |
| `litgraph.agents.deep.create_deep_agent(...)` (retries+budget+tracing wired) | ✅ | — |
| **💡 `litgraph.recipes.rag(corpus_path, model=...)`** → working RAG agent in 1 call | 💡 | M |
| **`litgraph.recipes.eval(target, cases)`** → harness with sane metrics | ✅ (iter 339) | S |
| **`litgraph.recipes.serve(graph, port=8080)`** → REST + SSE in 1 call | ⏳ (renders cmd; full impl pending) | S |
| **💡 `litgraph.recipes.multi_agent(roles=[...])`** → supervisor pre-wired | 💡 | M |

**Why Claude Code cares:** when the user says "build me an X", the
agent's first try should be `litgraph.recipes.X(...)`. If that exists
and works, the agent ships in 5 lines instead of 50.

---

## <a id="doc-search"></a>14. Doc search CLI

Even with MCP, sometimes the agent isn't connected. A grep-friendly
doc index is a fallback.

| Feature | Status | Effort |
|---|---|---|
| `litgraph search-docs <query>` returns Markdown chunks | 💡 | M |
| Bundled offline doc index in the wheel | 💡 | M |
| `litgraph show <symbol>` prints docstring + sig + nearest example | 💡 | S |

**Why Claude Code cares:** offline / sandboxed environments still need
discoverability. A CLI works when MCP doesn't.

---

## <a id="skill-plugin"></a>15. Claude Code skill plugin

A `litgraph` slash command that ships the right context into the
session.

| Feature | Status | Effort |
|---|---|---|
| `claude mcp add` config snippet in README | 💡 | S |
| Skill plugin: `/litgraph` slash command activates litGraph context | 💡 | M |
| Plugin includes: AGENTS.md, top examples, common errors → fixes table | 💡 | M |
| Plugin auto-detects `litgraph` in `pyproject.toml` and offers itself | 💡 | M |

**Why Claude Code cares:** when a user types `/litgraph` in their
session, Claude immediately gets the framework's mental model loaded.
No "let me read the README" detour.

---

## <a id="test-fixtures"></a>16. Test fixtures the agent can copy-paste

The agent often writes tests. Make the test infrastructure obvious.

| Feature | Status | Effort |
|---|---|---|
| `ScriptedModel` for deterministic chat tests (already used internally) | ⏳ (Rust-only) | S to expose |
| Python `MockChatModel` with scripted replies | ✅ (iter 339) | S |
| `MockEmbeddings(dim, deterministic=True)` | ✅ (iter 339) | S |
| `MockTool(returns=...)` | ✅ (iter 339) | S |
| Pytest fixtures auto-discovered: `mock_chat`, `mock_emb`, `mock_store` | 💡 | M |
| `litgraph.testing` module documented in USAGE.md | 🚧 | S |

**Why Claude Code cares:** the agent's tests need deterministic
behaviour. If `MockChatModel` is one import away, the agent writes
tests that pass on the first run.

---

## <a id="versioning"></a>17. Versioning ergonomics

Coding agents are sensitive to version drift. Make the version contract
explicit.

| Feature | Status | Effort |
|---|---|---|
| `litgraph.__version__` matches Cargo workspace version | ✅ | — |
| Stable v1.0 with SemVer guarantees | 🚧 | L |
| Changelog with code-level migration notes | ⏳ | S |
| `litgraph migrate` — automated source rewrite for breaking changes | 💡 | M |
| Deprecation warnings cite the replacement | ⏳ | S |

**Why Claude Code cares:** when the user is on `litgraph==0.1.1` and
the agent reads docs for `0.4.0`, code breaks subtly. Version-aware
docs + migration tooling fix it.

---

## Highest-impact subset (if you only ship 5)

If the team can only do 5 of these, do these:

1. **`AGENTS.md` at repo root** (S) — every agent reads this on first
   touch. Highest leverage per minute of work.
2. **`litgraph mcp serve`** (M) — turns the framework into something
   agents can query, not just read.
3. **`litgraph init <template>`** (M) — scaffolds a working repo so
   the agent starts at green tests, not from scratch.
4. **CI-tested examples for the top 10 patterns** (M) — copy-paste
   never ships broken code.
5. **`litgraph doctor`** (S) — turns env mysteries into one-line
   diagnostics.

These five together handle ~80% of "agent gets stuck" cases:
discovery, scaffolding, env, examples, queryability.

---

## Anti-features — DO NOT add these

Coding agents tempt frameworks into bloat. Resist:

- **🚫 Per-feature `AutoX` magic** that "just works" by guessing
  config from the environment. Magic is great for humans, terrible for
  agents — the agent can't verify what it doesn't see.
- **🚫 Hidden global state** (singletons, default configs that monkey-
  patch on import). Agents can't reason about side effects from import.
- **🚫 Abstract base classes for everything**. Agents construct
  concrete classes; deep abstraction makes them pick the wrong layer.
- **🚫 Per-tool config files** (`.litgraphrc`, etc.). One config
  format max — pyproject.toml.
- **🚫 Auto-detected behaviour that varies by env** (e.g. "fast mode
  in CI, slow mode locally"). Agents debug what's reproducible.

---

## Where to file issues

If you want one of these moved up the priority list, open an issue at
<https://github.com/plutonium-guy/litGraph/issues> with:

- Which feature (link to section)
- A concrete use-case ("I'm using Claude Code to build X, and Y blocks me")
- Effort vs. impact rationale

Coding-agent DX is a strict subset of human DX, but the ROI is bigger:
every feature here scales across millions of agent runs.

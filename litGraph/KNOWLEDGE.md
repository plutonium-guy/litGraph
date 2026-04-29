# litGraph — Session Checkpoint

Dense knowledge dump. Read first, then iterate. Sister docs:
- `FEATURES.md` — what's done, what's left (status block at top)
- `~/.claude/projects/-Volumes-external-storage-simple-agent/memory/project_litgraph.md` — per-iter log (auto-memory)

---

## 1. Philosophy

LangChain + LangGraph wedge points:

- **Bloat**: 200+ providers, 150+ loaders, 60+ vector stores, 100+ tool wrappers in one tree. Most users pull 5% of it.
- **Slow path**: Python asyncio + GIL + 6-layer `Runnable.invoke` stack. Per-token latency dominated by glue, not the LLM.
- **Abstraction debt**: legacy `LLMChain` / `SequentialChain` / `MultiPromptChain` zoo. Hard to deprecate without breaking users.
- **Observability lock-in**: LangSmith-shaped events. OTel is bolted on.

litGraph thesis:

1. **Rust core, Python shim** — every hot path (HTTP, SSE parse, tokenize, embed math, vector search, JSON parse, graph scheduling) lives in Rust. PyO3 layer is a thin dispatch.
2. **GIL-free parallelism** — `tokio::JoinSet` for I/O fan-out, `rayon::par_iter` for CPU batching. `py.allow_threads()` around every heavy block. Free-threaded Python 3.13 ready.
3. **Shallow stacks** — ≤ 2 frames from user code to HTTP. No `Runnable | Runnable | Runnable` magic.
4. **Split crates, zero default features** — pay for what you import. Workspace = 37 crates.
5. **Graph-first** — `StateGraph` is the headline. Class zoo collapses to functions + nodes.
6. **Inspectable** — `on_request` hook exposes final HTTP body. Solves 50% of debug pain.
7. **OTel-native** — tracing spans + future OTLP exporter (still TODO).

---

## 2. Architecture

```
crates/
├── litgraph-core/          types: Message, Tool, ChatModel, Embeddings, Document, errors, parsers, evaluators
├── litgraph-graph/         StateGraph + Kahn scheduler + interrupt + checkpointer trait
├── litgraph-providers-*/   one crate per provider (openai/anthropic/gemini/bedrock/cohere/voyage/jina)
├── litgraph-stores-*/      one crate per vector store (memory/hnsw/pgvector/chroma/qdrant/weaviate)
├── litgraph-rerankers-*/   one crate per reranker (cohere/voyage/jina)
├── litgraph-retrieval/     traits + bm25 + hybrid + 9 retrievers + transformers
├── litgraph-loaders/       15 sources (text/pdf/docx/web/notion/slack/confluence/github/gmail/gdrive/...)
├── litgraph-splitters/     recursive char + markdown header
├── litgraph-tools-*/       tool catalog (utils + search)
├── litgraph-agents/        ReactAgent + SupervisorAgent + TextReActAgent
├── litgraph-checkpoint-*/  sqlite/postgres/redis
├── litgraph-cache/         memory + sqlite + embedding + semantic
├── litgraph-memory-sqlite/ SqliteChatHistory (durable per-session)
├── litgraph-tokenizers/    tiktoken-rs + HF + trim_messages
├── litgraph-resilience/    Retry + RateLimit + Fallback wrappers
├── litgraph-observability/ CostTracker + GraphEvent + AgentEvent
├── litgraph-mcp/           MCP client (HTTP/SSE)
├── litgraph-macros/        proc-macro #[tool] (schemars-derived)
├── litgraph-bench/         criterion benches
└── litgraph-py/            ALL PyO3 bindings live here. Zero PyO3 elsewhere.
```

**Hard rule**: PyO3 imports are forbidden outside `litgraph-py`. Every other crate must compile + test as pure Rust (cargo can use them without Python).

---

## 3. Naming + module conventions

- File-level docstrings explain WHY, not WHAT. The "why this exists vs the alternative" framing is consistent throughout.
- Public API: `pub use submod::{TypeA, TypeB};` re-exported from each crate's `lib.rs`. Users import from crate root.
- Async: `async-trait` everywhere on traits. Concrete impls await freely.
- Errors: every crate uses `litgraph_core::Error` / `Result`. Common variants: `Provider`, `RateLimited`, `Timeout`, `InvalidInput`, `Template`, `Serde`, `ToolNotFound`, `ToolFailed`, `Parse`, `Cancelled`, `Other`. Constructors: `Error::provider(s)`, `Error::invalid(s)`, `Error::parse(s)`, `Error::other(s)`.
- Test pattern: hand-rolled `TcpListener` fakes for HTTP-shaped providers (avoids httpmock dep). For Python E2E: `http.server.ThreadingHTTPServer` with scripted handler classes.

---

## 4. PyO3 patterns + gotchas

### Subpackages
```rust
fn add_sub(py, parent, name, register_fn) { ... sys.modules["litgraph.<name>"] = sub ... }
```
Without `sys.modules` insert, `from litgraph.X import Y` fails even when attribute access works. Every submodule (`providers`, `tools`, `agents`, `parsers`, `evaluators`, ...) registered via `add_sub`.

### Function naming gotcha
`#[pyfunction]` exports the Rust function name verbatim. `fn mmr_select_py` becomes Python `mmr_select_py`. Always use `#[pyfunction(name = "mmr_select")]` to strip the `_py` suffix. Hit this in iter 112; fixed.

### Tokio runtime
`crate::runtime::{block_on_compat, rt}` — single shared runtime. `block_on_compat` lets sync Python methods drive async Rust without spinning a fresh runtime per call. `rt().spawn(...)` for streaming pumps.

### GIL release
`py.allow_threads(|| block_on_compat(async move { ... }))` around every long Rust call. Releases the GIL so other Python threads make progress. Critical for free-threaded 3.13 + concurrent agent invocations.

### Streams to Python
Pattern: spawn a tokio task that pumps the inner Stream into an `mpsc::channel`. Wrap the receiver in a `#[pyclass]` with `__iter__` + `__next__`. `__next__` blocks on `rx.recv()` with GIL released. Used by ReactAgent.stream(), TextReActAgent.stream(), LLM token streams.

### Polymorphism via extract_*
`extract_chat_model(bound)` and `extract_tool_arc(bound)` are central type-dispatch helpers. Add ALL new chat model / tool classes to both sites (in `litgraph-py/src/agents.rs` AND `litgraph-py/src/tools.rs::extract_tool_arc`).

### ChatResponse fields
`{message: Message, finish_reason: FinishReason, usage: TokenUsage, model: String}`. NOT `content` / `tool_calls` directly — they're inside `message`. `Message::text_content()` (NOT `text()`) returns the concatenated text view.

### Tool trait shape
`fn schema(&self) -> ToolSchema; async fn run(&self, args: Value) -> Result<Value>`. NOT `invoke()`. JSON args → JSON value out. Errors surface as `is_error: true` tool messages in agent traces.

### Pyright noise
Pyright always complains "Import 'litgraph.X' could not be resolved" for maturin-editable installs. False positive. Fix is `pyo3-stub-gen` (`.pyi` generation) — listed as TODO in FEATURES.md.

---

## 5. Provider patterns

All ChatModels share:
- `name(&self) -> &str`
- `async fn invoke(messages, opts) -> Result<ChatResponse>`
- `async fn stream(messages, opts) -> Result<ChatStream>`
- `async fn batch(...)` — default impl is naive serial; providers override for true parallel

Auth families covered:
- API key (OpenAI, Anthropic, Gemini AI Studio, Cohere, etc)
- SigV4 (Bedrock native + Bedrock Converse) — hand-rolled, no AWS SDK
- OAuth2 Bearer (Vertex AI, Gmail, Drive)
- PAT (GitHub, Confluence Bearer DC)
- Basic (Confluence Cloud)

Streaming:
- OpenAI-compat: SSE + `[DONE]` sentinel
- Anthropic: SSE + per-event types
- Bedrock: AWS event-stream binary frames (CRC32 + headers + payload). Reused for both Bedrock-native + Converse.
- Gemini: SSE
- Cohere: SSE

`response_to_py_dict` returns `{text, finish_reason, usage: {prompt, completion, total, cache_creation, cache_read}, model}`. Anthropic prompt-cache fields default 0 for non-supporting providers.

Resilience composition: wrap each provider in `RetryingChatModel` first (transient retry with backoff), then wrap the chain in `FallbackChatModel` (cross-provider failover). `RateLimitedChatModel` adds token-bucket throttling at any layer.

---

## 6. Output-parser surface

Parsers complement each other; `format_instructions` helpers tell the LLM how to produce the right format:

| Parser | Use | Format helper |
|---|---|---|
| `StructuredChatModel` (iter 89) | JSON via tool-call or response_format | schema-driven |
| `parse_xml_tags` / `parse_nested_xml` (105) | Anthropic XML idiom | `xml_format_instructions` |
| `parse_comma_list` (106) | "Give me 5 X" | `comma_list_format_instructions` |
| `parse_numbered_list` (106) | "1. ... 2. ..." | `numbered_list_format_instructions` |
| `parse_markdown_list` (106) | "- foo\n- bar" | `markdown_list_format_instructions` |
| `parse_boolean` (106) | yes/no, substring-safe | `boolean_format_instructions` |
| `parse_react_step` (107) | Thought/Action/Action Input/Final Answer | `react_format_instructions` |
| **OutputFixingParser** | LLM repair on parse fail | TODO |

All parsers tolerant of LLM noise (loose-prose tolerance, case-insensitive labels, code-fence stripping, markdown-bold labels).

---

## 7. Agent matrix

| Agent | Tool calling | Use |
|---|---|---|
| `ReactAgent` (iter 13) | provider-native | GPT-4, Claude, Gemini, Cohere R+, Mistral Large, etc |
| `TextReActAgent` (iter 109) | text-mode parse_react_step | Ollama / vLLM / llama.cpp / base-completion / older fine-tunes |
| `SupervisorAgent` | delegates to ReactAgents | multi-agent orchestration |

Both ReactAgent + TextReActAgent expose `.stream()` returning event streams (parallel tool calls in former; serial in latter).

Loop shape (text): user input → system-prompt-with-tool-catalog → LLM prose → `parse_react_step` → either Final (return) or Action (run tool, append `Observation: ...`, loop).

Loop shape (native): system prompt → LLM with `tools` arg → tool_calls collected → JoinSet runs them in parallel (capped by Semaphore) → tool messages appended → repeat.

---

## 8. Modality matrix (complete as of iter 116)

|  | Input | Output |
|---|---|---|
| Text | every chat model | every chat model |
| Image | `ContentPart::Image` (base64 or URL) on multimodal providers | `DalleImageTool` (iter 115) |
| Audio | `WhisperTranscribeTool` (iter 114) | `TtsAudioTool` (iter 116) |

All three new tools share the OpenAI-compat HTTP shape so they swap to Groq/Together/self-hosted equivalents by changing `base_url`.

---

## 9. Vector store + retrieval

Vector stores (6): memory, hnsw, pgvector, chroma, qdrant, weaviate.

Retrievers (9): Vector, Bm25, Hybrid (RRF + weighted), Reranking, ParentDocument, MultiQuery, ContextualCompression, SelfQuery, TimeWeighted.

Document transformers (iter 112, post-retrieval refinement): `mmr_select` (Maximal Marginal Relevance), `embedding_redundant_filter` (drop near-dupes), `long_context_reorder` (Liu et al "Lost in the Middle" workaround). All pure functions over `Vec<Document>`; compose freely.

Loaders (15): text, markdown, json, jsonl, csv, html, pdf, docx, directory, web, notion, slack, confluence, github-issues, github-files, gmail, gdrive.

Splitters: recursive char, markdown header.

---

## 10. Cache + observability

Cache:
- `MemoryCache`, `SqliteCache` — exact-key by `(model, prompt, params)` hash
- `embedding_cache` — caches embedding API calls
- `SemanticCachedModel` — embed query, return prior response if cosine sim above threshold
- `CachedTool` (iter 117) — TTL+LRU wrapper around any Tool

Observability:
- `CostTracker` — token + dollar accounting per provider
- `GraphEvent` enum — `node_start` / `node_end` / `node_error` / `state_update` / `interrupt`
- `AgentEvent` enum — `iteration_start` / `llm_message` / `tool_call_start` / `tool_call_result` / `final` / `max_iterations_reached` / `token_delta`
- `TextReactEvent` enum — text-mode equivalent (10 variants)
- `tracing` spans wrap every node + LLM call + tool invocation

---

## 11. Resilience matrix (iter 113 completed)

| Wrapper | Concern | When |
|---|---|---|
| `RetryingChatModel` | same provider, retry transient (rate-limit, timeout, 5xx) | OpenAI 429s — backoff + retry |
| `RateLimitedChatModel` | self-throttle to RPM cap | stay under provider quota |
| `FallbackChatModel` | DIFFERENT provider, immediate switch | OpenAI down → try Anthropic |

`is_transient(e)` is the shared classifier. Stream method only tries the first inner — token streams can't gracefully fail-over mid-stream.

Compose: `Fallback([Retry(openai), Retry(anthropic), Retry(gemini)])` — robust prod chain.

---

## 12. StateGraph

`petgraph::StableGraph` backend. Kahn topological scheduler with `JoinSet` + `Semaphore(max_parallel)` + `CancellationToken`.

Node = `async fn(state) -> NodeOutput`. NodeOutput carries partial state update + optional `goto` (override default edge).

Edges: static (always), conditional (fn → enum variant string), entry (from `START`), END.

Reducers via `#[derive(GraphState)]` macro: `#[reduce(append)]`, `#[reduce(replace)]`, custom fn. State updates are merged, not replaced.

`Send` fan-out: one node emits N parallel child invocations, reducer collects.

Checkpointers: every node-completion writes a snapshot keyed by `(thread_id, step)`. Restart resumes from latest. SQLite + Postgres + Redis impls. Bincode serialization (NOT JSON — perf reasons).

HITL: `interrupt(payload)` inside a node returns a serializable interrupt; resume via `Command::with(...)` from outside. State editing + branch fork + replay-from-checkpoint listed but **state-history API not yet built** (gap from FEATURES.md).

---

## 13. Build + dev commands

```bash
# Rust-only test (per crate)
cargo test -p litgraph-core <filter>
cargo test -p litgraph-tools-utils <filter>

# Workspace test
cargo test --workspace

# Build + install Python wheel (editable)
VIRTUAL_ENV=/Volumes/external_storage/simple_agent/litGraph/.venv \
PYO3_PYTHON=/Volumes/external_storage/simple_agent/litGraph/.venv/bin/python \
/Volumes/external_storage/simple_agent/litGraph/.venv/bin/maturin develop \
  -m crates/litgraph-py/Cargo.toml --release

# Run a single Python test (no pytest installed; tests have __main__ runners)
/Volumes/external_storage/simple_agent/litGraph/.venv/bin/python python_tests/test_<name>.py

# Bench
cargo bench -p litgraph-bench --bench bm25 -- --quick
```

**Venv quirk**: shell's `VIRTUAL_ENV=/Volumes/external_storage/simple_agent/.venv` is wrong (doesn't exist). Real venv lives at `/Volumes/external_storage/simple_agent/litGraph/.venv`. Always override with the env vars above when running cargo for PyO3 builds.

---

## 14. Iteration methodology (Ralph loop)

Each iter follows this rhythm:

1. **Pick a gap** — scan FEATURES.md status block + memory log. Aim for 1 self-contained capability per iter (small surface, broad applicability).
2. **Rust core first** — implement in the right crate (often `litgraph-core` or `litgraph-tools-utils` or a provider crate). Add unit tests using hand-rolled fakes (no httpmock / no big test deps).
3. **PyO3 binding** — add `Py*` wrapper class or `#[pyfunction(name = "...")]`. Wire into `extract_chat_model` / `extract_tool_arc` if applicable. Update agent extract paths (BOTH PyReactAgent's inline list AND `extract_tools` helper).
4. **Build wheel** — `maturin develop --release`. ~50s on this box.
5. **Python E2E test** — `python_tests/test_<feature>.py` with `__main__` runner. Use `http.server.ThreadingHTTPServer` for HTTP fakes. Always include round-trip-via-ReactAgent test for tools/agents.
6. **Update memory** — append to `~/.claude/projects/.../memory/project_litgraph.md` with iter summary + counts.
7. **Update FEATURES.md** — flip the entry to ✅.

**Test naming**: behavior-stating, not numbered. `final_answer_wins_over_action_when_both_present` not `test_react_5`.

**Comment policy**: WHY not WHAT. Module docstrings explain "why this exists vs alternatives" — referenced explicitly elsewhere ("the Whisper iter-114 pattern").

**No emojis in code/files** unless user asks. (User has asked: only in this README-style doc, sparingly.)

---

## 15. Critical gotchas + their fixes

- `#[pyfunction]` without `name=` exports `_py` suffix → ImportError in Python (iter 112)
- `Message::text_content()` NOT `text()` — `text()` is a constructor (iter 109)
- `ChatResponse.message.text_content()` not `.content` — `content` lives inside `Message` (iter 109)
- `Tool::run()` not `Tool::invoke()` (iter 109)
- `cargo` from `cd <wd>` triggers permission prompt — use absolute paths
- Borrow checker on closures touching `g` while `g.entries` is borrowed: clone the key set into `live: HashSet`, retain on it (iter 117)
- Markdown-bold labels like `**Action:**` need `find_label_at_line_start` to return the newline offset, not the label start, so trailing `**` doesn't bleed into the prior block (iter 107)
- `get_weather(city: str)` — calling with `{}` from LLM args missing city → @tool raises → captured as observation, not fatal (iter 109 test)
- Python's `"this is not python"` parses as a chained-comparison expression and raises NameError, not SyntaxError. Use `"def broken(:"` for actual SyntaxError tests (iter 118)

---

## 16. What to NOT build (deferred indefinitely)

- LanceDB / Pinecone vector stores — heavy deps (arrow-rs, datafusion, gRPC), low marginal win over qdrant/pgvector
- LangChain `Callbacks` API parity — wide surface, CostTracker + GraphEvent + AgentEvent already cover prod observability needs
- Streaming tool execution (tools emit progress events) — requires Tool trait extension, deep reach
- 100+ community provider integrations — current 7 + OpenAI-compat helpers cover 95% of use; rest go in user crates

---

## 17. Open gaps — read FEATURES.md status block for current list

Top picks for next iters (in rough leverage order):

1. **OutputFixingParser** — LLM-repair-on-parse-fail. FEATURES.md line 78. Real prod pain. Small surface.
2. **Time travel + state history API** — `state_history(thread_id)`, `fork_at(checkpoint_id)`, `replay_from(...)`. FEATURES.md line 126. The LangGraph-vs-litGraph differentiator that's listed but unbuilt.
3. **OpenTelemetry OTLP exporter** — `litgraph-tracing-otel` crate. Real prod observability needs it.
4. **`pyo3-stub-gen` `.pyi` generation** — kills every Pyright import warning; universal DX.
5. **Streaming JSON parser** — parse structured output as tokens arrive.

After v1.0 closes:
- fastembed-rs (local embeddings)
- candle / mistral.rs (local chat)
- ort (local cross-encoder rerankers)
- LangSmith OTel compat shim
- Webhook notifier tool

---

## 18. The wedge sentence

> Nobody ships LangGraph-quality typed StateGraph + checkpointers + HITL with Python bindings in Rust.

Build that well. Wrap it with slim LangChain equivalents. Release GIL everywhere. Benchmark brutally. The value prop sells itself.

litGraph is ~85% of the way there as of iter 118. Focus the remaining 15% on the listed gaps (especially the LangGraph-differentiator pieces: time travel + OTel) before chasing nice-to-haves.

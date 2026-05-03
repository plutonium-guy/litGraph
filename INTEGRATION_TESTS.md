# Integration testing — what's tested, what's blocked, why

Source of truth for **live-API** integration tests. The unit-test
suite (`python_tests/test_*.py`, `cargo test --workspace`) covers
shape + behaviour with mocks; this doc tracks the tests that hit
real provider endpoints.

**Provider used today: DeepSeek** — OpenAI-compatible REST at
`https://api.deepseek.com/v1`. Set `DEEPSEEK_API_KEY` to enable;
tests skip cleanly when it's missing.

**Model alias used in tests:** `deepseek-chat` (the general-purpose
model). The reasoning model `deepseek-reasoner` is exercised in the
streaming + structured-output tests where its different finish
semantics matter.

**Snapshot date:** 2026-05-02 · iter 366.

---

## Run

```bash
export DEEPSEEK_API_KEY=sk-...
source .venv/bin/activate

# Live tests only:
pytest python_tests/integration/ -v

# Skip live tests (CI default):
pytest python_tests/integration/ -v --no-deepseek
```

Each test is decorated with `@pytest.mark.integration` and gated on
`DEEPSEEK_API_KEY`. They make real API calls and cost real money
(small — DeepSeek is cheap; full suite < 5 cents at the time of
writing). Don't run on every commit; run on release-prep + after
provider changes.

---

## Tested ✅

88 live integration tests against DeepSeek pass as of iter 366.
10 cleanly skipped:
- `TokenBudgetChatModel`, `CostCappedChatModel`, `PiiScrubbingChatModel`
  not exposed on the Python surface today (4 cases — the TokenBudget
  wrapper has two cases).
- `test_react_agent_with_python_wrapped_tools_blocked` — documented
  Python `HookedTool` blocker (1 case).
- `LlmJudge` (2 cases) and `synthesize_eval_cases` (2 cases) — both
  use `StructuredChatModel` → `response_format=json_schema`, which
  DeepSeek does not support today. See Gotchas.
- `broadcast_chat_stream` (1 case) — Python binding fix landed in
  iter 360; re-enable after the next maturin build picks it up.

| Feature | Test file | Notes |
|---|---|---|
| `OpenAIChat.invoke` (single + multi-turn) | `test_chat_basic.py` (5 cases) | text / usage / model / finish_reason shapes |
| `OpenAIChat.stream` | `test_chat_stream.py` (4 cases) | SSE → `delta` events + `done` event with usage |
| Native tool calling via `ReactAgent` | `test_tool_calling.py` (2 cases) | full agent loop: tool_call → tool result → final |
| `ReactAgent.stream` + `stream_tokens` | `test_react_stream.py` (2 cases) | iteration / final / token_delta events |
| `TextReActAgent` + `PlanAndExecuteAgent` | `test_agent_variants.py` (2 cases) | transcript-mode + plan-step executor |
| Structured output (`json_object` mode) | `test_structured_output.py` (2 cases) | DeepSeek requires "json" in prompt — see Gotchas |
| `CostTracker` instrumentation | `test_cost_tracker.py` (3 cases) | per-call + per-model breakdown + USD helper |
| `batch_chat` fan-out | `test_batch.py` (3 cases) | order preservation + concurrency = 1/3/4 |
| `model.with_cache(...)` | `test_caching.py` (2 cases) | hit returns identical text; miss differs |
| `model.with_retry()` / `with_rate_limit()` | `test_resilience.py` (3 cases) | wrappers compose; happy path passes through |
| `StateGraph` with model node | `test_state_graph.py` (2 cases) | linear + parallel-branches both call DeepSeek |
| `recipes.summarize` + `multi_agent` | `test_recipes_live.py` (2 cases) | end-to-end recipe drivers |
| `recipes.eval` with DeepSeek as target | `test_eval_live.py` (2 cases) | per_case / aggregate report shape |
| `@entrypoint` / `@task` functional API | `test_functional_api.py` (2 cases) | async workflow + 2-task combine |
| `ChatPromptTemplate.from_messages` | `test_prompts_memory.py` (2 cases) | minijinja `{{ var }}` substitution |
| `TokenBufferMemory` round-trip | `test_prompts_memory.py` (2 cases) | recall + clear |
| `tool_hooks.{Before,After,ToolBudget,wrap_tool}` | `test_tool_hooks_live.py` (1 case) | hooks fire on a real-derived input |
| `lcel.Pipe` composition | `test_lcel_pipe.py` (2 cases) | `Pipe(fn) \| model_call \| extract` runs end-to-end |
| `MiddlewareChain` + `MiddlewareChat` via ReactAgent | `test_middleware_live.py` (2 cases) | `with_(SystemPromptMiddleware(...))` plugged into agent loop |
| `parsers.parse_json_with_retry` + `fix_with_llm` | `test_parsers_live.py` (2 cases) | repairs malformed JSON via real DeepSeek round-trip |
| Multi-tool `ReactAgent` selection | `test_react_multi_tool.py` (2 cases) | model picks between `add` / `subtract` correctly |
| `StateGraph.add_conditional_edges` | `test_conditional_routing.py` (2 cases) | router callable driven by model classifier output |
| `SupervisorAgent` worker routing | `test_supervisor_agent_live.py` (1 case) | supervisor delegates math → `math` worker |
| `tool_dispatch_concurrent` | `test_tool_dispatch_concurrent.py` (2 cases) | parallel tool dispatch outside agent loop + per-call error sentinel |
| `deep_agent.create_deep_agent` | `test_deep_agent_live.py` (2 cases) | factory with + without auto-injected Planning/VFS tools |
| `tools.PythonReplTool` via ReactAgent | `test_python_repl_tool_live.py` (1 case) | sandboxed subprocess + agent loop computes `19*23*17 = 7429` |
| `compat.{RunnableLambda,RunnableParallel}` | `test_compat_runnables.py` (2 cases) | LangChain shims compose with `Pipe` + a real model |
| `streaming.stream_events` translator | `test_streaming_translator.py` (1 case) | LCEL-style `{event,name,run_id,data}` lifecycle records over real stream |
| Built-in `CalculatorTool` via ReactAgent | `test_react_calculator.py` (1 case) | sandboxed math expression eval (no FunctionTool wrap needed) |
| `PlanningTool` + `VirtualFilesystemTool` attach | `test_react_planning_vfs.py` (1 case) | both deep-agents primitives accepted by ReactAgent; `snapshot()` accessors |
| `lcel.parallel(...)` over multiple steps | `test_lcel_parallel.py` (1 case) | result list aligned 1:1 with step order; mix model-call + pure fn |
| `ReactAgent.max_iterations` cap | `test_react_max_iterations.py` (1 case) | agent terminates even if instructed to loop forever |
| `multiplex_chat_streams` fan-in | `test_broadcast_multiplex.py` (1 case) | two-label tagged stream interleaved (broadcast variant skipped) |
| `compat.AgentExecutor` LangChain shim | `test_compat_agent_executor.py` (1 case) | `agent.llm + tools` constructor maps to ReactAgent loop |
| `ReactAgent.stream` lifecycle | `test_react_agent_event_stream.py` (1 case) | iterator yields iteration→final without stream_tokens=True |
| Chat-options shaping (`temperature`, `max_tokens`) | `test_chat_options.py` (2 cases) | temperature=0 reproducibility + max_tokens cap signals via `finish_reason` |
| Tool wrappers `Cached` / `Timeout` / `Retry` | `test_tool_wrappers.py` (3 cases) | wrappers compose around `FunctionTool` and survive ReactAgent dispatch |
| `coerce_one` (dataclass) + `coerce_stream` (TypedDict) | `test_coerce_adapters.py` (2 cases) | typed-state coercion over real DeepSeek output (stream needs an async wrapper) |
| `prompt_hub.{register,get,search,render}` | `test_prompt_hub_live.py` (2 cases) | round-trip register → render → invoke; substring/tag search |
| `compat.RunnableBranch` conditional dispatch | `test_compat_runnable_branch.py` (1 case) | first-truthy-predicate routing + default fall-through |
| `tools.SubagentTool` parent → child delegation | `test_subagent_tool_live.py` (1 case) | parent agent dispatches a math question to a sub-`ReactAgent` |
| `ReadFileTool` + `WriteFileTool` + `ListDirectoryTool` via agent | `test_filesystem_tools_live.py` (2 cases) | sandbox-rooted FS access; agent reads seed file + write→read round-trip |
| `Workflow(fn)` direct construction | `test_workflow_direct.py` (2 cases) | sync `invoke` and async `ainvoke` paths without `@entrypoint` decorator |
| `Workflow.astream` final-chunk contract | `test_workflow_astream.py` (1 case) | v1 yields `{"final": <result>}` chunk after a real model call |
| `compat.RunnablePassthrough` in parallel | `test_compat_runnable_passthrough.py` (1 case) | identity-branch alongside model-call branch in `RunnableParallel` |
| Multi-turn context preservation | `test_chat_long_messages.py` (1 case) | system + 2 user/assistant turns + dependent follow-up |
| `recipes.serve` stub command-renderer | `test_recipes_serve_stub.py` (3 cases) | accepts both StateGraph + CompiledGraph, rejects non-graphs (fixed a real bug — checker was looking for the wrong attrs) |

---

## Blocked / not tested ❌

Features deliberately not exercised against DeepSeek. Reason in each row.

| Feature | Why not tested |
|---|---|
| **Anthropic thinking blocks** | DeepSeek doesn't emit `thinking` events. Test against Anthropic when its key is available. |
| **Anthropic prompt caching** | Provider-specific (Anthropic's `cache_control` field). |
| **Gemini Vertex AI** | Different auth (Service Account JSON), different request shape. Needs `GOOGLE_APPLICATION_CREDENTIALS`. |
| **AWS Bedrock SigV4** | Needs AWS credentials + region; out of scope for a single-key integration run. |
| **OpenAI Responses API** | DeepSeek implements only `/chat/completions`, not `/responses`. |
| **OpenAI image generation (DALL·E)** | Image-gen path on `/images`; DeepSeek's catalog doesn't include it. |
| **Whisper / TTS tools** | Audio I/O endpoints; not on DeepSeek. |
| **Embeddings** (`/embeddings`) | DeepSeek doesn't expose an embedding model. Use Cohere / Voyage / Jina / OpenAI / FastEmbed for embedding tests. |
| **Vector stores live** | Need a running Qdrant / pgvector / Chroma / Weaviate / Milvus / Redis-search / Neo4j — out of scope here, separate compose-up integration suite. |
| **Postgres / SQLite checkpointers live** | Same — needs a DB; covered by mock-state unit tests. |
| **MCP server live** | Needs an MCP server endpoint; tested with the in-process fake server in `python_tests/test_mcp_*.py`. |
| **`litgraph-serve` HTTP** | Needs to spawn the binary; covered by Rust integration tests in `crates/litgraph-serve/tests/`. |
| **Free-threaded Python 3.13t** | Build matrix, not a per-call thing. Tested by running the full suite on 3.13t in CI. |
| **Vision / multimodal** | DeepSeek-VL is a separate model + endpoint shape; current tests use `deepseek-chat` only. |
| **Evaluator `LlmJudge` live** | Uses `StructuredChatModel` → `response_format=json_schema`. DeepSeek rejects schema-mode (`"This response_format type is unavailable now"`). Re-enable when DeepSeek adds it OR when `StructuredChatModel` falls back to `json_object` + post-validate. Test stubs are in `test_evaluators_llm_judge.py` (skipped). |

---

## Conditionally testable (provider key required)

Add the key, run the corresponding test file. None of these are
on by default.

| Provider | Env var | Adds tests for |
|---|---|---|
| OpenAI | `OPENAI_API_KEY` | `OpenAIChat`, `OpenAIEmbeddings`, `OpenAIResponses`, DALL·E, Whisper, TTS |
| Anthropic | `ANTHROPIC_API_KEY` | `AnthropicChat`, thinking blocks, prompt caching |
| Cohere | `COHERE_API_KEY` | `CohereChat`, `CohereEmbeddings`, `CohereReranker` |
| Voyage | `VOYAGE_API_KEY` | `VoyageEmbeddings`, `VoyageReranker` |
| Jina | `JINA_API_KEY` | `JinaEmbeddings`, `JinaReranker` |
| Tavily | `TAVILY_API_KEY` | `TavilyTool` web search |
| Gemini (AI Studio) | `GOOGLE_API_KEY` | `GeminiChat`, `GeminiEmbeddings` |
| AWS Bedrock | AWS standard chain | `BedrockChat`, Converse API, Bedrock embeddings |

---

## How tests are isolated from real money

- Each test sets `max_tokens` to a small value (≤ 50 unless the
  test specifically exercises long-output behaviour).
- Tests prefer `temperature=0` for determinism + cache-friendly
  re-runs.
- The whole `integration/` suite caps roughly at 5–10 cents per
  full run on DeepSeek's published prices.
- A `--no-deepseek` pytest CLI flag is honoured so CI can opt out
  even when the env var is set (the registered fixture flips
  every test to `pytest.skip`).

---

## DeepSeek-specific gotchas discovered while building these tests

- **`response_format=json_object` requires the prompt to contain
  the substring `"json"`** (case-insensitive). DeepSeek rejects
  the request with `400 invalid_request_error` otherwise. Tests
  set the system prompt to e.g. `'Reply with valid json: {...}'`.
- **`response_format=json_schema` is not supported on DeepSeek**
  (`"This response_format type is unavailable now"`). This blocks
  `LlmJudge` and any consumer of `StructuredChatModel.with_strict(true)`
  against DeepSeek. Workarounds: use `json_object` mode + manual
  validation, or use a provider that supports schema mode (OpenAI,
  Anthropic via tool-calls).
- **`MiddlewareChat` does NOT expose `.invoke()` on the Python surface.**
  It is an opaque chat-protocol wrapper for use inside `ReactAgent`,
  `SupervisorAgent`, etc. Drive it through an agent — direct
  `wrapped.invoke(messages, ...)` raises `AttributeError`.
- **`StateGraph.compile()` is single-shot per builder.** Calling
  `.compile()` a second time on the same `StateGraph` raises
  `RuntimeError: graph already compiled`. The returned `CompiledGraph`
  IS reusable, so save it: `compiled = g.compile(); compiled.invoke(...)`
  for each input. Don't call `g.compile()` inline inside the invoke
  loop.
- **`prompt_hub.Prompt.render(**vars)` uses Python `str.format()`**
  (single-brace `{var}`), NOT minijinja. For full Jinja semantics
  use `litgraph.prompts.ChatPromptTemplate.from_messages([...])`.
  `prompt_hub.search(query)` is a positional substring/tag matcher;
  no `tag=` kwarg.
- **`OpenAIChat.invoke()` accepts only `temperature`, `max_tokens`,
  `response_format`** as kwargs today. No `stop`, `top_p`, `seed`,
  `presence_penalty`, etc. To shape requests further, drop down to
  the underlying provider crate or add support in
  `crates/litgraph-providers-openai`.
- **`coerce_stream` requires an async iterable**, but
  `model.stream(...)` returns a SYNC `ChatStream`. To pipe model
  deltas through `coerce_stream`, wrap with a one-line async
  generator (`async def to_async(it): for x in it: yield x`).
  `coerce_one` works directly on dict outputs without any wrapping.
- **`streaming.stream_events` only emits lifecycle events**
  (`on_chat_model_start`, `on_chat_model_end`) for a raw chat stream.
  The intermediate `delta` chunks are NOT bridged into the LCEL-event
  shape today — `data` is empty `{}` on the end record. To consume
  delta tokens, iterate the underlying `model.stream(...)` directly
  (which yields native `{type:"delta",text:...}` dicts) instead of
  going through `stream_events`.
- **`OpenAIChat.invoke` does NOT take `tools=` directly.** Tools
  flow through `ReactAgent` (the agent loop owns the
  `tool_calls` protocol). Don't try
  `chat.invoke(messages, tools=[...])`.
- **`tool_calls[i].args`** is the canonical key on the response
  shape; some shims see `arguments` (string) — both are tolerated.
  Tool implementations receive args **as kwargs** (FunctionTool
  unpacks the JSON body against the schema).
- **`CostTracker` `instrument(model)` mutates in place + returns
  `None`.** Use the original `model` reference after calling
  `model.instrument(tracker)`. The bus drain is async — `time.sleep(0.5)`
  before reading `tracker.snapshot()` in tests.
- **Stream events are dicts**, shape `{type: "delta"|"done",
  text: ...}`. The `done` event carries the assembled text +
  `finish_reason` + `usage` + `model`.
- **`max_tokens=10` is plenty** for the smoke tests. Whole suite
  (30 cases) runs in ~ 30s at < 1 cent on DeepSeek's published
  rates.
- **All `model.with_*(...)` wrappers mutate in place + return None.**
  `with_cache(cache)`, `with_retry(...)`, `with_rate_limit(...)`,
  `instrument(tracker)` — keep using the original `model`
  reference; the side-effect installs the wrapper inside.
- **`recipes.summarize._content_of`** was reading `content` (LangChain
  shape) but native litGraph providers return `text`. Fixed in
  iter 354 — now tolerates both keys.
- **`ChatPromptTemplate` uses minijinja syntax (`{{ var }}`)**, not
  Python str.format `{var}`. Tests that use the template must double
  the braces.
- **`TokenBufferMemory(max_tokens, counter)`** takes a counter
  callable, not a `model_name` arg. Use a `lambda m: len(m["content"])
  // 4` for rough estimates or a tiktoken counter for exactness.
- **`@entrypoint` requires parens** — `@entrypoint()` (no args) wraps
  an `async def` workflow into a `Workflow`; `@entrypoint` without
  parens treats the function-or-anything as the first arg and
  blows up. The wrapped function must be `async def`.
- **Native `ReactAgent` rejects Python-side `HookedTool`** wrappers —
  its `extract_tools` only accepts the registered Rust tool types
  (FunctionTool, ShellTool, …). To attach middleware to the
  agent loop, use the Rust `ToolMiddlewareChain`
  (`litgraph_agents::middleware`) wired via `ReactAgentConfig.
  tool_middleware`. The Python `tool_hooks.wrap_tools` API is for
  manual / custom dispatchers outside the native loop.

## Failure triage

If a test fails, check in this order:

1. **Provider down?** `curl https://api.deepseek.com/v1/models -H
   "Authorization: Bearer $DEEPSEEK_API_KEY"` should return JSON.
2. **Rate-limited?** DeepSeek's free tier has per-minute caps;
   wait 60s and re-run.
3. **API drift?** DeepSeek occasionally changes the response
   shape (e.g., `usage.completion_tokens_details`). Update the
   adapter in `crates/litgraph-providers-openai` if the field is
   new.
4. **Model deprecated?** `deepseek-chat` is the stable alias;
   the underlying model rotates. If the test expects a specific
   model id in the response, check DeepSeek's release notes.

---

## Maintenance

- Add a row to "Tested" when you add a passing test.
- Move from "Tested" to "Blocked" with a reason if a provider
  change breaks a test for non-bug reasons (e.g., DeepSeek removes
  function calling).
- When a row moves from "Blocked" to "Tested" because we added
  a different provider key, update both this doc and the test's
  `pytest.mark`.

This doc is hand-maintained — there's no auto-sync between it and
the test files. PRs that add live tests must update this table.

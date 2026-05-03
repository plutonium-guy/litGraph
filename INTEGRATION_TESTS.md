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

**Snapshot date:** 2026-05-03 · iter 353.

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

19 live integration tests against DeepSeek pass as of iter 353.

| Feature | Test file | Notes |
|---|---|---|
| `OpenAIChat.invoke` (single + multi-turn) | `test_chat_basic.py` (5 cases) | text / usage / model / finish_reason shapes |
| `OpenAIChat.stream` | `test_chat_stream.py` (4 cases) | SSE → `delta` events + `done` event with usage |
| Native tool calling via `ReactAgent` | `test_tool_calling.py` (2 cases) | full agent loop: tool_call → tool result → final |
| Structured output (`json_object` mode) | `test_structured_output.py` (2 cases) | DeepSeek requires "json" in prompt — see Gotchas |
| `CostTracker` instrumentation | `test_cost_tracker.py` (3 cases) | per-call + per-model breakdown + USD helper |
| `batch_chat` fan-out | `test_batch.py` (3 cases) | order preservation + concurrency = 1/3/4 |

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
| **Evaluator LLM-judge** | Would call the model recursively; covered by mock unit tests. Add a smoke test once eval-suite is wired. |

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
  runs at < 1 cent on DeepSeek's published rates.

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

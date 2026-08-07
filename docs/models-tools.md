---
layout: default
title: Models and tools
description: Configure chat providers, streaming, structured output, tool schemas, middleware, and resilience without hidden state.
eyebrow: Compose primitives
---

# Models and tools

Models and tools share small contracts so they can be decorated, tested, observed, and replaced independently. Start with the simplest explicit model, then layer only the policies the solution needs.

## Chat providers

```python
from litgraph.providers import (
    OpenAIChat,
    AnthropicChat,
    GeminiChat,
    BedrockChat,
    CohereChat,
)

model = OpenAIChat(model="gpt-5")
message = model.invoke([
    {"role": "user", "content": "Explain Kahn scheduling in two sentences."}
])
print(message["content"])
```

Hosted providers use their conventional credential environment variables unless `api_key` is supplied. OpenAI-compatible services reuse `OpenAIChat` with an explicit `base_url`.

| Need | Configuration |
|---|---|
| OpenAI | `OpenAIChat(model="...")` |
| Anthropic | `AnthropicChat(model="...")` |
| Gemini | `GeminiChat(model="...")` |
| AWS Bedrock | `BedrockChat(...)` plus the AWS credential chain |
| Cohere | `CohereChat(model="...")` |
| Ollama / vLLM / LM Studio | `OpenAIChat(..., base_url="http://...")` |

Provider adapters implement the shared chat-model contract, which is why the same model can be placed behind caching, retries, budgets, tracing, structured output, or an agent.

## Streaming

```python
async for event in model.stream([
    {"role": "user", "content": "Count to five."}
]):
    if event.kind == "text":
        print(event.text, end="", flush=True)
```

Stream events cover text, tool-call deltas and completions, thinking, usage, and finish signals. `stream_tokens(...)` provides a text-only path for simple interfaces.

The Python streaming helpers support common fan-out patterns:

- `broadcast(stream, n)` copies one stream to multiple consumers.
- `race(streams)` returns the first stream to produce output and cancels the rest.
- `multiplex(streams)` interleaves several streams while preserving origin labels.
- `parse_stream_part(...)` and `aparse_stream_parts(...)` convert native events into typed `Delta`, `ToolCallDelta`, and `Done` variants.

Do not cache a token stream as if it were a completed message. Hash and semantic model caches intentionally bypass streaming.

## Structured output

```python
from pydantic import BaseModel

class Verdict(BaseModel):
    answer: str
    confidence: float

structured = model.with_structured_output(Verdict)
verdict = structured.invoke([
    {"role": "user", "content": "Is this incident customer-visible?"}
])

print(verdict.confidence)
```

Supported contracts include Pydantic v2 models, dataclasses, `TypedDict`, and raw JSON Schema. `coerce_one(...)` and `coerce_stream(...)` provide schema-aware conversion outside a model wrapper.

Treat the schema as part of the application boundary: keep field names meaningful, express required values directly, and test malformed or partial model output.

## Define tools with a decorator

```python
from litgraph.tools import tool

@tool
def search_orders(customer_id: str, limit: int = 10) -> list[dict]:
    """Return the customer's most recent orders."""
    return database.search_orders(customer_id, limit=limit)
```

The decorator derives a tool name, description, and argument schema from the Python function. Clear type annotations and docstrings improve both runtime validation and the model’s tool selection.

For an explicit schema or a callable accepting one argument dictionary, use `FunctionTool`:

```python
from litgraph.tools import FunctionTool

add = FunctionTool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"},
        },
        "required": ["a", "b"],
    },
    lambda args: {"sum": args["a"] + args["b"]},
)
```

Built-in families cover filesystem and shell operations, an in-memory virtual filesystem, SQLite, JSON Patch, search and web fetches, webhooks, audio, image generation, email, and utilities. Give an agent only the capabilities required for its task.

## Tool middleware

Middleware is the policy seam around tool execution. Use it for concerns that apply consistently across tools: retries, authorization, budgets, logging, redaction, or result shaping.

Python hooks in `litgraph.tool_hooks` support before/after/error interception. Rust middleware in `litgraph-agents` applies the same idea at the native trait boundary. Choose one side based on where the tool executes; avoid implementing the same policy twice.

<div class="callout warning"><strong>Security boundary.</strong> A model deciding to call a tool is not authorization. Validate identity, scope, arguments, and spend at the tool or middleware boundary before performing external side effects.</div>

## Resilience wrappers

Model policies compose as decorators:

```python
from litgraph.resilience import (
    RetryingChatModel,
    FallbackChatModel,
    RateLimitedChatModel,
    TokenBudgetChatModel,
    CostCappedChatModel,
    PiiScrubbingChatModel,
    TimeoutChatModel,
)

reliable = RetryingChatModel(model, max_attempts=3, base_delay_ms=200)
reliable = FallbackChatModel([reliable, backup_model])
reliable = RateLimitedChatModel(reliable, requests_per_minute=60)
reliable = TokenBudgetChatModel(reliable, max_input_tokens=20_000)
reliable = CostCappedChatModel(reliable, ceiling_usd=10.0)
reliable = PiiScrubbingChatModel(reliable)
reliable = TimeoutChatModel(reliable, timeout_ms=30_000)
```

Order matters. For example, place a request-wide cost cap outside retry logic if every retry must count against the same budget. Test the composed behavior with a scripted model that emits failures and recoveries in a known order.

## Caching

```python
from litgraph.cache import CachedChatModel, SqliteCache

cached = CachedChatModel(model, cache=SqliteCache("./state/model-cache.db"))
```

Hash caching keys the model, messages, and options for exact reuse. Semantic caching embeds the latest user message and uses a similarity threshold; reserve it for tolerant workloads such as FAQ traffic, never for tool calls or requests where a near match changes meaning.

## Use models from Rust

Every non-binding crate is usable without Python:

```rust
use litgraph_core::{ChatModel, ChatOptions, Message};
use litgraph_providers_openai::OpenAIChat;

let model = OpenAIChat::new(api_key, "gpt-5");
let reply = model
    .chat(&[Message::user("Say hello")], ChatOptions::default())
    .await?;
```

The [Python and Rust guide](/litGraph/python-rust/) explains the crate boundary and how to add an adapter without coupling the core to PyO3.

# litgraph-gateway

`litgraph-gateway` is a self-hosted OpenAI Chat Completions gateway. It gives
clients virtual API keys while routing a model alias across one or more
physical deployments with weighted selection, circuit-breaker failover,
per-key rate limits, and daily spend ceilings.

The gateway exposes:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`, including SSE streaming

## Quick start with Ollama

Start Ollama and pull a model:

```bash
ollama serve
ollama pull qwen2.5:7b
```

Mint a virtual key. Save the plaintext value; only its Argon2id hash is
stored in the gateway config.

```bash
cargo run -p litgraph-gateway -- keygen --id local --group local-chat
```

Create `gateway.toml` from the emitted key stanza and an Ollama deployment:

```toml
[[deployment]]
id = "ollama-local"
group = "local-chat"
provider = "ollama"
model = "qwen2.5:7b"
base_url = "http://127.0.0.1:11434/v1"
weight = 1

[[key]]
id = "local"
prefix = "<prefix printed by keygen>"
hash = "<argon2id hash printed by keygen>"
groups = ["local-chat"]
rpm = 60
max_usd_per_day = 5.0
```

`provider = "ollama"` deliberately needs no credential environment variable.
For OpenAI or any authenticated OpenAI-compatible endpoint, use
`provider = "openai"` and add `api_key_env = "UPSTREAM_API_KEY"`.

Run the gateway:

```bash
cargo run -p litgraph-gateway -- serve --config gateway.toml --bind 127.0.0.1:8080
```

Existing OpenAI clients work without modification beyond the base URL and
virtual key:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://127.0.0.1:8080/v1",
    api_key="<plaintext virtual key>",
)
response = client.chat.completions.create(
    model="local-chat",
    messages=[{"role": "user", "content": "Say hello"}],
)
```

## Routing and policy semantics

Deployments with the same `group` serve one client-visible model alias.
Selection is weighted. Transport failures, timeouts, upstream rate limits,
and 5xx responses may fail over; client request errors do not, because the
same invalid request would fail on every deployment. Each deployment owns a
shared circuit breaker.

Authentication, group allowlists, request rate limits, and spend accounting
are tenant-scoped by virtual key. Daily budgets are intentionally
"reject once over": a request that starts below its ceiling is allowed to
finish and may cross it; the next request is rejected. This avoids cutting a
completion off after tokens have already been generated.

Streaming can fail over only while establishing the upstream stream. After
the first byte, partial output belongs to the client, so a later failure is
reported as an in-band SSE error followed by `[DONE]`. Relayed usage is still
metered; if the upstream dies before its usage event, completion tokens are
estimated from the relayed text.

## Verification and benchmarks

The real-HTTP Rust test covers authentication plus streaming and non-streaming
wire shapes. `python_tests/gateway/test_openai_sdk_compat.py` exercises the
official OpenAI Python SDK against a running gateway.

On an Apple M2 Max with a mock upstream, Criterion measured approximately
7.25 microseconds for a non-streaming in-process round trip, 128 microseconds
to relay 100 SSE chunks, and 1.25 milliseconds for 1,000 chunks. These numbers
measure gateway work only, not network or model latency.

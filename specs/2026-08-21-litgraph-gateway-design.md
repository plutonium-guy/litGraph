# litgraph-gateway — design

**Status:** implemented and verified 2026-08-22
**Date:** 2026-08-21
**Scope:** v1 = routing + metering core

Implementation notes: v1 also includes the `serve` and `keygen` CLI, explicit
`provider = "ollama"` dispatch without an upstream credential, real-HTTP wire
tests, and official OpenAI Python SDK coverage. Streaming failover is limited
to upstream setup; after establishment, errors are reported in-band and
partial output is metered. The measured mock-upstream overhead is about
7.25 µs non-streaming and 1.25 ms for a 1,000-chunk SSE relay.

> Kept outside `docs/` deliberately: `docs/**` triggers the GitHub Pages
> build and `_config.yml` excludes only `README.md`, so anything added
> there publishes to the public site. This is an internal design doc.

---

## 1. Motivation

litGraph positions itself against LangChain, LangGraph, and LiteLLM.
The first two are covered in `COMPARISON.md`; **LiteLLM appears nowhere
in the repo** — not in `COMPARISON.md`, `README.md`, or `ROADMAP.md`.

What LiteLLM actually ships in production is not its SDK but its
**Proxy**: an org-wide OpenAI-compatible gateway with virtual API keys,
per-key budgets, spend tracking, and load-balanced routing across
deployments. That is the unclaimed axis.

litGraph already owns every ingredient — `fallback`, `race`,
`circuit_breaker`, `rate_limit`, `budget`, `cache`, `singleflight`,
`PriceSheet`, `CostTracker` — but they are not assembled into a
gateway, and `litgraph-serve` binds exactly one model
(`AppState { model: Arc<dyn ChatModel> }`).

A gateway is also where Rust earns the most rent. It is pure framework
overhead — accept, parse, route, relay, meter — with no model math.
That is precisely the workload where a Python proxy hits a ceiling and
where `COMPARISON.md`'s existing measurements already point (SSE parse
~12 µs vs ~2 ms per 16 KB chunk).

## 2. Goals

1. OpenAI-compatible `/v1/chat/completions`, streaming and non-streaming.
2. Virtual API keys with per-key authorization, rate limits, and spend caps.
3. Routing across N deployments per model alias, with failover.
4. Materially higher concurrent-stream ceiling and flatter p99 than
   LiteLLM Proxy, proven by a reproducible benchmark.
5. Ships as one Rust binary — no Python at the edge.

## 3. Non-goals (v1)

| Excluded | Rationale |
|---|---|
| Admin HTTP API / key CRUD | Keys come from config. Admin plane is v2. |
| Spend database | In-memory counters behind a `SpendStore` seam. |
| `/v1/embeddings` | Shares all machinery and `Embeddings` has parallel decorators, so it is cheap — but it is surface beyond the agreed v1 boundary. The trait seam is designed to accept it without reshaping the edge. |
| Least-latency routing | Needs per-deployment latency tracking, pathological on cold start. Weighted-random first, behind a trait so it can be swapped. |
| Managed hosting / UI | Out of scope indefinitely. |

## 4. Architecture

A deployment is an `Arc<dyn ChatModel>`, and every litGraph policy is
already a `ChatModel` decorator. The gateway is therefore a thin edge
over a pool of them, not a new execution engine.

### 4.1 The load-bearing decision: policy scope

litGraph's existing wrappers are **per-instance, not per-tenant**.
`RateLimitedChatModel::new(inner, cfg)` holds one bucket shared by all
callers. A gateway needs per-key limits over *shared* deployments.
Wrapping deployments in these decorators would either make all tenants
share a bucket (wrong) or require N-keys × M-deployments instances
(wasteful).

Policy therefore splits by scope:

| Scope | Lives on | Contains | Why |
|---|---|---|---|
| **Deployment** | the pooled `Arc<dyn ChatModel>`, shared | circuit breaker, upstream rate limit, cache, singleflight | Protects the provider. All tenants' failures are evidence about the same upstream, so one shared breaker is correct. |
| **Tenant** | registry keyed by `key_id`, applied at the edge | rate limit, spend cap, group allowlist | Protects fairness. Must never be shared; must survive deployment swaps. |

Existing decorators are reused as-is for the deployment tier. The
tenant tier is the only genuinely new state in the system.

### 4.2 Crate layout

New crate `litgraph-gateway`, feature-flagged, per "each capability is
its own crate; default features stay tight."

Depends on `litgraph-core` (the `ChatModel` trait),
`litgraph-resilience` (routing policy), `litgraph-observability`
(pricing), and reuses `litgraph_serve::auth` rather than
reimplementing bearer handling.

## 5. Components

### 5.1 Config

```toml
[[deployment]]
id = "gpt4o-openai"
group = "gpt-4o"              # the alias clients request
provider = "openai"
model = "gpt-4o"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_KEY"    # env indirection; secrets never in config
weight = 2
rpm = 3000                    # deployment-scoped: protects upstream

[[deployment]]
id = "gpt4o-azure"            # same group, second provider
group = "gpt-4o"
provider = "openai"
model = "gpt-4o"
base_url = "https://acme.openai.azure.com/v1"
api_key_env = "AZURE_KEY"
weight = 1

[[key]]
id = "team-research"
hash = "$argon2id$v=19$..."   # never plaintext
groups = ["gpt-4o", "claude-sonnet-4-5"]   # exact match; no globs in v1
rpm = 600                     # tenant-scoped
max_usd_per_day = 50.0
```

### 5.2 Types

- `Deployment { id, group, model: Arc<dyn ChatModel>, weight, breaker }`
  — `model` is constructed once at startup by the existing provider
  constructors, then wrapped in deployment-scoped decorators.
- `ModelGroup { name, deployments: Vec<Arc<Deployment>> }` — routing unit.
- `VirtualKey { id, hash, groups, rpm, budget }` — hash only, never a secret.
- `TenantPolicy` — `DashMap<KeyId, TenantState>`, where
  `TenantState { bucket: TokenBucket, spend: SpendCounter }`.
- `SpendStore` trait — `record(key_id, usd, tokens)` /
  `spent_today(key_id)`. In-memory for v1; mirrors the `Checkpointer`
  seam so Postgres lands later without touching the edge.
- `RoutingStrategy` trait — weighted-random in v1; the seam for
  least-latency later.

### 5.3 Key format

`lg-sk-<8-char prefix>.<32-byte secret>`

The prefix is an indexed lookup, so verification is one hash rather
than a scan across every key. Only the prefix is ever logged. Config
stores the argon2id hash — a leaked config file yields no working keys.

## 6. Request flow

```
POST /v1/chat/completions  {"model":"gpt-4o","messages":[...],"stream":true}
  1. extract bearer, split prefix/secret; lookup by prefix (O(1));
     constant-time argon2 verify                        -> 401
  2. authorize group against key.groups                 -> 403
  3. tenant gate: bucket.try_acquire(key_id)            -> 429 + Retry-After
     spend check: spent_today < max_usd_per_day         -> 402
  4. route: pick deployment from group, weighted, skipping open breakers
  5. dispatch: invoke() or stream()
     failure before first byte -> next deployment
  6. meter: usd = PriceSheet::lookup(model).cost(usage);
     spend.record(key_id, usd, tokens)
  7. respond in OpenAI wire shape
```

### 6.1 Two properties worth stating explicitly

**Streaming spend metering depends on a fix landed 2026-08-20.** Usage
only arrives on a stream when the client sends
`stream_options.include_usage`; `litgraph-providers-openai` did not
send it, so streamed calls reported zero tokens. Before that fix a
gateway would have metered every streaming request as free. Spend
enforcement on streams was not implementable against a compliant
upstream.

**Budgets are "reject once over", not a hard cap.** Cost cannot be
known before the tokens exist, so step 3 is a pre-flight check and a
single request may overshoot the ceiling. This matches LiteLLM's
guarantee and is stated rather than implied.

## 7. Performance design

Rust does not make the LLM call faster; upstream latency dominates
time-to-first-token. What changes is the **concurrency ceiling and p99
under load**.

| Work | Cost driver | Response |
|---|---|---|
| Auth | argon2id = 10–50 ms | Verify once; cache `prefix -> key_id` in a bounded TTL map. **Uncached this alone makes the gateway slower than LiteLLM.** |
| Request JSON | histories 10–100 KB+ | `serde_json` into borrowed types; no intermediate `Value` trees. |
| SSE relay | ~1 event per token | See 7.1 — the main event. |
| Counters | cross-core contention | Per-key atomics in a sharded map; no global mutex. |
| Upstream conn | TLS handshake ~100 ms | One `reqwest::Client` per deployment at startup, HTTP/2 keepalive. Never per request. |

### 7.1 Don't deserialize chunks you don't need

A relaying gateway needs two things from a stream: forward the bytes,
and read `usage` from the final chunk. It does not need to understand
the other 999.

Forward `bytes::Bytes` (refcount bump, not memcpy); parse only the
terminal usage chunk. Python proxies typically full-parse and
re-serialize every chunk — the ~12 µs vs ~2 ms per-chunk gap compounds
to ~12 ms vs ~2 s of gateway CPU over a 1k-chunk stream.

Wrinkle: the gateway returns the client's alias while upstream chunks
carry the deployment's real model name. When they differ, rewrite that
field with a byte-level splice rather than a JSON round-trip; when they
match, skip entirely.

### 7.2 The structural win: no GIL

Python's per-chunk parse holds the interpreter lock, so N concurrent
streams serialize their parsing onto one core. Tokio spreads the same
work across all cores with work-stealing, and each stream is a task
(~KBs), not a thread. This is a different scaling curve, not a
constant factor — and it is the reason the crate is worth building.

Supporting choices: `mimalloc`, reused buffers instead of per-chunk
`String`, and end-to-end backpressure by piping `reqwest`'s
`bytes_stream` straight into the axum body without buffering.

### 7.3 Proving it

Benchmark against a **mock upstream**, not a real provider — otherwise
the measurement is of OpenAI, not the gateway. Same hardware, same
request mix, litGraph vs LiteLLM Proxy:

- added latency p50 / p99 (gateway overhead only)
- RPS per core at fixed p99
- concurrent streams held before p99 degrades
- RSS per 1k streams

Target: sub-millisecond added p50, and p99 that stays flat where
LiteLLM's climbs. Numbers go in `COMPARISON.md` only after measurement.

**Where little or no win is expected, to be stated up front:**
time-to-first-token, low-RPS deployments, and workloads dominated by a
single fat upstream call.

## 8. Error handling

Errors use the OpenAI wire shape or client SDKs mishandle them:

```json
{"error": {"message": "...", "type": "rate_limit_error", "code": "rate_limit_exceeded"}}
```

| Condition | Status | Fails over |
|---|---|---|
| Unknown / malformed key | 401 | — |
| Group not allowed for key | 403 | — |
| Group not in config | 404 `model_not_found` | — |
| Tenant rate limit | 429 + `Retry-After` | — |
| Tenant budget exhausted | 402 | — |
| Upstream 4xx | passthrough | **No** |
| Upstream 5xx / timeout / conn refused | — | **Yes** |
| All deployments open or exhausted | 503 | — |

A 400 is the client's fault and identical at every deployment;
retrying burns quota everywhere and returns the same error slower. The
same rule governs the breaker: 5xx, timeouts and connection errors
count toward opening it, 4xx never does — otherwise one tenant sending
malformed requests trips the breaker for everyone.

### 8.1 Mid-stream failure

Once `200 OK` and the first chunk are sent, the status code is spent.

- **No failover after first byte.** The client holds partial tokens;
  restarting elsewhere would duplicate or contradict them. Failover
  applies only before first byte. A later upstream failure terminates
  the stream with an SSE error event.
- **Partial usage is still metered.** If a stream dies before the usage
  chunk, record the tokens actually relayed — otherwise disconnecting
  early avoids all billing.
- **No internals in client errors.** Upstream keys, deployment ids and
  base URLs go to the trace, never the response. A client learns
  "gpt-4o is unavailable", not which of three deployments failed.

## 9. Security

The gateway is a security boundary: a routing bug sends one tenant's
traffic through another tenant's upstream key.

- Keys stored as argon2id hashes; plaintext never persisted or logged.
- Constant-time comparison on the secret — no timing oracle.
- Only the key prefix appears in logs and traces.
- Upstream credentials resolved from env at startup, never from config
  or client input.
- The authz suite (§10) lands in the first commit, not later.

## 10. Testing

The rubric axis "testable without a paid LLM" is fully satisfiable: a
`Deployment` holds an `Arc<dyn ChatModel>`, so most of the gateway is
testable with `ScriptedChatModel` and zero network. `cassette.rs`
covers record/replay for the rest.

**Authz (first commit):**
- Key A cannot invoke a group only Key B allows
- Unknown, malformed, revoked keys rejected
- Verification is constant-time
- No key material in logs or errors at any level (assert on captured output)
- Tenant counters never bleed: A's spend never moves B's

**Routing** — seeded RNG for deterministic weighted selection: weight
distribution over N draws, open breakers skipped, failover on 5xx, no
failover on 400, all-down → 503.

**Metering** — accrual for streaming and non-streaming, budget
rejection at threshold, early-disconnect partial metering.

**Rate limiting** — fake clock, never `sleep`. Burst, refill, per-key
isolation.

**Wire compat** — the official `openai-python` client run unmodified
against the gateway, including streaming and error paths. Golden
fixtures alone miss SDK-level assumptions.

**Bench** — the §7.3 mock-upstream harness, reproducible in CI.

## 11. Required doc correction (independent of this work)

`COMPARISON.md` §23 claims:

> Multi-tenant auth scaffolding | ✅ (`litgraph_serve::auth::{bearer_layer,forwarded_user_layer}`)

This is inaccurate today. `ForwardedUser` is extracted by the
middleware and read by nothing; `studio_router` applies no tenant
scoping, so any caller can read or mutate any `thread_id`. Both layer
constructors also failed to compile for any caller until 2026-08-20.

The row should remain partial after gateway v1: the gateway enforces its own
virtual-key tenant boundary, but `litgraph-serve` still does not apply identity
to per-thread ACLs.

## 12. Decisions on points left open during design

**Group matching is exact-match in v1; no globs.** A glob is an authz
surface — `claude-*` silently grants every future model whose name
starts with `claude`, including ones added after the key was issued.
Over-granting is the failure mode that matters at a security boundary,
so v1 requires each group be named. Globs can land later with their own
authz tests. (`§5.1`'s example is exact-match accordingly.)

**No per-deployment concurrency caps (bulkhead) in v1.** Upstream `rpm`
plus the circuit breaker covers the protective case. `bulkhead.rs`
already exists and can be composed onto the deployment tier without
reshaping anything if load testing shows rpm alone is insufficient.
Adding it speculatively would mean tuning two interacting limiters with
no evidence about either.

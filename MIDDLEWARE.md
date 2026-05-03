# Tool middleware in litGraph

litGraph ships parallel middleware systems on both sides of the
PyO3 boundary. Both implement the same contract:

- **`before_tool(name, args)`** runs before every tool dispatch.
  May rewrite args, log, or short-circuit (abort with an error).
- **`after_tool(name, args, result)`** runs after every dispatch.
  May rewrite the result (PII scrub, redaction, response shape) or
  pass through unchanged.

The two implementations share semantics so you can prototype in
Python and migrate hot paths to Rust without rewriting your hooks.

---

## Rust — `litgraph_agents::middleware`

The trait + chain live in
[`crates/litgraph-agents/src/middleware.rs`](crates/litgraph-agents/src/middleware.rs).
Six built-ins ship out of the box:

| Built-in | Purpose |
|---|---|
| `LogToolCallsMiddleware` | `tracing::info!` events on every call (audit / observability sink) |
| `MetricsToolMiddleware` | Counters per tool name + per error; `snapshot()` for dashboards |
| `ToolBudgetMiddleware` | Hard cap on calls per turn; aborts with `ToolBudgetExceeded` |
| `PiiScrubMiddleware` | Email / SSN / credit-card pattern redaction; extensible regexes |
| `CachingToolMiddleware` | moka-LRU keyed on `(name, blake3(args))`; skips errors |
| `RetryOnMiddlewareErrorMiddleware<M>` | Retry shell over a flaky inner middleware |

### Wiring

```rust
use std::sync::Arc;
use litgraph_agents::{ReactAgent, ReactAgentConfig};
use litgraph_agents::middleware::{
    LogToolCallsMiddleware, PiiScrubMiddleware, ToolBudgetMiddleware,
    ToolMiddlewareChain,
};

let cfg = ReactAgentConfig::default()
    .add_middleware(Arc::new(LogToolCallsMiddleware))
    .add_middleware(Arc::new(PiiScrubMiddleware::new()))
    .add_middleware(Arc::new(ToolBudgetMiddleware::new(10)));

let agent = ReactAgent::new(model, tools, cfg)?;
```

`ReactAgentConfig::with_middleware(chain)` replaces the chain
wholesale; `add_middleware(mw)` appends. `TextReactAgentConfig`
exposes the same API for the transcript-mode ReAct flavour.

### Order

`before_tool` runs in **registration order**.
`after_tool` runs in **reverse** order — matches tower / express
semantics so wrappers properly nest.

### Custom middleware

```rust
use litgraph_agents::middleware::{ToolMiddleware, MiddlewareError};
use serde_json::Value;

struct AddTenantHeader { tenant_id: String }

impl ToolMiddleware for AddTenantHeader {
    fn before_tool(&self, _name: &str, args: &Value)
        -> Result<Option<Value>, MiddlewareError>
    {
        let mut new = args.clone();
        if let Some(obj) = new.as_object_mut() {
            obj.insert("tenant_id".into(), Value::String(self.tenant_id.clone()));
        }
        Ok(Some(new))
    }
}
```

Default impls of both methods are no-ops (return `Ok(None)`), so
override only what you need.

---

## Python — `litgraph.tool_hooks`

Pure-Python wrapper; runs in the agent loop without crossing the
PyO3 boundary. Useful when you can't rebuild the native module or
want to compose Python-defined hooks against existing tools.

```python
from litgraph.tool_hooks import (
    BeforeToolHook, AfterToolHook, ToolBudget, wrap_tools,
)
from litgraph.agents import ReactAgent

def audit(name, args):
    print(f"calling {name}({args})")

def redact(name, args, result):
    if isinstance(result, dict):
        result.pop("api_key", None)
    return result

tools = wrap_tools(
    [my_search_tool, my_sql_tool],
    before=BeforeToolHook(audit),
    after=AfterToolHook(redact),
    budget=ToolBudget(max_calls_per_turn=5),
)
agent = ReactAgent(model, tools, system_prompt="...")
```

Returns `HookedTool` wrappers that mirror the underlying tool's
`name` / `description` / `schema` so the agent loop sees them as
drop-in replacements.

---

## Choosing a side

| Use case | Pick |
|---|---|
| Cross-cutting concern across many agents | **Rust** — declared once in `ReactAgentConfig`, inherited everywhere |
| Need to intercept *one* specific tool | **Python `wrap_tools`** — call-site obvious, no agent rebuild |
| Hot path / high-throughput | **Rust** — no Python dispatch per tool call |
| Quick prototype / one-off | **Python** — no maturin rebuild |
| Custom regex / per-tenant config | **Either** — Rust if you want it shared; Python if you want it dynamic |

The two implementations don't share state. If you wire both, both
chains run (Python first, since it intercepts at the tool wrapper
layer; Rust second, since it intercepts at the dispatch layer).

---

## See also

- [USAGE.md §5](USAGE.md#tools--tool-calling-agents) — tools + agents 101
- [COMPARISON.md §7](COMPARISON.md) — feature comparison vs LangChain `before_tool` / `after_tool` callbacks
- [AGENT_DX.md §11](AGENT_DX.md) — schema-first contract design

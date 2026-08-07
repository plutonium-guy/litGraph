---
layout: default
title: Agent harness
description: Create, run, stream, trace, and evaluate batteries-included litGraph agents through one stable harness.
eyebrow: Build agents
---

# One harness for the development loop

`AgentHarness` wraps any object with `invoke(input)` and gives every run a stable result shape. `create_agent` builds the default deep ReAct agent and returns it already wrapped.

## Create an agent

```python
from litgraph import create_agent
from litgraph.providers import AnthropicChat
from litgraph.tools import WebFetchTool

harness = create_agent(
    AnthropicChat(model="claude-sonnet-4-5"),
    tools=[WebFetchTool()],
    instructions="Research carefully and cite the supplied sources.",
    agents_md_path="AGENTS.md",
    skills_dir="skills",
    max_iterations=12,
    planning=True,
    virtual_filesystem=True,
    trace_path=".litgraph/traces.jsonl",
)
```

| Option | Meaning | Default |
|---|---|---|
| `model` | An explicit chat model or compatible test double. | Required |
| `tools` | Native, decorated, or adapted tools available to the agent. | Empty |
| `instructions` | System-level behavior for the agent. | `None` |
| `agents_md_path` | Repository instructions loaded into agent context. | `None` |
| `skills_dir` | Directory containing reusable agent skills. | `None` |
| `max_iterations` | Maximum model/tool loop count; must be positive. | `15` |
| `planning` | Include planning support in the deep agent. | `True` |
| `virtual_filesystem` | Include an in-memory scratch filesystem. | `True` |
| `trace_path` | Append lifecycle and agent events as JSONL. | `None` |
| `on_event` | Receive each normalized event in process. | `None` |

The factory keeps provider selection and capabilities explicit. It does not guess a model or discover tools from global state.

## Run and inspect

```python
run = harness.run("Prepare a concise incident summary")

print(run.output)
print(run.elapsed_ms)
print(run.run_id)
print(run.success)
```

Each `AgentRun` contains:

- `run_id`: a unique hexadecimal identifier.
- `input`: the original text passed to the harness.
- `output`: normalized text extracted from common agent result shapes.
- `state`: the complete native result for advanced inspection.
- `elapsed_ms`: wall-clock execution time.
- `success` and `error`: a uniform outcome pair.

By default, `run()` re-raises agent exceptions after recording the failed run. Set `raise_errors=False` when failures should become data:

```python
run = harness.run("risky task", raise_errors=False)
if not run.success:
    logger.error("agent failed", extra=run.to_dict())
```

`harness.last_run` always points to the latest completed or captured run.

## Stream native events

```python
for event in harness.stream("Draft the report"):
    if event.get("kind") == "text":
        print(event.get("text", ""), end="", flush=True)
```

`stream(input, tokens=True)` prefers the agent’s `stream_tokens()` method and falls back to `stream()`. Set `tokens=False` to request the broader event stream directly. The harness forwards every yielded event to the same tracing and hook pipeline used by ordinary runs.

For model-level async streaming, litGraph also exposes typed stream parts and utilities such as `broadcast`, `race`, and `multiplex`. See [models and tools](/litGraph/models-tools/#streaming).

## Add an event hook

```python
events = []

def capture(event):
    events.append(event)

harness = create_agent(model, on_event=capture)
harness.run("hello")

assert events[0]["type"] == "run_start"
assert events[-1]["type"] == "run_end"
```

Lifecycle records receive a Unix `timestamp` automatically. Event hooks run synchronously, so keep them fast or enqueue work for another consumer.

## Persist JSONL traces

```python
harness = create_agent(model, trace_path=".litgraph/traces.jsonl")
harness.run("Investigate the alert")
```

Each event is serialized as one JSON object per line. Writes are protected by a lock, so multiple harness runs in the same process do not interleave partial records. Inspect a trace with:

```bash
pixi run litgraph trace .litgraph/traces.jsonl
pixi run litgraph trace .litgraph/traces.jsonl --json
```

Use JSONL traces for local debugging and reproducible fixtures; use OpenTelemetry when traces must cross process or service boundaries.

## Evaluate the same agent

```python
report = harness.evaluate(
    [
        {"input": "What is 2 + 2?", "expected": "4"},
        {"input": "What is 7 × 6?", "expected": "42"},
    ],
    scorers=[{"name": "exact_match"}],
    max_parallel=4,
)

print(report["aggregate"]["means"])
```

Evaluation invokes the same `harness.run(prompt).output` path used in production. That prevents test-only adapters from hiding a mismatch in instructions, tools, output normalization, or tracing.

## Wrap an existing agent

The harness is not limited to the built-in deep-agent factory:

```python
from litgraph import AgentHarness
from litgraph.agents import ReactAgent

agent = ReactAgent(model, tools, system_prompt="Be concise.")
harness = AgentHarness(agent, trace_path="runs.jsonl")
```

Any object with a callable `invoke(input)` is accepted. Streaming is available when it also exposes `stream()` or `stream_tokens()`.

## Test agent behavior deterministically

Use `ScriptedChatModel` to specify model responses in order. This is the preferred unit-test seam because the actual agent loop, tool execution, and result normalization still run.

```python
from litgraph.testing import ScriptedChatModel

model = ScriptedChatModel(["draft", "revision"])
harness = create_agent(model)

assert harness.run("first").output == "draft"
assert model.calls[0]["messages"][-1]["content"] == "first"
```

For end-to-end provider tests, keep live credentials and network calls in a separate integration suite. See [observe and evaluate](/litGraph/observability-evaluation/).

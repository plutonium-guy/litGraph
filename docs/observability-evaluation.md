---
layout: default
title: Observe and evaluate
description: Trace model, tool, retrieval, and graph work; track cost; and evaluate agents with deterministic and statistical tests.
eyebrow: Operate confidently
---

# Observe and evaluate

Agent quality is an operational property, not a prompt screenshot. litGraph connects local harness traces, a batched callback bus, OpenTelemetry, cost accounting, deterministic test doubles, and concurrent evaluation.

## Three levels of visibility

| Layer | Best for | Mechanism |
|---|---|---|
| Harness lifecycle | Local development and test fixtures | `on_event`, JSONL `trace_path` |
| Framework events | Model, tool, retrieval, and graph callbacks | callback bus, `CostTracker` |
| Distributed traces | Cross-service production diagnosis | OpenTelemetry / OTLP |

Use stable run and thread identifiers across layers so a failed evaluation case can be followed into model, tool, and checkpoint events.

## Local event traces

```python
from litgraph import create_agent

harness = create_agent(
    model,
    tools=tools,
    trace_path=".litgraph/traces.jsonl",
    on_event=lambda event: print(event["type"]),
)
harness.run("Analyze the incident")
```

Inspect the trace without running a collector:

```bash
pixi run litgraph trace .litgraph/traces.jsonl
pixi run litgraph trace .litgraph/traces.jsonl --json
```

JSONL is intentionally append-friendly and streamable. Treat traces as sensitive: prompts, tool arguments, retrieved documents, and outputs can all contain credentials or personal data.

## Callback bus and cost tracking

```python
from litgraph.observability import CostTracker, on_request

@on_request
def inspect_request(body):
    print("provider request", body)

tracker = CostTracker()
observed_model = tracker.wrap(model)
response = observed_model.invoke(messages)
print(tracker.totals())
```

The native callback bus batches events before crossing into Python. Batching reduces interpreter acquisitions during high-frequency token streams while preserving one coherent event surface for providers, graphs, and agent loops.

Keep price sheets versioned and include provider-reported usage when available. Cost estimates are operational signals, not billing records.

## OpenTelemetry

Set the OTLP endpoint used by your collector:

```bash
export LITGRAPH_OTLP_ENDPOINT=http://localhost:4317
```

The OpenTelemetry integration emits spans for provider requests, tool calls, retrievers, and graph steps. The exporter is feature-gated on the Rust side so deployments that do not use OTLP do not pay the binary-size cost.

Recommended span attributes include:

- run, thread, graph, node, and tool identifiers;
- provider and model names;
- input/output token counts and estimated cost;
- retry count, cache status, and latency;
- checkpoint identifiers and interruption state;
- error category without raw secret values.

## Deterministic unit tests

```python
from litgraph import create_agent
from litgraph.testing import ScriptedChatModel

model = ScriptedChatModel(["known response"])
harness = create_agent(model)

result = harness.run("known request")
assert result.output == "known response"
assert model.call_count == 1
```

Script provider replies, tool calls, errors, and retries so tests cover the actual orchestration path without a live model. This is the right layer for routing, budgets, middleware, schema conversion, and tool-loop behavior.

## Evaluation harness

The high-level agent harness can evaluate its own normalized output:

```python
report = harness.evaluate(
    cases=[
        {"input": "Question A", "expected": "Answer A"},
        {"input": "Question B", "expected": "Answer B"},
    ],
    scorers=[{"name": "exact_match"}],
    max_parallel=4,
)
```

For a custom target and metric suite, use the lower-level evaluation API:

```python
from litgraph.eval import EvalHarness, ExactMatch, BLEU, ROUGE

suite = EvalHarness(
    cases=cases,
    target=lambda case: answer(case["input"]),
    metrics=[ExactMatch(), BLEU(), ROUGE.l()],
)
report = suite.run()
```

Available metrics cover exact match, BLEU, ROUGE, chrF, METEOR-lite, BERTScore-lite, word and character error rates, TER, and relaxed word-mover distance. Statistical helpers include paired permutation tests and rank or linear correlations. LLM judges and trajectory evaluation plug into the same reporting surface.

## A practical quality loop

<ol class="steps">
  <li><strong>Capture representative cases.</strong> Include normal work, edge cases, unsafe requests, tool failures, and long contexts.</li>
  <li><strong>Choose observable criteria.</strong> Separate retrieval recall, tool correctness, state transitions, final-answer quality, latency, and cost.</li>
  <li><strong>Run deterministic tests first.</strong> Catch orchestration regressions without provider noise.</li>
  <li><strong>Run model evaluations.</strong> Pin model identifiers and record prompts, policies, and dataset versions.</li>
  <li><strong>Inspect regressions.</strong> Use traces and checkpoint state, not only aggregate means.</li>
  <li><strong>Gate releases.</strong> Define acceptable quality, safety, latency, and spend changes before reviewing results.</li>
</ol>

Avoid optimizing one aggregate score. Agent systems fail in clusters—one tool, route, language, or data source—so always retain per-case outputs and error categories.

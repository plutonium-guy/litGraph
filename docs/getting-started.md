---
layout: default
title: Getting started
description: Install litGraph, use Pixi for source development, and run a deterministic first agent.
eyebrow: Start here
---

# Getting started

For application use, install the published wheel:

```bash
pip install litgraph
```

Use Pixi for the most reproducible source-development setup. It supplies
Python, Rust, maturin, pytest, and the project tasks from one lockfile.

## Requirements

- Git.
- [Pixi](https://pixi.sh/latest/installation/).
- An API key only when you call a hosted model. The offline quickstart needs no key.

The workspace supports macOS arm64, Linux x86-64, and Linux arm64. Python is constrained to 3.9–3.13 and Rust to 1.75 or newer.

## Clone and build from source

```bash
git clone https://github.com/plutonium-guy/litGraph.git
cd litGraph
pixi run develop
```

`pixi run develop` executes `maturin develop --release`. Maturin compiles `crates/litgraph-py`, links the Rust workspace, and installs the native module into Pixi’s Python environment in editable mode.

Confirm the environment:

```bash
pixi run litgraph doctor
pixi run python -c "import litgraph; print(litgraph.__version__)"
```

<div class="callout"><strong>Why Pixi?</strong> The checked-in <code>pixi.lock</code> keeps the Python and Rust toolchain inputs consistent across contributors and CI. You can still use a conventional virtual environment and maturin if that better fits your system.</div>

## Run an agent without an API key

The scripted model makes the first run deterministic while exercising the real tool loop and harness.

```bash
pixi run python examples/scripted_agent.py
```

Its complete source is small enough to understand at a glance:

```python
from litgraph import create_agent
from litgraph.testing import ScriptedChatModel
from litgraph.tools import tool

@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b

model = ScriptedChatModel([
    {
        "tool_calls": [{
            "id": "add-1",
            "name": "add",
            "arguments": {"a": 17, "b": 25},
        }]
    },
    "17 + 25 = 42",
])

harness = create_agent(
    model,
    tools=[add],
    instructions="Use tools for arithmetic.",
)
result = harness.run("What is 17 + 25?")
assert result.output == "17 + 25 = 42"
```

This verifies tool schema generation, the model/tool loop, normalized results, and the native runtime without network variability.

## Connect a provider

Provider credentials can come from conventional environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`, or the AWS credential chain for Bedrock.

```bash
export OPENAI_API_KEY="..."
```

```python
from litgraph import create_agent
from litgraph.providers import OpenAIChat
from litgraph.tools import CalculatorTool

harness = create_agent(
    OpenAIChat(model="gpt-5"),
    tools=[CalculatorTool()],
    instructions="Use the calculator for arithmetic.",
)
print(harness.run("Compute (41 × 19) + 7").output)
```

For Ollama, vLLM, LM Studio, or another OpenAI-compatible service, keep the same class and make the endpoint explicit:

```python
model = OpenAIChat(
    model="llama3",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
```

## Scaffold a new solution

The CLI can create a Pixi-ready project:

```bash
pixi run litgraph init chat-agent my-agent
cd my-agent
pixi run test
pixi run start
```

The scaffold keeps configuration in `pyproject.toml`, makes the model choice visible, and includes a testable entry point. Run `litgraph doctor` when an environment or native-module mismatch is suspected.

## Development commands

| Command | Purpose |
|---|---|
| `pixi run develop` | Rebuild and install the release-mode PyO3 extension. |
| `pixi run check-rust` | Run `cargo check --workspace`. |
| `pixi run test-python` | Rebuild, then run the Python suite. |
| `pixi run test-stubs` | Rebuild, then check native bindings against PEP 561 stubs. |
| `pixi run test` | Run all configured Pixi checks. |

After changing Rust code, rebuild before Python tests; otherwise Python may load an older native module. For IDE autocomplete and static checking, install the separate stub package with `pip install ./litgraph-stubs`.

## Where next?

- [Agent harness](/litGraph/agent-harness/) for runs, streams, traces, and evaluation.
- [Graphs and workflows](/litGraph/graphs/) for stateful or branching systems.
- [Serve and deploy](/litGraph/deployment/) for graph APIs and the OpenAI-compatible LLM gateway.
- [Models and tools](/litGraph/models-tools/) for providers, schemas, and middleware.
- [Troubleshooting](/litGraph/troubleshooting/) for build, interpreter, and native-module issues.

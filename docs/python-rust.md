---
layout: default
title: Python and Rust
description: Work safely across litGraph's PyO3 boundary, rebuild with maturin, expose new native APIs, and keep stubs and tests synchronized.
eyebrow: Native boundary
---

# Python and Rust

litGraph uses PyO3 0.22 and maturin to ship a Rust runtime behind a Python-first package. The boundary is intentionally narrow: Python handles application composition, while Rust owns performance-sensitive and concurrency-heavy work.

## Package shape

```text
python/litgraph/__init__.py
        │ imports and augments
        ▼
litgraph.litgraph               native extension
        │ built from
        ▼
crates/litgraph-py              PyO3 conversion + registration
        │ delegates to
        ▼
litgraph-core + graph + agents + adapters
```

Maturin configuration lives in `pyproject.toml`:

- `manifest-path` points to `crates/litgraph-py/Cargo.toml`.
- `module-name` installs the extension as `litgraph.litgraph`.
- `python-source` packages the surrounding `python/litgraph` code.
- the `pyo3/extension-module` feature configures native linking.

The pure-Python package can still expose helpers in a development checkout where the native module has not been built, but production installs include the extension.

## Native module registration

`crates/litgraph-py/src/lib.rs` registers focused submodules such as providers, graph, tools, agents, retrieval, observability, cache, memory, MCP, middleware, deep-agent, and serve.

Submodules are inserted into `sys.modules` in addition to being attached to the parent module. This is what makes both forms work:

```python
import litgraph
from litgraph.graph import StateGraph
```

Keep registration declarative and local. Adding a new Python surface normally means adding or extending one module file and registering its classes or functions from that module’s `register` function.

## Interpreter discipline

The safe call shape is:

<div class="flow"><span>Python values</span><i>→</i><span>validate / convert</span><i>→</i><span>release Python</span><i>→</i><span>native I/O or compute</span><i>→</i><span>convert result</span></div>

Do not hold the GIL across provider requests, database work, graph execution, or other blocking native operations. PyO3 0.22 bindings use the version-appropriate thread-release API; when the project upgrades PyO3, preserve the behavior with the corresponding detached execution API.

Python callbacks are the exception: a Python graph node or tool callable must reacquire Python long enough to invoke it and convert its return value. Drop the acquisition again before awaiting more native work.

## Shared async runtime

The binding owns one lazily initialized Tokio runtime for the process. Provider methods, graph scheduling, and async storage use it rather than creating a runtime per call.

Do not call a fresh `Runtime::new()` inside each PyO3 method. Beyond startup overhead, nested runtimes make cancellation and spawned task ownership difficult to reason about.

## Add a native API

<ol class="steps">
  <li><strong>Define the Rust contract.</strong> Put shared types or traits in <code>litgraph-core</code>; put implementation in the focused crate.</li>
  <li><strong>Test it in Rust.</strong> Cover success, error, cancellation, and serialization behavior before adding Python conversion.</li>
  <li><strong>Add the binding.</strong> Convert arguments while Python is held, release it around native work, and map domain errors to useful Python exceptions.</li>
  <li><strong>Register the surface.</strong> Extend the relevant <code>register</code> function and, for a new submodule, add it in <code>lib.rs</code>.</li>
  <li><strong>Add Python ergonomics only when needed.</strong> Decorators and type-aware wrappers belong under <code>python/litgraph</code>.</li>
  <li><strong>Update PEP 561 stubs.</strong> Keep signatures, defaults, return types, and module placement synchronized.</li>
  <li><strong>Add one Python test file per public surface.</strong> Test the rebuilt extension inside the project environment.</li>
</ol>

## Rebuild and verify

```bash
pixi run develop
pixi run test-python
pixi run test-stubs
```

For focused Rust work:

```bash
cargo check --workspace
cargo test --workspace --lib
cargo clippy --workspace --all-targets
```

Always run Python tests through the Pixi environment or the project `.venv` that maturin updated. A separate Homebrew or system interpreter can have an older installed wheel and create convincing false failures.

## abi3 compatibility

The wheel targets Python’s stable ABI starting at 3.9. Avoid using an API that requires a newer CPython-specific ABI unless the minimum is intentionally changed. Free-threaded Python also makes interpreter-release discipline and thread-safe shared state essential.

The repository’s [free-threading notes](https://github.com/plutonium-guy/litGraph/blob/main/FREE_THREADING.md) document the current audit and opt-in path.

## Error design

Native errors should reach Python with:

- a stable exception category;
- enough context to identify the provider, node, tool, or backend;
- a corrective action when one is known;
- no secrets, raw credentials, or unnecessarily large payloads.

Avoid returning sentinel dictionaries for errors that callers cannot safely ignore. Conversely, do not translate ordinary model refusals or empty retrieval results into runtime exceptions when they are valid domain outcomes.

## Keep the boundary thin

If a PyO3 method starts owning retries, scheduling, storage policy, or provider logic, move that behavior into a pure Rust crate and leave only argument/result conversion in the binding. This keeps Rust reuse, testing, and concurrency properties intact.

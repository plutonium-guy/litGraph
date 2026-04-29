# Free-threading (PEP 703) status

CPython 3.13t / 3.14 free-threaded builds run without the GIL. Native
extensions need to be audited (and opt in) before they can run safely under
free-threading.

## tl;dr

litGraph is **expected to work** under free-threaded CPython, pending the
PyO3 0.23+ upgrade that adds the formal `gil_used = false` opt-in. The
audit below shows we have no `unsafe`, no `static mut`, and no shared
mutable state outside `parking_lot::RwLock` / `tokio::sync::Mutex`.

## Audit (iter 30, PyO3 0.22.6, edition 2021)

### `unsafe`

```bash
$ grep -rn "unsafe " --include="*.rs" crates/
(no results)
```

Zero `unsafe` blocks across the entire workspace. Free-threading concerns
that would otherwise apply to `unsafe` (data races, torn reads, Send/Sync
unsafety) do not apply.

### `static mut` and `UnsafeCell`

```bash
$ grep -rn "static mut\|UnsafeCell" --include="*.rs" crates/
(no results)
```

Zero direct uses. The only `static` is:

- `RT: OnceCell<Runtime>` in `crates/litgraph-py/src/runtime.rs` — `OnceCell`
  is thread-safe by construction.

### `#[pyclass]` Send/Sync posture

All `#[pyclass]`es are now sendable (no `unsendable` flag remaining as of
iter 30). Inner state is one of:

- `Arc<dyn Trait + Send + Sync>` (model implementations)
- `Arc<Mutex<...>>` / `RwLock<...>` (queue receivers, store-owned data)
- `Py<PyAny>` (PyO3 ref-counted Python handle; `Send + Sync` in 0.22+)
- Plain `Copy` config fields (Strings, ints, Durations)

There is no field whose mutation is guarded only by the GIL.

### Static state

| Location                         | Type                       | Thread-safety                                  |
| -------------------------------- | -------------------------- | ---------------------------------------------- |
| `litgraph-py::runtime::RT`       | `OnceCell<Runtime>`        | OnceCell is sound; tokio runtime is Send+Sync. |
| `litgraph-cache::semantic`       | `RwLock<Vec<Entry>>`       | Standard sync primitive.                       |
| `litgraph-stores-memory`         | `RwLock<HashMap<...>>`     | Standard sync primitive.                       |
| `litgraph-stores-hnsw`           | `RwLock<Inner>`            | Standard sync primitive.                       |
| `litgraph-checkpoint-{sqlite,postgres,redis}` | `Mutex<Connection>` / pool | Per-conn mutex or deadpool. |
| `litgraph-observability::CallbackBus` | `Mutex<Vec<...>>` for subscribers | Lock dropped before await. |

### Python-callback-from-Rust patterns

- `litgraph.tools.FunctionTool` → `Tool::run` re-acquires GIL via
  `Python::with_gil`, dispatches to a Python callable, releases.
- `litgraph.graph.StateGraph` → node functions go through the same path.
- `litgraph.providers.*.on_request` → the request-inspector hook does the
  same. Errors raised by the Python callback are logged via `tracing` rather
  than panicked, so a broken hook can't take down the request path.

Under free-threading these still serialize per-callable on the Python
interpreter's per-object locks, but the rest of our work continues
GIL-free in parallel.

## Opt-in path (PyO3 0.23+ upgrade)

PyO3 0.23 added a `#[pymodule(gil_used = false)]` attribute that opts the
extension into the free-threaded build. After upgrading:

```rust
#[pymodule(gil_used = false)]
fn litgraph(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // ...
}
```

Then maturin needs the `--features pyo3/abi3-py313` (or higher) and a
non-abi3 wheel for the free-threaded interpreter (`*-cp313t-*.whl`).

We're holding off on the upgrade until 0.23+ is stable in our pinned
toolchain and the rest of the PyO3 ecosystem (pyo3-async-runtimes,
rust-numpy if added later, pyo3-stub-gen) catches up. See iter 30 memory
notes for the latest status.

## Known not-yet-audited areas

- `pyo3-async-runtimes` interaction with `asyncio.run()` under free-threading
  (we currently use `asyncio.run()` to drive Python async nodes — verifying
  this stays correct when multiple Rust threads run nodes concurrently).
- Behavior of `Py<PyAny>` drops from off-Python threads under free-threading
  (PyO3 0.23 changed the rules).
- Rayon thread pool interaction with free-threaded interpreter for our batch
  splitter / embedding paths.

These will be revisited at the 0.23 upgrade.

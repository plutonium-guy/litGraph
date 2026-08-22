---
layout: default
title: Troubleshooting
description: Diagnose native builds, Python interpreter mismatches, credentials, graph execution, stubs, external-drive artifacts, and deployments.
eyebrow: Diagnose
---

# Troubleshooting

Start with the project diagnostic, then narrow the failing layer.

```bash
pixi run litgraph doctor
git status --short --branch
```

## Native module is not built

**Symptom:** importing `litgraph` warns that `litgraph.litgraph` is missing, or a native submodule cannot be imported.

```bash
pixi run develop
pixi run python -c "import litgraph; print(litgraph.__version__)"
```

Maturin must compile and install the extension into the same interpreter used for the application.

## Python sees old behavior after a Rust change

**Cause:** Python is loading an existing wheel or a different environment instead of the freshly built extension.

```bash
pixi run develop
pixi run python -c "import sys, litgraph; print(sys.executable); print(litgraph.__file__)"
```

On macOS, Homebrew Python and the project environment are separate installations. Do not validate a Pixi/maturin build with `/opt/homebrew/bin/python3` unless that is intentionally the deployment interpreter.

## Maturin or Rust compilation fails

Run the narrower native check first:

```bash
pixi run check-rust
cargo check -p litgraph-py
```

Confirm Rust is at least 1.75, Python is within the supported range, and the platform matches a Pixi environment in `pixi.toml`. For a full compiler diagnostic, rerun `maturin develop --release` without suppressing output.

## Provider authentication fails

Verify the variable in the same process environment used by Pixi:

```bash
pixi run python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

Use the appropriate variable for the provider, or pass `api_key` explicitly. Bedrock follows the AWS credential chain. Do not print the credential itself into a terminal recording or trace.

## OpenAI-compatible endpoint fails

Make the model, key placeholder, and base URL explicit:

```python
model = OpenAIChat(
    model="llama3",
    api_key="ollama",
    base_url="http://localhost:11434/v1",
)
```

Check whether the service includes `/v1` in its compatibility endpoint and whether it implements tool calls or streaming used by the application.

## Gateway returns 401, 403, 429, or 503

- `401` means the virtual key is missing, malformed, or failed Argon2id
  verification. Use the plaintext emitted by `keygen`, not the stored hash.
- `403` means the key's exact `groups` allowlist does not contain the requested
  model alias.
- `429` means the tenant request limit or daily spend ceiling rejected the
  request. Budgets reject the next request after the ceiling is crossed; they
  do not cut off an active completion.
- `503` means no healthy deployment could serve the group. Check upstream
  health, model names, API-key environment variables, and circuit-breaker
  cooldowns. Ollama deployments should use `provider = "ollama"` and need no
  `api_key_env`.

For an SSE stream that fails after it begins, inspect the in-band `error` event.
HTTP status can no longer change after response headers and partial tokens have
been sent; the gateway follows the error with `[DONE]` and meters partial usage.

## Graph does not terminate

Inspect conditional routes and cycles, then lower the recursion limit while debugging. Every cycle needs an explicit state change and exit condition. Generate `to_ascii()` or `to_mermaid()` output to confirm edges match the intended workflow.

For agents, also inspect `max_iterations`; repeated tool calls may indicate that tool output is not being added in the shape the model expects.

## Parallel branches overwrite state

Parallel nodes return partial updates that are combined by the reducer. If several branches write the same key, configure an append/merge reducer or give each branch a distinct field and combine them in a join node.

Do not depend on branch completion order. Network and tool latency can change it between runs.

## Resume pauses at the same interrupt

Resume through the compiled graph with the same thread identifier and stored checkpoint. Starting a new invocation without the checkpoint context is a new run and will correctly hit `interrupt_before` again.

## Native API and type stubs disagree

```bash
pixi run develop
pixi run test-stubs
```

Update the matching file under `litgraph-stubs` when a public binding is added or changed. External-drive AppleDouble files named `._*.pyi` are not stubs and must be ignored by glob-based tooling.

## Tests use the wrong environment

Prefer Pixi tasks:

```bash
pixi run test-python
pixi run test-stubs
```

If using `.venv`, activate it before rebuilding and testing. Confirm `sys.executable` and `litgraph.__file__` whenever results contradict the current source.

## Git reports corrupt objects on an external macOS drive

macOS can create AppleDouble files named `._*`, including under `.git`, on non-native filesystems. They are metadata companions, not Git objects. Remove only those precisely identified files, keep them ignored, then run:

```bash
git fsck --no-dangling
```

Never delete ordinary object files or reset the repository to solve this symptom.

## Documentation deployment fails

Run the dependency-free source check locally:

```bash
python tools/check_docs.py
```

Then inspect the **Documentation** workflow in GitHub Actions. GitHub Pages must use **GitHub Actions** as its build source, and the workflow needs `pages: write` plus `id-token: write` permissions.

## Still stuck?

Check [MISSING_FEATURES](https://github.com/plutonium-guy/litGraph/blob/main/MISSING_FEATURES.md) to see whether the behavior is intentionally unimplemented, search the matching `python_tests/test_<feature>.py`, and open an issue with:

- litGraph version and installation method;
- Python, Rust, OS, and architecture;
- the smallest reproducible example;
- complete error text with secrets removed;
- whether `pixi run litgraph doctor` and the focused tests pass.

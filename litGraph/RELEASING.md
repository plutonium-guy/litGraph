# Releasing litGraph

Tag-driven. CI does the rest.

## What gets released

A litGraph release ships **two PyPI packages**:

| Package          | Source                | Built by                                |
| ---------------- | --------------------- | --------------------------------------- |
| `litgraph`       | `crates/litgraph-py`  | `.github/workflows/CI.yml`              |
| `litgraph-stubs` | `litgraph-stubs/`     | `.github/workflows/release-stubs.yml`   |

Both publish to PyPI via [trusted publishing](https://docs.pypi.org/trusted-publishers/).
Set up the publisher entries on PyPI (one per package) before the first
release — workflow + repo are configured for `id-token: write`.

## Pre-tag checklist

1. `cargo test --workspace` is green.
2. `cargo clippy --workspace --all-targets -- -D warnings` is clean (or
   accepted as carry-over noise).
3. Every file in `python_tests/` passes (`maturin develop --release` then
   loop the directory).
4. `cargo bench -p litgraph-bench --bench bm25 -- --quick` numbers haven't
   regressed; if a hot path moved, update `FEATURES.md`.
5. `CHANGELOG.md` `[Unreleased]` section moved under a new `## [X.Y.Z]`
   heading with today's date.
6. Bump versions:
   - `Cargo.toml` → `[workspace.package] version = "X.Y.Z"`
   - `litgraph-stubs/setup.py` → `version="X.Y.Z"` (only if public API changed)
   - Run `cargo build --workspace` once to refresh `Cargo.lock`.

## Tagging

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

This triggers:
- `CI.yml` — builds wheels for Linux (x86_64, x86, aarch64, armv7, s390x,
  ppc64le), musllinux (4 targets), Windows, macOS, sdist; signs them with
  build provenance; publishes to PyPI as `litgraph`.
- `release-stubs.yml` — builds + publishes `litgraph-stubs`.

## SemVer policy

litGraph follows SemVer.

- **Patch (X.Y.Z+1)**: bug fixes, doc-only changes, perf wins that don't
  change the public API.
- **Minor (X.Y+1.0)**: new methods, new submodules, new providers/stores.
  Adding optional kwargs is minor; making old kwargs required is breaking.
- **Major (X+1.0.0)**: removing/renaming public types or methods, breaking
  trait signatures, dropping a provider/store crate.

Deprecate before removing. Two minor versions is the minimum
deprecation window for any documented public API.

## Hotfix

For an urgent fix on a released version:

```bash
git checkout vX.Y.Z
git checkout -b hotfix/X.Y.(Z+1)
# apply fix, bump versions, update CHANGELOG
git tag vX.Y.(Z+1)
git push origin vX.Y.(Z+1)
```

CI auto-publishes; `main` gets the same fix via a follow-up cherry-pick or PR.

## Manual sanity check after publish

```bash
pip install --upgrade litgraph litgraph-stubs
python -c "import litgraph; print(litgraph.__version__)"
```

If anything looks off, file an issue and yank the bad version with
`twine yank litgraph==X.Y.Z` (PyPI keeps the file listed but marks it
unavailable for new installs — safer than deletion).

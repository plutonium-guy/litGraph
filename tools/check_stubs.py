#!/usr/bin/env python3
"""Stub-drift checker — catch rot in `litgraph-stubs/*.pyi` against
the runtime native module.

Run from the repo root:

    python tools/check_stubs.py

Exit code 0 if stubs are in sync, 1 if drift detected. CI-runnable.

# What this is

A bridge primitive between "manual stubs rot" (the existing state)
and "pyo3-stub-gen ships full auto-generation" (still on roadmap).
This tool catches drift early — when a Rust dev adds a new
`#[pyfunction]` or `#[pyclass]` and forgets to update the
corresponding `.pyi`, this script reports it before the IDE goes
silent.

# What this is NOT

Not a stub generator. Doesn't write `.pyi` files. Doesn't
introspect Rust source — it walks the BUILT native module via
Python introspection, which means:

- Requires `maturin develop` to have been run (or a wheel
  install) so the native module is importable.
- If the native module isn't built, the script logs a warning
  and exits 0 (so CI on a Rust-only branch doesn't false-fail).

# What "drift" means here

For each public attribute on `litgraph.litgraph` (the native
submodule) and on each class within it, check whether a
corresponding stub entry exists in `litgraph-stubs/litgraph-stubs/`.
Report:

- **missing_in_stubs**: attributes present at runtime but absent
  from any .pyi file. These cause IDE autocomplete to be silent
  for newly-added bindings.
- **missing_in_runtime**: attributes named in stubs but absent at
  runtime. These cause IDE autocomplete to suggest dead names.

The output is a flat sorted list — easy to diff in PR review.

# Limitations

- Compares attribute names only, not signatures. A stub with the
  wrong arg types still passes this check (drift in signature
  semantics is what pyo3-stub-gen would catch with full type
  parity).
- Skips dunder attrs (`__doc__`, `__loader__`, etc) and any
  `_private` names.
- Walks one level deep into classes (their methods). Doesn't
  recurse into nested classes — uncommon in PyO3 bindings.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Set


REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_DIR = REPO_ROOT / "litgraph-stubs" / "litgraph-stubs"


_KEPT_DUNDERS = {"__version__", "__all__"}
# A package's own name appears as `litgraph.litgraph` (the native
# extension submodule). The package surface is already captured by
# walking the package directly, and the native submodule's contents
# are checked when collect_runtime_attrs is called with native=...
# Skip the self-import to avoid expecting a `litgraph.pyi`.
_SELF_NAMES = {"litgraph"}


def collect_runtime_attrs(module: object) -> Set[str]:
    """Return a set of "qualified" attribute names from a native module.

    Top-level functions: `funcname`.
    Top-level classes: `ClassName` and each method as `ClassName.method`.
    Submodules are reported as a single name (`submodname`) — their
    contents live in their own .pyi file and are checked separately
    via the stub-file basenames.
    """
    import types

    out: Set[str] = set()
    for name in dir(module):
        if name.startswith("_") and name not in _KEPT_DUNDERS:
            continue
        if name in _SELF_NAMES and getattr(module, "__name__", "") == "litgraph":
            # `litgraph.litgraph` — the native submodule — is checked
            # separately. Don't expect a top-level stub for it.
            continue
        obj = getattr(module, name)
        if isinstance(obj, types.ModuleType):
            out.add(name)
        elif isinstance(obj, type):
            out.add(name)
            for member in dir(obj):
                if member.startswith("_"):
                    continue
                out.add(f"{name}.{member}")
        else:
            out.add(name)
    return out


def collect_stub_attrs(stub_dir: Path) -> Set[str]:
    """Parse all .pyi files in `stub_dir` and return the set of names
    they declare. Same shape as `collect_runtime_attrs`.

    A `.pyi` file at top level is treated as a stub for a submodule
    of the same basename — so `agents.pyi` covers `litgraph.agents`
    (the runtime submodule), and the basename is added to the
    declared-names set."""
    out: Set[str] = set()
    if not stub_dir.is_dir():
        return out
    for pyi in stub_dir.glob("*.pyi"):
        # Skip macOS AppleDouble sidecars (`._foo.pyi`) that appear
        # when working off external drives. They're not real stubs
        # and the binary metadata blows up `ast.parse`.
        if pyi.name.startswith("._"):
            continue
        # Filename minus extension is itself a covered name (the
        # submodule). Skip `__init__` which represents the package.
        if pyi.stem != "__init__":
            out.add(pyi.stem)
        try:
            tree = ast.parse(pyi.read_text(encoding="utf-8"))
        except SyntaxError:
            # Malformed stub — surface as drift later (every name
            # in this file will show as missing).
            continue
        for node in tree.body:
            _walk_top(node, out)
    return out


def _walk_top(node: ast.AST, out: Set[str]) -> None:
    """Top-level handler — captures function defs + class defs."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        out.add(node.name)
    elif isinstance(node, ast.ClassDef):
        out.add(node.name)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.add(f"{node.name}.{child.name}")
    elif isinstance(node, ast.Assign):
        # `x: T = ...` or `x = ...` — module-level constant or alias.
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                out.add(tgt.id)
    elif isinstance(node, ast.AnnAssign):
        # `x: T` annotation-only.
        if isinstance(node.target, ast.Name):
            out.add(node.target.id)


def main() -> int:
    try:
        # The native module is an attribute of the outer Python
        # package — `litgraph.litgraph`.
        sys.path.insert(0, str(REPO_ROOT / "python"))
        import litgraph as _pkg

        native = getattr(_pkg, "litgraph", None)
        if native is None:
            print(
                "stub-drift: native module `litgraph.litgraph` not built. "
                "Run `maturin develop` to build, or skip this check on "
                "Rust-only branches.",
                file=sys.stderr,
            )
            return 0
    except ImportError as e:
        print(f"stub-drift: cannot import litgraph package: {e}", file=sys.stderr)
        return 0

    # Runtime: union of (a) native PyO3 module attrs and (b) the
    # outer Python package's attrs. The two together represent
    # everything an IDE will see when the user types `litgraph.X`.
    runtime = collect_runtime_attrs(native) | collect_runtime_attrs(_pkg)
    # Stubs: every name declared by any .pyi in the stub dir, plus
    # the basename of each .pyi (which covers submodule presence).
    stubs = collect_stub_attrs(STUB_DIR)

    # We only flag drift on names that *should* exist on the package
    # surface. Stub-side names from per-feature .pyi files (e.g.,
    # `AnthropicChat` in `providers.pyi`) describe attributes of the
    # `litgraph.providers` submodule — checking them against the
    # top-level `litgraph` namespace is structurally wrong and
    # produces false positives. Limit `missing_in_runtime` to
    # `__init__.pyi`-declared names plus .pyi basenames.
    init_pyi = STUB_DIR / "__init__.pyi"
    init_names: Set[str] = set()
    if init_pyi.is_file():
        try:
            for node in ast.parse(init_pyi.read_text(encoding="utf-8")).body:
                _walk_top(node, init_names)
        except SyntaxError:
            pass
    submodule_names = {p.stem for p in STUB_DIR.glob("*.pyi") if p.stem != "__init__"}
    relevant_stub_side = init_names | submodule_names

    missing_in_stubs = sorted(runtime - stubs)
    missing_in_runtime = sorted(relevant_stub_side - runtime)

    if not missing_in_stubs and not missing_in_runtime:
        print(f"stub-drift: in sync ({len(runtime)} runtime attrs, {len(stubs)} stub attrs)")
        return 0

    print("stub-drift: DRIFT DETECTED")
    if missing_in_stubs:
        print(f"  Missing from stubs ({len(missing_in_stubs)}):")
        for name in missing_in_stubs:
            print(f"    + {name}")
    if missing_in_runtime:
        print(f"  Missing from runtime ({len(missing_in_runtime)}):")
        for name in missing_in_runtime:
            print(f"    - {name}")
    return 1


if __name__ == "__main__":
    sys.exit(main())

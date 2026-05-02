"""`litgraph` CLI dispatcher.

Subcommands:
    doctor     env diagnostic (see _doctor.py)
    version    print the installed litGraph version

Designed for AI coding agents. Output is grep-friendly; `--json`
where it makes sense.
"""
from __future__ import annotations

import sys


__all__ = ["main"]


_USAGE = """\
usage: litgraph <command> [options]

commands:
  doctor             Diagnose env (native module, API keys, Python build).
                     --json prints a structured report.
  version            Print the installed litGraph version.
  examples [list]    List bundled example files (one per pattern).
  show <symbol>      Print signature + docstring for `litgraph.<symbol>`.
                     Example: `litgraph show ReactAgent`.
  init <tmpl> <dir>  Scaffold a new project from a template
                     (chat-agent | rag | eval-suite).
  help               Show this message.

Examples:
  litgraph doctor
  litgraph doctor --json
  litgraph version
  litgraph examples
  litgraph show ReactAgent
"""


def _cmd_doctor(argv: list[str]) -> int:
    from ._doctor import main as doctor_main
    return doctor_main(argv)


def _cmd_version(_argv: list[str]) -> int:
    try:
        import litgraph
        print(getattr(litgraph, "__version__", "<unknown>"))
        return 0
    except ImportError as e:
        print(f"litgraph not importable: {e}", file=sys.stderr)
        return 1


def _cmd_help(_argv: list[str]) -> int:
    print(_USAGE)
    return 0


def _cmd_examples(_argv: list[str]) -> int:
    """Print the index of bundled examples. Looks in `examples/` next
    to the source checkout; if not found (wheel install), prints a
    pointer at the GitHub examples directory."""
    import os
    import litgraph

    pkg_dir = os.path.dirname(os.path.abspath(litgraph.__file__))
    # Walk up looking for examples/ (only present in source checkout).
    candidate = pkg_dir
    examples_dir: str | None = None
    for _ in range(5):
        candidate = os.path.dirname(candidate)
        d = os.path.join(candidate, "examples")
        if os.path.isdir(d):
            examples_dir = d
            break

    if examples_dir is None:
        print("Examples are bundled with the source checkout; for a "
              "wheel install browse:")
        print("  https://github.com/plutonium-guy/litGraph/tree/main/examples")
        return 0

    names = sorted(
        f for f in os.listdir(examples_dir)
        if f.endswith(".py")
        and not f.startswith("_")
        # Skip macOS AppleDouble sidecars (`._foo.py`) — they're not
        # real source.
        and not f.startswith(".")
    )
    print(f"litgraph examples ({examples_dir}):")
    for name in names:
        # First non-blank line of the docstring is the one-line summary.
        path = os.path.join(examples_dir, name)
        summary = "(no summary)"
        try:
            with open(path, "r", encoding="utf-8") as f:
                src = f.read(2048)
            # Pull the first triple-quoted line.
            if '"""' in src:
                inner = src.split('"""', 2)[1]
                first = next(
                    (ln for ln in inner.splitlines() if ln.strip()),
                    "",
                )
                summary = first.strip()
        except OSError:
            pass
        print(f"  {name:<30s}  {summary}")
    return 0


def _cmd_show(argv: list[str]) -> int:
    """`litgraph show <symbol>` — print the signature + docstring for a
    public name on the `litgraph` package. Helps coding agents browse
    without running help() in a REPL."""
    if not argv:
        print("usage: litgraph show <symbol>", file=sys.stderr)
        return 2
    name = argv[0]
    try:
        import litgraph
    except ImportError as e:
        print(f"litgraph not importable: {e}", file=sys.stderr)
        return 1

    obj: object | None = getattr(litgraph, name, None)
    found_at = f"litgraph.{name}"
    if obj is None and "." in name:
        # Try dotted path: `submodule.symbol`.
        head, tail = name.split(".", 1)
        sub = getattr(litgraph, head, None)
        if sub is not None:
            obj = getattr(sub, tail, None)
            if obj is not None:
                found_at = f"litgraph.{name}"
    if obj is None:
        # Fallback: search through known submodules for a matching name.
        import types
        for attr in dir(litgraph):
            sub = getattr(litgraph, attr, None)
            if isinstance(sub, types.ModuleType):
                cand = getattr(sub, name, None)
                if cand is not None:
                    obj = cand
                    found_at = f"litgraph.{attr}.{name}"
                    break
    if obj is None:
        print(f"litgraph: no public symbol named {name!r}", file=sys.stderr)
        print("Try `litgraph examples` for a list of patterns.", file=sys.stderr)
        return 1

    import inspect
    print(found_at)
    try:
        sig = str(inspect.signature(obj))  # type: ignore[arg-type]
        print(f"  signature: {name}{sig}")
    except (ValueError, TypeError):
        print(f"  type:      {type(obj).__name__}")
    doc = inspect.getdoc(obj) or "(no docstring)"
    print()
    for line in doc.splitlines():
        print(f"  {line}")
    return 0


def _cmd_init(argv: list[str]) -> int:
    from ._init import main as init_main
    return init_main(argv)


_DISPATCH = {
    "doctor": _cmd_doctor,
    "version": _cmd_version,
    "examples": _cmd_examples,
    "show": _cmd_show,
    "init": _cmd_init,
    "help": _cmd_help,
    "-h": _cmd_help,
    "--help": _cmd_help,
}


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(_USAGE)
        return 0
    cmd, rest = argv[0], argv[1:]
    handler = _DISPATCH.get(cmd)
    if handler is None:
        print(f"litgraph: unknown command {cmd!r}", file=sys.stderr)
        print(_USAGE, file=sys.stderr)
        return 2
    return handler(rest)


if __name__ == "__main__":
    sys.exit(main())

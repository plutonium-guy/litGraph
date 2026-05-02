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
  doctor       Diagnose env (native module, API keys, Python build).
               --json prints a structured report.
  version      Print the installed litGraph version.
  help         Show this message.

Examples:
  litgraph doctor
  litgraph doctor --json
  litgraph version
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


_DISPATCH = {
    "doctor": _cmd_doctor,
    "version": _cmd_version,
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

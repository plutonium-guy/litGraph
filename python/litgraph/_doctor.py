"""`litgraph doctor` — env diagnostic CLI.

Run as:
    python -m litgraph.doctor [--json]

Exit code 0 on a clean bill of health, 1 if any check failed. Designed
for AI coding agents: `--json` emits a structured report
agents can grep / parse.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, asdict
from typing import Any


__all__ = ["main", "Check", "run_checks"]


@dataclass
class Check:
    name: str
    ok: bool
    detail: str


def _native_module() -> Check:
    try:
        import litgraph
        native = getattr(litgraph, "litgraph", None)
        if native is None:
            return Check("native_module", False, "litgraph.litgraph not built — run `maturin develop --release`")
        version = getattr(litgraph, "__version__", "<unknown>")
        return Check("native_module", True, f"version {version}")
    except ImportError as e:
        return Check("native_module", False, f"import failed: {e}")


def _api_keys() -> list[Check]:
    expected = {
        "OPENAI_API_KEY": "OpenAI",
        "ANTHROPIC_API_KEY": "Anthropic",
        "GOOGLE_API_KEY": "Gemini (AI Studio)",
        "COHERE_API_KEY": "Cohere",
        "VOYAGE_API_KEY": "Voyage",
        "JINA_API_KEY": "Jina",
        "TAVILY_API_KEY": "Tavily search",
    }
    out: list[Check] = []
    for env, label in expected.items():
        present = bool(os.environ.get(env))
        # API keys are advisory — missing one isn't a failure unless
        # the user actually tried to use that provider. Mark `ok=True`
        # when missing (informational only) so the overall exit stays 0.
        out.append(
            Check(
                name=f"api_key:{env}",
                ok=True,
                detail=f"{label}: {'set' if present else 'not set'}",
            )
        )
    return out


def _python_build() -> Check:
    free_threaded = (
        getattr(sys, "_is_gil_enabled", lambda: True)() is False
    )
    return Check(
        name="python_build",
        ok=True,
        detail=f"{sys.version.split()[0]} ({'free-threaded' if free_threaded else 'GIL'})",
    )


def _stub_drift() -> Check:
    """Best-effort stub-drift check. Skipped if the package isn't a
    source checkout (no `tools/check_stubs.py`)."""
    try:
        import litgraph
        pkg_dir = os.path.dirname(os.path.abspath(litgraph.__file__))
        # Walk up looking for tools/check_stubs.py — only present in
        # source checkouts.
        cur = pkg_dir
        for _ in range(5):
            cur = os.path.dirname(cur)
            tool = os.path.join(cur, "tools", "check_stubs.py")
            if os.path.isfile(tool):
                return Check("stub_drift", True, f"checker available at {tool}")
        return Check("stub_drift", True, "skipped (not a source checkout)")
    except Exception as e:
        return Check("stub_drift", True, f"skipped ({e})")


def run_checks() -> list[Check]:
    checks: list[Check] = []
    checks.append(_python_build())
    checks.append(_native_module())
    checks.extend(_api_keys())
    checks.append(_stub_drift())
    return checks


def _format_human(checks: list[Check]) -> str:
    lines = ["litgraph doctor:"]
    width = max(len(c.name) for c in checks) + 2
    for c in checks:
        mark = "✓" if c.ok else "✗"
        lines.append(f"  {mark} {c.name:<{width}} {c.detail}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    as_json = "--json" in argv
    checks = run_checks()
    if as_json:
        payload: dict[str, Any] = {
            "ok": all(c.ok for c in checks),
            "checks": [asdict(c) for c in checks],
        }
        print(json.dumps(payload, indent=2))
    else:
        print(_format_human(checks))
    return 0 if all(c.ok for c in checks) else 1


if __name__ == "__main__":
    sys.exit(main())

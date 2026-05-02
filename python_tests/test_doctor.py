"""Tests for `litgraph.doctor` — env diagnostic CLI.

Locks the JSON shape so AI coding agents that parse `--json` output
don't break on minor refactors.
"""
from __future__ import annotations

import json
import sys

from litgraph._doctor import Check, run_checks, main as doctor_main
from litgraph._cli import main as cli_main


def test_doctor_returns_list_of_checks():
    checks = run_checks()
    assert all(isinstance(c, Check) for c in checks)
    names = {c.name for c in checks}
    assert "python_build" in names
    assert "native_module" in names
    assert any(n.startswith("api_key:") for n in names)


def test_doctor_native_module_check_passes_in_built_env():
    checks = run_checks()
    by_name = {c.name: c for c in checks}
    # Tests run against a `maturin develop`-built native, so this
    # must pass; if it doesn't, the user's env is the bug, not the code.
    assert by_name["native_module"].ok


def test_doctor_human_output_contains_marks(capsys):
    rc = doctor_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "litgraph doctor:" in out
    assert "✓" in out
    assert "native_module" in out


def test_doctor_json_output_parses(capsys):
    rc = doctor_main(["--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "ok" in payload
    assert "checks" in payload
    assert payload["ok"] is True
    by_name = {c["name"]: c for c in payload["checks"]}
    assert by_name["native_module"]["ok"] is True


def test_cli_version_prints_installed_version(capsys):
    rc = cli_main(["version"])
    out = capsys.readouterr().out.strip()
    assert rc == 0
    import litgraph
    assert out == litgraph.__version__


def test_cli_no_args_prints_usage(capsys):
    rc = cli_main([])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage:" in out
    assert "doctor" in out


def test_cli_unknown_command_returns_2(capsys):
    rc = cli_main(["nope"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown command" in err


def test_cli_dispatches_doctor(capsys):
    rc = cli_main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "litgraph doctor:" in out


def test_cli_dispatches_doctor_json(capsys):
    rc = cli_main(["doctor", "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert "checks" in payload


def test_cli_help_flag(capsys):
    rc = cli_main(["--help"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage:" in out

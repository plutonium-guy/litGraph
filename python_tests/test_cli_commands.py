"""Tests for `litgraph examples`, `litgraph show`, and `litgraph init`."""
from __future__ import annotations

from litgraph._cli import main as cli_main


def test_examples_command_prints_index(capsys):
    rc = cli_main(["examples"])
    out = capsys.readouterr().out
    # Either lists files (source checkout) or points at the GitHub URL
    # (wheel install). Both are 0-exit.
    assert rc == 0
    assert "examples" in out.lower() or "github.com" in out


def test_show_command_finds_top_level_symbol(capsys):
    rc = cli_main(["show", "entrypoint"])  # part of __all__
    out = capsys.readouterr().out
    err = capsys.readouterr().err
    # entrypoint may be a function — we accept either signature or
    # type output.
    assert rc == 0, err
    assert "entrypoint" in out


def test_show_command_finds_dotted_symbol(capsys):
    rc = cli_main(["show", "recipes.eval"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "litgraph.recipes.eval" in out
    assert "exact_match" in out  # docstring mentions default scorers


def test_show_command_falls_back_through_submodules(capsys):
    rc = cli_main(["show", "MockChatModel"])  # lives in litgraph.testing
    out = capsys.readouterr().out
    assert rc == 0
    assert "litgraph.testing.MockChatModel" in out


def test_show_command_unknown_returns_1(capsys):
    rc = cli_main(["show", "definitely_not_a_real_symbol_xyz"])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no public symbol" in err


def test_show_command_no_arg_returns_2(capsys):
    rc = cli_main(["show"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "usage:" in err.lower()


def test_init_command_help(capsys):
    rc = cli_main(["init"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "templates:" in out
    assert "chat-agent" in out


def test_init_command_creates_files(tmp_path, capsys):
    target = tmp_path / "demo"
    rc = cli_main(["init", "chat-agent", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "wrote" in out
    # Required files all present.
    assert (target / "pyproject.toml").is_file()
    assert (target / "README.md").is_file()
    assert (target / "AGENTS.md").is_file()
    assert (target / ".env.example").is_file()
    assert (target / "tests" / "test_agent.py").is_file()
    # Slug-named package created (filter out the tests/ pkg).
    pkg = [p for p in target.glob("*/__init__.py") if p.parent.name != "tests"]
    assert len(pkg) == 1
    pkg_dir = pkg[0].parent
    assert (pkg_dir / "__main__.py").is_file()


def test_init_command_unknown_template(tmp_path, capsys):
    rc = cli_main(["init", "no-such-template", str(tmp_path / "x")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "unknown template" in err


def test_init_command_refuses_non_empty_target(tmp_path, capsys):
    target = tmp_path / "exists"
    target.mkdir()
    (target / "marker").write_text("hi")
    rc = cli_main(["init", "chat-agent", str(target)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "not empty" in err


def test_init_scaffold_pyproject_uses_slug(tmp_path):
    target = tmp_path / "My-Cool-App"
    cli_main(["init", "chat-agent", str(target)])
    pyproject = (target / "pyproject.toml").read_text()
    assert "my_cool_app" in pyproject  # slug derived from dir name
    assert "litgraph" in pyproject


def test_init_scaffold_test_file_imports_from_slug(tmp_path):
    target = tmp_path / "alpha"
    cli_main(["init", "chat-agent", str(target)])
    test_src = (target / "tests" / "test_agent.py").read_text()
    assert "from alpha import" in test_src

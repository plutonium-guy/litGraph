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


# ---- new templates ----


def test_init_rag_template_scaffolds(tmp_path):
    target = tmp_path / "rag-app"
    rc = cli_main(["init", "rag", str(target)])
    assert rc == 0
    pkg = list((target).glob("*/__init__.py"))
    pkg = [p for p in pkg if p.parent.name != "tests"]
    assert len(pkg) == 1
    src = pkg[0].read_text()
    assert "build_agent" in src
    assert "SAMPLE_DOCS" in src


def test_init_eval_suite_template_scaffolds(tmp_path):
    target = tmp_path / "eval-suite-app"
    rc = cli_main(["init", "eval-suite", str(target)])
    assert rc == 0
    pkg = [p for p in target.glob("*/__init__.py") if p.parent.name != "tests"]
    assert len(pkg) == 1
    src = pkg[0].read_text()
    assert "run_eval" in src
    assert "SAMPLE_CASES" in src


def test_init_help_lists_all_three_templates(capsys):
    rc = cli_main(["init"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "chat-agent" in out
    assert "rag" in out
    assert "eval-suite" in out


# ---- add-tool ----


def test_add_tool_writes_stub_and_test(tmp_path, capsys):
    # Scaffold a project first.
    target = tmp_path / "myapp"
    cli_main(["init", "chat-agent", str(target)])
    capsys.readouterr()  # drain
    rc = cli_main(["add-tool", "weather", str(target)])
    out = capsys.readouterr().out
    assert rc == 0
    pkg_dirs = [p for p in target.glob("*/__init__.py") if p.parent.name != "tests"]
    slug = pkg_dirs[0].parent.name
    tool_path = target / slug / "tools" / "weather.py"
    test_path = target / "tests" / "test_tool_weather.py"
    assert tool_path.is_file()
    assert test_path.is_file()
    assert "def weather(args" in tool_path.read_text()
    assert "make_tool" in tool_path.read_text()
    assert f"from {slug}.tools.weather" in test_path.read_text()


def test_add_tool_rejects_bad_name(tmp_path):
    target = tmp_path / "myapp"
    cli_main(["init", "chat-agent", str(target)])
    rc = cli_main(["add-tool", "BadName", str(target)])
    assert rc == 2


def test_add_tool_no_args_prints_usage(capsys):
    rc = cli_main(["add-tool"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "usage:" in out


def test_add_tool_no_package_returns_1(tmp_path, capsys):
    rc = cli_main(["add-tool", "x", str(tmp_path)])
    err = capsys.readouterr().err
    assert rc == 1
    assert "no Python package" in err


def test_add_tool_refuses_overwrite(tmp_path):
    target = tmp_path / "myapp"
    cli_main(["init", "chat-agent", str(target)])
    cli_main(["add-tool", "x", str(target)])
    rc = cli_main(["add-tool", "x", str(target)])
    assert rc == 1


# ---- add-node ----


def test_add_node_writes_stub_and_test(tmp_path):
    target = tmp_path / "myapp"
    cli_main(["init", "chat-agent", str(target)])
    rc = cli_main(["add-node", "score", str(target)])
    assert rc == 0
    pkg_dirs = [p for p in target.glob("*/__init__.py") if p.parent.name != "tests"]
    slug = pkg_dirs[0].parent.name
    node_path = target / slug / "nodes" / "score.py"
    test_path = target / "tests" / "test_node_score.py"
    assert node_path.is_file()
    assert test_path.is_file()
    assert "def score(state" in node_path.read_text()


def test_add_node_rejects_bad_name(tmp_path):
    target = tmp_path / "myapp"
    cli_main(["init", "chat-agent", str(target)])
    rc = cli_main(["add-node", "Bad-Name", str(target)])
    assert rc == 2

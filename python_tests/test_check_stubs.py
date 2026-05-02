"""Tests for `tools/check_stubs.py` — the stub-drift checker.

Exercises the pure functions (`collect_runtime_attrs`,
`collect_stub_attrs`) against synthetic fixtures so the test
runs without a built native module.
"""

import sys
import textwrap
from pathlib import Path
from types import ModuleType

sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))

from check_stubs import collect_runtime_attrs, collect_stub_attrs  # noqa: E402


# ─── collect_runtime_attrs ─────────────────────────────────────────


def _make_module():
    """Build a fake module with the shape of a PyO3 native module:
    top-level fn, top-level class with methods, dunder attrs that
    must be filtered out."""
    m = ModuleType("fake_native")
    m.__doc__ = "fake module"

    def public_fn():
        pass

    def _private_fn():
        pass

    class Foo:
        def method_a(self):
            pass

        def method_b(self):
            pass

        def _private_method(self):
            pass

    m.public_fn = public_fn
    m._private_fn = _private_fn
    m.Foo = Foo
    return m


def test_runtime_attrs_walk_top_level_fns():
    m = _make_module()
    attrs = collect_runtime_attrs(m)
    assert "public_fn" in attrs


def test_runtime_attrs_skip_private_top_level():
    m = _make_module()
    attrs = collect_runtime_attrs(m)
    assert "_private_fn" not in attrs


def test_runtime_attrs_walk_class_methods():
    m = _make_module()
    attrs = collect_runtime_attrs(m)
    assert "Foo" in attrs
    assert "Foo.method_a" in attrs
    assert "Foo.method_b" in attrs


def test_runtime_attrs_skip_private_methods():
    m = _make_module()
    attrs = collect_runtime_attrs(m)
    assert "Foo._private_method" not in attrs


def test_runtime_attrs_skip_dunder():
    m = _make_module()
    attrs = collect_runtime_attrs(m)
    # __doc__, __name__, etc. must NOT appear.
    assert not any(a.startswith("__") for a in attrs)


# ─── collect_stub_attrs ────────────────────────────────────────────


def test_stub_attrs_top_level_function(tmp_path):
    stub = tmp_path / "mod.pyi"
    stub.write_text(
        textwrap.dedent(
            """
            def public_fn() -> int: ...
            def _private() -> int: ...
            """
        ).strip()
    )
    attrs = collect_stub_attrs(tmp_path)
    # Both names found — collect_stub_attrs doesn't filter privates;
    # filtering happens at the runtime side. Stubs may declare
    # private things explicitly.
    assert "public_fn" in attrs
    assert "_private" in attrs


def test_stub_attrs_class_methods(tmp_path):
    stub = tmp_path / "mod.pyi"
    stub.write_text(
        textwrap.dedent(
            """
            class Foo:
                def method_a(self) -> int: ...
                def method_b(self) -> str: ...
            """
        ).strip()
    )
    attrs = collect_stub_attrs(tmp_path)
    assert "Foo" in attrs
    assert "Foo.method_a" in attrs
    assert "Foo.method_b" in attrs


def test_stub_attrs_async_function(tmp_path):
    stub = tmp_path / "mod.pyi"
    stub.write_text("async def fetcher() -> str: ...")
    attrs = collect_stub_attrs(tmp_path)
    assert "fetcher" in attrs


def test_stub_attrs_module_level_constant(tmp_path):
    stub = tmp_path / "mod.pyi"
    stub.write_text(
        textwrap.dedent(
            """
            VERSION: str
            DEFAULT_K: int = 10
            """
        ).strip()
    )
    attrs = collect_stub_attrs(tmp_path)
    assert "VERSION" in attrs
    assert "DEFAULT_K" in attrs


def test_stub_attrs_multiple_files_merged(tmp_path):
    (tmp_path / "a.pyi").write_text("def fn_a() -> int: ...")
    (tmp_path / "b.pyi").write_text("def fn_b() -> str: ...")
    attrs = collect_stub_attrs(tmp_path)
    assert "fn_a" in attrs
    assert "fn_b" in attrs


def test_stub_attrs_malformed_file_doesnt_crash(tmp_path):
    (tmp_path / "good.pyi").write_text("def fn_good() -> int: ...")
    (tmp_path / "bad.pyi").write_text("def fn_bad(  # syntax error")
    # Should still surface the good file's attrs without crashing
    # on the bad file.
    attrs = collect_stub_attrs(tmp_path)
    assert "fn_good" in attrs


def test_stub_attrs_empty_dir(tmp_path):
    attrs = collect_stub_attrs(tmp_path)
    assert attrs == set()


def test_stub_attrs_nonexistent_dir(tmp_path):
    attrs = collect_stub_attrs(tmp_path / "nonexistent")
    assert attrs == set()


# ─── End-to-end drift detection ────────────────────────────────────


def test_drift_detection_finds_missing_in_stubs(tmp_path):
    # Runtime has `new_function` that the stubs don't mention.
    m = ModuleType("fake")

    def existing_function():
        pass

    def new_function():
        pass

    m.existing_function = existing_function
    m.new_function = new_function
    runtime = collect_runtime_attrs(m)

    stub_dir = tmp_path
    (stub_dir / "mod.pyi").write_text("def existing_function() -> int: ...")
    stubs = collect_stub_attrs(stub_dir)

    # Drift: new_function is in runtime but not in stubs.
    missing_in_stubs = runtime - stubs
    assert "new_function" in missing_in_stubs


def test_drift_detection_finds_missing_in_runtime(tmp_path):
    # Stubs declare `removed_function` that's no longer in runtime.
    m = ModuleType("fake")

    def existing_function():
        pass

    m.existing_function = existing_function
    runtime = collect_runtime_attrs(m)

    stub_dir = tmp_path
    (stub_dir / "mod.pyi").write_text(
        textwrap.dedent(
            """
            def existing_function() -> int: ...
            def removed_function() -> str: ...
            """
        ).strip()
    )
    stubs = collect_stub_attrs(stub_dir)

    missing_in_runtime = stubs - runtime
    assert "removed_function" in missing_in_runtime


def test_in_sync_case_no_drift(tmp_path):
    m = ModuleType("fake")

    def fn():
        pass

    class Foo:
        def method(self):
            pass

    m.fn = fn
    m.Foo = Foo
    runtime = collect_runtime_attrs(m)

    stub_dir = tmp_path
    (stub_dir / "mod.pyi").write_text(
        textwrap.dedent(
            """
            def fn() -> int: ...
            class Foo:
                def method(self) -> int: ...
            """
        ).strip()
    )
    stubs = collect_stub_attrs(stub_dir)

    # Runtime sees only what the fake module declares; stubs additionally
    # carry the file's basename ("mod") to mark the submodule itself as
    # covered. So runtime must be a subset of stubs.
    assert runtime <= stubs
    assert stubs - runtime == {"mod"}

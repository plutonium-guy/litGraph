"""VirtualFilesystemTool — sandboxed in-memory FS for agent scratch space."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.tools import VirtualFilesystemTool  # noqa: E402


def test_starts_empty():
    vfs = VirtualFilesystemTool()
    assert vfs.snapshot() == {}
    assert vfs.total_bytes() == 0


def test_repr_includes_counts():
    vfs = VirtualFilesystemTool()
    r = repr(vfs)
    assert "VirtualFilesystemTool" in r
    assert "files=0" in r


def test_max_total_bytes_construction_accepted():
    vfs = VirtualFilesystemTool(max_total_bytes=1_000)
    assert vfs.total_bytes() == 0


def test_zero_bytes_means_unlimited():
    # Default is 0 (unlimited).
    VirtualFilesystemTool()


def test_name_is_vfs():
    assert VirtualFilesystemTool().name == "vfs"


def test_drops_into_react_agent():
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    ReactAgent(base, [VirtualFilesystemTool()], max_iterations=1)


def test_two_instances_isolated():
    a = VirtualFilesystemTool()
    b = VirtualFilesystemTool()
    assert a.snapshot() == {}
    assert b.snapshot() == {}
    assert id(a) != id(b)

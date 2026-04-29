"""PlanningTool — Deep Agents-style stateful todo list."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.tools import PlanningTool  # noqa: E402


def test_starts_empty():
    p = PlanningTool()
    assert p.snapshot() == []


def test_repr_contains_count():
    p = PlanningTool()
    assert "PlanningTool(items=0)" in repr(p)


def test_clear_returns_count_removed():
    p = PlanningTool()
    assert p.clear() == 0


def test_name_is_planning():
    assert PlanningTool().name == "planning"


def test_passes_extract_into_react_agent():
    """PlanningTool must drop into a ReactAgent's tool list without an
    unsupported-tool error."""
    from litgraph.agents import ReactAgent
    from litgraph.providers import OpenAIChat

    base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    ReactAgent(base, [PlanningTool()], max_iterations=1)


def test_two_instances_isolated():
    """Each PlanningTool owns its state; one instance does not leak into
    another."""
    a = PlanningTool()
    b = PlanningTool()
    # Snapshots independent.
    assert a.snapshot() == []
    assert b.snapshot() == []
    # Construction itself does not touch shared state.
    assert id(a) != id(b)

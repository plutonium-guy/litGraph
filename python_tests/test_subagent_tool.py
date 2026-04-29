"""Subagent tool — wraps a ReactAgent so a parent agent can spawn it."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.agents import ReactAgent  # noqa: E402
from litgraph.providers import OpenAIChat  # noqa: E402
from litgraph.tools import CalculatorTool, SubagentTool  # noqa: E402


def _agent():
    base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    return ReactAgent(base, [CalculatorTool()], max_iterations=2)


def test_construct_with_react_agent():
    a = _agent()
    s = SubagentTool("research", "Does research", a)
    assert s.name == "research"


def test_repr_includes_name():
    a = _agent()
    s = SubagentTool("worker", "desc", a)
    assert "worker" in repr(s)


def test_drops_into_parent_react_agent():
    """Parent agent must accept a SubagentTool as a tool — that's the whole
    point of dynamic subagent spawn."""
    inner = _agent()
    sub = SubagentTool("sub", "delegate to me", inner)
    parent_base = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    ReactAgent(parent_base, [sub, CalculatorTool()], max_iterations=2)


def test_two_subagents_with_distinct_names():
    a1 = _agent()
    a2 = _agent()
    s1 = SubagentTool("researcher", "researches", a1)
    s2 = SubagentTool("calculator_agent", "does math", a2)
    assert s1.name != s2.name
    parent = OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")
    ReactAgent(parent, [s1, s2], max_iterations=1)

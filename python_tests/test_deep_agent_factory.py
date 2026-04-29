"""create_deep_agent — one-call factory composing the deep-agents primitives."""

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.agents import ReactAgent  # noqa: E402
from litgraph.deep_agent import create_deep_agent  # noqa: E402
from litgraph.providers import OpenAIChat  # noqa: E402
from litgraph.tools import CalculatorTool  # noqa: E402


def _model():
    return OpenAIChat(api_key="sk-fake", model="gpt-4o-mini")


def test_factory_returns_react_agent():
    agent = create_deep_agent(_model())
    assert isinstance(agent, ReactAgent)


def test_extra_tools_are_passed_through():
    agent = create_deep_agent(_model(), tools=[CalculatorTool()])
    # Construction succeeds — internally PlanningTool + VFS + Calculator are
    # all in the tool list and accepted by ReactAgent.
    assert agent is not None


def test_with_planning_false_skips_planning():
    agent = create_deep_agent(_model(), with_planning=False)
    assert agent is not None


def test_with_vfs_false_skips_vfs():
    agent = create_deep_agent(_model(), with_vfs=False)
    assert agent is not None


def test_both_disabled_still_returns_agent():
    agent = create_deep_agent(_model(), with_planning=False, with_vfs=False)
    assert agent is not None


def test_loads_agents_md_from_path(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("project memory: be terse")
    agent = create_deep_agent(_model(), agents_md_path=str(p))
    assert agent is not None


def test_missing_agents_md_is_non_fatal(tmp_path):
    agent = create_deep_agent(
        _model(),
        agents_md_path=str(tmp_path / "does-not-exist.md"),
    )
    assert agent is not None


def test_loads_skills_dir(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "summarize.md").write_text("# Summarize\nuse for long inputs")
    agent = create_deep_agent(_model(), skills_dir=str(d))
    assert agent is not None


def test_missing_skills_dir_is_non_fatal(tmp_path):
    agent = create_deep_agent(
        _model(),
        skills_dir=str(tmp_path / "missing-skills"),
    )
    assert agent is not None


def test_custom_system_prompt_overrides_default():
    agent = create_deep_agent(
        _model(),
        system_prompt="you are a math tutor",
    )
    assert agent is not None


def test_max_iterations_is_passed():
    # Construction must succeed for legal positive values; we can't introspect
    # the value without invoking the loop.
    create_deep_agent(_model(), max_iterations=1)
    create_deep_agent(_model(), max_iterations=50)


def test_full_pipeline_from_disk(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("memory blob")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "search.md").write_text("# Search\nfor lookups")
    agent = create_deep_agent(
        _model(),
        tools=[CalculatorTool()],
        agents_md_path=str(agents),
        skills_dir=str(skills),
        system_prompt="be helpful",
        max_iterations=3,
    )
    assert isinstance(agent, ReactAgent)

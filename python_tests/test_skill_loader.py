"""AGENTS.md / skills loader + SystemPromptBuilder."""

import textwrap

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.prompts import (  # noqa: E402
    Skill,
    SystemPromptBuilder,
    load_agents_md,
    load_skills_dir,
)


def test_load_agents_md_returns_none_for_missing(tmp_path):
    assert load_agents_md(str(tmp_path / "AGENTS.md")) is None


def test_load_agents_md_reads_file(tmp_path):
    p = tmp_path / "AGENTS.md"
    p.write_text("be terse")
    assert load_agents_md(str(p)) == "be terse"


def test_load_skills_dir_empty_for_missing(tmp_path):
    assert load_skills_dir(str(tmp_path / "skills")) == []


def test_load_skills_dir_reads_md_files_sorted(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "b.md").write_text("# Beta\nbody b")
    (d / "a.md").write_text("# Alpha\nbody a")
    (d / "ignored.txt").write_text("nope")
    skills = load_skills_dir(str(d))
    assert [s.name for s in skills] == ["a", "b"]
    assert skills[0].description == "Alpha"
    assert "body a" in skills[0].content


def test_skill_frontmatter_overrides_filename(tmp_path):
    d = tmp_path / "skills"
    d.mkdir()
    (d / "x.md").write_text(
        textwrap.dedent(
            """\
            ---
            name: Custom Name
            description: Custom desc
            ---
            # Heading
            actual content
            """
        )
    )
    [s] = load_skills_dir(str(d))
    assert s.name == "Custom Name"
    assert s.description == "Custom desc"
    assert "actual content" in s.content


def test_skill_repr_includes_name():
    s = Skill("foo", "desc", "body")
    r = repr(s)
    assert "foo" in r and "desc" in r


def test_builder_emits_sections_in_order():
    b = SystemPromptBuilder("base prompt")
    b.with_agents_md("memory body")
    b.with_skill(Skill("s1", "desc 1", "skill body 1"))
    out = b.build()
    assert out.index("base prompt") < out.index("## Memory (AGENTS.md)")
    assert out.index("## Memory (AGENTS.md)") < out.index("## Skills")
    assert "### s1" in out
    assert "skill body 1" in out


def test_builder_skips_empty_sections():
    out = SystemPromptBuilder("only base").build()
    assert "## Memory" not in out
    assert "## Skills" not in out


def test_builder_with_skills_list():
    b = SystemPromptBuilder("base")
    b.with_skills([Skill("a", "da", "ca"), Skill("b", "db", "cb")])
    out = b.build()
    assert "### a" in out
    assert "### b" in out


def test_builder_extra_section():
    b = SystemPromptBuilder("base")
    b.with_section("Notes", "remember this")
    out = b.build()
    assert "## Notes" in out
    assert "remember this" in out


def test_full_pipeline_from_disk(tmp_path):
    agents = tmp_path / "AGENTS.md"
    agents.write_text("project memory line")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "summarizer.md").write_text("# Summarize\nlong stuff into short")

    b = SystemPromptBuilder("you are an assistant")
    body = load_agents_md(str(agents))
    if body:
        b.with_agents_md(body)
    b.with_skills(load_skills_dir(str(skills)))
    out = b.build()
    assert "you are an assistant" in out
    assert "project memory line" in out
    assert "### summarizer" in out
    assert "long stuff into short" in out


def test_build_is_deterministic():
    def mk():
        b = SystemPromptBuilder("base")
        b.with_agents_md("m")
        b.with_skill(Skill("n", "d", "c"))
        return b.build()

    assert mk() == mk()

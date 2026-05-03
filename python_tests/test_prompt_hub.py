"""Tests for litgraph.prompt_hub — local prompt registry."""
from __future__ import annotations

import pytest

from litgraph.prompt_hub import (
    register,
    get,
    search,
    list_prompts,
    clear,
    Prompt,
    HUB_URL,
)


@pytest.fixture(autouse=True)
def _isolated_registry():
    clear()
    yield
    clear()


def test_register_and_get_round_trip():
    register("a/b", "hello {name}")
    p = get("a/b")
    assert p.name == "a/b"
    assert p.template == "hello {name}"


def test_render_substitutes_vars():
    register("greet", "hello {name}")
    assert get("greet").render(name="world") == "hello world"


def test_register_refuses_overwrite_by_default():
    register("dup", "v1")
    with pytest.raises(ValueError, match="already registered"):
        register("dup", "v2")


def test_register_with_overwrite_replaces():
    register("dup", "v1")
    register("dup", "v2", overwrite=True)
    assert get("dup").template == "v2"


def test_get_missing_raises_keyerror():
    with pytest.raises(KeyError):
        get("nope")


def test_search_matches_name_substring():
    register("rag/sql", "...", tags=["rag"])
    register("rag/web", "...", tags=["rag"])
    register("agent/react", "...", tags=["agent"])
    matches = {p.name for p in search("rag")}
    assert matches == {"rag/sql", "rag/web"}


def test_search_matches_tag():
    register("misc", "...", tags=["sql", "rag"])
    matches = [p.name for p in search("sql")]
    assert "misc" in matches


def test_search_matches_description():
    register("x", "...", description="involves SQL queries")
    matches = [p.name for p in search("sql")]
    assert "x" in matches


def test_list_prompts_sorted():
    register("z", "...")
    register("a", "...")
    register("m", "...")
    names = [p.name for p in list_prompts()]
    assert names == ["a", "m", "z"]


def test_clear_empties_registry():
    register("foo", "...")
    clear()
    assert list_prompts() == []


def test_prompt_is_immutable():
    register("x", "...", tags=["t"])
    p = get("x")
    with pytest.raises((AttributeError, TypeError)):
        p.template = "mutated"  # type: ignore[misc]


def test_hub_url_points_at_github():
    assert "github.com" in HUB_URL


def test_prompt_dataclass_accepts_default_factory():
    """Direct construction works with default tags."""
    p = Prompt(name="x", template="t")
    assert p.tags == ()
    assert p.version == "1"

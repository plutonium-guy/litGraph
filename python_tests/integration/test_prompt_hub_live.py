"""Live integration: `prompt_hub` register/get/render → invoke flow.

Wire a registered minijinja prompt through DeepSeek end-to-end:
register → fetch → render with vars → send to model.
"""
from __future__ import annotations

import uuid

import pytest


pytestmark = pytest.mark.integration


def test_prompt_hub_render_then_invoke(deepseek_chat):
    """`Prompt.render(**vars)` uses Python `str.format()` (single-brace
    `{var}`), NOT minijinja. For full Jinja semantics use
    `litgraph.prompts.ChatPromptTemplate` instead."""
    from litgraph import prompt_hub

    name = f"capital_q_{uuid.uuid4().hex[:8]}"
    p = prompt_hub.register(
        name,
        template="What is the capital of {country}? Reply with one word.",
        tags=["geography"],
        version="1",
        description="Single-shot capital lookup.",
    )
    assert p.name == name

    fetched = prompt_hub.get(name)
    assert fetched.template == p.template
    assert fetched.version == "1"

    rendered = fetched.render(country="France")
    assert "France" in rendered
    assert "{" not in rendered, f"unrendered braces: {rendered!r}"

    out = deepseek_chat.invoke(
        [{"role": "user", "content": rendered}],
        max_tokens=10,
    )
    assert "Paris" in out["text"]


def test_prompt_hub_search_substring_match(deepseek_chat):
    """`search(query)` is a positional substring/tag matcher; case-insensitive."""
    from litgraph import prompt_hub

    name = f"taggy_{uuid.uuid4().hex[:8]}"
    prompt_hub.register(name, template="x", tags=["smoke-test"], version="1")
    hits = prompt_hub.search("smoke-test")
    names = [h.name for h in hits]
    assert name in names

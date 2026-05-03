"""Live integration: more parsers exercised on real DeepSeek output.

- `parse_react_step` — coax a ReAct-format reply, parse it
- `parse_xml_tags` — `<answer>...</answer>` extraction
- `parse_markdown_list` / `parse_comma_list` / `parse_boolean`
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_parse_react_step_on_modeled_reply(deepseek_chat):
    """Make the model emit ReAct format and parse it."""
    from litgraph.parsers import parse_react_step

    out = deepseek_chat.invoke(
        [
            {
                "role": "system",
                "content": (
                    "Reply in ReAct format. Output exactly:\n"
                    "Thought: <one sentence>\n"
                    "Action: get_weather\n"
                    "Action Input: {\"city\": \"Paris\"}"
                ),
            },
            {"role": "user", "content": "What's the weather in Paris?"},
        ],
        max_tokens=80,
    )
    step = parse_react_step(out["text"])
    assert step["kind"] == "action", f"unexpected step: {step!r}"
    assert step["tool"] == "get_weather"


def test_parse_xml_tags_extracts_answer(deepseek_chat):
    from litgraph.parsers import parse_xml_tags

    out = deepseek_chat.invoke(
        [
            {
                "role": "system",
                "content": (
                    "You MUST wrap your answer in XML tags. Output exactly:\n"
                    "<thinking>brief reasoning</thinking>\n"
                    "<answer>integer here</answer>\n"
                    "Do not output anything outside the tags."
                ),
            },
            {"role": "user", "content": "What is 6 * 7?"},
        ],
        max_tokens=80,
    )
    tags = parse_xml_tags(out["text"], ["answer"])
    # Tolerate the case where the model ignores the format; verify only
    # if it cooperated, but always check the parser handles missing
    # tags as an absent key (NOT an error).
    if "answer" in tags:
        assert "42" in tags["answer"]
    else:
        # Parser contract: missing tag → absent key, not an exception.
        assert tags == {} or all(v is not None for v in tags.values())


def test_parse_markdown_list(deepseek_chat):
    from litgraph.parsers import parse_markdown_list

    out = deepseek_chat.invoke(
        [
            {
                "role": "user",
                "content": (
                    "List exactly 3 primary colors as a markdown list "
                    "(`- red`, `- green`, `- blue`). No other text."
                ),
            }
        ],
        max_tokens=30,
    )
    items = parse_markdown_list(out["text"])
    assert isinstance(items, list)
    # Be lenient: model may return more or fewer items.
    assert len(items) >= 2, f"too few list items: {items!r}"


def test_parse_comma_list(deepseek_chat):
    from litgraph.parsers import parse_comma_list

    out = deepseek_chat.invoke(
        [
            {
                "role": "user",
                "content": "Reply with exactly: red, green, blue (comma-separated, no other text).",
            }
        ],
        max_tokens=15,
    )
    items = parse_comma_list(out["text"])
    assert "red" in items or "Red" in items
    assert "blue" in items or "Blue" in items


def test_parse_boolean(deepseek_chat):
    from litgraph.parsers import parse_boolean

    out = deepseek_chat.invoke(
        [
            {
                "role": "user",
                "content": "Is the sky blue? Reply with exactly one word: yes or no.",
            }
        ],
        max_tokens=5,
    )
    val = parse_boolean(out["text"])
    assert val is True, f"parse_boolean got {val!r} for {out['text']!r}"

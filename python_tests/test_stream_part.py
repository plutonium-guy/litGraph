"""Tests for `litgraph.StreamPart` — typed mirror of native
ChatStreamEvent dict events (iter 379). Closes Tier-1 #7 StreamPart
half."""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from litgraph import (  # noqa: E402
    Delta,
    Done,
    ToolCallDelta,
    aparse_stream_parts,
    parse_stream_part,
    parse_stream_parts,
)


# ─── single-event parsing ─────────────────────────────────────────


def test_delta_parses_with_text():
    part = parse_stream_part({"type": "delta", "text": "hello"})
    assert isinstance(part, Delta)
    assert part.text == "hello"
    assert part.type == "delta"


def test_tool_call_delta_parses_with_optional_fields():
    part = parse_stream_part(
        {
            "type": "tool_call_delta",
            "index": 0,
            "id": "call_abc",
            "name": "calculator",
            "arguments_delta": '{"a":',
        }
    )
    assert isinstance(part, ToolCallDelta)
    assert part.index == 0
    assert part.id == "call_abc"
    assert part.name == "calculator"
    assert part.arguments_delta == '{"a":'


def test_tool_call_delta_parses_with_only_index():
    """The provider's first chunk for a tool call typically has just
    `index` + `id`. Later chunks have just `index` + `arguments_delta`.
    All three permutations must parse with `None` for omitted keys."""
    part = parse_stream_part({"type": "tool_call_delta", "index": 1})
    assert isinstance(part, ToolCallDelta)
    assert part.index == 1
    assert part.id is None
    assert part.name is None
    assert part.arguments_delta is None


def test_done_parses_with_usage_dict():
    part = parse_stream_part(
        {
            "type": "done",
            "text": "all set",
            "finish_reason": "stop",
            "model": "claude-x",
            "usage": {"prompt": 10, "completion": 5, "total": 15},
        }
    )
    assert isinstance(part, Done)
    assert part.text == "all set"
    assert part.finish_reason == "stop"
    assert part.model == "claude-x"
    assert part.usage == {"prompt": 10, "completion": 5, "total": 15}


def test_done_defaults_usage_to_empty_dict():
    """Some providers omit usage on `done` — the typed variant must
    not crash; the dataclass default supplies `{}`."""
    part = parse_stream_part(
        {
            "type": "done",
            "text": "x",
            "finish_reason": "stop",
            "model": "m",
        }
    )
    assert isinstance(part, Done)
    assert part.usage == {}


# ─── error surface ────────────────────────────────────────────────


def test_unknown_type_raises_with_known_list():
    with pytest.raises(ValueError, match="unknown stream event type"):
        parse_stream_part({"type": "garbage", "text": "x"})


def test_missing_type_raises():
    with pytest.raises(ValueError, match="unknown stream event type"):
        parse_stream_part({"text": "x"})


def test_non_dict_input_raises():
    with pytest.raises(ValueError, match="expected dict"):
        parse_stream_part("not a dict")  # type: ignore[arg-type]


def test_extra_top_level_keys_raise_typeerror():
    """Provider drift sanity-check: a stray key on a known variant
    raises TypeError (dataclass kwargs validation) so we notice
    silently-broken upstream changes."""
    with pytest.raises(TypeError):
        parse_stream_part(
            {"type": "delta", "text": "x", "experimental_field": True}
        )


# ─── stream adapters ──────────────────────────────────────────────


def test_parse_stream_parts_iterates_in_order():
    events = [
        {"type": "delta", "text": "hi"},
        {"type": "delta", "text": " there"},
        {"type": "done", "text": "hi there", "finish_reason": "stop", "model": "m"},
    ]
    parts = list(parse_stream_parts(events))
    assert len(parts) == 3
    assert all(isinstance(p, Delta) for p in parts[:2])
    assert isinstance(parts[2], Done)
    assert "".join(p.text for p in parts[:2]) == "hi there"


def test_parse_stream_parts_propagates_error_eagerly():
    events = [
        {"type": "delta", "text": "ok"},
        {"type": "unknown"},  # blows up here
        {"type": "delta", "text": "never reached"},
    ]
    it = parse_stream_parts(events)
    assert isinstance(next(it), Delta)
    with pytest.raises(ValueError, match="unknown stream event type"):
        next(it)


def test_aparse_stream_parts_async():
    async def events():
        yield {"type": "delta", "text": "a"}
        yield {"type": "done", "text": "a", "finish_reason": "stop", "model": "m"}

    async def collect():
        return [p async for p in aparse_stream_parts(events())]

    parts = asyncio.run(collect())
    assert len(parts) == 2
    assert isinstance(parts[0], Delta)
    assert isinstance(parts[1], Done)


# ─── pattern-match exhaustiveness (smoke test) ────────────────────


def test_match_pattern_narrows_correctly():
    """Python 3.10+ structural match must narrow each variant."""
    events = [
        {"type": "delta", "text": "hi"},
        {"type": "tool_call_delta", "index": 0, "name": "x"},
        {"type": "done", "text": "hi", "finish_reason": "stop", "model": "m"},
    ]
    seen = []
    for ev in parse_stream_parts(events):
        match ev:
            case Delta(text=t):
                seen.append(("delta", t))
            case ToolCallDelta(index=i, name=n):
                seen.append(("tool", i, n))
            case Done(model=m):
                seen.append(("done", m))
    assert seen == [
        ("delta", "hi"),
        ("tool", 0, "x"),
        ("done", "m"),
    ]


# ─── equality / hashability ───────────────────────────────────────


def test_dataclass_equality_and_hashable():
    """`@dataclass(frozen=True)` gives free __eq__ + __hash__."""
    a = Delta(text="x")
    b = Delta(text="x")
    c = Delta(text="y")
    assert a == b
    assert a != c
    assert hash(a) == hash(b)
    # Equal-by-value → usable as dict keys / set members.
    s = {a, b, c}
    assert len(s) == 2

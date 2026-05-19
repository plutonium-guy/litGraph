"""Typed enum for `ChatModel.stream` events (iter 379).

The native ``ChatModel.stream`` API yields ``dict`` events with a
``type`` discriminator: ``"delta"`` (partial assistant text),
``"tool_call_delta"`` (provider-specific partial tool-call args), and
``"done"`` (final ChatResponse summary). The dict shape is JSON-safe
and wire-stable, but loses static typing — readers can't see which
keys are populated for which variant without consulting the docs or
the Rust source.

This module ships a Python-side typed mirror of those events plus a
discriminator (`parse_stream_part`) that turns one dict into one
dataclass instance. The dataclasses are `frozen=True` so they're
hashable + cheap to copy; their constructors validate that the
``type`` field matches the variant.

Typical use:

    from litgraph import parse_stream_parts

    async for part in parse_stream_parts(model.astream(messages)):
        match part:
            case Delta(text=text):
                ui.append(text)
            case ToolCallDelta(name=name, arguments_delta=args):
                ...
            case Done(usage=usage):
                billing.charge(usage["prompt"], usage["completion"])

Callers who don't want the typed mirror keep iterating the native
``dict`` events — nothing in the framework forces them to wrap.

Why a hand-rolled typed-union instead of a Pydantic model:
the dataclasses path has zero runtime dep (Pydantic is optional in
litGraph), matches the cassette / cache JSON shape exactly, and lets
``match`` narrow correctly without a runtime validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterable, AsyncIterator, Iterable, Iterator, Optional, Union


@dataclass(frozen=True)
class Delta:
    """A token-level chunk of assistant text from the model stream."""
    text: str
    type: str = "delta"


@dataclass(frozen=True)
class ToolCallDelta:
    """A partial tool-call payload — provider-specific.

    ``index`` identifies which parallel tool call this fragment belongs
    to (LLMs may emit multiple tool calls in one turn; deltas
    interleave). ``id`` and ``name`` arrive once in the first chunk for
    that index; ``arguments_delta`` accumulates the JSON args one chunk
    at a time and must be aggregated client-side before parsing.
    """
    index: int
    id: Optional[str] = None
    name: Optional[str] = None
    arguments_delta: Optional[str] = None
    type: str = "tool_call_delta"


@dataclass(frozen=True)
class Done:
    """Terminal event with the assembled final response.

    `usage` is a dict with `prompt` / `completion` / `total` token
    counts. `finish_reason` is the provider's terminal reason in
    snake_case (`stop`, `length`, `tool_calls`, etc).
    """
    text: str
    finish_reason: str
    model: str
    usage: dict = field(default_factory=dict)
    type: str = "done"


# Tagged-union type alias. The `type` field on each variant is the
# discriminator; `parse_stream_part` uses it for dispatch.
StreamPart = Union[Delta, ToolCallDelta, Done]


_VARIANT_BY_TYPE: dict[str, Any] = {
    "delta": Delta,
    "tool_call_delta": ToolCallDelta,
    "done": Done,
}


def parse_stream_part(event: dict) -> StreamPart:
    """Convert one native stream-event dict into its typed variant.

    Raises ``ValueError`` if the dict lacks a ``type`` field or its
    value isn't one of the three known variants — surfaces protocol
    drift early instead of letting downstream ``match`` arms silently
    skip unknown events.

    Field bookkeeping: each variant drops ``type`` from the kwargs
    before constructing (the dataclass declares its own constant
    default), and ``Done`` swallows unknown keys via dict-passthrough
    on ``usage`` only — extra top-level keys raise ``TypeError`` so a
    provider that adds a new field on ``done`` is loud, not silent.
    """
    if not isinstance(event, dict):
        raise ValueError(
            f"parse_stream_part: expected dict event, got {type(event).__name__}"
        )
    tag = event.get("type")
    cls = _VARIANT_BY_TYPE.get(tag)
    if cls is None:
        raise ValueError(
            f"parse_stream_part: unknown stream event type {tag!r}. "
            f"Known: {sorted(_VARIANT_BY_TYPE.keys())}"
        )
    payload = {k: v for k, v in event.items() if k != "type"}
    return cls(**payload)


def parse_stream_parts(stream: Iterable[dict]) -> Iterator[StreamPart]:
    """Map a sync iterable of native dict events to typed StreamParts.

    Errors in ``parse_stream_part`` propagate (the stream stops at
    the first malformed event so the caller sees the protocol
    mismatch instead of silently skipping data — same contract as
    ``coerce_stream``).
    """
    for event in stream:
        yield parse_stream_part(event)


async def aparse_stream_parts(
    stream: AsyncIterable[dict],
) -> AsyncIterator[StreamPart]:
    """Async variant of ``parse_stream_parts`` for ``ChatModel.astream``."""
    async for event in stream:
        yield parse_stream_part(event)

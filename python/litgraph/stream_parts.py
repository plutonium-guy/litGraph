"""Typed names for native chat-stream event dictionaries."""
from __future__ import annotations

from enum import Enum
from typing import Any, Mapping


__all__ = ["StreamPart", "stream_part"]


class StreamPart(str, Enum):
    """Stable event kinds emitted by ``ChatModel.stream``."""

    DELTA = "delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    DONE = "done"

    @classmethod
    def from_event(cls, event: Mapping[str, Any]) -> "StreamPart":
        try:
            value = event["type"]
        except KeyError as error:
            raise ValueError("stream event is missing a 'type' field") from error
        try:
            return cls(str(value))
        except ValueError as error:
            raise ValueError(f"unknown chat stream event type: {value!r}") from error


def stream_part(event: Mapping[str, Any]) -> StreamPart:
    """Return the typed :class:`StreamPart` for a native event dict."""
    return StreamPart.from_event(event)

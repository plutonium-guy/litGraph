"""Typed chat stream event names."""
import pytest

from litgraph import StreamPart, stream_part


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({"type": "delta", "text": "hi"}, StreamPart.DELTA),
        ({"type": "tool_call_delta", "index": 0}, StreamPart.TOOL_CALL_DELTA),
        ({"type": "done", "text": "hi"}, StreamPart.DONE),
    ],
)
def test_stream_part_parses_native_event_types(raw, expected):
    assert stream_part(raw) is expected
    assert StreamPart.from_event(raw) is expected


def test_stream_part_is_a_string_enum():
    assert StreamPart.DELTA == "delta"


def test_stream_part_rejects_missing_or_unknown_type():
    with pytest.raises(ValueError, match="missing"):
        stream_part({})
    with pytest.raises(ValueError, match="unknown"):
        stream_part({"type": "graph_start"})

"""Tests for `litgraph trace` (iter 386). Pure-Python OTel JSON
viewer — no native dep, no live OTel collector needed."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from litgraph._trace import render_trace  # noqa: E402


def _sdk_span(name, span_id, parent=None, start_ms=0, dur_ms=10, model=None,
              prompt=None, completion=None, status_code=1):
    base = "2026-05-20T12:01:03"
    return {
        "name": name,
        "context": {"trace_id": "trace-1", "span_id": span_id},
        "parent_id": parent,
        "start_time": f"{base}.{start_ms:03d}000Z" if start_ms else f"{base}.000000Z",
        "end_time": f"{base}.{start_ms + dur_ms:03d}000Z",
        "status": {"code": status_code},
        "attributes": {
            **({"model": model} if model else {}),
            **({"prompt_excerpt": prompt} if prompt else {}),
            **({"completion_excerpt": completion} if completion else {}),
        },
    }


def test_renders_single_span_with_excerpts():
    span = _sdk_span("chat.invoke", "s1", model="gpt-5",
                     prompt="user: hi", completion="hello")
    out = render_trace(json.dumps(span), use_colour=False)
    assert "chat.invoke" in out
    assert "ok" in out
    assert "gpt-5" in out
    assert "user: hi" in out
    assert "hello" in out


def test_nests_children_under_parent_by_id():
    parent = _sdk_span("chat.invoke", "s1", start_ms=0, dur_ms=200)
    child_a = _sdk_span("tool.calc", "s2", parent="s1", start_ms=20, dur_ms=10)
    child_b = _sdk_span("retriever.bm25", "s3", parent="s1", start_ms=50, dur_ms=15)
    grandchild = _sdk_span("vector.search", "s4", parent="s2", start_ms=25, dur_ms=5)
    doc = "\n".join(json.dumps(s) for s in [parent, child_a, grandchild, child_b])
    out = render_trace(doc, use_colour=False)
    lines = out.splitlines()
    parent_line = next(i for i, l in enumerate(lines) if "chat.invoke" in l)
    calc_line = next(i for i, l in enumerate(lines) if "tool.calc" in l)
    bm25_line = next(i for i, l in enumerate(lines) if "retriever.bm25" in l)
    vector_line = next(i for i, l in enumerate(lines) if "vector.search" in l)
    # Children appear after parent, indented (start with at least 2 spaces).
    assert parent_line < calc_line < vector_line
    assert parent_line < bm25_line
    assert lines[calc_line].startswith("  ")
    assert lines[bm25_line].startswith("  ")
    # Grandchild indented one more level than child.
    assert lines[vector_line].startswith("    ")
    # Sibling order by start_ns: calc (20ms) before bm25 (50ms).
    assert calc_line < bm25_line


def test_error_status_marked_red_in_label():
    span = _sdk_span("bad.call", "s1", status_code=2)
    span["attributes"]["error"] = "boom"
    out = render_trace(json.dumps(span), use_colour=False)
    assert "error" in out.lower()
    assert "boom" in out


def test_otlp_envelope_shape_works():
    # OTLP JSON envelope — alternative shape from collectors.
    envelope = {
        "resourceSpans": [
            {
                "scopeSpans": [
                    {
                        "spans": [
                            {
                                "name": "agent.run",
                                "traceId": "abc",
                                "spanId": "deadbeef",
                                "parentSpanId": "0" * 16,
                                "startTimeUnixNano": "1747742463402000000",
                                "endTimeUnixNano": "1747742463526000000",
                                "status": {"code": 1},
                                "attributes": [
                                    {"key": "model", "value": {"stringValue": "gpt-5"}},
                                    {
                                        "key": "prompt_excerpt",
                                        "value": {"stringValue": "user: hello"},
                                    },
                                ],
                            }
                        ]
                    }
                ]
            }
        ]
    }
    out = render_trace(json.dumps(envelope), use_colour=False)
    assert "agent.run" in out
    assert "gpt-5" in out
    assert "user: hello" in out


def test_jsonl_input_with_multiple_traces():
    a = _sdk_span("agent.run", "s1", start_ms=0, dur_ms=100)
    b_dict = _sdk_span("agent.run", "s9", start_ms=0, dur_ms=80)
    b_dict["context"]["trace_id"] = "trace-2"
    doc = json.dumps(a) + "\n" + json.dumps(b_dict)
    out = render_trace(doc, use_colour=False)
    assert "trace trace-1" in out
    assert "trace trace-2" in out


def test_unparseable_input_returns_friendly_message():
    out = render_trace("not json at all", use_colour=False)
    assert "no spans found" in out


def test_empty_input_returns_friendly_message():
    out = render_trace("", use_colour=False)
    assert "no spans found" in out


def test_unset_status_renders_without_crash():
    span = _sdk_span("misc.span", "s1", status_code=0)
    out = render_trace(json.dumps(span), use_colour=False)
    assert "misc.span" in out
    assert "unset" in out


def test_duration_label_in_ms_for_span():
    span = _sdk_span("quick", "s1", start_ms=0, dur_ms=42)
    out = render_trace(json.dumps(span), use_colour=False)
    assert "42.0ms" in out


def test_otlp_attributes_array_normalised():
    """OTLP key/value list must be flattened to dict for excerpts."""
    span = {
        "name": "kv.test",
        "traceId": "t",
        "spanId": "s",
        "startTimeUnixNano": "0",
        "endTimeUnixNano": "1000000",
        "status": {"code": 1},
        "attributes": [
            {"key": "model", "value": {"stringValue": "gpt-x"}},
            {"key": "prompt_excerpt", "value": {"stringValue": "hi"}},
            {"key": "n", "value": {"intValue": 7}},
        ],
    }
    out = render_trace(json.dumps(span), use_colour=False)
    assert "gpt-x" in out
    assert "hi" in out

"""`litgraph trace <file>` — terminal timeline viewer for OTel JSON
dumps (iter 386). Closes the Tier-2 "litgraph trace viewer" CLI gap.

# Why a pure-Python viewer

litGraph's `init_stdout` exporter (and any OTLP collector dump) emits
spans as JSON. To debug a single agent run today, a user has to pipe
that through `jq` and read raw fields. This module turns the same
JSON into a colour-coded indented timeline:

    [12:01:03.402 +0.000] chat.invoke (124ms) gpt-5 ok
                          prompt: "user: what's photosynthesis?"
                          completion: "Photosynthesis is the process …"
      [+0.013s] tool.calculator (8ms) ok
      [+0.041s] retriever.bm25 (15ms) ok
    [+0.156s] chat.invoke (89ms) gpt-5 ok
                          prompt: "tool: …"
                          completion: "In summary, …"

Stdlib-only — no `rich` / `textual` dep. Falls back to plain text
when stdout isn't a TTY.

# Input formats

Accepts either:
- JSONL — one OTel span (or batch envelope) per line
- A single JSON document — either one span, or an OTLP
  `{"resourceSpans": [{"scopeSpans": [{"spans": [...]}]}]}` envelope

Auto-detect by sniffing the first non-whitespace char + key.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


_ANSI = {
    "reset": "\x1b[0m",
    "dim": "\x1b[2m",
    "bold": "\x1b[1m",
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "cyan": "\x1b[36m",
}


@dataclass
class _Span:
    """Span normalized across the exporter shapes we accept."""

    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int
    status: str
    attributes: dict[str, Any] = field(default_factory=dict)
    children: list["_Span"] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1_000_000.0


def _normalize_attribute_value(v: Any) -> Any:
    """OTLP encodes attribute values as `{"stringValue": ...}` or
    `{"intValue": ...}` etc. Pure stdout exporters often dump them as
    plain Python values. Accept either."""
    if isinstance(v, dict):
        for k in ("stringValue", "intValue", "doubleValue", "boolValue"):
            if k in v:
                return v[k]
        if "arrayValue" in v:
            arr = v["arrayValue"].get("values", [])
            return [_normalize_attribute_value(x) for x in arr]
    return v


def _normalize_attributes(attrs: Any) -> dict[str, Any]:
    """OTLP attrs come as `[{"key": ..., "value": {...}}, ...]`.
    Stdout-exporter attrs come as `{name: value}`. Accept both."""
    if isinstance(attrs, dict):
        return {k: _normalize_attribute_value(v) for k, v in attrs.items()}
    if isinstance(attrs, list):
        out = {}
        for entry in attrs:
            if isinstance(entry, dict) and "key" in entry:
                out[entry["key"]] = _normalize_attribute_value(entry.get("value"))
        return out
    return {}


def _parse_time(node: Any) -> int:
    """Extract a nanosecond timestamp from either a string ISO time,
    a numeric nanosecond value, or an OTLP `{"unixNano": ...}`."""
    if isinstance(node, dict) and "unixNano" in node:
        return int(node["unixNano"])
    if isinstance(node, (int, float)):
        return int(node)
    if isinstance(node, str):
        # OTel stdout: "2026-05-20T12:01:03.402Z" or epoch-ns as str.
        if node.isdigit():
            return int(node)
        from datetime import datetime, timezone

            # Permit a trailing Z and fractional seconds; assume UTC.
        s = node.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1_000_000_000)
        except ValueError:
            return 0
    return 0


def _coerce_span(raw: dict[str, Any]) -> _Span | None:
    """Convert one raw span dict into the normalized `_Span`. Returns
    `None` if the shape doesn't look like a span (skip silently —
    OTLP envelopes carry resource / scope wrappers we want to ignore)."""
    name = raw.get("name")
    if not isinstance(name, str):
        return None
    # OTLP names trace/span/parent ids; SDK stdout names them in
    # `context: {trace_id, span_id}` or directly on the span.
    ctx = raw.get("context") or {}
    trace_id = (
        raw.get("trace_id")
        or raw.get("traceId")
        or ctx.get("trace_id")
        or ctx.get("traceId")
        or ""
    )
    span_id = (
        raw.get("span_id")
        or raw.get("spanId")
        or ctx.get("span_id")
        or ctx.get("spanId")
        or ""
    )
    parent_id = raw.get("parent_id") or raw.get("parentSpanId") or None
    if isinstance(parent_id, str) and parent_id in ("", "0" * 16):
        parent_id = None
    start_ns = _parse_time(
        raw.get("start_time")
        or raw.get("startTime")
        or raw.get("startTimeUnixNano")
    )
    end_ns = _parse_time(
        raw.get("end_time") or raw.get("endTime") or raw.get("endTimeUnixNano")
    )
    status_raw = raw.get("status") or {}
    if isinstance(status_raw, dict):
        status_code = status_raw.get("code") or status_raw.get("status_code")
        if isinstance(status_code, int):
            # OTLP: 0=Unset, 1=Ok, 2=Error
            status = {0: "unset", 1: "ok", 2: "error"}.get(status_code, "unset")
        elif isinstance(status_code, str):
            status = status_code.lower().replace("status_code_", "")
        else:
            status = "unset"
    elif isinstance(status_raw, str):
        status = status_raw.lower()
    else:
        status = "unset"
    return _Span(
        name=name,
        trace_id=str(trace_id),
        span_id=str(span_id),
        parent_id=parent_id,
        start_ns=start_ns,
        end_ns=end_ns,
        status=status,
        attributes=_normalize_attributes(raw.get("attributes")),
    )


def _iter_spans_from_root(root: Any) -> Iterable[dict[str, Any]]:
    """Walk an OTLP envelope or a flat span. Yields raw span dicts."""
    if isinstance(root, list):
        for item in root:
            yield from _iter_spans_from_root(item)
        return
    if not isinstance(root, dict):
        return
    # OTLP envelope: resource_spans → scope_spans → spans
    for top_key in ("resourceSpans", "resource_spans"):
        if top_key in root:
            for rs in root[top_key] or []:
                scope_key = (
                    "scopeSpans" if "scopeSpans" in rs else "instrumentationLibrarySpans"
                )
                for ss in rs.get(scope_key) or []:
                    for span in ss.get("spans") or []:
                        yield span
            return
    # Flat span — `name` + timing fields at the top level
    if "name" in root and ("start_time" in root or "startTime" in root or "startTimeUnixNano" in root):
        yield root


def _load_spans(text: str) -> list[_Span]:
    text = text.strip()
    if not text:
        return []
    raw_spans: list[dict[str, Any]] = []
    # Try as a single JSON document first; fall back to JSONL.
    try:
        doc = json.loads(text)
        raw_spans = list(_iter_spans_from_root(doc))
    except json.JSONDecodeError:
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                doc = json.loads(ln)
            except json.JSONDecodeError:
                continue
            raw_spans.extend(_iter_spans_from_root(doc))
    out = []
    for raw in raw_spans:
        s = _coerce_span(raw)
        if s is not None:
            out.append(s)
    return out


def _build_tree(spans: list[_Span]) -> list[_Span]:
    """Wire `children` lists from `parent_id` references. Returns the
    list of root spans (no parent OR parent not in the dump)."""
    by_id = {s.span_id: s for s in spans if s.span_id}
    roots: list[_Span] = []
    for s in spans:
        if s.parent_id and s.parent_id in by_id:
            by_id[s.parent_id].children.append(s)
        else:
            roots.append(s)
    # Sort each level by start_ns for deterministic timeline.
    for s in spans:
        s.children.sort(key=lambda x: x.start_ns)
    roots.sort(key=lambda x: x.start_ns)
    return roots


def _colourise(text: str, colour: str, use_colour: bool) -> str:
    if not use_colour:
        return text
    return f"{_ANSI[colour]}{text}{_ANSI['reset']}"


def _render(span: _Span, depth: int, base_ns: int, use_colour: bool, out: list[str]) -> None:
    indent = "  " * depth
    offset_s = (span.start_ns - base_ns) / 1e9 if base_ns else 0.0
    duration_label = f"{span.duration_ms:.1f}ms"
    status_colour = {
        "ok": "green",
        "error": "red",
        "unset": "dim",
    }.get(span.status, "dim")
    status_label = _colourise(span.status, status_colour, use_colour)
    header = (
        f"{indent}[{offset_s:+0.3f}s] "
        f"{_colourise(span.name, 'cyan', use_colour)} "
        f"({duration_label}) "
        f"{status_label}"
    )
    model = span.attributes.get("model")
    if model:
        header += f" {_colourise(str(model), 'yellow', use_colour)}"
    out.append(header)
    # Surface exemplars + a few other useful attrs without dumping
    # the whole bag. Anything else the user can pipe into `jq`.
    prompt = span.attributes.get("prompt_excerpt") or span.attributes.get("litgraph.prompt_excerpt")
    completion = span.attributes.get("completion_excerpt") or span.attributes.get(
        "litgraph.completion_excerpt"
    )
    if prompt:
        out.append(f"{indent}  {_colourise('prompt:', 'dim', use_colour)} {prompt}")
    if completion:
        out.append(f"{indent}  {_colourise('completion:', 'dim', use_colour)} {completion}")
    error = span.attributes.get("error") or span.attributes.get("exception.message")
    if error:
        out.append(f"{indent}  {_colourise('error:', 'red', use_colour)} {error}")
    for child in span.children:
        _render(child, depth + 1, base_ns, use_colour, out)


def render_trace(text: str, *, use_colour: bool | None = None) -> str:
    """Pure-functional renderer — for tests + library callers. Takes
    the raw file text, returns the rendered timeline string."""
    if use_colour is None:
        use_colour = sys.stdout.isatty()
    spans = _load_spans(text)
    if not spans:
        return "litgraph trace: no spans found in input\n"
    roots = _build_tree(spans)
    nonzero_starts = [s.start_ns for s in spans if s.start_ns]
    base_ns = min(nonzero_starts) if nonzero_starts else 0
    out: list[str] = []
    # Group roots by trace id for legibility when multiple traces are
    # concatenated in the same dump.
    by_trace: dict[str, list[_Span]] = {}
    for r in roots:
        by_trace.setdefault(r.trace_id, []).append(r)
    for trace_id, trace_roots in by_trace.items():
        if trace_id:
            out.append(_colourise(f"trace {trace_id}", "bold", use_colour))
        for r in trace_roots:
            _render(r, 0, base_ns, use_colour, out)
        out.append("")
    return "\n".join(out)


def main(argv: list[str]) -> int:
    """CLI entry: `litgraph trace <file>` or `litgraph trace -` to read stdin."""
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: litgraph trace <file>   (use - for stdin)")
        return 0
    path = argv[0]
    if path == "-":
        text = sys.stdin.read()
    else:
        try:
            text = Path(path).read_text()
        except OSError as e:
            print(f"litgraph trace: cannot read {path!r}: {e}", file=sys.stderr)
            return 1
    sys.stdout.write(render_trace(text))
    return 0

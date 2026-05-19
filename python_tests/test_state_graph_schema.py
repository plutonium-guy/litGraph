"""Tests for `litgraph.StateGraph(state_schema=...)` — implicit coerce
on invoke / resume (iter 378). Closes Tier-1 #7 partial-status."""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

# Native module is required for these tests — they exercise the real
# StateGraph end-to-end. Skip with a clear message if maturin develop
# hasn't been run.
litgraph = pytest.importorskip("litgraph")
if not hasattr(litgraph, "litgraph"):
    pytest.skip(
        "native litgraph module not built; run `maturin develop`",
        allow_module_level=True,
    )

from litgraph import CompiledGraph, StateGraph  # noqa: E402
from litgraph.graph import END, START  # noqa: E402


# ─── dataclass-based state ────────────────────────────────────────


@dataclass
class CounterState:
    n: int = 0
    note: str = ""


def _build_counter_graph(state_schema=None) -> CompiledGraph:
    g = StateGraph(state_schema=state_schema)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    return g.compile()


def test_no_schema_is_dict_passthrough():
    compiled = _build_counter_graph()
    out = compiled.invoke({"n": 0, "note": "hi"})
    assert isinstance(out, dict)
    assert out["n"] == 1
    assert out["note"] == "hi"


def test_dataclass_schema_dumps_input_and_coerces_output():
    compiled = _build_counter_graph(state_schema=CounterState)
    out = compiled.invoke(CounterState(n=0, note="hi"))
    assert isinstance(out, CounterState)
    assert out.n == 1
    assert out.note == "hi"


def test_dataclass_schema_accepts_dict_input_too():
    """Schema doesn't lock the input — a plain dict still works.

    Keeps the migration path frictionless: enable `state_schema` for
    the output coercion benefit without forcing every call-site to
    construct an instance up front.
    """
    compiled = _build_counter_graph(state_schema=CounterState)
    out = compiled.invoke({"n": 5, "note": "dict-in"})
    assert isinstance(out, CounterState)
    assert out.n == 6
    assert out.note == "dict-in"


# ─── Pydantic v2 model ────────────────────────────────────────────


try:
    from pydantic import BaseModel
except ImportError:  # pragma: no cover — pydantic should be installed
    BaseModel = None  # type: ignore


@pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
def test_pydantic_v2_model_round_trips_through_invoke():
    class PState(BaseModel):
        n: int = 0
        tag: str = ""

    g = StateGraph(state_schema=PState)
    g.add_node("inc", lambda s: {"n": s["n"] + 2, "tag": "done"})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    compiled = g.compile()

    out = compiled.invoke(PState(n=3, tag="init"))
    assert isinstance(out, PState)
    assert out.n == 5
    assert out.tag == "done"


@pytest.mark.skipif(BaseModel is None, reason="pydantic not installed")
def test_pydantic_validates_coerced_output_types():
    """Pydantic must enforce its declared types on the way out.

    If a node mistakenly emits a string where the model expects an int,
    Pydantic's validator surfaces the error at coerce time rather than
    silently passing through.
    """
    class PState(BaseModel):
        n: int

    g = StateGraph(state_schema=PState)
    # Bug-on-purpose: returns a str instead of int.
    g.add_node("bad", lambda _s: {"n": "not-an-int"})
    g.add_edge(START, "bad")
    g.add_edge("bad", END)
    compiled = g.compile()

    with pytest.raises(Exception):
        # Pydantic raises ValidationError; we don't depend on the
        # specific exception type so the test stays decoupled from
        # the Pydantic major version.
        compiled.invoke(PState(n=0))


# ─── TypedDict ─────────────────────────────────────────────────────


class TDState(TypedDict, total=False):
    n: int
    tag: str


def test_typeddict_schema_returns_dict_unchanged():
    """TypedDict has no runtime instance — coerce_one returns the dict
    untouched. Verify the wrapper doesn't accidentally break that."""
    g = StateGraph(state_schema=TDState)
    g.add_node("inc", lambda s: {"n": s["n"] + 1, "tag": "x"})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    compiled = g.compile()
    out = compiled.invoke({"n": 0, "tag": ""})
    # Still a dict (TypedDict is dict at runtime), but the test
    # validates that `state_schema=TDState` doesn't blow up.
    assert isinstance(out, dict)
    assert out["n"] == 1
    assert out["tag"] == "x"


# ─── non-schema instance input is rejected ────────────────────────


def test_invoke_rejects_non_dict_when_schema_unset():
    """Without `state_schema`, the wrapper is pure passthrough — a
    bare object input would normally fail at the native boundary with
    a less helpful error. Schema=None preserves that surface (no
    silent magic), so the user sees the native error."""
    compiled = _build_counter_graph()
    with pytest.raises(Exception):
        # Native expects a dict; instance of a random class isn't one.
        class Bare:
            pass
        compiled.invoke(Bare())


def test_dump_state_rejects_unknown_shape_with_clear_error():
    """When state_schema is set but the input isn't dict /
    Pydantic / dataclass, the wrapper raises a clear TypeError at the
    Python layer rather than passing garbage to the native side."""
    compiled = _build_counter_graph(state_schema=CounterState)

    class Mystery:
        n = 1
        note = "x"

    with pytest.raises(TypeError, match="dict.*Pydantic.*dataclass"):
        compiled.invoke(Mystery())


# ─── resume coercion ───────────────────────────────────────────────


def test_resume_coerces_output_when_schema_set():
    """A resumed graph should also yield a typed state — same contract
    as invoke. Uses interrupt_before to checkpoint mid-run."""
    g = StateGraph(state_schema=CounterState)
    g.add_node("a", lambda s: {"n": s["n"] + 1})
    g.add_node("b", lambda s: {"n": s["n"] + 10, "note": "after-b"})
    g.add_edge(START, "a")
    g.add_edge("a", "b")
    g.add_edge("b", END)
    g.interrupt_before("b")
    compiled = g.compile()

    # First call hits the interrupt → raises with thread_id usable for
    # resume. The native API surfaces this as an exception today; we
    # capture and check resume separately.
    try:
        compiled.invoke(CounterState(n=0, note=""), thread_id="t-resume")
    except Exception as e:
        # Sanity-check that we hit an interrupt, not some other error.
        assert "b" in str(e) or "interrupt" in str(e).lower(), str(e)

    out = compiled.resume("t-resume", update=None)
    assert isinstance(out, CounterState)
    assert out.n == 11
    assert out.note == "after-b"


# ─── delegation parity ────────────────────────────────────────────


def test_to_mermaid_passes_through():
    g = StateGraph(state_schema=CounterState)
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_edge(START, "inc")
    g.add_edge("inc", END)
    mermaid = g.to_mermaid()
    assert "inc" in mermaid
    assert "graph" in mermaid.lower()


def test_compiled_to_ascii_passes_through():
    compiled = _build_counter_graph(state_schema=CounterState)
    ascii_repr = compiled.to_ascii()
    assert "inc" in ascii_repr

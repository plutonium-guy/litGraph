"""Tests for `litgraph.coerce` — Pydantic / dataclass / TypedDict
coercion of stream chunks."""

import asyncio
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "python"))

from litgraph.coerce import coerce_one, coerce_stream  # noqa: E402


# ─── dataclass ─────────────────────────────────────────────────────


@dataclass
class State:
    step: str
    value: int


def test_coerce_one_dataclass():
    s = coerce_one({"step": "init", "value": 42}, State)
    assert isinstance(s, State)
    assert s.step == "init"
    assert s.value == 42


def test_coerce_one_dataclass_extra_field_errors():
    with pytest.raises(TypeError):
        coerce_one({"step": "init", "value": 42, "extra": "nope"}, State)


def test_coerce_one_dataclass_missing_field_errors():
    with pytest.raises(TypeError):
        coerce_one({"step": "init"}, State)  # missing `value`


# ─── TypedDict ─────────────────────────────────────────────────────


class StateTD(TypedDict):
    step: str
    value: int


def test_coerce_one_typeddict_passes_through():
    # TypedDict has no runtime instance — coerce_one returns the dict
    # unchanged, since TypedDict is purely a type-checker hint.
    chunk = {"step": "init", "value": 42}
    result = coerce_one(chunk, StateTD)
    assert result is chunk
    assert result == {"step": "init", "value": 42}


# ─── Pydantic (optional) ───────────────────────────────────────────


def _pydantic_available() -> bool:
    try:
        import pydantic  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _pydantic_available(), reason="pydantic not installed")
def test_coerce_one_pydantic_v2():
    from pydantic import BaseModel

    class StateP(BaseModel):
        step: str
        value: int

    s = coerce_one({"step": "init", "value": 42}, StateP)
    assert isinstance(s, StateP)
    assert s.step == "init"
    assert s.value == 42


@pytest.mark.skipif(not _pydantic_available(), reason="pydantic not installed")
def test_coerce_one_pydantic_validation_coerces_types():
    # Pydantic coerces "42" → 42 for int fields. dataclass would NOT.
    from pydantic import BaseModel

    class StateP(BaseModel):
        step: str
        value: int

    s = coerce_one({"step": "init", "value": "42"}, StateP)
    assert s.value == 42  # coerced from str → int


# ─── Non-dict passthrough ──────────────────────────────────────────


def test_coerce_one_non_dict_passthrough():
    # Plain values (sentinels, progress numbers) aren't dicts;
    # coerce_one returns them unchanged regardless of `cls`.
    assert coerce_one(42, State) == 42
    assert coerce_one("done", State) == "done"
    assert coerce_one(None, State) is None


# ─── coerce_stream ─────────────────────────────────────────────────


def test_coerce_stream_dataclass():
    async def source():
        yield {"step": "init", "value": 1}
        yield {"step": "process", "value": 2}
        yield {"step": "done", "value": 3}

    async def collect():
        results = []
        async for s in coerce_stream(source(), State):
            results.append(s)
        return results

    results = asyncio.run(collect())
    assert len(results) == 3
    assert all(isinstance(r, State) for r in results)
    assert results[0].step == "init"
    assert results[2].value == 3


def test_coerce_stream_passes_through_non_dict_chunks():
    # Non-dict chunks (e.g. progress sentinels) flow through unchanged.
    async def source():
        yield {"step": "a", "value": 1}
        yield "PROGRESS-50%"
        yield {"step": "b", "value": 2}
        yield 999  # raw int sentinel

    async def collect():
        results = []
        async for chunk in coerce_stream(source(), State):
            results.append(chunk)
        return results

    results = asyncio.run(collect())
    assert isinstance(results[0], State)
    assert results[1] == "PROGRESS-50%"
    assert isinstance(results[2], State)
    assert results[3] == 999


def test_coerce_stream_propagates_errors_immediately():
    async def source():
        yield {"step": "ok", "value": 1}
        yield {"step": "bad", "value": 2, "extra": "field"}  # invalid for State
        yield {"step": "never_reached", "value": 3}

    async def collect():
        results = []
        async for chunk in coerce_stream(source(), State):
            results.append(chunk)
        return results

    with pytest.raises(TypeError):
        asyncio.run(collect())


# ─── Integration with @entrypoint ──────────────────────────────────


def test_coerce_stream_with_workflow_astream():
    from litgraph.functional import entrypoint

    @entrypoint()
    async def workflow(x):
        return {"step": "result", "value": x * 2}

    async def collect():
        results = []
        async for s in coerce_stream(workflow.astream(5), dict):
            # workflow.astream wraps result in {"final": <result>}.
            # Inner result is the {"step", "value"} dict — but the
            # OUTER chunk shape is {"final": <inner>}, so coerce
            # against dict passes through.
            results.append(s)
        return results

    results = asyncio.run(collect())
    assert len(results) == 1
    assert results[0] == {"final": {"step": "result", "value": 10}}

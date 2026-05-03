"""Live integration: `coerce_one` + `coerce_stream` over real DeepSeek output.

Coerce a model's `out["text"]` into a typed dataclass / TypedDict.
Verifies the adapter chain works on real provider data.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest


pytestmark = pytest.mark.integration


@dataclass
class Reply:
    text: str
    finish_reason: str


def test_coerce_one_into_dataclass(deepseek_chat):
    from litgraph import coerce_one

    out = deepseek_chat.invoke(
        [{"role": "user", "content": "Reply: ok"}],
        max_tokens=10,
    )
    # The model returns extra keys (`usage`, `model`); we strip to
    # what the dataclass declares.
    pruned = {k: out[k] for k in ("text", "finish_reason")}
    typed = coerce_one(pruned, Reply)
    assert isinstance(typed, Reply)
    assert typed.text.strip()
    assert typed.finish_reason


@pytest.mark.asyncio
async def test_coerce_stream_over_async_wrapped_native_stream(deepseek_chat):
    """`coerce_stream` requires an async iterable. `model.stream(...)` is
    a SYNC iterator (`ChatStream`), so wrap it in a one-line async
    generator before piping through. Tests the integration boundary,
    not a deficiency in `coerce_stream` — async-only is a deliberate
    contract on coerce_stream."""
    from typing import TypedDict

    from litgraph import coerce_stream

    class Delta(TypedDict, total=False):
        type: str
        text: str

    raw = deepseek_chat.stream(
        [{"role": "user", "content": "Reply: hi"}],
        max_tokens=10,
    )

    async def to_async(it):
        for x in it:
            yield x

    coerced = []
    async for chunk in coerce_stream(to_async(raw), Delta):
        coerced.append(chunk)
    assert coerced, "no chunks emitted"
    for chunk in coerced:
        assert isinstance(chunk, dict)

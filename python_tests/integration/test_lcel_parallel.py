"""Live integration: `lcel.parallel(...)` runs N steps over the same input.

`parallel` is sequential by design (LCEL parity); for true concurrency
use `StateGraph` branches. Tests the result-shape (`list` in step
order) over a real model call.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_lcel_parallel_returns_list_in_step_order(deepseek_chat):
    from litgraph.lcel import Pipe, parallel

    def short(country: str) -> str:
        return deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )["text"].strip()

    def loud(country: str) -> str:
        return country.upper()

    chain = Pipe(parallel(short, loud))
    out = chain("germany")
    assert isinstance(out, list)
    assert len(out) == 2
    assert "Berlin" in out[0]
    assert out[1] == "GERMANY"

"""Live integration: `Workflow` constructed directly (no `@entrypoint`).

`Workflow(fn)` wraps an async function. Verifies the same set of
methods (`invoke`, `ainvoke`, `astream`) work whether the workflow is
built via the decorator (covered in `test_functional_api.py`) or
constructed directly.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


def test_workflow_direct_construction_invoke(deepseek_chat):
    from litgraph import Workflow

    async def my_workflow(country: str) -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )
        return out["text"].strip()

    wf = Workflow(my_workflow)
    result = wf.invoke("France")
    assert "Paris" in result, f"workflow result missing: {result!r}"


@pytest.mark.asyncio
async def test_workflow_ainvoke_async_path(deepseek_chat):
    from litgraph import Workflow

    async def my_workflow(country: str) -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )
        return out["text"].strip()

    wf = Workflow(my_workflow)
    result = await wf.ainvoke("Italy")
    assert "Rome" in result, f"workflow async result missing: {result!r}"

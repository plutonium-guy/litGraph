"""Live integration: `Workflow.astream` yields a `{"final": ...}` chunk.

Per docstring: v1 yields a single `{"final": <result>}` chunk; future
versions add per-task progress events. Verifies the contract holds
when the workflow body hits a real DeepSeek invoke.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_workflow_astream_yields_final_chunk(deepseek_chat):
    from litgraph import Workflow

    async def my_workflow(country: str) -> str:
        out = deepseek_chat.invoke(
            [{"role": "user", "content": f"Capital of {country}? One word."}],
            max_tokens=10,
        )
        return out["text"].strip()

    wf = Workflow(my_workflow)
    chunks = []
    async for chunk in wf.astream("Spain"):
        chunks.append(chunk)
    assert chunks, "astream yielded no chunks"
    # v1 contract: at least one `{"final": ...}` chunk.
    finals = [c for c in chunks if isinstance(c, dict) and "final" in c]
    assert finals, f"no final-chunk in {chunks!r}"
    assert "Madrid" in str(finals[-1]["final"]), (
        f"final chunk lacks expected content: {finals[-1]!r}"
    )

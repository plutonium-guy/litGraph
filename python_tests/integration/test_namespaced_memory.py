"""Live integration: `NamespacedMemory` — BLOCKED against native backends.

`NamespacedMemory` requires the inner backend to:
1. Expose `add_user(text, metadata=...)` / `add_ai(text, metadata=...)`,
   AND
2. Preserve the `metadata` dict on messages so reads can filter by
   `metadata[NS_KEY] == namespace`.

Native litGraph memory classes (`BufferMemory`, `TokenBufferMemory`,
`SummaryBufferMemory`, …) expose only `append({"role", "content"})`
and silently DROP the `metadata` field. Result: `NamespacedMemory`
either raises `AttributeError: ... does not support 'add_user'` or
the metadata-filter on read returns an empty list.

Use NamespacedMemory with backends that preserve metadata
(LangChain-compat memories, custom dict stores). Tests can be added
once a metadata-preserving native backend lands.
"""
from __future__ import annotations

import pytest


pytestmark = pytest.mark.integration


@pytest.mark.skip(reason="NamespacedMemory needs a metadata-preserving inner backend; native litGraph memories drop metadata")
def test_namespaced_memory_isolates_threads(deepseek_chat):  # pragma: no cover
    pass

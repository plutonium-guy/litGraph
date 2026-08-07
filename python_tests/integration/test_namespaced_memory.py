"""Native-memory integration for `NamespacedMemory`."""
from __future__ import annotations

import pytest

from litgraph.memory import BufferMemory
from litgraph.memory_extras import NamespacedMemory

pytestmark = pytest.mark.integration


def test_namespaced_memory_isolates_threads_on_native_backend():
    shared = BufferMemory()
    alice = NamespacedMemory(shared, "tenant/alice")
    bob = NamespacedMemory(shared, "tenant/bob")

    alice.add_user("hello from alice", {"request_id": "a1"})
    bob.add_user("hello from bob")
    alice.add_ai("alice reply")

    assert [m["content"] for m in alice.messages()] == [
        "hello from alice",
        "alice reply",
    ]
    assert [m["content"] for m in bob.messages()] == ["hello from bob"]
    assert alice.messages()[0]["metadata"]["request_id"] == "a1"

    alice.clear()
    assert alice.messages() == []
    assert [m["content"] for m in bob.messages()] == ["hello from bob"]

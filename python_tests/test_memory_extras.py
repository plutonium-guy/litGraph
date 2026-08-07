"""Tests for litgraph.memory_extras.NamespacedMemory."""
from __future__ import annotations

from typing import Any, Mapping

import pytest

from litgraph.memory_extras import NamespacedMemory, NS_KEY
from litgraph.memory import BufferMemory


class _FakeBackend:
    """Minimal backend implementing the duck shape NamespacedMemory wraps."""

    def __init__(self) -> None:
        self.store: list[dict[str, Any]] = []

    def add_user(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.store.append({"role": "user", "content": text, "metadata": dict(metadata or {})})

    def add_ai(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.store.append({"role": "assistant", "content": text, "metadata": dict(metadata or {})})

    def add_tool(self, text: str, metadata: Mapping[str, Any] | None = None) -> None:
        self.store.append({"role": "tool", "content": text, "metadata": dict(metadata or {})})

    def messages(self) -> list[dict[str, Any]]:
        return list(self.store)

    def clear(self) -> None:
        self.store.clear()


class _AppendOnlyBackend:
    def __init__(self) -> None:
        self.store: list[dict[str, Any]] = []

    def append(self, message: Mapping[str, Any]) -> None:
        self.store.append(dict(message))

    def messages(self) -> list[dict[str, Any]]:
        return list(self.store)

    def clear(self) -> None:
        self.store.clear()


def test_namespace_required_non_empty():
    inner = _FakeBackend()
    with pytest.raises(ValueError):
        NamespacedMemory(inner, namespace="")


def test_writes_stamp_namespace():
    inner = _FakeBackend()
    mem = NamespacedMemory(inner, namespace="tenant_42")
    mem.add_user("hello")
    assert inner.store[0]["metadata"][NS_KEY] == "tenant_42"


def test_reads_filter_by_namespace():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    b = NamespacedMemory(inner, namespace="bob")
    a.add_user("a-msg")
    b.add_user("b-msg")
    a.add_ai("a-ai")
    a_msgs = [m["content"] for m in a.messages()]
    b_msgs = [m["content"] for m in b.messages()]
    assert a_msgs == ["a-msg", "a-ai"]
    assert b_msgs == ["b-msg"]


def test_clear_drops_only_own_namespace():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    b = NamespacedMemory(inner, namespace="bob")
    a.add_user("a")
    b.add_user("b")
    a.clear()
    assert [m["content"] for m in b.messages()] == ["b"]
    assert a.messages() == []


def test_set_system_lives_outside_inner_backend():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    b = NamespacedMemory(inner, namespace="bob")
    a.set_system("you are alice")
    a.add_user("hi")
    b.add_user("hi")
    a_msgs = a.messages()
    b_msgs = b.messages()
    assert a_msgs[0]["role"] == "system"
    assert a_msgs[0]["content"] == "you are alice"
    # Bob should NOT see Alice's system pin.
    assert b_msgs[0]["role"] == "user"
    # Inner backend never received the system pin → no cross-tenant leak.
    assert all("you are alice" not in str(m) for m in inner.store)


def test_set_system_dict_is_passed_through():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    a.set_system({"role": "system", "content": "x"})
    assert a.messages()[0]["content"] == "x"


def test_metadata_passes_through_to_inner():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    a.add_user("hi", metadata={"correlation_id": "c1"})
    md = inner.store[0]["metadata"]
    assert md["correlation_id"] == "c1"
    assert md[NS_KEY] == "alice"


def test_repr_useful():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice")
    s = repr(a)
    assert "alice" in s
    assert "_FakeBackend" in s


def test_namespace_property_returns_value():
    inner = _FakeBackend()
    a = NamespacedMemory(inner, namespace="alice/session_1")
    assert a.namespace == "alice/session_1"


def test_append_only_backend_uses_shared_namespace_sidecar():
    inner = _AppendOnlyBackend()
    a = NamespacedMemory(inner, namespace="alice")
    b = NamespacedMemory(inner, namespace="bob")
    a.add_user("a", {"request_id": "1"})
    b.add_ai("b")

    assert [m["content"] for m in a.messages()] == ["a"]
    assert [m["content"] for m in b.messages()] == ["b"]
    assert a.messages()[0]["metadata"]["request_id"] == "1"
    assert inner.store == [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
    ]


def test_append_only_clear_is_logical_and_keeps_other_namespace():
    inner = _AppendOnlyBackend()
    a = NamespacedMemory(inner, namespace="alice")
    b = NamespacedMemory(inner, namespace="bob")
    a.add_user("a")
    b.add_user("b")
    a.clear()

    assert a.messages() == []
    assert [m["content"] for m in b.messages()] == ["b"]


def test_native_buffer_memory_uses_sidecar_isolation():
    inner = BufferMemory()
    a = NamespacedMemory(inner, namespace="native/a")
    b = NamespacedMemory(inner, namespace="native/b")
    a.add_user("a", {"request_id": "a1"})
    b.add_user("b")

    assert [m["content"] for m in a.messages()] == ["a"]
    assert [m["content"] for m in b.messages()] == ["b"]
    assert a.messages()[0]["metadata"]["request_id"] == "a1"

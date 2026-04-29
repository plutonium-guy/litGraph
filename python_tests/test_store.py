"""Long-term memory `Store` (namespace+key JSON document store).

Mirrors LangGraph's BaseStore API. This file pins the public Python contract;
backend implementations (Postgres, Redis) must satisfy the same shape.
"""

import time

import pytest

litgraph = pytest.importorskip("litgraph")
from litgraph.store import InMemoryStore  # noqa: E402


def test_put_get_roundtrip():
    store = InMemoryStore()
    ns = ("users", "alice")
    store.put(ns, "pref:lang", {"value": "rust"})
    item = store.get(ns, "pref:lang")
    assert item is not None
    assert item["value"] == {"value": "rust"}
    assert tuple(item["namespace"]) == ns
    assert item["key"] == "pref:lang"
    assert item["created_at_ms"] > 0
    assert item["updated_at_ms"] >= item["created_at_ms"]


def test_get_missing_returns_none():
    assert InMemoryStore().get(("nope",), "k") is None


def test_delete_returns_truth_only_when_present():
    store = InMemoryStore()
    ns = ("t",)
    assert store.delete(ns, "k") is False
    store.put(ns, "k", 1)
    assert store.delete(ns, "k") is True
    assert store.get(ns, "k") is None


def test_pop_raises_keyerror_when_missing():
    store = InMemoryStore()
    with pytest.raises(KeyError):
        store.pop(("t",), "missing")


def test_pop_returns_and_removes():
    store = InMemoryStore()
    ns = ("t",)
    store.put(ns, "k", {"x": 1})
    item = store.pop(ns, "k")
    assert item["value"] == {"x": 1}
    assert store.get(ns, "k") is None


def test_search_filters_by_match():
    store = InMemoryStore()
    ns = ("docs",)
    store.put(ns, "a", {"role": "admin", "name": "alice"})
    store.put(ns, "b", {"role": "user", "name": "bob"})
    hits = store.search(ns, matches=[("/role", "admin")])
    assert len(hits) == 1
    assert hits[0]["key"] == "a"


def test_search_filters_by_query_text_case_insensitive():
    store = InMemoryStore()
    ns = ("docs",)
    store.put(ns, "a", {"body": "Hello WORLD"})
    store.put(ns, "b", {"body": "goodbye"})
    hits = store.search(ns, query_text="world")
    assert len(hits) == 1
    assert hits[0]["key"] == "a"


def test_search_respects_namespace_prefix():
    store = InMemoryStore()
    store.put(("t", "a"), "k", 1)
    store.put(("t", "b"), "k", 2)
    store.put(("other",), "k", 3)
    hits = store.search(("t",))
    assert len(hits) == 2
    assert all(tuple(h["namespace"])[0] == "t" for h in hits)


def test_search_pagination_and_limit():
    store = InMemoryStore()
    ns = ("t",)
    for i in range(5):
        store.put(ns, f"k{i}", i)
    hits = store.search(ns, limit=2, offset=1)
    assert len(hits) == 2


def test_ttl_evicts_on_read():
    store = InMemoryStore()
    ns = ("t",)
    store.put(ns, "k", "v", ttl_ms=0)
    assert store.get(ns, "k") is None


def test_ttl_kept_for_long_horizon():
    store = InMemoryStore()
    ns = ("t",)
    store.put(ns, "k", "v", ttl_ms=10_000)
    item = store.get(ns, "k")
    assert item is not None
    assert item["expires_at_ms"] is not None
    assert item["expires_at_ms"] >= int(time.time() * 1000)


def test_list_namespaces():
    store = InMemoryStore()
    store.put(("t", "a"), "k", 1)
    store.put(("t", "a"), "k2", 2)
    store.put(("t", "b"), "k", 3)
    nss = store.list_namespaces(("t",))
    nss_tuples = {tuple(n) for n in nss}
    assert nss_tuples == {("t", "a"), ("t", "b")}


def test_list_namespaces_root_returns_all():
    store = InMemoryStore()
    store.put(("a",), "k", 1)
    store.put(("b", "c"), "k", 2)
    nss = store.list_namespaces()
    assert any(tuple(n) == ("a",) for n in nss)
    assert any(tuple(n) == ("b", "c") for n in nss)


def test_list_form_namespace_accepted():
    store = InMemoryStore()
    store.put(["users", "alice"], "k", "v")
    item = store.get(["users", "alice"], "k")
    assert item is not None and item["value"] == "v"


def test_string_namespace_accepted_as_single_segment():
    store = InMemoryStore()
    store.put("users", "k", 1)
    item = store.get("users", "k")
    assert item is not None and item["value"] == 1


def test_overwrite_preserves_created_updates_updated():
    store = InMemoryStore()
    ns = ("t",)
    store.put(ns, "k", 1)
    first = store.get(ns, "k")
    time.sleep(0.005)
    store.put(ns, "k", 2)
    second = store.get(ns, "k")
    assert second["value"] == 2
    assert second["created_at_ms"] == first["created_at_ms"]
    assert second["updated_at_ms"] >= first["updated_at_ms"]


def test_len_and_repr():
    store = InMemoryStore()
    store.put(("t",), "k1", 1)
    store.put(("t",), "k2", 2)
    assert len(store) == 2
    assert "InMemoryStore" in repr(store)

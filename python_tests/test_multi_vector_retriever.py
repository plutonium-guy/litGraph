"""MultiVectorRetriever — store N caller-supplied perspectives per
parent doc, retrieve the parent on a hit against any of them.

Distinct from ParentDocumentRetriever (which derives perspectives via
a ChildSplitter). Uses FunctionEmbeddings + MemoryVectorStore for a
network-free, deterministic test."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    MemoryDocStore, MemoryVectorStore, MultiVectorRetriever,
)


def _len_embedder():
    """2-D embedding `[len(t), 1.0]`. Cosine similarity on 2-D positive
    vectors orders by the angle, which here is monotone in `len(t)`,
    so longer texts rank closer to longer queries (and vice versa).
    1-D would collapse to constant cosine = 1.0 for any positive pair."""
    def embed(texts):
        return [[float(len(t)), 1.0] for t in texts]
    return FunctionEmbeddings(embed, dimensions=2, name="len2d")


def _fixture():
    return MultiVectorRetriever(
        vector_store=MemoryVectorStore(),
        embeddings=_len_embedder(),
        parent_store=MemoryDocStore(),
    )


def test_index_returns_assigned_parent_ids():
    r = _fixture()
    items = [
        {
            "parent": {"id": "p1", "content": "PARENT ONE"},
            "perspectives": ["short", "this is medium-length"],
        },
        {
            "parent": {"id": "p2", "content": "PARENT TWO"},
            "perspectives": ["aaa"],
        },
    ]
    pids = r.index(items)
    assert pids == ["p1", "p2"]


def test_retrieve_returns_parent_on_perspective_hit():
    r = _fixture()
    r.index([
        {
            "parent": {"id": "p1", "content": "PARENT ONE"},
            "perspectives": ["x", "yy", "zzz"],  # lengths 1, 2, 3
        }
    ])
    # Query of length 1 → matches "x" → parent returned.
    hits = r.retrieve("a", k=1)
    assert len(hits) == 1
    assert hits[0]["id"] == "p1"
    assert hits[0]["content"] == "PARENT ONE"


def test_retrieve_dedups_multiple_perspective_hits():
    """Multiple perspectives of the same parent should collapse to a
    single parent hit."""
    r = _fixture()
    r.index([
        {
            "parent": {"id": "p1", "content": "PARENT ONE"},
            "perspectives": ["aa", "bb", "cc"],  # all length 2
        }
    ])
    hits = r.retrieve("xx", k=5)
    assert len(hits) == 1
    assert hits[0]["id"] == "p1"


def test_distinct_parents_each_returnable_by_perspective():
    r = _fixture()
    r.index([
        {
            "parent": {"id": "p1", "content": "PARENT ONE"},
            "perspectives": ["a"],  # length 1
        },
        {
            "parent": {"id": "p2", "content": "PARENT TWO"},
            "perspectives": ["a" * 30],  # length 30
        },
    ])
    short = r.retrieve("x", k=5)
    assert short and short[0]["id"] == "p1"
    long_q = r.retrieve("x" * 30, k=5)
    assert long_q and long_q[0]["id"] == "p2"


def test_index_empty_list_returns_empty_pids():
    r = _fixture()
    assert r.index([]) == []


def test_empty_perspectives_persists_parent_but_no_hits():
    r = _fixture()
    pids = r.index([
        {
            "parent": {"id": "p1", "content": "STILL HERE"},
            "perspectives": [],
        }
    ])
    assert pids == ["p1"]
    assert r.retrieve("anything", k=5) == []


def test_index_assigns_uuid_when_parent_id_omitted():
    r = _fixture()
    pids = r.index([
        {
            "parent": {"content": "ANON PARENT"},  # no id
            "perspectives": ["foo"],
        }
    ])
    assert len(pids) == 1
    assert pids[0]  # non-empty


def test_index_rejects_non_dict_item():
    r = _fixture()
    try:
        r.index(["not a dict"])
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError")


def test_index_rejects_missing_keys():
    r = _fixture()
    try:
        r.index([{"parent": {"content": "X"}}])  # missing 'perspectives'
    except (ValueError, TypeError):
        pass
    else:
        raise AssertionError("expected ValueError/TypeError on missing 'perspectives'")


def test_repr_shows_child_k_factor():
    r = _fixture()
    assert "child_k_factor=4" in repr(r)


if __name__ == "__main__":
    fns = [
        test_index_returns_assigned_parent_ids,
        test_retrieve_returns_parent_on_perspective_hit,
        test_retrieve_dedups_multiple_perspective_hits,
        test_distinct_parents_each_returnable_by_perspective,
        test_index_empty_list_returns_empty_pids,
        test_empty_perspectives_persists_parent_but_no_hits,
        test_index_assigns_uuid_when_parent_id_omitted,
        test_index_rejects_non_dict_item,
        test_index_rejects_missing_keys,
        test_repr_shows_child_k_factor,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

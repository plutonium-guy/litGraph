"""E2E tests for the Python bindings. Invoked via pytest or directly."""
import json
import os
import tempfile

import litgraph
from litgraph.graph import StateGraph, START, END
from litgraph.retrieval import Bm25Index
from litgraph.splitters import RecursiveCharacterSplitter, MarkdownHeaderSplitter
from litgraph.loaders import TextLoader, DirectoryLoader
from litgraph.observability import CostTracker
from litgraph.cache import MemoryCache, SqliteCache


def test_sanity():
    assert litgraph.sum_as_string(2, 3) == "5"
    # Version is bumped per release; assert shape, not value.
    assert isinstance(litgraph.__version__, str)
    assert litgraph.__version__.count(".") >= 2


def test_state_graph_simple_linear():
    g = StateGraph()
    g.add_node("inc", lambda s: {"n": s["n"] + 1})
    g.add_node("double", lambda s: {"n": s["n"] * 2})
    g.add_edge(START, "inc")
    g.add_edge("inc", "double")
    g.add_edge("double", END)
    compiled = g.compile()
    result = compiled.invoke({"n": 5})
    assert result["n"] == 12


def test_state_graph_conditional():
    g = StateGraph()
    g.add_node("classify", lambda s: {"class": "big" if s["x"] > 10 else "small"})
    g.add_node("big_branch", lambda s: {"result": "is big"})
    g.add_node("small_branch", lambda s: {"result": "is small"})
    g.add_edge(START, "classify")
    g.add_conditional_edges(
        "classify",
        lambda s: "big_branch" if s["class"] == "big" else "small_branch",
    )
    g.add_edge("big_branch", END)
    g.add_edge("small_branch", END)
    compiled = g.compile()

    assert compiled.invoke({"x": 5})["result"] == "is small"
    # Same compiled graph, different input — state independence verified.
    assert compiled.invoke({"x": 50})["result"] == "is big"


def test_state_graph_parallel_fanout():
    """Two branches run concurrently in one superstep; append reducer merges lists."""
    g = StateGraph()
    g.add_node("a", lambda s: {"items": [1]})
    g.add_node("b", lambda s: {"items": [2]})
    g.add_node("c", lambda s: {"items": [3]})
    g.add_node("join", lambda s: {})
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge(START, "c")
    g.add_edge("a", "join")
    g.add_edge("b", "join")
    g.add_edge("c", "join")
    g.add_edge("join", END)
    compiled = g.compile()
    result = compiled.invoke({"items": []})
    assert sorted(result["items"]) == [1, 2, 3]


def test_bm25():
    idx = Bm25Index()
    idx.add([
        {"content": "the quick brown fox", "id": "a"},
        {"content": "lazy dogs sleep all day", "id": "b"},
        {"content": "foxes love chickens", "id": "c"},
    ])
    hits = idx.search("fox", 2)
    assert len(hits) >= 1
    assert hits[0]["id"] in {"a", "c"}


def test_recursive_splitter():
    sp = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=5)
    chunks = sp.split_text("a" * 100)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c) <= 35  # chunk_size + overlap slack


def test_markdown_splitter_preserves_metadata():
    sp = MarkdownHeaderSplitter(max_depth=2)
    docs = sp.split_documents([{"content": "# Root\n\nintro\n\n## Sec\n\nbody", "id": "d"}])
    assert len(docs) == 2
    assert docs[1]["metadata"]["h1"] == "Root"
    assert docs[1]["metadata"]["h2"] == "Sec"


def test_directory_loader():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(5):
            with open(os.path.join(tmp, f"f{i}.txt"), "w") as f:
                f.write(f"doc {i}")
        docs = DirectoryLoader(tmp, "*.txt").load()
        assert len(docs) == 5


def test_cost_tracker_basics():
    t = CostTracker({"gpt-5": (2.5, 10.0)})
    assert t.usd() == 0.0
    snap = t.snapshot()
    assert snap["calls"] == 0
    assert snap["usd"] == 0.0


def test_caches_instantiate():
    MemoryCache(max_capacity=100)
    SqliteCache.in_memory()


if __name__ == "__main__":
    fns = [
        test_sanity,
        test_state_graph_simple_linear,
        test_state_graph_conditional,
        test_state_graph_parallel_fanout,
        test_bm25,
        test_recursive_splitter,
        test_markdown_splitter_preserves_metadata,
        test_directory_loader,
        test_cost_tracker_basics,
        test_caches_instantiate,
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

"""SemanticChunker — embedding-based splitting with FunctionEmbeddings.

Uses a deterministic toy embedder so the chunker's percentile logic is
testable without hitting any network.
"""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.splitters import SemanticChunker


def topic_embedder(texts):
    """Returns 3-dim vectors keyed off keyword in window text."""
    out = []
    for t in texts:
        lower = t.lower()
        if "dog" in lower:
            out.append([1.0, 0.0, 0.0])
        elif "car" in lower:
            out.append([0.0, 1.0, 0.0])
        elif "plant" in lower:
            out.append([0.0, 0.0, 1.0])
        else:
            out.append([0.5, 0.5, 0.5])
    return out


def test_semantic_chunker_splits_at_topic_shift():
    e = FunctionEmbeddings(topic_embedder, dimensions=3, name="toy")
    chunker = SemanticChunker(e, buffer_size=0, breakpoint_percentile=50.0)
    text = ("I love my dog. The dog plays fetch. "
            "My car is fast. The car needs gas. "
            "The plant needs water. My plant grew tall.")
    chunks = chunker.split_text(text)
    assert len(chunks) >= 2
    assert "dog" in chunks[0].lower()
    assert "plant" in chunks[-1].lower()


def test_semantic_chunker_single_sentence_returns_unchanged():
    e = FunctionEmbeddings(topic_embedder, dimensions=3, name="toy")
    chunker = SemanticChunker(e)
    chunks = chunker.split_text("Just one sentence here.")
    assert chunks == ["Just one sentence here."]


def test_semantic_chunker_empty_text_returns_empty():
    e = FunctionEmbeddings(topic_embedder, dimensions=3, name="toy")
    chunker = SemanticChunker(e)
    assert chunker.split_text("   \n\t  ") == []


def test_semantic_chunker_split_documents_preserves_metadata():
    e = FunctionEmbeddings(topic_embedder, dimensions=3, name="toy")
    chunker = SemanticChunker(e, buffer_size=0, breakpoint_percentile=50.0)
    docs = [{
        "content": ("I love my dog. The dog plays. "
                    "My car is fast. The car needs gas."),
        "id": "doc-1",
        "metadata": {"origin": "test"},
    }]
    out = chunker.split_documents(docs)
    assert len(out) >= 2
    # docs_to_pylist round-trips metadata through string form; numeric values
    # come back as strings.
    for i, d in enumerate(out):
        assert d["metadata"]["chunk_index"] == str(i)
        assert d["metadata"]["source_id"] == "doc-1"
        assert d["metadata"]["origin"] == "test"
        assert d["id"] == f"doc-1#{i}"


def test_semantic_chunker_higher_percentile_yields_fewer_chunks():
    """Same text; p=100 ⇒ 1 chunk (no break crosses), p=50 ⇒ ≥2 chunks."""
    e = FunctionEmbeddings(topic_embedder, dimensions=3, name="toy")
    text = ("I love my dog. The dog plays fetch. "
            "My car is fast. The car needs gas. "
            "The plant needs water. My plant grew tall.")
    aggressive = SemanticChunker(e, buffer_size=0, breakpoint_percentile=50.0).split_text(text)
    conservative = SemanticChunker(e, buffer_size=0, breakpoint_percentile=100.0).split_text(text)
    assert len(aggressive) >= len(conservative)
    assert len(conservative) == 1


def test_semantic_chunker_rejects_non_embeddings():
    try:
        SemanticChunker("not an embedder")  # type: ignore[arg-type]
    except TypeError as exc:
        assert "embeddings must be one of" in str(exc)
    else:
        raise AssertionError("expected TypeError")


if __name__ == "__main__":
    fns = [
        test_semantic_chunker_splits_at_topic_shift,
        test_semantic_chunker_single_sentence_returns_unchanged,
        test_semantic_chunker_empty_text_returns_empty,
        test_semantic_chunker_split_documents_preserves_metadata,
        test_semantic_chunker_higher_percentile_yields_fewer_chunks,
        test_semantic_chunker_rejects_non_embeddings,
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

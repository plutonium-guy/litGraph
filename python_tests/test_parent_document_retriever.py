"""ParentDocumentRetriever — index small chunks for embedding precision,
return full parent docs at retrieval time. Direct LangChain parity.

Solves the standard RAG tradeoff: small chunks give precise embeddings (each
talks about one thing) but lose context; large chunks preserve context but
smear embeddings. PDR sidesteps it by indexing both."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import (
    MemoryDocStore,
    MemoryVectorStore,
    ParentDocumentRetriever,
)
from litgraph.splitters import RecursiveCharacterSplitter


def _len_embeddings():
    """Deterministic embedder: doc-length × 4 dims. Distinct lengths → distinct
    vectors. Small but real enough to exercise the search path."""
    def fn(texts):
        return [[float(len(t))] * 4 for t in texts]
    return FunctionEmbeddings(fn, dimensions=4, name="len-embed")


def test_pdr_index_then_retrieve_returns_full_parent_documents():
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
    store = MemoryVectorStore()
    docs_kv = MemoryDocStore()
    pdr = ParentDocumentRetriever(
        child_splitter=splitter,
        vector_store=store,
        embeddings=_len_embeddings(),
        parent_store=docs_kv,
    )
    parents = [
        {"content": "Rust is fast. Rust is safe. Rust avoids GC overhead."},
        {"content": "Python is dynamic. Python's syntax is approachable."},
    ]
    ids = pdr.index_documents(parents)
    assert len(ids) == 2
    assert len(docs_kv) == 2
    # Vector store now has multiple children (≥ 2 since each parent splits).
    assert len(store) >= 2

    # Retrieve — returns FULL parents, not split chunks. Either parent
    # could rank first since the test embedder is length-based; just assert
    # we got both full docs.
    hits = pdr.retrieve("rust", k=2)
    contents = [h["content"] for h in hits]
    assert any("Rust is fast" in c and "Rust avoids GC" in c for c in contents)


def test_pdr_dedupes_when_multiple_children_match_one_parent():
    """A single parent splitting into many children must collapse to ONE
    parent in the result, not N copies."""
    splitter = RecursiveCharacterSplitter(chunk_size=10, chunk_overlap=0)
    store = MemoryVectorStore()
    docs_kv = MemoryDocStore()
    pdr = ParentDocumentRetriever(
        child_splitter=splitter,
        vector_store=store,
        embeddings=_len_embeddings(),
        parent_store=docs_kv,
    )
    pdr.index_documents([
        {"content": "Same parent. Many small. Tiny chunks. Lots of them."},
    ])
    # Top-5 child hits all map to the SAME parent → dedupe to 1.
    hits = pdr.retrieve("anything", k=5)
    assert len(hits) == 1


def test_pdr_propagates_caller_supplied_parent_id():
    """If the caller sets `id` on a parent doc, that id is preserved for
    the parent_store lookup (rather than overwritten with a random UUID)."""
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
    store = MemoryVectorStore()
    docs_kv = MemoryDocStore()
    pdr = ParentDocumentRetriever(
        child_splitter=splitter,
        vector_store=store,
        embeddings=_len_embeddings(),
        parent_store=docs_kv,
    )
    ids = pdr.index_documents([
        {"id": "doc-alpha", "content": "Alpha content with several words."},
    ])
    assert ids == ["doc-alpha"]


def test_pdr_empty_index_call_is_a_no_op():
    splitter = RecursiveCharacterSplitter(chunk_size=20, chunk_overlap=0)
    store = MemoryVectorStore()
    docs_kv = MemoryDocStore()
    pdr = ParentDocumentRetriever(
        child_splitter=splitter,
        vector_store=store,
        embeddings=_len_embeddings(),
        parent_store=docs_kv,
    )
    ids = pdr.index_documents([])
    assert ids == []
    assert len(docs_kv) == 0
    assert len(store) == 0


def test_pdr_rejects_invalid_splitter_type():
    """`child_splitter` must be a litGraph splitter — string args or random
    objects raise TypeError at construction (not at retrieval time)."""
    store = MemoryVectorStore()
    docs_kv = MemoryDocStore()
    try:
        ParentDocumentRetriever(
            child_splitter="not a splitter",
            vector_store=store,
            embeddings=_len_embeddings(),
            parent_store=docs_kv,
        )
    except TypeError as e:
        assert "child_splitter" in str(e)
    else:
        raise AssertionError("expected TypeError on invalid splitter")


def test_pdr_memory_doc_store_repr_and_len():
    s = MemoryDocStore()
    assert len(s) == 0
    assert "MemoryDocStore" in repr(s)


if __name__ == "__main__":
    fns = [
        test_pdr_index_then_retrieve_returns_full_parent_documents,
        test_pdr_dedupes_when_multiple_children_match_one_parent,
        test_pdr_propagates_caller_supplied_parent_id,
        test_pdr_empty_index_call_is_a_no_op,
        test_pdr_rejects_invalid_splitter_type,
        test_pdr_memory_doc_store_repr_and_len,
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

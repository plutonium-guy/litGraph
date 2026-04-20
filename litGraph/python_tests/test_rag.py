"""Full Python RAG pipeline: loader → splitter → embed → store → retrieve."""
from litgraph.embeddings import FunctionEmbeddings
from litgraph.retrieval import MemoryVectorStore, HnswVectorStore, VectorRetriever
from litgraph.splitters import RecursiveCharacterSplitter
from litgraph.providers import AnthropicChat


def test_function_embeddings_instantiates():
    def embed(texts):
        return [[float(len(t)), float(t.count(" "))] for t in texts]

    e = FunctionEmbeddings(embed, dimensions=2, name="charlen")
    assert e.name == "charlen"
    assert e.dimensions == 2
    v = e.embed_query("hello world")
    assert v == [11.0, 1.0]


def test_memory_vector_store_add_and_search():
    store = MemoryVectorStore()
    docs = [
        {"content": "cat", "id": "1"},
        {"content": "dog", "id": "2"},
        {"content": "car", "id": "3"},
    ]
    embs = [[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]]
    store.add(docs, embs)
    assert len(store) == 3

    results = store.similarity_search([1.0, 0.0, 0.0], k=2)
    assert len(results) == 2
    assert results[0]["id"] == "1"  # cat is nearest


def test_end_to_end_rag_pipeline():
    """Split text → embed → store → retrieve. All in Python, runs on Rust."""
    # 1) Split
    sp = RecursiveCharacterSplitter(chunk_size=30, chunk_overlap=5)
    text = (
        "Rust has no garbage collector. "
        "Python uses reference counting and cycle detection. "
        "Rust provides ownership and borrowing. "
        "Async Rust avoids threads for I/O."
    )
    chunks = sp.split_text(text)
    docs = [{"content": c, "id": f"c{i}"} for i, c in enumerate(chunks)]

    # 2) Embed (toy BOW-style)
    vocab = ["rust", "python", "ownership", "borrowing", "async", "threads"]

    def embed(texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append([float(low.count(w)) for w in vocab])
        return out

    embeddings = FunctionEmbeddings(embed, dimensions=len(vocab), name="bow")

    # 3) Store (hit the rust rayon-parallel cosine path)
    store = MemoryVectorStore()
    embs = embed([d["content"] for d in docs])
    store.add(docs, embs)

    # 4) Retrieve
    retriever = VectorRetriever(embeddings, store)
    hits = retriever.retrieve("async I/O in rust", k=2)
    assert len(hits) == 2
    # At least one chunk mentioning "async" should be retrieved.
    assert any("async" in h["content"].lower() for h in hits), \
        f"expected an async chunk, got {[h['content'] for h in hits]}"


def test_hnsw_store_add_and_search():
    store = HnswVectorStore()
    docs = [
        {"content": "apple", "id": "a"},
        {"content": "banana", "id": "b"},
        {"content": "cherry", "id": "c"},
        {"content": "applet", "id": "d"},
    ]
    embs = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.95, 0.05, 0.0],
    ]
    store.add(docs, embs)
    assert len(store) == 4
    hits = store.similarity_search([1.0, 0.0, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0]["id"] in {"a", "d"}
    assert hits[0]["score"] > 0.9


def test_vector_retriever_accepts_hnsw_store():
    def embed(texts):
        # 3-dim bag-of-words
        vocab = ["rust", "python", "graph"]
        return [[float(t.lower().count(w)) for w in vocab] for t in texts]

    e = FunctionEmbeddings(embed, dimensions=3, name="bow")
    store = HnswVectorStore()
    docs = [
        {"content": "rust graph is fast", "id": "1"},
        {"content": "python dict is simple", "id": "2"},
        {"content": "graph algorithms in rust", "id": "3"},
    ]
    embs = embed([d["content"] for d in docs])
    store.add(docs, embs)

    retriever = VectorRetriever(e, store)
    hits = retriever.retrieve("rust graph perf", k=2)
    assert len(hits) == 2
    ids = {h["id"] for h in hits}
    assert "1" in ids or "3" in ids


def test_anthropic_chat_constructs():
    model = AnthropicChat(api_key="dummy", model="claude-opus-4-7")
    assert "claude-opus-4-7" in repr(model)


if __name__ == "__main__":
    fns = [
        test_function_embeddings_instantiates,
        test_memory_vector_store_add_and_search,
        test_end_to_end_rag_pipeline,
        test_hnsw_store_add_and_search,
        test_vector_retriever_accepts_hnsw_store,
        test_anthropic_chat_constructs,
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

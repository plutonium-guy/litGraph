"""Validate new Python bindings for iter 15: PgVectorStore construction +
VectorRetriever polymorphism."""

def test_pg_vector_store_api_exists():
    from litgraph.retrieval import PgVectorStore
    # Static method exists; we can't actually connect without a live Postgres.
    assert hasattr(PgVectorStore, "connect")


def test_vector_retriever_type_errors_on_wrong_store():
    from litgraph.retrieval import VectorRetriever, MemoryVectorStore
    from litgraph.embeddings import FunctionEmbeddings

    e = FunctionEmbeddings(lambda texts: [[0.0] for _ in texts], dimensions=1)
    # Correct: works
    _ = VectorRetriever(e, MemoryVectorStore())

    # Wrong: passing a bare string should raise TypeError from Rust.
    try:
        _ = VectorRetriever(e, "not a store")
        raised = False
    except TypeError:
        raised = True
    assert raised, "expected TypeError when passing a non-store"


if __name__ == "__main__":
    fns = [
        test_pg_vector_store_api_exists,
        test_vector_retriever_type_errors_on_wrong_store,
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

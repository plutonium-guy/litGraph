"""VectorStoreMemory — embed each turn, retrieve top-K most-relevant
past messages by cosine. Topic-relevance memory for long-running agents."""
from litgraph.memory import VectorStoreMemory
from litgraph.embeddings import FunctionEmbeddings


def keyword_embedder(keywords):
    def embed(texts):
        out = []
        for text in texts:
            lower = text.lower()
            out.append([1.0 if kw in lower else 0.0 for kw in keywords])
        return out
    return FunctionEmbeddings(embed, len(keywords))


KEYWORDS = ["rust", "borrow", "lifetime", "css", "flexbox", "python", "comprehension"]


def test_retrieves_top_k_by_relevance():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.append({"role": "user", "content": "rust borrow checker question"})
    mem.append({"role": "assistant", "content": "use clone or rework lifetimes"})
    mem.append({"role": "user", "content": "css flexbox alignment"})
    mem.append({"role": "assistant", "content": "use justify-content"})
    mem.append({"role": "user", "content": "python list comprehension"})
    mem.append({"role": "assistant", "content": "use [x for x in ...]"})

    r = mem.retrieve_for("rust lifetime trouble", k=2)
    assert len(r) == 2
    for item in r:
        text = item["message"]["content"].lower()
        assert "rust" in text or "borrow" in text or "lifetime" in text or "clone" in text


def test_auto_flush_on_retrieve():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.append({"role": "user", "content": "rust borrow"})
    assert mem.embedded_len() == 0
    assert mem.pending_len() == 1
    r = mem.retrieve_for("rust", k=1)
    assert len(r) == 1
    assert mem.pending_len() == 0
    assert mem.embedded_len() == 1


def test_explicit_flush_returns_count():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    for i in range(5):
        mem.append({"role": "user", "content": f"rust msg {i}"})
    n = mem.flush()
    assert n == 5
    assert mem.embedded_len() == 5
    assert mem.pending_len() == 0


def test_flush_idempotent_when_pending_empty():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    n = mem.flush()
    assert n == 0


def test_system_role_sets_pin_not_embedded():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.append({"role": "system", "content": "you are a helpful assistant"})
    mem.append({"role": "user", "content": "rust borrow"})
    mem.flush()
    assert mem.embedded_len() == 1


def test_set_system_explicitly():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.set_system({"role": "system", "content": "be terse"})
    mem.set_system(None)


def test_build_context_includes_system_retrieved_and_current():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.set_system({"role": "system", "content": "be terse"})
    mem.append({"role": "user", "content": "rust borrow tip?"})
    mem.append({"role": "assistant", "content": "use clone"})
    mem.append({"role": "user", "content": "css question"})

    ctx = mem.build_context(
        query="rust",
        current={"role": "user", "content": "more rust help"},
        k=2,
    )
    # [system, ...top2, current] = 4
    assert len(ctx) == 4
    assert ctx[0]["role"] == "system"
    assert ctx[0]["content"] == "be terse"
    assert ctx[-1]["content"] == "more rust help"


def test_k_zero_uses_default_top_k():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=3)
    for i in range(5):
        mem.append({"role": "user", "content": f"rust {i}"})
    r = mem.retrieve_for("rust", k=0)
    assert len(r) == 3


def test_empty_store_returns_empty_results():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=5)
    r = mem.retrieve_for("anything", k=3)
    assert r == []


def test_clear_drops_everything():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    mem.append({"role": "user", "content": "rust"})
    mem.flush()
    mem.append({"role": "user", "content": "pending"})
    mem.clear()
    assert mem.embedded_len() == 0
    assert mem.pending_len() == 0


def test_results_sorted_descending_by_score():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=5)
    mem.append({"role": "user", "content": "rust borrow lifetime"})  # 3 hits
    mem.append({"role": "user", "content": "rust borrow"})            # 2
    mem.append({"role": "user", "content": "rust"})                   # 1
    mem.append({"role": "user", "content": "nothing"})                # 0
    r = mem.retrieve_for("rust borrow lifetime", k=4)
    scores = [item["score"] for item in r]
    assert scores == sorted(scores, reverse=True)
    assert r[0]["score"] > r[-1]["score"]


def test_repr():
    mem = VectorStoreMemory(embeddings=keyword_embedder(KEYWORDS), default_top_k=2)
    assert "VectorStoreMemory" in repr(mem)


if __name__ == "__main__":
    import traceback
    fns = [
        test_retrieves_top_k_by_relevance,
        test_auto_flush_on_retrieve,
        test_explicit_flush_returns_count,
        test_flush_idempotent_when_pending_empty,
        test_system_role_sets_pin_not_embedded,
        test_set_system_explicitly,
        test_build_context_includes_system_retrieved_and_current,
        test_k_zero_uses_default_top_k,
        test_empty_store_returns_empty_results,
        test_clear_drops_everything,
        test_results_sorted_descending_by_score,
        test_repr,
    ]
    failed = []
    for fn in fns:
        try:
            fn()
            print(f"  ok  {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"FAIL  {fn.__name__}: {e!r}")
            traceback.print_exc()
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    if failed:
        raise SystemExit(1)

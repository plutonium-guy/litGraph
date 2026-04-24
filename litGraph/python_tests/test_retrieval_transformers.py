"""Document transformers — MMR, embedding redundancy filter, long-context
reorder. Post-retrieval RAG quality improvements. Pure functions over
candidate lists; compose freely."""
from litgraph.retrieval import (
    embedding_redundant_filter,
    long_context_reorder,
    mmr_select,
)


def _doc(content):
    return {"content": content}


def test_mmr_pure_relevance_picks_most_similar_first():
    q = [1.0, 0.0]
    docs = [_doc("d0"), _doc("d1"), _doc("d2")]
    embs = [[1.0, 0.0], [0.7, 0.7], [0.0, 1.0]]
    out = mmr_select(q, docs, embs, k=3, lambda_mult=1.0)
    assert out[0]["content"] == "d0"
    assert len(out) == 3


def test_mmr_with_diversity_avoids_near_dupes():
    q = [1.0, 0.0]
    docs = [_doc("d0"), _doc("d1-dup"), _doc("d2-diverse")]
    embs = [
        [1.0, 0.0],
        [0.99, 0.05],   # near-dup of d0
        [0.6, 0.5],
    ]
    out = mmr_select(q, docs, embs, k=2, lambda_mult=0.3)
    names = [d["content"] for d in out]
    assert names[0] == "d0"
    assert names[1] != "d1-dup"


def test_mmr_default_lambda_is_05():
    q = [1.0, 0.0]
    docs = [_doc("a"), _doc("b")]
    embs = [[1.0, 0.0], [0.5, 0.5]]
    # Just verify the default kw works.
    out = mmr_select(q, docs, embs, k=2)
    assert len(out) == 2


def test_mmr_caps_k_at_candidates_len():
    q = [1.0]
    docs = [_doc("a"), _doc("b")]
    embs = [[1.0], [0.5]]
    out = mmr_select(q, docs, embs, k=100, lambda_mult=0.5)
    assert len(out) == 2


def test_redundant_filter_drops_near_dupes_at_threshold():
    docs = [_doc("a"), _doc("a-dup"), _doc("b")]
    embs = [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0]]
    kept = embedding_redundant_filter(docs, embs, threshold=0.95)
    names = [d["content"] for d in kept]
    assert names == ["a", "b"]


def test_redundant_filter_default_threshold_095():
    docs = [_doc("a"), _doc("a-dup")]
    embs = [[1.0, 0.0], [0.99, 0.01]]
    kept = embedding_redundant_filter(docs, embs)  # default 0.95
    assert len(kept) == 1


def test_redundant_filter_threshold_above_one_keeps_perfect_dupes():
    docs = [_doc("a"), _doc("a")]
    embs = [[1.0], [1.0]]
    kept = embedding_redundant_filter(docs, embs, threshold=1.01)
    assert len(kept) == 2


def test_long_context_reorder_one_or_two_docs_unchanged():
    one = [_doc("a")]
    assert long_context_reorder(one)[0]["content"] == "a"

    two = [_doc("a"), _doc("b")]
    out = long_context_reorder(two)
    assert [d["content"] for d in out] == ["a", "b"]


def test_long_context_reorder_places_top_at_edges():
    """5 docs ranked a..e (a best). Result puts a/b at edges, e in center."""
    docs = [_doc(c) for c in "abcde"]
    out = long_context_reorder(docs)
    names = [d["content"] for d in out]
    assert names == ["a", "c", "e", "d", "b"]


def test_full_rag_pipeline_compose():
    """Simulate retriever → MMR → long-context reorder."""
    q = [1.0, 0.0]
    retrieved = [_doc(f"doc{i}") for i in range(5)]
    embs = [
        [1.0, 0.0],
        [0.8, 0.4],
        [0.99, 0.05],
        [0.5, 0.5],
        [0.0, 1.0],
    ]
    diverse = mmr_select(q, retrieved, embs, k=3, lambda_mult=0.5)
    assert len(diverse) == 3
    # Reorder for the LLM context window.
    reordered = long_context_reorder(diverse)
    assert len(reordered) == 3
    assert reordered[0]["content"] == diverse[0]["content"]


if __name__ == "__main__":
    import traceback

    fns = [
        test_mmr_pure_relevance_picks_most_similar_first,
        test_mmr_with_diversity_avoids_near_dupes,
        test_mmr_default_lambda_is_05,
        test_mmr_caps_k_at_candidates_len,
        test_redundant_filter_drops_near_dupes_at_threshold,
        test_redundant_filter_default_threshold_095,
        test_redundant_filter_threshold_above_one_keeps_perfect_dupes,
        test_long_context_reorder_one_or_two_docs_unchanged,
        test_long_context_reorder_places_top_at_edges,
        test_full_rag_pipeline_compose,
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

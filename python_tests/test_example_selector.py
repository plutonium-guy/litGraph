"""SemanticSimilarityExampleSelector — embedding-based top-K example
selection for FewShot prompts. Uses FunctionEmbeddings (deterministic
keyword-hash) so we can reason about which examples should win."""
from litgraph.prompts import (
    FewShotChatPromptTemplate,
    SemanticSimilarityExampleSelector,
)
from litgraph.embeddings import FunctionEmbeddings


def keyword_embedder(keywords):
    """Embed text(s) → fixed-dim vectors marking each keyword's presence (1.0).
    The litGraph FunctionEmbeddings callback receives `list[str]` (batch) and
    must return `list[list[float]]`."""
    def embed(texts):
        out = []
        for text in texts:
            lower = text.lower()
            out.append([1.0 if kw in lower else 0.0 for kw in keywords])
        return out
    return FunctionEmbeddings(embed, len(keywords))


POOL = [
    {"input": "fix borrow checker error in rust", "output": "use clone or rework lifetimes"},
    {"input": "css flexbox alignment problem", "output": "use justify-content + align-items"},
    {"input": "rust lifetime annotation syntax", "output": "use 'a notation"},
    {"input": "javascript promise chaining", "output": "use .then or async/await"},
    {"input": "python list comprehension syntax", "output": "use [x for x in ...]"},
]


def test_select_top_k_picks_rust_examples():
    embedder = keyword_embedder(["rust", "borrow", "lifetime", "css", "javascript", "promise", "python", "list"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    picked = sel.select("how do I fix a rust borrow problem?", k=2)
    assert len(picked) == 2
    for ex in picked:
        assert "rust" in ex["input"]


def test_select_returns_full_pool_when_k_exceeds_pool_size():
    embedder = keyword_embedder(["x"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    picked = sel.select("anything", k=100)
    assert len(picked) == 5


def test_select_k_zero_returns_empty_list():
    embedder = keyword_embedder(["x"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    picked = sel.select("rust", k=0)
    assert picked == []


def test_empty_pool_returns_empty_list_no_error():
    embedder = keyword_embedder(["x"])
    sel = SemanticSimilarityExampleSelector([], embedder, key_field="input")
    assert sel.select("anything", k=3) == []
    assert sel.pool_size() == 0


def test_missing_key_field_raises_runtime_error():
    embedder = keyword_embedder(["x"])
    bad_pool = [{"output": "no input field"}]
    sel = SemanticSimilarityExampleSelector(bad_pool, embedder, key_field="input")
    try:
        sel.select("anything", k=1)
        raise AssertionError("expected error")
    except RuntimeError as e:
        assert "input" in str(e)


def test_warmup_pre_embeds_pool():
    call_log = []
    def embed(texts):
        call_log.extend(texts)
        return [[1.0 if "rust" in t.lower() else 0.0] for t in texts]
    embedder = FunctionEmbeddings(embed, 1)

    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    sel.warmup()
    # All 5 pool entries embedded; query NOT yet.
    assert len(call_log) == 5

    # Warmup again is no-op.
    sel.warmup()
    assert len(call_log) == 5


def test_pool_embeddings_cached_across_select_calls():
    call_log = []
    def embed(texts):
        call_log.extend(texts)
        return [[1.0 if "rust" in t.lower() else 0.0] for t in texts]
    embedder = FunctionEmbeddings(embed, 1)

    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    sel.select("rust", k=1)
    after_first = len(call_log)  # 5 docs + 1 query = 6
    assert after_first == 6

    sel.select("css", k=1)
    after_second = len(call_log)  # +1 query (no doc re-embed)
    assert after_second == 7


def test_pool_size_reports_pool_length():
    embedder = keyword_embedder(["x"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    assert sel.pool_size() == 5


def test_repr_shows_pool_size():
    embedder = keyword_embedder(["x"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    r = repr(sel)
    assert "SemanticSimilarityExampleSelector" in r
    assert "pool_size=5" in r


def test_picks_feed_into_few_shot_template():
    """End-to-end: select examples → render through FewShotChatPromptTemplate."""
    from litgraph.prompts import ChatPromptTemplate

    embedder = keyword_embedder(["rust", "borrow", "lifetime", "css", "python"])
    sel = SemanticSimilarityExampleSelector(POOL, embedder, key_field="input")
    picked = sel.select("rust borrow trouble", k=2)

    example_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ input }}"),
        ("assistant", "{{ output }}"),
    ])
    input_prompt = ChatPromptTemplate.from_messages([
        ("user", "{{ question }}"),
    ])
    few_shot = FewShotChatPromptTemplate(
        examples=picked,
        example_prompt=example_prompt,
        input_prompt=input_prompt,
        system_prefix="You are a helpful coding assistant.",
    )
    msgs = few_shot.format({"question": "how do I fix this borrow error?"})
    # Layout: [system, user(ex1.input), assistant(ex1.output),
    #         user(ex2.input), assistant(ex2.output), user(question)]
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"
    assert "borrow error" in msgs[-1]["content"]
    # Picked examples must be rust-themed (semantic match).
    user_ex_contents = [m["content"] for m in msgs if m["role"] == "user"][:-1]
    for c in user_ex_contents:
        assert "rust" in c


if __name__ == "__main__":
    import traceback
    fns = [
        test_select_top_k_picks_rust_examples,
        test_select_returns_full_pool_when_k_exceeds_pool_size,
        test_select_k_zero_returns_empty_list,
        test_empty_pool_returns_empty_list_no_error,
        test_missing_key_field_raises_runtime_error,
        test_warmup_pre_embeds_pool,
        test_pool_embeddings_cached_across_select_calls,
        test_pool_size_reports_pool_length,
        test_repr_shows_pool_size,
        test_picks_feed_into_few_shot_template,
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

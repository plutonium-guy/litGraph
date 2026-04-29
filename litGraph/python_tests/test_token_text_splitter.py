"""TokenTextSplitter — splits by exact token count via litgraph-tokenizers.
Use after structural splits to enforce hard token budgets per chunk."""
from litgraph.splitters import TokenTextSplitter


def test_small_text_returns_one_chunk():
    s = TokenTextSplitter(chunk_size=1000, chunk_overlap=0, model="gpt-4o")
    chunks = s.split_text("hello world")
    assert chunks == ["hello world"]


def test_empty_text_returns_empty():
    s = TokenTextSplitter(chunk_size=100, chunk_overlap=0, model="gpt-4o")
    assert s.split_text("") == []


def test_long_text_splits_into_multiple_chunks():
    text = "the quick brown fox jumps over the lazy dog. " * 50
    s = TokenTextSplitter(chunk_size=20, chunk_overlap=0, model="gpt-4o")
    chunks = s.split_text(text)
    assert len(chunks) > 1


def test_zero_overlap_chunks_concatenate_to_source():
    text = "Lorem ipsum dolor sit amet. " * 30
    s = TokenTextSplitter(chunk_size=15, chunk_overlap=0, model="gpt-4o")
    chunks = s.split_text(text)
    joined = "".join(chunks)
    assert joined == text


def test_overlap_produces_at_least_as_many_chunks():
    text = "the quick brown fox jumps over the lazy dog " * 20
    no_overlap = TokenTextSplitter(chunk_size=15, chunk_overlap=0, model="gpt-4o").split_text(text)
    with_overlap = TokenTextSplitter(chunk_size=15, chunk_overlap=30, model="gpt-4o").split_text(text)
    assert len(with_overlap) >= len(no_overlap)


def test_split_documents_carries_metadata():
    text = "the quick brown fox jumps over the lazy dog. " * 10
    s = TokenTextSplitter(chunk_size=15, chunk_overlap=0, model="gpt-4o")
    chunks = s.split_documents([
        {"id": "src.txt", "content": text, "metadata": {"source": "src.txt"}}
    ])
    assert len(chunks) > 1
    for c in chunks:
        assert c["metadata"]["source"] == "src.txt"
        assert "chunk_index" in c["metadata"]


def test_unicode_text_splits_at_valid_utf8_boundaries():
    text = "hello 你好世界 🌍 lorem ipsum dolor sit amet " * 20
    s = TokenTextSplitter(chunk_size=10, chunk_overlap=0, model="gpt-4o")
    chunks = s.split_text(text)
    assert len(chunks) > 0
    # If any chunk had invalid UTF-8, this would have errored earlier.
    for c in chunks:
        # Each chunk decodes back to valid Python str.
        assert isinstance(c, str)


def test_different_models_both_work():
    text = "lorem ipsum " * 50
    openai = TokenTextSplitter(chunk_size=20, chunk_overlap=0, model="gpt-4o").split_text(text)
    anthropic = TokenTextSplitter(chunk_size=20, chunk_overlap=0,
                                  model="claude-opus-4-7").split_text(text)
    assert len(openai) > 0
    assert len(anthropic) > 0


def test_chunk_overlap_clamped_to_size_minus_one():
    """Constructor should clamp overlap >= chunk_size to chunk_size-1."""
    s = TokenTextSplitter(chunk_size=10, chunk_overlap=100, model="gpt-4o")
    r = repr(s)
    assert "chunk_overlap=9" in r


def test_repr_shows_config():
    s = TokenTextSplitter(chunk_size=512, chunk_overlap=50, model="gpt-4o")
    r = repr(s)
    assert "TokenTextSplitter" in r
    assert "chunk_size=512" in r
    assert "chunk_overlap=50" in r
    assert "gpt-4o" in r


def test_default_constructor():
    """Defaults should produce a sensible splitter."""
    s = TokenTextSplitter()
    chunks = s.split_text("hello")
    assert chunks == ["hello"]


if __name__ == "__main__":
    import traceback
    fns = [
        test_small_text_returns_one_chunk,
        test_empty_text_returns_empty,
        test_long_text_splits_into_multiple_chunks,
        test_zero_overlap_chunks_concatenate_to_source,
        test_overlap_produces_at_least_as_many_chunks,
        test_split_documents_carries_metadata,
        test_unicode_text_splits_at_valid_utf8_boundaries,
        test_different_models_both_work,
        test_chunk_overlap_clamped_to_size_minus_one,
        test_repr_shows_config,
        test_default_constructor,
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

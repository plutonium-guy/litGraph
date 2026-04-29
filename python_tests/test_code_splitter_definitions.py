"""CodeSplitter (iter 142) — definition-boundary code splitter for
Python/Rust/JS/TS/Go/Java/Cpp/Ruby/PHP. Distinct from
RecursiveCharacterSplitter.for_language: this one identifies whole
def/class/fn/impl boundaries and packs them, falling back to recursive
char split only for oversize single definitions."""
from litgraph.splitters import CodeSplitter


def test_python_groups_small_defs_into_one_chunk():
    src = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n"
    s = CodeSplitter(language="python", chunk_size=200, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) == 1
    assert "def a" in chunks[0]
    assert "def b" in chunks[0]
    assert "def c" in chunks[0]


def test_python_splits_large_input_at_def_boundaries():
    src = "def alpha():\n    return 1\n\ndef beta():\n    return 2\n\nclass Gamma:\n    def method(self):\n        return 3\n"
    s = CodeSplitter(language="python", chunk_size=40, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) >= 2
    for c in chunks:
        assert ("def " in c) or ("class " in c)


def test_rust_splits_at_fn_impl_struct():
    src = "fn one() -> i32 { 1 }\n\nimpl Foo {\n    fn two(&self) {}\n}\n\nstruct Bar;\n\nenum E { A }\n"
    s = CodeSplitter(language="rust", chunk_size=40, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    for kw in ["fn one", "impl Foo", "struct Bar", "enum E"]:
        assert kw in joined


def test_javascript_splits_at_function_class_const():
    src = "function alpha() { return 1; }\n\nclass Beta {}\n\nconst gamma = () => 2;\n\nexport function delta() {}\n"
    s = CodeSplitter(language="js", chunk_size=40, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) >= 2
    joined = "\n".join(chunks)
    for kw in ["function alpha", "class Beta", "gamma", "delta"]:
        assert kw in joined


def test_oversize_def_falls_back_to_recursive():
    body = "    print('x')\n" * 100
    src = f"def big():\n{body}"
    s = CodeSplitter(language="python", chunk_size=120, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) > 1


def test_no_definitions_falls_back_to_recursive():
    src = "x = 1\ny = 2\nprint(x + y)\n"
    s = CodeSplitter(language="python", chunk_size=200, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) == 1
    assert "print" in chunks[0]


def test_empty_input_returns_empty_list():
    s = CodeSplitter(language="python", chunk_size=100)
    assert s.split_text("") == []


def test_unknown_language_raises_value_error():
    try:
        CodeSplitter(language="brainfuck", chunk_size=100)
        raise AssertionError("expected ValueError")
    except ValueError as e:
        assert "brainfuck" in str(e)


def test_split_documents_preserves_metadata():
    src = "def a():\n    pass\n\ndef b():\n    pass\n"
    s = CodeSplitter(language="python", chunk_size=15, chunk_overlap=0)
    chunks = s.split_documents([
        {"id": "module.py", "content": src, "metadata": {"language": "python"}},
    ])
    assert len(chunks) >= 2
    for c in chunks:
        assert c["metadata"]["language"] == "python"
        assert c["metadata"]["source_id"] == "module.py"


def test_repr_shows_chunk_size_and_overlap():
    s = CodeSplitter(language="python", chunk_size=500, chunk_overlap=50)
    r = repr(s)
    assert "CodeSplitter" in r
    assert "chunk_size=500" in r
    assert "chunk_overlap=50" in r


def test_language_aliases_accepted():
    for lang in ["py", "rs", "js", "ts", "golang", "c++", "rb"]:
        s = CodeSplitter(language=lang, chunk_size=100)
        assert isinstance(repr(s), str)


if __name__ == "__main__":
    import traceback
    fns = [
        test_python_groups_small_defs_into_one_chunk,
        test_python_splits_large_input_at_def_boundaries,
        test_rust_splits_at_fn_impl_struct,
        test_javascript_splits_at_function_class_const,
        test_oversize_def_falls_back_to_recursive,
        test_no_definitions_falls_back_to_recursive,
        test_empty_input_returns_empty_list,
        test_unknown_language_raises_value_error,
        test_split_documents_preserves_metadata,
        test_repr_shows_chunk_size_and_overlap,
        test_language_aliases_accepted,
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

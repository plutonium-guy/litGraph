"""RecursiveCharacterSplitter.for_language(...) — language-aware code chunking."""
from litgraph.splitters import RecursiveCharacterSplitter


def test_python_splits_at_def_boundaries():
    src = "\n".join([
        "def foo():",
        "    return 1",
        "",
        "def bar():",
        "    return 2",
        "",
        "def baz():",
        "    return 3",
    ])
    s = RecursiveCharacterSplitter.for_language("python", chunk_size=40, chunk_overlap=0)
    chunks = s.split_text(src)
    assert len(chunks) >= 2
    assert any("def foo" in c for c in chunks)
    assert any("def baz" in c for c in chunks)


def test_rust_splits_at_fn_boundaries():
    src = "fn alpha() { 1 }\n\nfn beta() { 2 }\n\nfn gamma() { 3 }\n"
    s = RecursiveCharacterSplitter.for_language("rust", chunk_size=20, chunk_overlap=0)
    chunks = s.split_text(src)
    assert any("fn alpha" in c for c in chunks)
    assert any("fn gamma" in c for c in chunks)


def test_javascript_typescript_aliases_both_work():
    src = "function a() {}\nfunction b() {}\nfunction c() {}\n"
    js = RecursiveCharacterSplitter.for_language("javascript", chunk_size=20, chunk_overlap=0)
    ts = RecursiveCharacterSplitter.for_language("typescript", chunk_size=20, chunk_overlap=0)
    assert js.split_text(src) == ts.split_text(src)
    # Short alias also works.
    js_short = RecursiveCharacterSplitter.for_language("js", chunk_size=20, chunk_overlap=0)
    assert js.split_text(src) == js_short.split_text(src)


def test_go_java_cpp_ruby_php_constructible():
    """Smoke: every supported language constructs a splitter without error."""
    for lang in ["go", "java", "cpp", "c++", "ruby", "rb", "php"]:
        s = RecursiveCharacterSplitter.for_language(lang, chunk_size=100, chunk_overlap=0)
        chunks = s.split_text("some\nplain\ntext")
        assert isinstance(chunks, list)


def test_markdown_and_html_languages_work():
    md = "# Title\n\n## Sub\nbody1\n\n## Sub2\nbody2\n"
    s = RecursiveCharacterSplitter.for_language("markdown", chunk_size=20, chunk_overlap=0)
    chunks = s.split_text(md)
    assert len(chunks) >= 2

    html = "<body><div>one</div><div>two</div><div>three</div></body>"
    h = RecursiveCharacterSplitter.for_language("html", chunk_size=15, chunk_overlap=0)
    hchunks = h.split_text(html)
    assert len(hchunks) >= 2


def test_unknown_language_raises():
    try:
        RecursiveCharacterSplitter.for_language("haskell", chunk_size=100, chunk_overlap=0)
    except ValueError as e:
        assert "haskell" in str(e)
        assert "python" in str(e)  # error message lists supported langs
    else:
        raise AssertionError("expected ValueError")


def test_default_constructor_still_works():
    """Regression: adding for_language must not break the plain constructor."""
    s = RecursiveCharacterSplitter(chunk_size=50, chunk_overlap=10)
    chunks = s.split_text("paragraph one.\n\nparagraph two.\n\nparagraph three.")
    assert len(chunks) >= 1


if __name__ == "__main__":
    fns = [
        test_python_splits_at_def_boundaries,
        test_rust_splits_at_fn_boundaries,
        test_javascript_typescript_aliases_both_work,
        test_go_java_cpp_ruby_php_constructible,
        test_markdown_and_html_languages_work,
        test_unknown_language_raises,
        test_default_constructor_still_works,
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

"""HtmlHeaderSplitter — heading-aware structural chunking with breadcrumbs."""
from litgraph.splitters import HtmlHeaderSplitter


def test_splits_at_h1_h2_boundaries():
    html = """
        <html><body>
        <h1>Intro</h1>
        <p>Welcome to the doc.</p>
        <h2>Setup</h2>
        <p>Install the thing.</p>
        <h2>Usage</h2>
        <p>Run it like this.</p>
        </body></html>
    """
    s = HtmlHeaderSplitter(max_depth=3)
    chunks = s.split_text(html)
    assert len(chunks) == 3
    assert "Welcome" in chunks[0]
    assert "Install" in chunks[1]
    assert "Run it" in chunks[2]


def test_strip_headers_omits_heading_from_chunk():
    s = HtmlHeaderSplitter(max_depth=3, strip_headers=True)
    chunks = s.split_text("<h1>Title</h1><p>body content</p>")
    assert len(chunks) == 1
    assert "body content" in chunks[0]
    assert "Title" not in chunks[0]


def test_max_depth_keeps_lower_headings_inside_section():
    """max_depth=2 → h3 doesn't trigger a new chunk."""
    html = "<h2>Top</h2><p>a</p><h3>Sub</h3><p>b</p><h2>Next</h2><p>c</p>"
    s = HtmlHeaderSplitter(max_depth=2)
    chunks = s.split_text(html)
    assert len(chunks) == 2
    assert "Sub" in chunks[0] and "b" in chunks[0]


def test_split_documents_emits_breadcrumb_metadata():
    docs = [{
        "content": (
            "<h1>API</h1><p>top intro</p>"
            "<h2>Endpoints</h2><p>list</p>"
            "<h2>Auth</h2><p>auth</p>"
        ),
        "id": "api",
        "metadata": {"origin": "test"},
    }]
    s = HtmlHeaderSplitter(max_depth=3)
    out = s.split_documents(docs)
    assert len(out) == 3
    # h1 propagates through every chunk.
    for d in out:
        assert d["metadata"]["h1"] == "API"
        assert d["metadata"]["origin"] == "test"
        assert d["metadata"]["source_id"] == "api"
    # h2 changes per section.
    assert out[1]["metadata"]["h2"] == "Endpoints"
    assert out[2]["metadata"]["h2"] == "Auth"
    # docs_to_pylist round-trips numeric metadata as strings (existing helper).
    assert out[0]["metadata"]["chunk_index"] == "0"
    assert out[2]["metadata"]["chunk_index"] == "2"


def test_no_headings_returns_one_chunk():
    chunks = HtmlHeaderSplitter().split_text("<p>just paragraphs</p><p>more</p>")
    assert len(chunks) == 1
    assert "just paragraphs" in chunks[0]


def test_breadcrumb_resets_when_returning_up():
    docs = [{
        "content": "<h1>A</h1><p>x</p><h2>B</h2><p>y</p><h3>C</h3><p>z</p><h2>D</h2><p>w</p>",
        "id": "d",
    }]
    out = HtmlHeaderSplitter(max_depth=3).split_documents(docs)
    last = out[-1]
    assert last["metadata"]["h1"] == "A"
    assert last["metadata"]["h2"] == "D"
    # h3 from the prior section MUST have been dropped.
    assert "h3" not in last["metadata"]


if __name__ == "__main__":
    fns = [
        test_splits_at_h1_h2_boundaries,
        test_strip_headers_omits_heading_from_chunk,
        test_max_depth_keeps_lower_headings_inside_section,
        test_split_documents_emits_breadcrumb_metadata,
        test_no_headings_returns_one_chunk,
        test_breadcrumb_resets_when_returning_up,
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

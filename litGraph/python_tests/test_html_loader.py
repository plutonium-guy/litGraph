"""HtmlLoader: extract plaintext from HTML strings + files."""
import os
import tempfile

from litgraph.loaders import HtmlLoader


def test_html_string_strips_script_style_comments_and_extracts_title():
    html = """
        <html><head><title>Hi there</title>
        <style>body{color:red}</style>
        <script>alert('no')</script>
        </head><body>
        <!-- comment -->
        <p>Hello world</p>
        <p>Second paragraph.</p>
        </body></html>
    """
    docs = HtmlLoader(html=html).load()
    assert len(docs) == 1
    c = docs[0]["content"]
    assert "Hello world" in c
    assert "Second paragraph." in c
    assert "alert" not in c
    assert "color:red" not in c
    assert "comment" not in c
    assert docs[0]["metadata"]["title"] == "Hi there"


def test_html_strips_boilerplate_by_default():
    html = """<body><nav>menu menu</nav><header>brand</header>
              <p>Real content</p>
              <footer>copyright</footer></body>"""
    docs = HtmlLoader(html=html).load()
    c = docs[0]["content"]
    assert "Real content" in c
    assert "menu" not in c
    assert "brand" not in c
    assert "copyright" not in c


def test_html_keep_boilerplate_preserves_nav():
    html = "<body><nav>menu</nav><p>x</p></body>"
    docs = HtmlLoader(html=html, strip_boilerplate=False).load()
    assert "menu" in docs[0]["content"]


def test_html_decodes_entities():
    html = "<p>Tom &amp; Jerry &lt;3 &#65; &#x42;</p>"
    docs = HtmlLoader(html=html).load()
    assert docs[0]["content"] == "Tom & Jerry <3 A B"


def test_html_from_file():
    fd, path = tempfile.mkstemp(prefix="page-", suffix=".html")
    os.write(fd, b"<html><title>T</title><body><p>file body</p></body></html>")
    os.close(fd)
    try:
        docs = HtmlLoader(path=path).load()
        assert "file body" in docs[0]["content"]
        assert docs[0]["metadata"]["title"] == "T"
        assert docs[0]["metadata"]["source"] == path
    finally:
        os.unlink(path)


def test_html_loader_requires_exactly_one_input():
    try:
        HtmlLoader()
    except ValueError as e:
        assert "either `path` or `html`" in str(e)
    else:
        raise AssertionError("expected ValueError")
    try:
        HtmlLoader(path="/x", html="<p>y</p>")
    except ValueError as e:
        assert "exactly one" in str(e)
    else:
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [
        test_html_string_strips_script_style_comments_and_extracts_title,
        test_html_strips_boilerplate_by_default,
        test_html_keep_boilerplate_preserves_nav,
        test_html_decodes_entities,
        test_html_from_file,
        test_html_loader_requires_exactly_one_input,
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

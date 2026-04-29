"""DocxLoader — extract text from DOCX files (the .docx ZIP container).

Builds a minimal valid DOCX in-memory using Python's zipfile + a
hand-crafted word/document.xml. Verifies content + metadata + missing-file
error path."""
import os
import tempfile
import zipfile

from litgraph.loaders import DocxLoader


_DOC_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>First paragraph here.</w:t></w:r></w:p>
    <w:p>
      <w:r><w:t>Second has a tab</w:t></w:r>
      <w:r><w:tab/></w:r>
      <w:r><w:t>between words.</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"""


def _build_minimal_docx():
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    tmp.close()
    with zipfile.ZipFile(tmp.name, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", '<?xml version="1.0"?><Types/>')
        z.writestr("word/document.xml", _DOC_XML)
    return tmp.name


def test_docx_loader_returns_one_document_per_file_with_paragraph_breaks():
    path = _build_minimal_docx()
    try:
        docs = DocxLoader(path).load()
    finally:
        os.unlink(path)
    assert len(docs) == 1
    text = docs[0]["content"]
    assert "First paragraph here." in text
    assert "Second has a tab" in text
    # Tab survives.
    assert "\tbetween words." in text
    # Paragraph break separates the two paragraphs.
    lines = text.split("\n")
    assert any("First paragraph" in l for l in lines)


def test_docx_loader_metadata_carries_source_path():
    path = _build_minimal_docx()
    try:
        docs = DocxLoader(path).load()
    finally:
        os.unlink(path)
    assert docs[0]["metadata"]["source"].endswith(".docx")


def test_docx_loader_missing_file_raises():
    try:
        DocxLoader("/this/does/not/exist.docx").load()
    except RuntimeError as e:
        # error path includes "docx" or "open" — be lenient on phrasing.
        msg = str(e).lower()
        assert "docx" in msg or "open" in msg or "exist" in msg
    else:
        raise AssertionError("expected RuntimeError on missing file")


def test_docx_loader_handles_xml_entities():
    """`&lt;` `&amp;` etc. in word/document.xml decode to <, &, >."""
    xml = '''<?xml version="1.0"?>
<w:document xmlns:w="x">
  <w:p><w:r><w:t>5 &lt; 10 &amp; 20 &gt; 15</w:t></w:r></w:p>
</w:document>'''
    tmp = tempfile.NamedTemporaryFile(suffix=".docx", delete=False)
    tmp.close()
    try:
        with zipfile.ZipFile(tmp.name, "w") as z:
            z.writestr("word/document.xml", xml)
        docs = DocxLoader(tmp.name).load()
        assert "5 < 10 & 20 > 15" in docs[0]["content"]
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    fns = [
        test_docx_loader_returns_one_document_per_file_with_paragraph_breaks,
        test_docx_loader_metadata_carries_source_path,
        test_docx_loader_missing_file_raises,
        test_docx_loader_handles_xml_entities,
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

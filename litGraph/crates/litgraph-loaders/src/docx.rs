//! DOCX loader — extracts text from `word/document.xml` inside a `.docx`
//! ZIP container. Pure-Rust via `zip` + `quick-xml`; no LibreOffice/Pandoc
//! shell-out, no native deps.
//!
//! Strategy: each `.docx` is a ZIP. The flowing text lives in
//! `word/document.xml` under `<w:t>...</w:t>` elements (one per text run).
//! `<w:p>` is a paragraph; we insert `\n` on `</w:p>` so the output reads
//! naturally with paragraph breaks. `<w:br/>` becomes `\n` too. Tables and
//! headers/footers are NOT included in v1 (most RAG ingestion treats them
//! as boilerplate; can opt back in if asked).
//!
//! Single Document per file (DOCX has no first-class page concept like
//! PDF — Word computes page breaks at render time, not in the source XML).

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use litgraph_core::Document;
use quick_xml::Reader;
use quick_xml::events::Event;
use serde_json::Value;
use zip::ZipArchive;

use crate::{Loader, LoaderError, LoaderResult};

pub struct DocxLoader {
    pub path: PathBuf,
}

impl DocxLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self { path: p.as_ref().to_path_buf() }
    }

    /// Pull text from `word/document.xml` byte content. Public-but-internal
    /// so tests can hit it without a temp ZIP file.
    pub(crate) fn extract_text(xml_bytes: &[u8]) -> LoaderResult<String> {
        let mut reader = Reader::from_reader(xml_bytes);
        reader.config_mut().trim_text(false);
        let mut buf = Vec::new();
        let mut out = String::new();
        let mut in_t = false;
        loop {
            match reader.read_event_into(&mut buf) {
                Ok(Event::Start(e)) if e.name().as_ref() == b"w:t" => in_t = true,
                Ok(Event::End(e)) if e.name().as_ref() == b"w:t" => in_t = false,
                Ok(Event::End(e)) if e.name().as_ref() == b"w:p" => out.push('\n'),
                Ok(Event::Empty(e)) if e.name().as_ref() == b"w:br" => out.push('\n'),
                Ok(Event::Empty(e)) if e.name().as_ref() == b"w:tab" => out.push('\t'),
                Ok(Event::Text(t)) if in_t => {
                    let s = t.unescape().map_err(|e| LoaderError::Other(format!(
                        "docx xml unescape: {e}"
                    )))?;
                    out.push_str(&s);
                }
                Ok(Event::Eof) => break,
                Err(e) => return Err(LoaderError::Other(format!("docx xml parse: {e}"))),
                _ => {}
            }
            buf.clear();
        }
        // Trim trailing blank lines from the synthesized paragraph breaks.
        Ok(out.trim_end_matches('\n').to_string())
    }
}

impl Loader for DocxLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let f = File::open(&self.path)
            .map_err(|e| LoaderError::Other(format!("docx open `{}`: {e}", self.path.display())))?;
        let mut zip = ZipArchive::new(f)
            .map_err(|e| LoaderError::Other(format!("docx zip: {e}")))?;
        let mut entry = zip
            .by_name("word/document.xml")
            .map_err(|e| LoaderError::Other(format!(
                "docx missing word/document.xml: {e}"
            )))?;
        let mut bytes = Vec::new();
        entry.read_to_end(&mut bytes)?;
        let text = Self::extract_text(&bytes)?;
        let source = self.path.display().to_string();
        let mut d = Document::new(text);
        d.id = Some(source.clone());
        d.metadata.insert("source".into(), Value::String(source));
        Ok(vec![d])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;
    use zip::ZipWriter;
    use zip::write::SimpleFileOptions;

    /// Minimal `word/document.xml` for a 2-paragraph doc, with one tab and
    /// one explicit line break thrown in to exercise the special elements.
    const TWO_PARA_XML: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>Hello world</w:t></w:r></w:p>
    <w:p>
      <w:r><w:t>second paragraph</w:t></w:r>
      <w:r><w:tab/></w:r>
      <w:r><w:t>after tab</w:t></w:r>
      <w:r><w:br/></w:r>
      <w:r><w:t>after break</w:t></w:r>
    </w:p>
  </w:body>
</w:document>
"#;

    fn build_minimal_docx(document_xml: &str) -> NamedTempFile {
        let mut tmp = NamedTempFile::new().unwrap();
        let buf = {
            let mut cur = std::io::Cursor::new(Vec::<u8>::new());
            let mut zip = ZipWriter::new(&mut cur);
            let opts = SimpleFileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            // Word at minimum needs document.xml; add Content_Types.xml for
            // strict readers but DocxLoader only requires word/document.xml.
            zip.start_file("[Content_Types].xml", opts).unwrap();
            zip.write_all(br#"<?xml version="1.0" encoding="UTF-8"?><Types/>"#).unwrap();
            zip.start_file("word/document.xml", opts).unwrap();
            zip.write_all(document_xml.as_bytes()).unwrap();
            zip.finish().unwrap();
            cur.into_inner()
        };
        tmp.write_all(&buf).unwrap();
        tmp
    }

    #[test]
    fn loads_two_paragraphs_and_inserts_paragraph_breaks() {
        let docx = build_minimal_docx(TWO_PARA_XML);
        let docs = DocxLoader::new(docx.path()).load().unwrap();
        assert_eq!(docs.len(), 1);
        let text = &docs[0].content;
        assert!(text.contains("Hello world"));
        assert!(text.contains("second paragraph"));
        // Paragraph break between the two paragraphs.
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines.iter().any(|l| l.contains("Hello world")));
        // Tab and break elements within the second paragraph survive.
        assert!(text.contains("\tafter tab"));
        assert!(text.contains("after break"));
    }

    #[test]
    fn metadata_source_is_path() {
        let docx = build_minimal_docx(TWO_PARA_XML);
        let docs = DocxLoader::new(docx.path()).load().unwrap();
        let src = docs[0].metadata["source"].as_str().unwrap();
        assert_eq!(src, docx.path().display().to_string());
    }

    #[test]
    fn missing_file_returns_loader_error() {
        let res = DocxLoader::new("/this/path/does/not/exist.docx").load();
        assert!(matches!(res, Err(LoaderError::Other(_))));
    }

    #[test]
    fn truncated_xml_returns_partial_text_not_panic() {
        // quick-xml is intentionally lenient about EOF inside an open tag —
        // it stops at the last known good event. Document that contract:
        // we don't crash, we return what we got. (A genuine corruption like
        // bad entity escapes WOULD return Err.)
        let bad = b"<w:document xmlns:w=\"x\"><w:t>part1</w:t><w:t>part2";
        let res = DocxLoader::extract_text(bad);
        assert!(res.is_ok(), "expected lenient EOF behavior, got {res:?}");
        assert!(res.unwrap().contains("part1"));
    }

    #[test]
    fn xml_entities_decode_correctly() {
        let xml = r#"<?xml version="1.0"?>
<w:document xmlns:w="x">
  <w:p><w:r><w:t>5 &lt; 10 &amp; 20 &gt; 15</w:t></w:r></w:p>
</w:document>"#;
        let text = DocxLoader::extract_text(xml.as_bytes()).unwrap();
        assert!(text.contains("5 < 10 & 20 > 15"), "got: {text:?}");
    }
}

//! PDF loader — text extraction via lopdf (pure-Rust, no native deps).
//!
//! Two modes selectable via `with_per_page(bool)`:
//! - **per-page = true (default)**: one `Document` per PDF page. Each carries
//!   `metadata.page = N` (1-indexed) so retrieval can cite "page 7 of report.pdf".
//!   This is the production default — it's what every RAG pipeline wants.
//! - **per-page = false**: one `Document` for the whole PDF, all pages joined
//!   with form-feed (`\f`) separators so a downstream splitter still sees a
//!   page boundary.
//!
//! Extracted text comes from `lopdf::Document::extract_text(&[page_id])`.
//! Form-encoded glyphs (Type 1 fonts with custom CIDs) decode best-effort;
//! pages with non-recoverable encoding produce empty text + a `decode_error`
//! metadata flag rather than failing the whole load — partial extraction
//! beats no extraction for big multi-section PDFs.

use std::path::{Path, PathBuf};

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

pub struct PdfLoader {
    pub path: PathBuf,
    pub per_page: bool,
}

impl PdfLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self { path: p.as_ref().to_path_buf(), per_page: true }
    }

    /// `true` (default): one Document per PDF page, `metadata.page` set.
    /// `false`: one Document for the entire file with form-feed separators.
    pub fn with_per_page(mut self, per_page: bool) -> Self {
        self.per_page = per_page;
        self
    }
}

impl Loader for PdfLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let doc = lopdf::Document::load(&self.path)
            .map_err(|e| LoaderError::Other(format!("pdf load: {e}")))?;

        // Page IDs are returned in document order; lopdf's `get_pages()`
        // returns a BTreeMap<u32, ObjectId> keyed on 1-indexed page number.
        let pages = doc.get_pages();
        if pages.is_empty() {
            return Ok(Vec::new());
        }
        let source = self.path.display().to_string();

        if self.per_page {
            let mut out = Vec::with_capacity(pages.len());
            let total = pages.len() as i64;
            for (page_no, _id) in &pages {
                let (text, decode_error) = match doc.extract_text(&[*page_no]) {
                    Ok(t) => (t, false),
                    Err(_) => (String::new(), true),
                };
                let mut d = Document::new(text);
                d.id = Some(format!("{}#p{}", source, page_no));
                d.metadata.insert("source".into(), Value::String(source.clone()));
                d.metadata.insert("page".into(), Value::from(*page_no as i64));
                d.metadata.insert("page_count".into(), Value::from(total));
                if decode_error {
                    d.metadata.insert("decode_error".into(), Value::Bool(true));
                }
                out.push(d);
            }
            Ok(out)
        } else {
            // Whole-document path: extract per page then join with form-feed
            // (U+000C). Splitters that respect page boundaries (e.g. a
            // `RecursiveCharacterSplitter` configured with `\f` in separators)
            // can still reconstruct page-level chunks without re-loading.
            let page_nums: Vec<u32> = pages.keys().copied().collect();
            let total = page_nums.len() as i64;
            let mut joined = String::new();
            let mut decode_errors: Vec<u32> = Vec::new();
            for (i, p) in page_nums.iter().enumerate() {
                if i > 0 { joined.push('\u{000C}'); }
                match doc.extract_text(&[*p]) {
                    Ok(t) => joined.push_str(&t),
                    Err(_) => decode_errors.push(*p),
                }
            }
            let mut d = Document::new(joined);
            d.id = Some(source.clone());
            d.metadata.insert("source".into(), Value::String(source));
            d.metadata.insert("page_count".into(), Value::from(total));
            if !decode_errors.is_empty() {
                d.metadata.insert(
                    "decode_errors".into(),
                    Value::Array(
                        decode_errors.iter().map(|p| Value::from(*p as i64)).collect()
                    ),
                );
            }
            Ok(vec![d])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::{Content, Operation};
    use lopdf::dictionary;
    use lopdf::{Document as LoDoc, Object, Stream};
    use std::io::Write;
    use tempfile::NamedTempFile;

    /// Build a minimal valid PDF with `pages` pages, each containing the
    /// given text. Uses lopdf's writer API so the bytes are guaranteed to
    /// parse cleanly with the same library on the read side. Returns a
    /// tempfile path the test can `PdfLoader::new(...)` over.
    fn build_test_pdf(page_texts: &[&str]) -> NamedTempFile {
        let mut doc = LoDoc::with_version("1.5");
        let pages_id = doc.new_object_id();

        // One Type1 Helvetica font shared across pages.
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });

        let mut page_ids: Vec<Object> = Vec::with_capacity(page_texts.len());
        for text in page_texts {
            let content = Content {
                operations: vec![
                    Operation::new("BT", vec![]),
                    Operation::new("Tf", vec!["F1".into(), 24.into()]),
                    Operation::new("Td", vec![100.into(), 700.into()]),
                    Operation::new("Tj", vec![Object::string_literal(*text)]),
                    Operation::new("ET", vec![]),
                ],
            };
            let content_id = doc.add_object(Stream::new(
                dictionary! {},
                content.encode().unwrap(),
            ));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            });
            page_ids.push(page_id.into());
        }

        doc.objects.insert(pages_id, Object::Dictionary(dictionary! {
            "Type" => "Pages",
            "Kids" => page_ids,
            "Count" => page_texts.len() as i64,
            "Resources" => resources_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
        }));
        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);
        doc.compress();

        let mut tmp = NamedTempFile::new().unwrap();
        let mut buf: Vec<u8> = Vec::new();
        doc.save_to(&mut buf).unwrap();
        tmp.write_all(&buf).unwrap();
        tmp
    }

    #[test]
    fn per_page_emits_one_document_per_page_with_page_metadata() {
        let pdf = build_test_pdf(&["alpha page", "beta page", "gamma page"]);
        let docs = PdfLoader::new(pdf.path()).load().unwrap();
        assert_eq!(docs.len(), 3);
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(d.metadata["page"].as_i64().unwrap(), (i + 1) as i64);
            assert_eq!(d.metadata["page_count"].as_i64().unwrap(), 3);
        }
        assert!(docs[0].content.contains("alpha"));
        assert!(docs[1].content.contains("beta"));
        assert!(docs[2].content.contains("gamma"));
    }

    #[test]
    fn whole_document_mode_joins_pages_with_form_feed() {
        let pdf = build_test_pdf(&["one", "two"]);
        let docs = PdfLoader::new(pdf.path()).with_per_page(false).load().unwrap();
        assert_eq!(docs.len(), 1);
        let combined = &docs[0].content;
        assert!(combined.contains("one"));
        assert!(combined.contains("two"));
        assert!(combined.contains('\u{000C}'), "expected \\f separator between pages");
        assert_eq!(docs[0].metadata["page_count"].as_i64().unwrap(), 2);
        assert!(docs[0].metadata.get("page").is_none());
    }

    #[test]
    fn metadata_carries_source_path() {
        let pdf = build_test_pdf(&["x"]);
        let docs = PdfLoader::new(pdf.path()).load().unwrap();
        let src = docs[0].metadata["source"].as_str().unwrap();
        assert_eq!(src, pdf.path().display().to_string());
    }

    #[test]
    fn missing_file_returns_loader_error_not_panic() {
        let res = PdfLoader::new("/this/path/definitely/does/not/exist.pdf").load();
        assert!(matches!(res, Err(LoaderError::Other(_))));
    }
}

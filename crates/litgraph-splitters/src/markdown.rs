//! Markdown header-aware splitter — splits by ATX headers (`#`, `##`, `###`, …),
//! carrying a header breadcrumb into chunk metadata.

use litgraph_core::Document;

use crate::Splitter;

#[derive(Debug, Clone)]
pub struct MarkdownHeaderSplitter {
    /// Max header depth to split on (inclusive). E.g. 3 splits on `#`, `##`, `###`.
    pub max_depth: u8,
    /// Strip header lines from chunk content (they're preserved in metadata).
    pub strip_headers: bool,
}

impl Default for MarkdownHeaderSplitter {
    fn default() -> Self { Self { max_depth: 3, strip_headers: false } }
}

impl MarkdownHeaderSplitter {
    pub fn new(max_depth: u8) -> Self { Self { max_depth, strip_headers: false } }
    pub fn strip_headers(mut self, on: bool) -> Self { self.strip_headers = on; self }
}

fn header_level(line: &str) -> Option<(u8, &str)> {
    let trimmed = line.trim_start();
    let mut hashes = 0u8;
    for b in trimmed.as_bytes() {
        if *b == b'#' { hashes += 1; } else { break; }
        if hashes > 6 { return None; }
    }
    if hashes == 0 { return None; }
    let rest = trimmed[hashes as usize..].trim_start();
    if rest.is_empty() || !trimmed.as_bytes().get(hashes as usize).is_some_and(|b| *b == b' ') {
        return None;
    }
    Some((hashes, rest))
}

impl Splitter for MarkdownHeaderSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut cur = String::new();
        for line in text.lines() {
            match header_level(line) {
                Some((lvl, _)) if lvl <= self.max_depth => {
                    if !cur.trim().is_empty() { chunks.push(std::mem::take(&mut cur)); }
                    if !self.strip_headers {
                        cur.push_str(line);
                        cur.push('\n');
                    }
                }
                _ => {
                    cur.push_str(line);
                    cur.push('\n');
                }
            }
        }
        if !cur.trim().is_empty() { chunks.push(cur); }
        chunks
    }

    fn split_document(&self, doc: &Document) -> Vec<Document> {
        // Emit header breadcrumb per-chunk.
        let mut out = Vec::new();
        let mut breadcrumb: Vec<(u8, String)> = Vec::new();
        let mut cur = String::new();
        let mut chunk_idx = 0usize;
        let flush = |breadcrumb: &[(u8, String)], cur: &mut String, out: &mut Vec<Document>, idx: &mut usize| {
            if cur.trim().is_empty() { cur.clear(); return; }
            let mut d = Document::new(std::mem::take(cur));
            d.metadata = doc.metadata.clone();
            d.metadata.insert("chunk_index".into(), serde_json::json!(*idx));
            for (lvl, title) in breadcrumb {
                d.metadata.insert(format!("h{lvl}"), serde_json::json!(title));
            }
            if let Some(id) = &doc.id {
                d.id = Some(format!("{id}#{idx}"));
                d.metadata.insert("source_id".into(), serde_json::json!(id));
            }
            out.push(d);
            *idx += 1;
        };

        for line in doc.content.lines() {
            match header_level(line) {
                Some((lvl, title)) if lvl <= self.max_depth => {
                    flush(&breadcrumb, &mut cur, &mut out, &mut chunk_idx);
                    breadcrumb.retain(|(l, _)| *l < lvl);
                    breadcrumb.push((lvl, title.to_string()));
                    if !self.strip_headers {
                        cur.push_str(line);
                        cur.push('\n');
                    }
                }
                _ => {
                    cur.push_str(line);
                    cur.push('\n');
                }
            }
        }
        flush(&breadcrumb, &mut cur, &mut out, &mut chunk_idx);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_by_headers() {
        let md = "# Title\n\nintro\n\n## Sec A\n\nA body\n\n## Sec B\n\nB body";
        let s = MarkdownHeaderSplitter::new(2);
        let r = s.split_text(md);
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn carries_breadcrumb() {
        let md = "# Root\n\nroot body\n\n## Child\n\nchild body";
        let doc = Document::new(md).with_id("doc1");
        let s = MarkdownHeaderSplitter::new(2);
        let chunks = s.split_document(&doc);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[1].metadata.get("h1").unwrap(), &serde_json::json!("Root"));
        assert_eq!(chunks[1].metadata.get("h2").unwrap(), &serde_json::json!("Child"));
    }
}

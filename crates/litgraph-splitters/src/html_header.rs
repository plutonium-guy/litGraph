//! HTML header-aware splitter. Splits on `<h1>` … `<h6>` tags and emits one
//! Document per section, with a breadcrumb of ancestor headings in metadata.
//!
//! Mirrors `MarkdownHeaderSplitter` for HTML — for the same reason we kept
//! that one: structural splits beat character-window splits when the document
//! has clear headings (API docs, technical articles, scraped pages).
//!
//! Section content is plaintext: scripts/styles/comments are stripped, block
//! tags become newlines, remaining tags become whitespace. (We don't pull in
//! `litgraph-loaders::html` to avoid the cross-crate dep — minimal stripping
//! is duplicated here. If you want richer HTML cleanup, run
//! `HtmlLoader::from_string(...)` first and then split the plain output with
//! `MarkdownHeaderSplitter`.)

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::Splitter;

#[derive(Debug, Clone)]
pub struct HtmlHeaderSplitter {
    /// Max heading depth to split on (1..=6, inclusive). Default 3 = split on
    /// h1/h2/h3 only — h4+ stays inside their parent section.
    pub max_depth: u8,
    /// Strip the heading text from chunk content. Heading text is preserved
    /// in metadata regardless. Default false.
    pub strip_headers: bool,
}

impl Default for HtmlHeaderSplitter {
    fn default() -> Self { Self { max_depth: 3, strip_headers: false } }
}

impl HtmlHeaderSplitter {
    pub fn new(max_depth: u8) -> Self {
        Self { max_depth: max_depth.clamp(1, 6), strip_headers: false }
    }
    pub fn strip_headers(mut self, on: bool) -> Self { self.strip_headers = on; self }
}

static HEADING_RE: Lazy<Regex> = Lazy::new(|| {
    // Captures level (1..=6) + heading inner text (lazy across newlines).
    Regex::new(r"(?is)<h([1-6])\b[^>]*>(.*?)</h[1-6]\s*>").unwrap()
});
static SCRIPT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<script\b[^>]*>.*?</script\s*>").unwrap());
static STYLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<style\b[^>]*>.*?</style\s*>").unwrap());
static COMMENT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
static MULTI_BLANK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static MULTI_SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").unwrap());

/// Strip an HTML fragment to plaintext. Replaces block-level tags with
/// newlines so paragraph structure survives the tag removal.
fn html_to_text(s: &str) -> String {
    let mut t = SCRIPT_RE.replace_all(s, " ").into_owned();
    t = STYLE_RE.replace_all(&t, " ").into_owned();
    t = COMMENT_RE.replace_all(&t, " ").into_owned();
    // Convert block-level tags to newlines (preserves paragraph breaks).
    for tag in &["br", "p", "div", "li", "tr", "section", "article"] {
        let open = Regex::new(&format!(r"(?i)<{tag}\b[^>]*>")).unwrap();
        let close = Regex::new(&format!(r"(?i)</{tag}\s*>")).unwrap();
        t = open.replace_all(&t, "\n").into_owned();
        t = close.replace_all(&t, "\n").into_owned();
    }
    t = TAG_RE.replace_all(&t, " ").into_owned();
    // Decode common entities that survived tag stripping.
    t = t.replace("&amp;", "&")
         .replace("&lt;", "<")
         .replace("&gt;", ">")
         .replace("&quot;", "\"")
         .replace("&apos;", "'")
         .replace("&nbsp;", " ");
    // Collapse whitespace.
    t = MULTI_SPACE_RE.replace_all(&t, " ").into_owned();
    t = t.lines().map(|l| l.trim()).collect::<Vec<_>>().join("\n");
    t = MULTI_BLANK_RE.replace_all(&t, "\n\n").into_owned();
    t.trim().to_string()
}

#[derive(Debug, Clone)]
struct HeadingMatch {
    level: u8,
    text: String,
    /// Byte index of the heading's full match in the source HTML.
    full_start: usize,
    /// Byte index of the heading's full match end (after `</hN>`).
    full_end: usize,
}

fn find_headings(html: &str) -> Vec<HeadingMatch> {
    HEADING_RE.captures_iter(html)
        .filter_map(|c| {
            let level: u8 = c.get(1)?.as_str().parse().ok()?;
            let text = html_to_text(c.get(2)?.as_str());
            let m = c.get(0)?;
            Some(HeadingMatch {
                level,
                text,
                full_start: m.start(),
                full_end: m.end(),
            })
        })
        .collect()
}

impl Splitter for HtmlHeaderSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let headings = find_headings(text);
        // Only respect headings at or above the configured max_depth.
        let splits: Vec<&HeadingMatch> = headings.iter()
            .filter(|h| h.level <= self.max_depth)
            .collect();

        if splits.is_empty() {
            // No splittable headings → treat whole input as one chunk.
            let plain = html_to_text(text);
            return if plain.is_empty() { vec![] } else { vec![plain] };
        }

        let mut chunks = Vec::new();
        // Preamble: anything BEFORE the first split-eligible heading.
        let preamble = html_to_text(&text[..splits[0].full_start]);
        if !preamble.is_empty() {
            chunks.push(preamble);
        }

        for (i, head) in splits.iter().enumerate() {
            let body_start = if self.strip_headers { head.full_end } else { head.full_start };
            let body_end = splits.get(i + 1).map(|h| h.full_start).unwrap_or(text.len());
            let body = html_to_text(&text[body_start..body_end]);
            if !body.is_empty() {
                chunks.push(body);
            }
        }
        chunks
    }

    fn split_document(&self, doc: &Document) -> Vec<Document> {
        let html = &doc.content;
        let headings = find_headings(html);
        let splits: Vec<&HeadingMatch> = headings.iter()
            .filter(|h| h.level <= self.max_depth)
            .collect();

        if splits.is_empty() {
            // Single chunk; carry parent metadata + chunk_index = 0.
            let plain = html_to_text(html);
            if plain.is_empty() { return vec![]; }
            let mut d = Document::new(plain);
            d.metadata = doc.metadata.clone();
            d.metadata.insert("chunk_index".into(), serde_json::json!(0));
            if let Some(id) = &doc.id {
                d.id = Some(format!("{id}#0"));
                d.metadata.insert("source_id".into(), serde_json::json!(id));
            }
            return vec![d];
        }

        let mut out = Vec::new();
        let mut breadcrumb: Vec<(u8, String)> = Vec::new();
        let mut chunk_idx = 0usize;

        // Preamble (no heading in scope yet).
        let preamble = html_to_text(&html[..splits[0].full_start]);
        if !preamble.is_empty() {
            let mut d = Document::new(preamble);
            d.metadata = doc.metadata.clone();
            d.metadata.insert("chunk_index".into(), serde_json::json!(chunk_idx));
            if let Some(id) = &doc.id {
                d.id = Some(format!("{id}#{chunk_idx}"));
                d.metadata.insert("source_id".into(), serde_json::json!(id));
            }
            out.push(d);
            chunk_idx += 1;
        }

        for (i, head) in splits.iter().enumerate() {
            // Update breadcrumb: pop entries at >= this level, then push self.
            breadcrumb.retain(|(lvl, _)| *lvl < head.level);
            breadcrumb.push((head.level, head.text.clone()));

            let body_start = if self.strip_headers { head.full_end } else { head.full_start };
            let body_end = splits.get(i + 1).map(|h| h.full_start).unwrap_or(html.len());
            let body = html_to_text(&html[body_start..body_end]);
            if body.is_empty() { continue; }

            let mut d = Document::new(body);
            d.metadata = doc.metadata.clone();
            d.metadata.insert("chunk_index".into(), serde_json::json!(chunk_idx));
            for (lvl, title) in &breadcrumb {
                d.metadata.insert(format!("h{lvl}"), serde_json::json!(title));
            }
            if let Some(id) = &doc.id {
                d.id = Some(format!("{id}#{chunk_idx}"));
                d.metadata.insert("source_id".into(), serde_json::json!(id));
            }
            out.push(d);
            chunk_idx += 1;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_at_h1_h2_boundaries() {
        let html = r#"
            <html><body>
            <h1>Intro</h1>
            <p>Welcome to the doc.</p>
            <h2>Setup</h2>
            <p>Install the thing.</p>
            <h2>Usage</h2>
            <p>Run it like this.</p>
            </body></html>
        "#;
        let s = HtmlHeaderSplitter::new(3);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 3, "got: {chunks:?}");
        assert!(chunks[0].contains("Welcome"));
        assert!(chunks[1].contains("Install"));
        assert!(chunks[2].contains("Run it"));
    }

    #[test]
    fn preamble_before_first_heading_becomes_its_own_chunk() {
        let html = "<p>Free-floating intro.</p><h1>Real start</h1><p>Body.</p>";
        let s = HtmlHeaderSplitter::new(3);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("Free-floating"));
        assert!(chunks[1].contains("Real start") && chunks[1].contains("Body"));
    }

    #[test]
    fn deeper_headings_below_max_depth_stay_inside_section() {
        // max_depth=2 → h3 doesn't trigger a new chunk; it stays in the h2 section.
        let html = "<h2>Section</h2><p>top</p><h3>Sub</h3><p>nested</p><h2>Next</h2><p>after</p>";
        let s = HtmlHeaderSplitter::new(2);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 2);
        // First chunk holds the h3 + its body inline.
        assert!(chunks[0].contains("Sub") && chunks[0].contains("nested"));
        assert!(chunks[1].contains("after"));
    }

    #[test]
    fn strip_headers_removes_heading_from_body() {
        let html = "<h1>Title</h1><p>body content</p>";
        let s = HtmlHeaderSplitter::new(3).strip_headers(true);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("body content"));
        assert!(!chunks[0].contains("Title"), "heading text leaked: {:?}", chunks[0]);
    }

    #[test]
    fn split_document_emits_breadcrumb_metadata() {
        let html = r#"
            <h1>API</h1>
            <p>top-level intro</p>
            <h2>Endpoints</h2>
            <p>list of endpoints</p>
            <h2>Auth</h2>
            <p>auth details</p>
        "#;
        let doc = Document::new(html.to_string()).with_id("api-doc");
        let s = HtmlHeaderSplitter::new(3);
        let docs = s.split_document(&doc);
        assert_eq!(docs.len(), 3);
        // First chunk: just h1.
        assert_eq!(docs[0].metadata.get("h1").unwrap(), &serde_json::json!("API"));
        assert!(docs[0].metadata.get("h2").is_none());
        // Second chunk: h1 + h2.
        assert_eq!(docs[1].metadata.get("h1").unwrap(), &serde_json::json!("API"));
        assert_eq!(docs[1].metadata.get("h2").unwrap(), &serde_json::json!("Endpoints"));
        // Third chunk: h1 + UPDATED h2 (the prior Endpoints replaced).
        assert_eq!(docs[2].metadata.get("h2").unwrap(), &serde_json::json!("Auth"));
        // chunk_index + source_id propagated.
        for (i, d) in docs.iter().enumerate() {
            assert_eq!(d.metadata.get("chunk_index").unwrap(), &serde_json::json!(i));
            assert_eq!(d.metadata.get("source_id").unwrap(), &serde_json::json!("api-doc"));
            assert_eq!(d.id.as_deref().unwrap(), format!("api-doc#{i}"));
        }
    }

    #[test]
    fn no_headings_returns_single_chunk_with_plaintext() {
        let html = "<p>just a paragraph</p><p>and another</p>";
        let s = HtmlHeaderSplitter::new(3);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("just a paragraph") && chunks[0].contains("and another"));
    }

    #[test]
    fn scripts_and_styles_stripped_from_section_bodies() {
        let html = r#"
            <h1>T</h1>
            <p>visible</p>
            <style>body{color:red}</style>
            <script>alert('x')</script>
        "#;
        let s = HtmlHeaderSplitter::new(3);
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("visible"));
        assert!(!chunks[0].contains("color:red"));
        assert!(!chunks[0].contains("alert"));
    }

    #[test]
    fn breadcrumb_resets_when_returning_to_higher_level() {
        // h1 → h2 → h3 → h2 should drop the h3 entry from the breadcrumb.
        let html = r#"
            <h1>A</h1><p>x</p>
            <h2>B</h2><p>y</p>
            <h3>C</h3><p>z</p>
            <h2>D</h2><p>w</p>
        "#;
        let s = HtmlHeaderSplitter::new(3);
        let docs = s.split_document(&Document::new(html.to_string()).with_id("d"));
        assert_eq!(docs.len(), 4);
        // Last chunk is under D — should have h1=A, h2=D, NO h3.
        let last = docs.last().unwrap();
        assert_eq!(last.metadata.get("h1").unwrap(), &serde_json::json!("A"));
        assert_eq!(last.metadata.get("h2").unwrap(), &serde_json::json!("D"));
        assert!(last.metadata.get("h3").is_none(), "h3 should have been dropped");
    }

    #[test]
    fn max_depth_clamps_to_1_through_6() {
        assert_eq!(HtmlHeaderSplitter::new(0).max_depth, 1);
        assert_eq!(HtmlHeaderSplitter::new(99).max_depth, 6);
    }
}

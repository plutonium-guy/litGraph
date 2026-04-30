//! `HtmlSectionSplitter` — split HTML by semantic-block tags
//! (`<article>`, `<section>`, `<main>`, `<aside>`, `<nav>`,
//! `<header>`, `<footer>`).
//!
//! # Distinct from `HtmlHeaderSplitter`
//!
//! - **`HtmlHeaderSplitter`** (existing): splits by `<h1>`–`<h6>`
//!   heading boundaries. Right when content is delimited by
//!   headings; fails when sections lack headings or have
//!   non-hierarchical structure.
//! - **`HtmlSectionSplitter`** (this iter): splits by HTML5
//!   semantic-block tags. Right when the HTML follows the
//!   semantic-tag convention (modern blog templates, docs sites,
//!   articles published with HTML5 structure).
//!
//! # Algorithm
//!
//! Walk the input, find each opening + closing pair of the
//! configured block tags, emit the inner content (with nested
//! tags stripped to plain text) as one chunk. Content outside
//! any tracked block tag is collected as a single trailing
//! chunk so nothing is lost.
//!
//! Non-content tags (`<script>`, `<style>`, `<!-- -->`) are
//! stripped first — same convention as `HtmlHeaderSplitter`.
//!
//! # Caveats
//!
//! - Nested same-tag pairs (`<section><section>...</section></section>`)
//!   are flattened into the outer match. To split nested
//!   sections, run this splitter recursively.
//! - The splitter does not parse HTML strictly — uses a
//!   forgiving regex-based scanner. For arbitrary or adversarial
//!   HTML, run a real HTML loader first and feed the cleaned
//!   text to a structural splitter instead.

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;

use crate::Splitter;

/// Default HTML5 semantic block tags to split on. Matches the
/// HTML living standard's "sectioning content" + the layout-
/// landmark tags. Lowercased; matching is case-insensitive.
pub const DEFAULT_SECTION_TAGS: &[&str] = &[
    "article", "section", "main", "aside", "nav", "header", "footer",
];

#[derive(Debug, Clone)]
pub struct HtmlSectionSplitter {
    /// Tag names to treat as section boundaries. Case-insensitive.
    pub tags: Vec<String>,
    /// If true, content outside any section tag is dropped.
    /// Default false — outside content emits as a final chunk.
    pub drop_outside: bool,
}

impl Default for HtmlSectionSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl HtmlSectionSplitter {
    pub fn new() -> Self {
        Self {
            tags: DEFAULT_SECTION_TAGS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            drop_outside: false,
        }
    }

    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags
            .into_iter()
            .map(|s| s.into().to_ascii_lowercase())
            .collect();
        self
    }

    pub fn with_drop_outside(mut self, drop: bool) -> Self {
        self.drop_outside = drop;
        self
    }

    /// Splitter that returns each section as a [`Document`] with
    /// `tag` and `index` metadata. Mirrors the
    /// `HtmlHeaderSplitter` shape so callers swap freely.
    pub fn split_to_documents(&self, text: &str) -> Vec<Document> {
        let stripped = strip_non_content(text);
        let sections = collect_sections(&stripped, &self.tags);
        let mut docs = Vec::new();
        let mut last_end = 0;
        for (i, sec) in sections.iter().enumerate() {
            // Emit any "outside" text before this section.
            if !self.drop_outside && sec.start > last_end {
                let outside = &stripped[last_end..sec.start];
                let cleaned = strip_tags(outside);
                if !cleaned.trim().is_empty() {
                    let mut d = Document::new(cleaned).with_id(format!("outside_{i}"));
                    d.metadata.insert("kind".into(), serde_json::json!("outside"));
                    docs.push(d);
                }
            }
            // Emit the section itself.
            let inner = &stripped[sec.content_start..sec.content_end];
            let cleaned = strip_tags(inner);
            if !cleaned.trim().is_empty() {
                let mut d = Document::new(cleaned).with_id(format!("{}_{i}", sec.tag));
                d.metadata.insert("tag".into(), serde_json::json!(sec.tag));
                d.metadata.insert("section_index".into(), serde_json::json!(i));
                docs.push(d);
            }
            last_end = sec.end;
        }
        // Trailing outside content.
        if !self.drop_outside && last_end < stripped.len() {
            let outside = &stripped[last_end..];
            let cleaned = strip_tags(outside);
            if !cleaned.trim().is_empty() {
                let mut d = Document::new(cleaned).with_id("outside_tail".to_string());
                d.metadata.insert("kind".into(), serde_json::json!("outside"));
                docs.push(d);
            }
        }
        docs
    }
}

impl Splitter for HtmlSectionSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_to_documents(text)
            .into_iter()
            .map(|d| d.content)
            .collect()
    }
}

#[derive(Debug, Clone)]
struct SectionMatch {
    tag: String,
    /// Byte offset where the opening `<tag>` starts.
    start: usize,
    /// Byte offset just after the closing `</tag>`.
    end: usize,
    /// Byte offset just after the opening `<tag>`'s `>`.
    content_start: usize,
    /// Byte offset where the closing `</tag>` begins.
    content_end: usize,
}

fn collect_sections(html: &str, tags: &[String]) -> Vec<SectionMatch> {
    // For each tag, find balanced (outermost) matches. We walk
    // character by character to track nesting depth per tag.
    let mut sections: Vec<SectionMatch> = Vec::new();
    for tag in tags {
        let opener = format!("<{tag}");
        let closer = format!("</{tag}");
        let lower = html.to_ascii_lowercase();
        let mut i = 0usize;
        while i < lower.len() {
            if let Some(open_rel) = lower[i..].find(&opener) {
                let open_start = i + open_rel;
                // Confirm next char is `>` or whitespace (avoid matching
                // `<sectionish>` etc).
                let after_open = open_start + opener.len();
                if after_open >= lower.len() {
                    break;
                }
                let next_ch = lower.as_bytes()[after_open];
                if next_ch != b' '
                    && next_ch != b'\t'
                    && next_ch != b'\n'
                    && next_ch != b'\r'
                    && next_ch != b'>'
                    && next_ch != b'/'
                {
                    i = open_start + 1;
                    continue;
                }
                // Find the `>` that closes this opener tag.
                let Some(open_end_rel) = lower[after_open..].find('>') else {
                    break;
                };
                let content_start = after_open + open_end_rel + 1;
                // Walk forward tracking depth.
                let mut depth = 1;
                let mut j = content_start;
                let mut content_end = content_start;
                while j < lower.len() && depth > 0 {
                    let next_open = lower[j..].find(&opener).map(|p| j + p);
                    let next_close = lower[j..].find(&closer).map(|p| j + p);
                    match (next_open, next_close) {
                        (Some(o), Some(c)) if o < c => {
                            // Validate it's a real opener (next char OK).
                            let after = o + opener.len();
                            if after < lower.len() {
                                let nb = lower.as_bytes()[after];
                                if nb == b' '
                                    || nb == b'\t'
                                    || nb == b'\n'
                                    || nb == b'\r'
                                    || nb == b'>'
                                    || nb == b'/'
                                {
                                    depth += 1;
                                }
                            }
                            j = o + opener.len();
                        }
                        (_, Some(c)) => {
                            depth -= 1;
                            if depth == 0 {
                                content_end = c;
                                // Find closing `>` after `</tag`.
                                let after_c = c + closer.len();
                                if let Some(close_end_rel) = lower[after_c..].find('>') {
                                    let end = after_c + close_end_rel + 1;
                                    sections.push(SectionMatch {
                                        tag: tag.clone(),
                                        start: open_start,
                                        end,
                                        content_start,
                                        content_end,
                                    });
                                    i = end;
                                    break;
                                } else {
                                    break;
                                }
                            }
                            j = c + closer.len();
                        }
                        _ => break,
                    }
                }
                if depth != 0 {
                    // Unbalanced — bail on this tag.
                    break;
                }
            } else {
                break;
            }
        }
    }
    // Sort by start offset, drop fully-nested sections (only keep
    // the outermost of overlapping ranges).
    sections.sort_by_key(|s| s.start);
    let mut filtered: Vec<SectionMatch> = Vec::new();
    for s in sections {
        if let Some(last) = filtered.last() {
            if s.start < last.end {
                continue; // nested inside the previous outer match
            }
        }
        filtered.push(s);
    }
    filtered
}

static SCRIPT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<script[^>]*>.*?</script>|<style[^>]*>.*?</style>|<!--.*?-->")
        .expect("valid")
});

static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").expect("valid"));

fn strip_non_content(html: &str) -> String {
    SCRIPT_RE.replace_all(html, "").into_owned()
}

fn strip_tags(html: &str) -> String {
    let no_tags = TAG_RE.replace_all(html, " ");
    // Collapse whitespace.
    let mut out = String::with_capacity(no_tags.len());
    let mut prev_space = false;
    for c in no_tags.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_top_level_articles() {
        let html = r#"<html><body>
<article>First article body</article>
<article>Second article body</article>
</body></html>"#;
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        assert!(chunks.iter().any(|c| c.contains("First article body")));
        assert!(chunks.iter().any(|c| c.contains("Second article body")));
    }

    #[test]
    fn splits_section_tags() {
        let html =
            "<section>Section A content</section><section>Section B content</section>";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("Section A content"));
        assert!(chunks[1].contains("Section B content"));
    }

    #[test]
    fn nested_sections_treated_as_one() {
        // Outer + inner section; only the outer is split.
        let html = "<section>Outer <section>Inner content</section> end</section>";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("Outer"));
        assert!(chunks[0].contains("Inner content"));
    }

    #[test]
    fn outside_content_emitted_when_drop_outside_false() {
        let html =
            "Lead-in prose. <article>The article</article> Trailing prose.";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        // 3 chunks: lead-in, article, trailing.
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.contains("Lead-in prose")));
        assert!(chunks.iter().any(|c| c.contains("The article")));
        assert!(chunks.iter().any(|c| c.contains("Trailing prose")));
    }

    #[test]
    fn drop_outside_excludes_outside_content() {
        let html =
            "Lead-in prose. <article>The article</article> Trailing prose.";
        let s = HtmlSectionSplitter::new().with_drop_outside(true);
        let chunks = s.split_text(html);
        // Just the article.
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("The article"));
    }

    #[test]
    fn strips_scripts_and_styles() {
        let html = r#"<article>visible<script>document.write('hidden');</script><style>p{color:red}</style></article>"#;
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("visible"));
        assert!(!chunks[0].contains("hidden"));
        assert!(!chunks[0].contains("color:red"));
    }

    #[test]
    fn nested_inline_tags_become_whitespace() {
        let html = "<article>Text with <strong>bold</strong> and <em>italic</em>.</article>";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        let c = &chunks[0];
        assert!(c.contains("Text with"));
        assert!(c.contains("bold"));
        assert!(c.contains("italic"));
        assert!(!c.contains("<strong>"));
    }

    #[test]
    fn custom_tags_override_defaults() {
        let html = "<article>ignored</article><div class='post'>kept</div>";
        let s = HtmlSectionSplitter::new().with_tags(["div"]);
        let chunks = s.split_text(html);
        // Only `<div>` matches now; article content shows up as outside.
        assert!(chunks.iter().any(|c| c.contains("kept")));
    }

    #[test]
    fn split_to_documents_carries_metadata() {
        let html =
            "<article>One body</article><section>Two body</section>";
        let s = HtmlSectionSplitter::new();
        let docs = s.split_to_documents(html);
        let article = docs
            .iter()
            .find(|d| d.content.contains("One body"))
            .unwrap();
        assert_eq!(
            article.metadata.get("tag").and_then(|v| v.as_str()),
            Some("article"),
        );
        let section = docs
            .iter()
            .find(|d| d.content.contains("Two body"))
            .unwrap();
        assert_eq!(
            section.metadata.get("tag").and_then(|v| v.as_str()),
            Some("section"),
        );
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = HtmlSectionSplitter::new();
        assert!(s.split_text("").is_empty());
        assert!(s.split_text("<html></html>").is_empty());
    }

    #[test]
    fn case_insensitive_tag_match() {
        let html = "<ARTICLE>upper</ARTICLE><Article>mixed</Article>";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].contains("upper"));
        assert!(chunks[1].contains("mixed"));
    }

    #[test]
    fn unmatched_section_doesnt_panic() {
        // Opening tag with no closer — should not match (depth never balances).
        let html = "<article>orphaned section start";
        let s = HtmlSectionSplitter::new();
        let chunks = s.split_text(html);
        // Either drops the section or treats whole thing as outside.
        // We just assert no panic and we get at most something sensible.
        let _ = chunks;
    }
}

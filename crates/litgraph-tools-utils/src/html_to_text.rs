//! `HtmlToTextTool` — strip HTML to clean text for RAG ingestion,
//! summarization, or any downstream text analysis.
//!
//! # Why a dedicated tool
//!
//! `WebFetchTool` returns raw HTML. Most agent workflows that fetch
//! a page don't *want* the HTML — they want the readable content.
//! Without a tool, the LLM either:
//! - Tries to parse HTML in its head (poor at it; often hallucinates
//!   what's inside vs outside tags), OR
//! - Asks for a custom extractor tool that the user has to write.
//!
//! `HtmlToTextTool` does the standard "strip and clean" in one call
//! and returns plain text the agent can summarize, embed, or
//! quote directly.
//!
//! # What this is NOT
//!
//! This is NOT a full-fidelity HTML-to-text converter. It does NOT:
//!
//! - Render JavaScript-built content (no headless browser).
//! - Preserve table structure as ASCII tables.
//! - Reconstruct the visual hierarchy beyond block-level boundaries.
//! - Handle malformed HTML beyond what regex-based stripping permits.
//!
//! For full-fidelity rendering, callers should use a real headless
//! browser (out of scope here). This tool covers the 90% of
//! "fetched a static page, want the prose" workflows.
//!
//! # Why regex and not html5ever
//!
//! `html5ever` (used by the `scraper` crate) is the real-deal HTML5
//! parser — handles malformed input via the official spec error-
//! recovery rules. But it pulls in ~30 transitive deps. For the
//! "strip-and-clean" scope here, regex over a sequence of well-
//! defined patterns produces equivalent output on real-world HTML
//! at a fraction of the dep weight. Iter-293 `HtmlSectionSplitter`
//! already established this pattern in the splitters crate.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

/// Block-level tags that produce a newline boundary in the output.
/// `<br>` is treated as soft-newline; the others as hard paragraph
/// breaks. This is how an LLM-friendly-text representation should
/// look — paragraphs separated by blank lines, list items by single
/// newlines.
const BLOCK_TAGS_HARD_BREAK: &[&str] = &[
    "p", "div", "section", "article", "main", "aside", "header", "footer", "nav", "h1", "h2",
    "h3", "h4", "h5", "h6", "li", "tr", "blockquote", "pre",
];
const BLOCK_TAGS_SOFT_BREAK: &[&str] = &["br"];

static SCRIPT_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<script\b[^>]*>.*?</script>").expect("regex compiles")
});
static STYLE_BLOCK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<style\b[^>]*>.*?</style>").expect("regex compiles"));
static NOSCRIPT_BLOCK_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<noscript\b[^>]*>.*?</noscript>").expect("regex compiles")
});
static COMMENT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?s)<!--.*?-->").expect("regex compiles"));
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").expect("regex compiles"));
static MULTI_NEWLINE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\n{3,}").expect("regex compiles"));
static MULTI_SPACE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"[ \t]+").expect("regex compiles"));

#[derive(Debug, Clone, Default)]
pub struct HtmlToTextTool;

impl HtmlToTextTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for HtmlToTextTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "html_to_text".into(),
            description: "Strip HTML to clean readable text. Removes <script>/<style>/<noscript> \
                blocks and HTML comments entirely (their content is never user-facing); converts \
                block-level tags (p, div, h1-h6, li, ...) into newline boundaries; converts <br> \
                to soft newlines; decodes common HTML entities (&amp;, &lt;, &gt;, &quot;, &apos;, \
                &nbsp;, numeric &#NN; / &#xHH;). Collapses runs of horizontal whitespace and \
                excess blank lines. Use after `web_fetch` to feed clean text to summarization, \
                RAG, or any downstream analysis."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "html": {
                        "type": "string",
                        "description": "The HTML string to convert."
                    }
                },
                "required": ["html"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let html = args
            .get("html")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("html_to_text: missing `html`"))?;
        let text = html_to_text(html);
        Ok(json!({ "text": text, "char_count": text.chars().count() }))
    }
}

/// Convert HTML to plain text. Pure function exposed for tests +
/// callers wanting the conversion outside the Tool trait.
pub fn html_to_text(html: &str) -> String {
    // 1. Strip script/style/noscript blocks AND their contents entirely.
    let s = SCRIPT_BLOCK_RE.replace_all(html, "");
    let s = STYLE_BLOCK_RE.replace_all(&s, "");
    let s = NOSCRIPT_BLOCK_RE.replace_all(&s, "");
    // 2. Strip HTML comments.
    let s = COMMENT_RE.replace_all(&s, "");
    // 3. Convert block-level open/close tags into newline placeholders.
    //    Done before the generic tag strip so we preserve paragraph boundaries.
    let mut buf = String::with_capacity(s.len());
    let mut i = 0;
    let bytes = s.as_bytes();
    while i < bytes.len() {
        if bytes[i] == b'<' && looks_like_tag_start(&s[i..]) {
            // Find tag end.
            if let Some(end_off) = s[i..].find('>') {
                let end = i + end_off + 1;
                let tag = &s[i..end];
                let kind = classify_tag(tag);
                match kind {
                    TagKind::HardBreak => buf.push_str("\n\n"),
                    TagKind::SoftBreak => buf.push('\n'),
                    TagKind::Other => {
                        // Leave the tag in place — generic stripper will handle.
                        buf.push_str(tag);
                    }
                }
                i = end;
                continue;
            }
            // Unterminated < — emit literally.
            buf.push('<');
            i += 1;
        } else {
            buf.push(bytes[i] as char);
            i += 1;
        }
    }
    // 4. Strip remaining tags.
    let s = TAG_RE.replace_all(&buf, "");
    // 5. Decode HTML entities.
    let s = decode_entities(&s);
    // 6. Collapse whitespace.
    let s = MULTI_SPACE_RE.replace_all(&s, " ");
    // Strip per-line trailing whitespace.
    let lines: Vec<String> = s.lines().map(|l| l.trim_end().to_string()).collect();
    let joined = lines.join("\n");
    // Collapse 3+ newlines to 2.
    let s = MULTI_NEWLINE_RE.replace_all(&joined, "\n\n");
    s.trim().to_string()
}

/// `<` followed by a name-character, `/`, or `!` looks like a tag.
/// `<` followed by whitespace or end-of-string is a literal `<` —
/// e.g. "Price < 5 dollars". This guard prevents the tag-finder
/// from gobbling everything up to the next `>` somewhere far down
/// the document.
fn looks_like_tag_start(s: &str) -> bool {
    let mut chars = s.chars();
    let _ = chars.next(); // consume '<'
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '/' || c == '!' => true,
        _ => false,
    }
}

enum TagKind {
    HardBreak,
    SoftBreak,
    Other,
}

fn classify_tag(tag: &str) -> TagKind {
    // Strip < and > and any leading slash for closing tags.
    let inner = tag.trim_start_matches('<').trim_end_matches('>').trim();
    let inner = inner.strip_prefix('/').unwrap_or(inner);
    // Tag name = first run of [A-Za-z0-9] (stops at whitespace OR `/` for
    // self-closing tags like `<br/>`).
    let name_end = inner
        .find(|c: char| c.is_whitespace() || c == '/')
        .unwrap_or(inner.len());
    let name = inner[..name_end].to_lowercase();
    if BLOCK_TAGS_HARD_BREAK.contains(&name.as_str()) {
        TagKind::HardBreak
    } else if BLOCK_TAGS_SOFT_BREAK.contains(&name.as_str()) {
        TagKind::SoftBreak
    } else {
        TagKind::Other
    }
}

/// Decode the named HTML entities the LLM is most likely to encounter,
/// plus numeric `&#NN;` / `&#xHH;` references. Anything else is left
/// literal — better to leave a stray `&hellip;` than corrupt valid
/// content with an over-eager replacement.
fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut iter = s.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        if c != '&' {
            out.push(c);
            continue;
        }
        // Lookahead for the closing semicolon (within 8 chars — common
        // entities are short; longer ones we leave as-is).
        let rest = &s[i..];
        let Some(semi_off) = rest.find(';') else {
            out.push(c);
            continue;
        };
        if semi_off > 8 {
            out.push(c);
            continue;
        }
        let entity = &rest[..=semi_off];
        let decoded = match entity {
            "&amp;" => Some("&".to_string()),
            "&lt;" => Some("<".to_string()),
            "&gt;" => Some(">".to_string()),
            "&quot;" => Some("\"".to_string()),
            "&apos;" => Some("'".to_string()),
            "&nbsp;" => Some(" ".to_string()),
            _ if entity.starts_with("&#x") || entity.starts_with("&#X") => {
                let hex = &entity[3..entity.len() - 1];
                u32::from_str_radix(hex, 16)
                    .ok()
                    .and_then(char::from_u32)
                    .map(|c| c.to_string())
            }
            _ if entity.starts_with("&#") => {
                let dec = &entity[2..entity.len() - 1];
                dec.parse::<u32>()
                    .ok()
                    .and_then(char::from_u32)
                    .map(|c| c.to_string())
            }
            _ => None,
        };
        if let Some(s) = decoded {
            out.push_str(&s);
            // Advance the iterator past the consumed entity.
            for _ in 0..semi_off {
                iter.next();
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_script_and_style_blocks_with_contents() {
        let html = r#"<html><head><style>body{color:red;}</style></head>
<body><script>alert("xss")</script><p>Hello</p></body></html>"#;
        let out = html_to_text(html);
        assert!(!out.contains("alert"));
        assert!(!out.contains("color:red"));
        assert!(out.contains("Hello"));
    }

    #[test]
    fn strips_html_comments() {
        let html = "<!-- secret --><p>Visible</p><!-- another -->";
        let out = html_to_text(html);
        assert!(!out.contains("secret"));
        assert!(!out.contains("another"));
        assert!(out.contains("Visible"));
    }

    #[test]
    fn block_tags_produce_paragraph_boundaries() {
        let html = "<p>First paragraph.</p><p>Second paragraph.</p>";
        let out = html_to_text(html);
        // Two paragraphs separated by a blank line.
        assert!(out.contains("First paragraph.\n\nSecond paragraph."));
    }

    #[test]
    fn br_produces_soft_newline() {
        let html = "Line one<br>Line two<br/>Line three";
        let out = html_to_text(html);
        assert_eq!(out, "Line one\nLine two\nLine three");
    }

    #[test]
    fn list_items_separated_by_newlines() {
        let html = "<ul><li>First</li><li>Second</li><li>Third</li></ul>";
        let out = html_to_text(html);
        assert!(out.contains("First"));
        assert!(out.contains("Second"));
        assert!(out.contains("Third"));
        // Each li produces a hard break.
        assert!(out.contains("First\n\nSecond"));
    }

    #[test]
    fn headings_produce_paragraph_boundaries() {
        let html = "<h1>Title</h1><p>Body text.</p>";
        let out = html_to_text(html);
        assert!(out.contains("Title\n\nBody text."));
    }

    #[test]
    fn entities_decoded() {
        let html =
            "<p>Tom &amp; Jerry &lt;3 &quot;life&quot; &apos;s &nbsp;great&nbsp;</p>";
        let out = html_to_text(html);
        assert!(out.contains("Tom & Jerry"));
        assert!(out.contains("<3"));
        assert!(out.contains("\"life\""));
        assert!(out.contains("'s"));
        // Two consecutive nbsp collapsed to one space.
        assert!(out.contains(" great"));
        // Verify the double-space "  great" was collapsed to single.
        assert!(!out.contains("  great"));
    }

    #[test]
    fn numeric_entities_decoded() {
        // &#39; is apostrophe; &#x27; is also apostrophe (hex).
        let html = "<p>It&#39;s nice; &#x27;quoted&#x27;</p>";
        let out = html_to_text(html);
        assert!(out.contains("It's nice"));
        assert!(out.contains("'quoted'"));
    }

    #[test]
    fn unknown_entity_left_literal() {
        // We don't handle &hellip; — left as-is rather than mis-decoded.
        let html = "<p>End&hellip;</p>";
        let out = html_to_text(html);
        assert!(out.contains("&hellip;"));
    }

    #[test]
    fn multiple_blank_lines_collapsed() {
        let html = "<p>One</p><p></p><p></p><p>Two</p>";
        let out = html_to_text(html);
        // Empty <p>s would otherwise produce 4+ newlines; we cap at 2.
        let blank_runs: Vec<&str> = out.split('\n').filter(|l| !l.is_empty()).collect();
        assert_eq!(blank_runs, vec!["One", "Two"]);
        // Verify no triple-newline survives.
        assert!(!out.contains("\n\n\n"));
    }

    #[test]
    fn horizontal_whitespace_collapsed() {
        let html = "<p>Lots    of    spaces\t\there</p>";
        let out = html_to_text(html);
        assert_eq!(out, "Lots of spaces here");
    }

    #[test]
    fn preserves_inline_text_around_inline_tags() {
        let html = "<p>This is <strong>bold</strong> and <em>italic</em> text.</p>";
        let out = html_to_text(html);
        assert_eq!(out, "This is bold and italic text.");
    }

    #[test]
    fn tag_with_attributes_handled() {
        let html = r#"<p class="foo" id="bar" data-x="1">Text</p>"#;
        let out = html_to_text(html);
        assert_eq!(out, "Text");
    }

    #[test]
    fn nested_blocks_produce_clean_output() {
        let html = "<div><p>Outer.</p><p>Inner.</p></div>";
        let out = html_to_text(html);
        // Outer + Inner separated by blank line; div boundaries collapse.
        assert!(out.contains("Outer.\n\nInner."));
    }

    #[test]
    fn empty_html_returns_empty() {
        assert_eq!(html_to_text(""), "");
        assert_eq!(html_to_text("   "), "");
    }

    #[test]
    fn plain_text_passes_through() {
        let out = html_to_text("Just plain text, no tags.");
        assert_eq!(out, "Just plain text, no tags.");
    }

    #[tokio::test]
    async fn tool_run_returns_text_and_char_count() {
        let t = HtmlToTextTool::new();
        let v = t
            .run(json!({"html": "<p>Hello</p>"}))
            .await
            .unwrap();
        assert_eq!(v.get("text").and_then(|x| x.as_str()), Some("Hello"));
        assert_eq!(v.get("char_count").and_then(|x| x.as_i64()), Some(5));
    }

    #[tokio::test]
    async fn tool_run_missing_html_errors() {
        let t = HtmlToTextTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[test]
    fn unterminated_angle_bracket_left_literal() {
        // < without a matching > should not eat the rest of the doc.
        let html = "Price < 5 dollars <p>here</p>";
        let out = html_to_text(html);
        assert!(out.contains("Price < 5 dollars"));
        assert!(out.contains("here"));
    }
}

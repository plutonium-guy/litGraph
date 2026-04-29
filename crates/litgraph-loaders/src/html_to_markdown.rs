//! HTML → Markdown converter — turn raw HTML (e.g., from `WebLoader` or
//! `SitemapLoader`) into clean Markdown for LLM context. Markdown is more
//! token-efficient than HTML, preserves structure (headings, lists, links,
//! code blocks) that matters for RAG quality, and is what most LLMs were
//! pretrained heavily on.
//!
//! Comparison vs `HtmlLoader::strip_html`:
//! - `strip_html` reduces to plain text — drops link URLs, list markers,
//!   heading levels, emphasis. Good for full-text search; lossy for RAG.
//! - `html_to_markdown` preserves structural signal as markdown syntax.
//!
//! Implementation: regex-based replacement, same pattern as the rest of
//! `html.rs`. Pure-Rust, no `html5ever`/`scraper` dep (~50 transitive
//! crates avoided). Trades exact correctness on pathological inputs for
//! a tight dep tree. Handles real-world docs sites (Read the Docs, Hugo,
//! Jekyll, Sphinx, MkDocs) cleanly.
//!
//! Tag coverage:
//! - `<h1>`..`<h6>` → `#`..`######`
//! - `<p>` → blank-line-separated paragraph
//! - `<br>` → hard break (two trailing spaces + newline)
//! - `<strong>`, `<b>` → `**bold**`
//! - `<em>`, `<i>` → `*italic*`
//! - `<code>` → `` `code` ``
//! - `<pre>` → fenced code block (```)
//! - `<a href="URL">text</a>` → `[text](URL)`
//! - `<img src="URL" alt="ALT">` → `![ALT](URL)`
//! - `<ul>`/`<ol>`/`<li>` → `- item` / `1. item`
//! - `<blockquote>` → `> quote`
//! - `<hr>` → `---`
//! - `<script>`, `<style>`, `<!-- comments -->` → removed
//! - `<nav>`, `<header>`, `<footer>`, `<aside>` → removed (when
//!   `strip_boilerplate=true`, default)
//!
//! Limitations:
//! - Nested lists flatten to single-level (regex can't track depth).
//! - Tables become text rows; if you need GFM table syntax, post-process.
//! - Whitespace in `<pre>` is preserved on a best-effort basis.

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::html::decode_entities;

static SCRIPT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<script\b[^>]*>.*?</script\s*>").unwrap());
static STYLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<style\b[^>]*>.*?</style\s*>").unwrap());
static COMMENT_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)<!--.*?-->").unwrap());

static NAV_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<nav\b[^>]*>.*?</nav\s*>").unwrap());
static HEADER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<header\b[^>]*>.*?</header\s*>").unwrap());
static FOOTER_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<footer\b[^>]*>.*?</footer\s*>").unwrap());
static ASIDE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<aside\b[^>]*>.*?</aside\s*>").unwrap());

// `<pre>` block — extract verbatim before any other transform so internal
// indentation/newlines survive. Captured group preserves inner content.
static PRE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<pre\b[^>]*>(.*?)</pre\s*>").unwrap());

// Headings — one regex per level; loop in `html_to_markdown` walks 1..=6.
static H1_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h1\b[^>]*>(.*?)</h1\s*>").unwrap());
static H2_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h2\b[^>]*>(.*?)</h2\s*>").unwrap());
static H3_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h3\b[^>]*>(.*?)</h3\s*>").unwrap());
static H4_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h4\b[^>]*>(.*?)</h4\s*>").unwrap());
static H5_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h5\b[^>]*>(.*?)</h5\s*>").unwrap());
static H6_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<h6\b[^>]*>(.*?)</h6\s*>").unwrap());

fn heading_regex(level: usize) -> &'static Regex {
    match level {
        1 => &H1_RE,
        2 => &H2_RE,
        3 => &H3_RE,
        4 => &H4_RE,
        5 => &H5_RE,
        6 => &H6_RE,
        _ => unreachable!(),
    }
}

// Inline formatting.
static STRONG_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<(?:strong|b)\b[^>]*>(.*?)</(?:strong|b)\s*>").unwrap());
static EM_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<(?:em|i)\b[^>]*>(.*?)</(?:em|i)\s*>").unwrap());
static CODE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<code\b[^>]*>(.*?)</code\s*>").unwrap());

// Links + images. `href`/`src` may use ' or "; pre-empt by accepting either.
static A_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<a\b[^>]*?href\s*=\s*["']([^"']*)["'][^>]*>(.*?)</a\s*>"#).unwrap()
});
// Capture the entire `<img ...>` attribute string; src/alt extracted via
// independent sub-regexes inside the closure (alt may appear before OR
// after src in real-world HTML, so a single ordered pattern would miss
// half the cases).
static IMG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<img\b([^>]*)/?>").unwrap());
static IMG_SRC_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)src\s*=\s*["']([^"']*)["']"#).unwrap());
static IMG_ALT_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?i)alt\s*=\s*["']([^"']*)["']"#).unwrap());

// Lists.
static OL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<ol\b[^>]*>(.*?)</ol\s*>").unwrap());
static UL_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<ul\b[^>]*>(.*?)</ul\s*>").unwrap());
static LI_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<li\b[^>]*>(.*?)</li\s*>").unwrap());

// Blockquote, hr, br.
static BLOCKQUOTE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<blockquote\b[^>]*>(.*?)</blockquote\s*>").unwrap());
static HR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<hr\b[^>]*/?>").unwrap());
static BR_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<br\b[^>]*/?>").unwrap());
static P_OPEN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)<p\b[^>]*>").unwrap());
static P_CLOSE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)</p\s*>").unwrap());
static DIV_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)</?div\b[^>]*>").unwrap());
static SPAN_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)</?span\b[^>]*>").unwrap());

// Title (kept for metadata).
static TITLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<title\b[^>]*>(.*?)</title\s*>").unwrap());

// Catch-all: any tag that survived the above stages.
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());

// Whitespace cleanup.
static MULTI_BLANK_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static MULTI_SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").unwrap());

/// Sentinel format for stashed `<pre>` blocks. `\u{0}` is illegal in
/// well-formed HTML, so it's safe to use as a marker. The whitespace
/// normalizer steps are line- and run-based and will leave a sentinel
/// alone — which is the whole point: pre-block bodies must NOT be
/// touched by `MULTI_SPACE_RE` / `lines().trim_end()` (they'd lose
/// indentation + trailing-space hard breaks). After normalization we
/// substitute each sentinel with the full fenced markdown block.
fn pre_marker(idx: usize) -> String {
    format!("\u{0}PRE_BLOCK_{idx}\u{0}")
}

/// Convert HTML to Markdown. Pure function — same input always produces
/// the same output. See module docs for tag coverage and limitations.
///
/// `strip_boilerplate=true` removes `<nav>`, `<header>`, `<footer>`,
/// `<aside>` blocks (with their content). Mirrors the `HtmlLoader` flag.
pub fn html_to_markdown(input: &str, strip_boilerplate: bool) -> String {
    let mut s = input.to_string();

    // 1) Drop scripts / styles / comments — content + tags both gone.
    s = SCRIPT_RE.replace_all(&s, "").into_owned();
    s = STYLE_RE.replace_all(&s, "").into_owned();
    s = COMMENT_RE.replace_all(&s, "").into_owned();

    // 2) Drop boilerplate sections if requested.
    if strip_boilerplate {
        s = NAV_RE.replace_all(&s, "").into_owned();
        s = HEADER_RE.replace_all(&s, "").into_owned();
        s = FOOTER_RE.replace_all(&s, "").into_owned();
        s = ASIDE_RE.replace_all(&s, "").into_owned();
    }

    // 3) Stash <pre> blocks under sentinel markers. The fenced markdown
    // gets restored AFTER all transform + whitespace passes, so internal
    // indentation / trailing spaces inside the code block survive intact.
    let mut pre_blocks: Vec<String> = Vec::new();
    s = PRE_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let body = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            // Strip a single leading <code> wrapper if present (common
            // GitHub/highlight.js pattern: <pre><code>...</code></pre>).
            let body = strip_leading_code_tag(body);
            let body = decode_entities(&body);
            let fenced = format!("```\n{}\n```", body.trim_end_matches('\n').trim_end_matches(' '));
            let idx = pre_blocks.len();
            pre_blocks.push(fenced);
            // Surround the marker with blank lines so it lands as its own
            // block paragraph regardless of surrounding tags.
            format!("\n\n{}\n\n", pre_marker(idx))
        })
        .into_owned();

    // 4) Headings — process h1..h6 in order.
    for level in 1..=6 {
        let prefix = "#".repeat(level);
        s = heading_regex(level)
            .replace_all(&s, |caps: &regex::Captures| {
                let inner = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
                let inner = strip_inline_tags(inner);
                format!("\n\n{prefix} {inner}\n\n")
            })
            .into_owned();
    }

    // 5) Lists — render <ol>/<ul> bodies before any other tag transforms
    // get to <li>. Numbering for <ol> resets per block.
    s = OL_RE
        .replace_all(&s, |caps: &regex::Captures| render_list(caps.get(1).map(|m| m.as_str()).unwrap_or(""), true))
        .into_owned();
    s = UL_RE
        .replace_all(&s, |caps: &regex::Captures| render_list(caps.get(1).map(|m| m.as_str()).unwrap_or(""), false))
        .into_owned();

    // 6) Inline formatting (innermost-first via lazy regex).
    s = STRONG_RE
        .replace_all(&s, |caps: &regex::Captures| {
            format!("**{}**", caps.get(1).map(|m| m.as_str()).unwrap_or("").trim())
        })
        .into_owned();
    s = EM_RE
        .replace_all(&s, |caps: &regex::Captures| {
            format!("*{}*", caps.get(1).map(|m| m.as_str()).unwrap_or("").trim())
        })
        .into_owned();
    s = CODE_RE
        .replace_all(&s, |caps: &regex::Captures| {
            format!("`{}`", caps.get(1).map(|m| m.as_str()).unwrap_or(""))
        })
        .into_owned();

    // 7) Links + images. Empty link text falls back to the URL itself
    // (using `[url](url)` rather than `<url>` autolink — angle-bracket
    // autolinks would be eaten by the catch-all tag stripper below).
    s = A_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let href = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let text = caps.get(2).map(|m| m.as_str()).unwrap_or("").trim();
            let text = strip_inline_tags(text);
            let label = if text.is_empty() { href.to_string() } else { text };
            format!("[{label}]({href})")
        })
        .into_owned();
    s = IMG_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let attrs = caps.get(1).map(|m| m.as_str()).unwrap_or("");
            let src = IMG_SRC_RE
                .captures(attrs)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");
            let alt = IMG_ALT_RE
                .captures(attrs)
                .and_then(|c| c.get(1))
                .map(|m| m.as_str())
                .unwrap_or("");
            format!("![{alt}]({src})")
        })
        .into_owned();

    // 8) Blockquote — prefix every non-empty line with `> `.
    s = BLOCKQUOTE_RE
        .replace_all(&s, |caps: &regex::Captures| {
            let inner = caps.get(1).map(|m| m.as_str()).unwrap_or("").trim();
            let prefixed = inner
                .lines()
                .map(|l| {
                    let t = l.trim();
                    if t.is_empty() { String::new() } else { format!("> {t}") }
                })
                .collect::<Vec<_>>()
                .join("\n");
            format!("\n\n{prefixed}\n\n")
        })
        .into_owned();

    // 9) Hr, br, p. Hard break uses `\\\n` (CommonMark backslash form)
    // rather than the trailing-two-spaces form, because the per-line
    // `trim_end()` in step 14 would eat the trailing spaces.
    s = HR_RE.replace_all(&s, "\n\n---\n\n").into_owned();
    s = BR_RE.replace_all(&s, "\\\n").into_owned();
    s = P_OPEN_RE.replace_all(&s, "\n\n").into_owned();
    s = P_CLOSE_RE.replace_all(&s, "\n\n").into_owned();
    s = DIV_RE.replace_all(&s, "\n").into_owned();
    s = SPAN_RE.replace_all(&s, "").into_owned();

    // 10) Drop title (already extracted as metadata).
    s = TITLE_RE.replace_all(&s, "").into_owned();

    // 11) Strip any remaining unknown tags.
    s = TAG_RE.replace_all(&s, "").into_owned();

    // 12) Decode entities (after tag stripping so e.g. `&lt;` doesn't get
    // re-interpreted as a tag start). Sentinel markers contain no
    // entities so this is a no-op for them.
    s = decode_entities(&s);

    // 13) Whitespace normalization. Sentinels survive since they contain
    // no whitespace runs.
    s = MULTI_SPACE_RE.replace_all(&s, " ").into_owned();
    s = s
        .lines()
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n");
    s = MULTI_BLANK_RE.replace_all(&s, "\n\n").into_owned();

    // 14) Restore <pre> blocks last — fenced code with original
    // whitespace + trailing-space hard breaks intact.
    for (idx, fenced) in pre_blocks.iter().enumerate() {
        s = s.replace(&pre_marker(idx), fenced);
    }

    s.trim().to_string()
}

/// Render a `<ol>` or `<ul>` body to markdown bullets. Strips inline tags
/// from each `<li>` body and decodes entities.
fn render_list(body: &str, ordered: bool) -> String {
    let mut out = String::from("\n\n");
    let mut idx = 1;
    for cap in LI_RE.captures_iter(body) {
        let inner = cap.get(1).map(|m| m.as_str()).unwrap_or("").trim();
        let inner = strip_inline_tags(inner);
        if inner.is_empty() {
            continue;
        }
        if ordered {
            out.push_str(&format!("{idx}. {inner}\n"));
            idx += 1;
        } else {
            out.push_str(&format!("- {inner}\n"));
        }
    }
    out.push('\n');
    out
}

/// Drop inline tags that markdown can't represent (e.g. `<span>`) but
/// keep the content. Used inside heading / link / list contexts where we
/// don't want HTML spans surviving into the output.
fn strip_inline_tags(s: &str) -> String {
    let cleaned = TAG_RE.replace_all(s, "").into_owned();
    cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// `<pre><code>...</code></pre>` is the standard HTML pattern for code
/// blocks. Strip the inner `<code>` wrapper so the verbatim body of the
/// pre block doesn't get backtick-wrapped twice.
fn strip_leading_code_tag(body: &str) -> String {
    let trimmed = body.trim_start();
    if trimmed.starts_with("<code") {
        if let Some(open_end) = trimmed.find('>') {
            let after_open = &trimmed[open_end + 1..];
            // Match a closing </code> at the end (allow trailing whitespace).
            let lower = after_open.to_ascii_lowercase();
            if let Some(close_pos) = lower.rfind("</code>") {
                let inner = &after_open[..close_pos];
                return inner.to_string();
            }
        }
    }
    body.to_string()
}

/// Extract `<title>` content (entity-decoded, whitespace-collapsed). Used
/// by `HtmlToMarkdownTransformer` to preserve title metadata.
pub fn extract_title(html: &str) -> Option<String> {
    TITLE_RE.captures(html).and_then(|c| c.get(1)).map(|m| {
        let raw = m.as_str();
        let cleaned = raw.split_whitespace().collect::<Vec<_>>().join(" ");
        decode_entities(&cleaned)
    })
}

/// Document transformer that converts each input doc's `content` from
/// HTML to Markdown. Use as the bridge between web/sitemap loaders and
/// downstream splitters/embedders.
///
/// ```ignore
/// let docs = WebLoader::new(url).load()?;            // raw HTML
/// let docs = HtmlToMarkdownTransformer::new()
///     .transform(docs);                               // clean markdown
/// let chunks = MarkdownHeaderSplitter::new(...)
///     .split_documents(&docs);                        // structure-aware chunks
/// ```
#[derive(Clone, Debug)]
pub struct HtmlToMarkdownTransformer {
    pub strip_boilerplate: bool,
    /// When `true` (default), preserves the `source` metadata key from the
    /// input doc. When `false`, drops all metadata.
    pub keep_metadata: bool,
}

impl Default for HtmlToMarkdownTransformer {
    fn default() -> Self {
        Self { strip_boilerplate: true, keep_metadata: true }
    }
}

impl HtmlToMarkdownTransformer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn keep_boilerplate(mut self) -> Self {
        self.strip_boilerplate = false;
        self
    }

    pub fn drop_metadata(mut self) -> Self {
        self.keep_metadata = false;
        self
    }

    /// Convert one document's HTML content to Markdown. If the source
    /// HTML has a `<title>`, its (entity-decoded) value is added under
    /// the `title` metadata key.
    pub fn transform_one(&self, doc: Document) -> Document {
        let title = extract_title(&doc.content);
        let md = html_to_markdown(&doc.content, self.strip_boilerplate);
        let mut out = if self.keep_metadata {
            let mut d = Document::new(md);
            d.id = doc.id;
            d.metadata = doc.metadata;
            d
        } else {
            let mut d = Document::new(md);
            d.id = doc.id;
            d
        };
        if let Some(t) = title {
            if !t.is_empty() {
                out.metadata.insert("title".into(), Value::String(t));
            }
        }
        out
    }

    pub fn transform(&self, docs: Vec<Document>) -> Vec<Document> {
        docs.into_iter().map(|d| self.transform_one(d)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn headings_become_pound_signs() {
        let html = "<h1>Title</h1><h2>Sub</h2><h3>Deeper</h3>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("# Title"));
        assert!(md.contains("## Sub"));
        assert!(md.contains("### Deeper"));
    }

    #[test]
    fn paragraphs_separated_by_blank_lines() {
        let html = "<p>First.</p><p>Second.</p>";
        let md = html_to_markdown(html, true);
        // Paragraphs separated by exactly one blank line.
        assert_eq!(md, "First.\n\nSecond.");
    }

    #[test]
    fn unordered_list_renders_bullets() {
        let html = "<ul><li>alpha</li><li>beta</li><li>gamma</li></ul>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("- alpha"));
        assert!(md.contains("- beta"));
        assert!(md.contains("- gamma"));
    }

    #[test]
    fn ordered_list_numbers_from_one() {
        let html = "<ol><li>first</li><li>second</li><li>third</li></ol>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("1. first"));
        assert!(md.contains("2. second"));
        assert!(md.contains("3. third"));
    }

    #[test]
    fn link_with_href_becomes_bracket_paren() {
        let html = r#"<p>see <a href="https://example.com">docs</a> for more</p>"#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("[docs](https://example.com)"));
    }

    #[test]
    fn link_without_text_falls_back_to_url_label() {
        // Empty <a> body → use the URL itself as label so the resulting
        // markdown isn't `<url>` autolink (would collide with the
        // catch-all tag stripper).
        let html = r#"<a href="https://x.com"></a>"#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("[https://x.com](https://x.com)"));
    }

    #[test]
    fn img_becomes_markdown_image() {
        let html = r#"<img src="https://x.com/cat.jpg" alt="A cat">"#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("![A cat](https://x.com/cat.jpg)"));
    }

    #[test]
    fn img_without_alt_renders_empty_alt() {
        let html = r#"<img src="x.png">"#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("![](x.png)"));
    }

    #[test]
    fn bold_and_italic_use_markdown_emphasis() {
        let html = "<p>This is <strong>bold</strong> and <em>italic</em></p>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("**bold**"));
        assert!(md.contains("*italic*"));
    }

    #[test]
    fn b_and_i_aliases_also_supported() {
        let html = "<p><b>hard</b> and <i>soft</i></p>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("**hard**"));
        assert!(md.contains("*soft*"));
    }

    #[test]
    fn inline_code_uses_backticks() {
        let html = "<p>Run <code>cargo build</code> first.</p>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("`cargo build`"));
    }

    #[test]
    fn pre_block_becomes_fenced_code() {
        let html = "<pre>let x = 1;\nlet y = 2;</pre>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("```\nlet x = 1;\nlet y = 2;\n```"));
    }

    #[test]
    fn pre_with_inner_code_strips_wrapper() {
        let html = "<pre><code>fn main() { println!(\"hi\"); }</code></pre>";
        let md = html_to_markdown(html, true);
        // No double-backticks around the code body.
        assert!(md.contains("```\nfn main() { println!(\"hi\"); }\n```"));
        assert!(!md.contains("`fn main"));
    }

    #[test]
    fn pre_block_preserves_internal_whitespace() {
        let html = "<pre>line1\n    indented\n        deep</pre>";
        let md = html_to_markdown(html, true);
        // The indentation must survive intact (no whitespace collapse).
        assert!(md.contains("line1\n    indented\n        deep"));
    }

    #[test]
    fn blockquote_prefixes_lines() {
        let html = "<blockquote>To be or not to be.</blockquote>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("> To be or not to be."));
    }

    #[test]
    fn hr_becomes_three_dashes() {
        let html = "<p>before</p><hr><p>after</p>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("---"));
    }

    #[test]
    fn br_becomes_hard_break() {
        let html = "<p>line1<br>line2</p>";
        let md = html_to_markdown(html, true);
        // CommonMark backslash-newline form. (The trailing-two-spaces
        // form would be eaten by the per-line trim_end pass.)
        assert!(md.contains("line1\\\nline2"));
    }

    #[test]
    fn script_and_style_blocks_are_removed_with_content() {
        let html = r#"
            <script>alert('x')</script>
            <style>body{}</style>
            <p>visible</p>
        "#;
        let md = html_to_markdown(html, true);
        assert!(!md.contains("alert"));
        assert!(!md.contains("body{}"));
        assert!(md.contains("visible"));
    }

    #[test]
    fn comments_are_removed() {
        let html = "<p>hello</p><!-- secret --><p>world</p>";
        let md = html_to_markdown(html, true);
        assert!(!md.contains("secret"));
        assert!(md.contains("hello"));
        assert!(md.contains("world"));
    }

    #[test]
    fn boilerplate_stripped_by_default() {
        let html = r#"
            <nav>menu</nav>
            <header>brand</header>
            <p>real content</p>
            <footer>copyright</footer>
        "#;
        let md = html_to_markdown(html, true);
        assert!(!md.contains("menu"));
        assert!(!md.contains("brand"));
        assert!(!md.contains("copyright"));
        assert!(md.contains("real content"));
    }

    #[test]
    fn keep_boilerplate_preserves_nav_etc() {
        let html = "<nav>menu</nav><p>x</p><footer>c</footer>";
        let md = html_to_markdown(html, false);
        assert!(md.contains("menu"));
        assert!(md.contains("c"));
    }

    #[test]
    fn html_entities_decoded_in_output() {
        let html = "<p>Tom &amp; Jerry &lt;3 &#65;</p>";
        let md = html_to_markdown(html, true);
        assert!(md.contains("Tom & Jerry <3 A"));
    }

    #[test]
    fn nested_inline_tags_inside_link_text_stripped() {
        let html = r#"<a href="/x"><span>label</span></a>"#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("[label](/x)"));
    }

    #[test]
    fn extract_title_returns_decoded_title() {
        let html = "<title>Hello &amp; world</title><p>x</p>";
        assert_eq!(extract_title(html).as_deref(), Some("Hello & world"));
    }

    #[test]
    fn extract_title_returns_none_when_missing() {
        let html = "<p>no title here</p>";
        assert!(extract_title(html).is_none());
    }

    #[test]
    fn transformer_preserves_metadata_and_adds_title() {
        let mut d = Document::new("<title>My Page</title><h1>Hi</h1><p>body</p>");
        d.metadata.insert("source".into(), Value::String("https://x.com".into()));
        let out = HtmlToMarkdownTransformer::new().transform_one(d);
        assert!(out.content.contains("# Hi"));
        assert!(out.content.contains("body"));
        assert_eq!(out.metadata.get("source").unwrap().as_str().unwrap(), "https://x.com");
        assert_eq!(out.metadata.get("title").unwrap().as_str().unwrap(), "My Page");
    }

    #[test]
    fn transformer_drop_metadata_clears_existing_keys() {
        let mut d = Document::new("<p>x</p>");
        d.metadata.insert("source".into(), Value::String("kept-original".into()));
        let out = HtmlToMarkdownTransformer::new().drop_metadata().transform_one(d);
        assert!(out.metadata.get("source").is_none());
    }

    #[test]
    fn transform_batch_processes_all_docs() {
        let docs = vec![
            Document::new("<h1>One</h1>"),
            Document::new("<h1>Two</h1>"),
            Document::new("<h1>Three</h1>"),
        ];
        let out = HtmlToMarkdownTransformer::new().transform(docs);
        assert_eq!(out.len(), 3);
        assert!(out[0].content.contains("# One"));
        assert!(out[1].content.contains("# Two"));
        assert!(out[2].content.contains("# Three"));
    }

    #[test]
    fn realistic_doc_page_renders_cleanly() {
        // Resembles a Read the Docs / MkDocs page.
        let html = r#"
            <html>
            <head><title>Quickstart</title></head>
            <body>
                <nav>Menu items</nav>
                <header>Site brand</header>
                <article>
                    <h1>Quickstart</h1>
                    <p>Install the package:</p>
                    <pre><code>pip install litgraph</code></pre>
                    <p>Then <a href="/api">read the API docs</a>.</p>
                    <h2>Features</h2>
                    <ul>
                        <li>Fast</li>
                        <li>Typed</li>
                        <li>Free</li>
                    </ul>
                </article>
                <footer>© 2026</footer>
            </body>
            </html>
        "#;
        let md = html_to_markdown(html, true);
        assert!(md.contains("# Quickstart"));
        assert!(md.contains("## Features"));
        assert!(md.contains("```\npip install litgraph\n```"));
        assert!(md.contains("[read the API docs](/api)"));
        assert!(md.contains("- Fast"));
        assert!(md.contains("- Typed"));
        assert!(md.contains("- Free"));
        // Boilerplate stripped.
        assert!(!md.contains("Menu items"));
        assert!(!md.contains("Site brand"));
        assert!(!md.contains("2026"));
    }

    #[test]
    fn empty_input_yields_empty_output() {
        assert_eq!(html_to_markdown("", true), "");
    }

    #[test]
    fn plain_text_input_round_trips() {
        let md = html_to_markdown("just plain text", true);
        assert_eq!(md, "just plain text");
    }
}

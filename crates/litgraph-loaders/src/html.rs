//! HTML loader — strip-and-extract for the common "give me the readable text"
//! use case in web-RAG pipelines. Avoids pulling html5ever/scraper (~50 transitive
//! deps) by using regex + a small entity-decode table. Good enough for boilerplate
//! removal + plaintext extraction; not a DOM library — if you need CSS selectors,
//! preprocess with scraper externally.
//!
//! What this does:
//!   - Removes `<script>` / `<style>` / `<!-- comments -->` blocks (with content).
//!   - Optionally removes `<nav>` / `<header>` / `<footer>` / `<aside>`.
//!   - Replaces remaining tags with whitespace.
//!   - Decodes common HTML entities (`&amp;` `&lt;` `&gt;` `&quot;` `&#NNN;` `&#xHH;`).
//!   - Collapses whitespace runs to single spaces; preserves paragraph breaks.
//!   - Extracts `<title>` into `metadata.title`.

use std::path::{Path, PathBuf};

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

static SCRIPT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<script\b[^>]*>.*?</script\s*>").unwrap()
});
static STYLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<style\b[^>]*>.*?</style\s*>").unwrap()
});
static COMMENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?s)<!--.*?-->").unwrap()
});
// `regex` crate doesn't support backrefs, so one regex per boilerplate tag.
static BOILERPLATE_NAV_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<nav\b[^>]*>.*?</nav\s*>").unwrap()
});
static BOILERPLATE_HEADER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<header\b[^>]*>.*?</header\s*>").unwrap()
});
static BOILERPLATE_FOOTER_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<footer\b[^>]*>.*?</footer\s*>").unwrap()
});
static BOILERPLATE_ASIDE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<aside\b[^>]*>.*?</aside\s*>").unwrap()
});
static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"<[^>]+>").unwrap());
static TITLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<title\b[^>]*>(.*?)</title\s*>").unwrap()
});
static ENTITY_NUMERIC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"&#(?:(?P<dec>\d+)|x(?P<hex>[0-9a-fA-F]+));").unwrap()
});
static MULTI_BLANK_LINE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\n{3,}").unwrap());
static MULTI_SPACE_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]{2,}").unwrap());

#[derive(Clone, Debug)]
pub struct HtmlLoader {
    /// `Some` for file-backed loader; `None` when the source is provided as a
    /// pre-fetched string (see `from_string`).
    pub path: Option<PathBuf>,
    inline_html: Option<String>,
    pub strip_boilerplate: bool,
}

impl HtmlLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self {
            path: Some(p.as_ref().to_path_buf()),
            inline_html: None,
            strip_boilerplate: true,
        }
    }
    /// Build from an in-memory HTML string (e.g. fetched via `WebLoader`).
    pub fn from_string(html: impl Into<String>) -> Self {
        Self {
            path: None,
            inline_html: Some(html.into()),
            strip_boilerplate: true,
        }
    }
    pub fn keep_boilerplate(mut self) -> Self {
        self.strip_boilerplate = false;
        self
    }

    fn read(&self) -> LoaderResult<String> {
        if let Some(s) = &self.inline_html {
            return Ok(s.clone());
        }
        let p = self.path.as_ref().ok_or_else(|| {
            LoaderError::Other("HtmlLoader: no source (neither path nor inline html)".into())
        })?;
        Ok(std::fs::read_to_string(p)?)
    }
}

impl Loader for HtmlLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let raw = self.read()?;
        let title = TITLE_RE
            .captures(&raw)
            .and_then(|c| c.get(1))
            .map(|m| decode_entities(&collapse_whitespace(m.as_str())));

        let stripped = strip_html(&raw, self.strip_boilerplate);
        let text = decode_entities(&stripped);

        let source = self
            .path
            .as_ref()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<inline>".into());

        let mut d = Document::new(text).with_id(source.clone());
        d.metadata.insert("source".into(), Value::String(source));
        if let Some(t) = title {
            if !t.is_empty() {
                d.metadata.insert("title".into(), Value::String(t));
            }
        }
        Ok(vec![d])
    }
}

/// Public for unit testing + direct use without instantiating a loader.
pub fn strip_html(input: &str, strip_boilerplate: bool) -> String {
    let mut s = SCRIPT_RE.replace_all(input, " ").into_owned();
    s = STYLE_RE.replace_all(&s, " ").into_owned();
    s = COMMENT_RE.replace_all(&s, " ").into_owned();
    if strip_boilerplate {
        s = BOILERPLATE_NAV_RE.replace_all(&s, " ").into_owned();
        s = BOILERPLATE_HEADER_RE.replace_all(&s, " ").into_owned();
        s = BOILERPLATE_FOOTER_RE.replace_all(&s, " ").into_owned();
        s = BOILERPLATE_ASIDE_RE.replace_all(&s, " ").into_owned();
    }
    // Replace block-level tags with newlines BEFORE generic tag stripping so
    // paragraph structure survives.
    let block_tags = [
        "br", "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "section", "article",
    ];
    for tag in block_tags {
        let pat = format!("(?i)<{tag}\\b[^>]*>");
        if let Ok(re) = Regex::new(&pat) {
            s = re.replace_all(&s, "\n").into_owned();
        }
        let pat = format!("(?i)</{tag}\\s*>");
        if let Ok(re) = Regex::new(&pat) {
            s = re.replace_all(&s, "\n").into_owned();
        }
    }
    s = TAG_RE.replace_all(&s, " ").into_owned();
    collapse_whitespace(&s)
}

fn collapse_whitespace(s: &str) -> String {
    let mut t = MULTI_SPACE_RE.replace_all(s, " ").into_owned();
    // Trim trailing/leading spaces on each line.
    t = t
        .lines()
        .map(|l| l.trim())
        .collect::<Vec<_>>()
        .join("\n");
    t = MULTI_BLANK_LINE_RE.replace_all(&t, "\n\n").into_owned();
    t.trim().to_string()
}

/// Decode the common named entities + `&#NNN;` / `&#xHH;` numeric references.
/// Anything unrecognized is left as-is.
pub fn decode_entities(s: &str) -> String {
    let mut t = s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ");
    t = ENTITY_NUMERIC_RE
        .replace_all(&t, |caps: &regex::Captures| {
            let cp: Option<u32> = if let Some(d) = caps.name("dec") {
                d.as_str().parse().ok()
            } else if let Some(h) = caps.name("hex") {
                u32::from_str_radix(h.as_str(), 16).ok()
            } else {
                None
            };
            cp.and_then(char::from_u32)
                .map(|c| c.to_string())
                .unwrap_or_else(|| caps.get(0).unwrap().as_str().to_string())
        })
        .into_owned();
    t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_script_style_and_comments() {
        let html = r#"
            <html><head><title>Hi</title>
            <style>body{color:red}</style>
            <script>alert('no')</script>
            </head><body>
            <!-- comment to remove -->
            <p>Hello world</p>
            <p>Second paragraph.</p>
            </body></html>
        "#;
        let docs = HtmlLoader::from_string(html).load().unwrap();
        assert_eq!(docs.len(), 1);
        let c = &docs[0].content;
        assert!(c.contains("Hello world"));
        assert!(c.contains("Second paragraph."));
        assert!(!c.contains("alert"), "script body must be gone: {c}");
        assert!(!c.contains("color:red"), "style body must be gone: {c}");
        assert!(!c.contains("comment to remove"));
        assert_eq!(docs[0].metadata.get("title").unwrap().as_str().unwrap(), "Hi");
    }

    #[test]
    fn strips_boilerplate_by_default() {
        let html = r#"<body>
            <nav>menu menu menu</nav>
            <header>brand</header>
            <p>Real content</p>
            <footer>copyright</footer>
        </body>"#;
        let docs = HtmlLoader::from_string(html).load().unwrap();
        let c = &docs[0].content;
        assert!(c.contains("Real content"));
        assert!(!c.contains("menu menu"));
        assert!(!c.contains("brand"));
        assert!(!c.contains("copyright"));
    }

    #[test]
    fn keep_boilerplate_preserves_nav_and_footer() {
        let html = r#"<body><nav>menu</nav><p>x</p><footer>©</footer></body>"#;
        let docs = HtmlLoader::from_string(html).keep_boilerplate().load().unwrap();
        let c = &docs[0].content;
        assert!(c.contains("menu"));
        assert!(c.contains("x"));
    }

    #[test]
    fn decodes_common_entities() {
        let html = "<p>Tom &amp; Jerry &lt;3 &#65; &#x42;</p>";
        let docs = HtmlLoader::from_string(html).load().unwrap();
        assert_eq!(docs[0].content, "Tom & Jerry <3 A B");
    }

    #[test]
    fn block_tags_become_newlines() {
        let html = "<p>one</p><p>two</p><br><p>three</p>";
        let docs = HtmlLoader::from_string(html).load().unwrap();
        // Three paragraphs separated by newlines.
        let lines: Vec<&str> = docs[0].content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines, vec!["one", "two", "three"]);
    }

    #[test]
    fn missing_title_omits_metadata_key() {
        let html = "<p>only body</p>";
        let docs = HtmlLoader::from_string(html).load().unwrap();
        assert!(docs[0].metadata.get("title").is_none());
    }
}

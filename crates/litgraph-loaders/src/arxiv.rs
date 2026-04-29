//! arXiv loader. Queries the public arXiv Atom API and emits one
//! [`Document`] per matched paper. Useful for RAG over a research corpus
//! without scraping HTML.
//!
//! # API
//!
//! - Endpoint: `http://export.arxiv.org/api/query`
//! - Returns Atom 1.0 XML — one `<entry>` per paper. We parse:
//!   `<title>`, `<summary>` (abstract), `<id>` (canonical URL),
//!   `<published>`, `<updated>`, `<author><name>`,
//!   `<arxiv:primary_category term="...">`, `<link rel="alternate">`.
//! - No auth, no API key. Public rate-limit is "be reasonable" — they
//!   ask for ≤1 request per 3 seconds (we don't enforce this; the
//!   caller's job).
//!
//! # Document shape
//!
//! - `content` = abstract (the `<summary>` field). Titles + authors land
//!   in metadata, NOT in content, so embedders index the abstract — the
//!   semantic gold — without title noise duplicating it.
//! - `id` = the arxiv id (`2104.12345v2`), stable across versions.
//! - `metadata`:
//!   - `title`: paper title (whitespace-collapsed)
//!   - `authors`: JSON array of author names
//!   - `published`: ISO-8601 string
//!   - `updated`: ISO-8601 string
//!   - `primary_category`: arXiv category code (e.g. `cs.LG`)
//!   - `categories`: JSON array of all category codes
//!   - `pdf_url`: direct PDF link (may be `null` if API didn't include one)
//!   - `abs_url`: HTML abstract page link
//!   - `source`: `"arxiv"`
//!
//! # Example
//!
//! ```no_run
//! use litgraph_loaders::{ArxivLoader, Loader};
//! let docs = ArxivLoader::new("transformer attention")
//!     .with_max_results(20)
//!     .with_sort_by("submittedDate")
//!     .with_sort_order("descending")
//!     .load()
//!     .unwrap();
//! for d in &docs {
//!     println!("{} - {}", d.id.as_deref().unwrap_or("?"),
//!              d.metadata.get("title").and_then(|v| v.as_str()).unwrap_or(""));
//! }
//! ```
//!
//! # Why parse Atom with regex
//!
//! arXiv's Atom output is well-formed and shallow. Pulling in a full XML
//! parser (quick-xml, roxmltree) for one loader bloats the dep tree.
//! The patterns here match the production-stable subset that arXiv has
//! emitted for ~15 years. If they change, tests catch it.

use std::time::Duration;

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const DEFAULT_MAX_RESULTS: usize = 10;
const ENDPOINT: &str = "http://export.arxiv.org/api/query";

/// Sort field accepted by the arXiv API. Re-exported as strings for
/// flexibility; we don't enum-wrap because the API may add new ones.
const VALID_SORT_BY: &[&str] = &["relevance", "lastUpdatedDate", "submittedDate"];
const VALID_SORT_ORDER: &[&str] = &["ascending", "descending"];

pub struct ArxivLoader {
    /// Free-form search query — passed as `search_query=all:<q>` if it
    /// doesn't already contain a field selector (`ti:`, `au:`, …).
    pub query: String,
    pub max_results: usize,
    pub start: usize,
    pub sort_by: Option<String>,
    pub sort_order: Option<String>,
    pub timeout: Duration,
    pub user_agent: String,
}

impl ArxivLoader {
    pub fn new(query: impl Into<String>) -> Self {
        Self {
            query: query.into(),
            max_results: DEFAULT_MAX_RESULTS,
            start: 0,
            sort_by: None,
            sort_order: None,
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }
    pub fn with_start(mut self, n: usize) -> Self {
        self.start = n;
        self
    }
    /// Validates against the API's accepted list and falls back silently
    /// if invalid — keeps loader resilient to typos at construction.
    pub fn with_sort_by(mut self, s: impl Into<String>) -> Self {
        let v = s.into();
        if VALID_SORT_BY.iter().any(|x| *x == v) {
            self.sort_by = Some(v);
        }
        self
    }
    pub fn with_sort_order(mut self, s: impl Into<String>) -> Self {
        let v = s.into();
        if VALID_SORT_ORDER.iter().any(|x| *x == v) {
            self.sort_order = Some(v);
        }
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }

    /// Override endpoint — useful for tests against a local mock.
    #[doc(hidden)]
    pub fn _endpoint_for_test(&self) -> &str {
        // Public hook is via env var so tests can swap the endpoint
        // without contaminating the public API.
        ENDPOINT
    }

    fn build_url(&self) -> String {
        let endpoint = std::env::var("LITGRAPH_ARXIV_ENDPOINT").unwrap_or_else(|_| ENDPOINT.into());
        let q = if self.query.contains(':') {
            // Caller supplied a field selector — pass through.
            self.query.clone()
        } else {
            format!("all:{}", self.query)
        };
        let mut url = format!(
            "{endpoint}?search_query={}&start={}&max_results={}",
            urlencode(&q),
            self.start,
            self.max_results
        );
        if let Some(sb) = &self.sort_by {
            url.push_str(&format!("&sortBy={sb}"));
        }
        if let Some(so) = &self.sort_order {
            url.push_str(&format!("&sortOrder={so}"));
        }
        url
    }

    fn fetch(&self) -> LoaderResult<String> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?;
        let url = self.build_url();
        let resp = client.get(&url).send()?;
        if !resp.status().is_success() {
            let s = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!("arxiv {s}: {body}")));
        }
        Ok(resp.text()?)
    }
}

impl Loader for ArxivLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let xml = self.fetch()?;
        Ok(parse_atom_feed(&xml))
    }
}

// -- Parsing -----------------------------------------------------------------

static ENTRY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<entry\b[^>]*>(.*?)</entry>").expect("entry regex")
});
static TITLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<title\b[^>]*>(.*?)</title>").expect("title regex")
});
static SUMMARY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<summary\b[^>]*>(.*?)</summary>").expect("summary regex")
});
static ID_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<id\b[^>]*>(.*?)</id>").expect("id regex")
});
static PUBLISHED_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<published\b[^>]*>(.*?)</published>").expect("published regex")
});
static UPDATED_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<updated\b[^>]*>(.*?)</updated>").expect("updated regex")
});
static AUTHOR_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<author\b[^>]*>.*?<name\b[^>]*>(.*?)</name>.*?</author>")
        .expect("author regex")
});
static PRIMARY_CAT_RE: Lazy<Regex> = Lazy::new(|| {
    // Order-tolerant: matches term="..." anywhere inside the tag.
    Regex::new(r#"(?is)<arxiv:primary_category\b[^>]*\bterm="([^"]+)""#)
        .expect("primary cat regex")
});
static CATEGORY_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<category\b[^>]*\bterm="([^"]+)""#).expect("cat regex")
});
/// All `<link ...>` self-closing tags. We then parse attrs out of capture
/// 1 (the inside) so attribute order doesn't matter — arXiv has emitted
/// both `href`-first and `rel`-first orderings over the years.
static LINK_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)<link\b([^>]*)/?>"#).expect("link regex"));
static ATTR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)\b([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)""#).expect("attr regex")
});

fn parse_link_attrs(inside: &str) -> std::collections::HashMap<String, String> {
    ATTR_RE
        .captures_iter(inside)
        .map(|c| (c[1].to_string(), c[2].to_string()))
        .collect()
}

/// Find the first `<link>` whose attribute predicate matches. Returns the
/// `href` value of that link.
fn find_link_href<F: Fn(&std::collections::HashMap<String, String>) -> bool>(
    entry_xml: &str,
    pred: F,
) -> Option<String> {
    for c in LINK_RE.captures_iter(entry_xml) {
        let attrs = parse_link_attrs(&c[1]);
        if pred(&attrs) {
            if let Some(h) = attrs.get("href") {
                return Some(h.clone());
            }
        }
    }
    None
}

fn first_capture<'a>(re: &Regex, hay: &'a str) -> Option<&'a str> {
    re.captures(hay).and_then(|c| c.get(1).map(|m| m.as_str()))
}

/// Parse a full Atom feed into a vec of Documents. Public (crate-vis) so
/// tests can exercise parsing on canned XML.
pub(crate) fn parse_atom_feed(xml: &str) -> Vec<Document> {
    ENTRY_RE
        .captures_iter(xml)
        .map(|c| parse_entry(&c[1]))
        .collect()
}

fn parse_entry(entry_xml: &str) -> Document {
    let title = first_capture(&TITLE_RE, entry_xml)
        .map(|s| collapse_ws(&decode_entities(s)))
        .unwrap_or_default();
    let summary = first_capture(&SUMMARY_RE, entry_xml)
        .map(|s| collapse_ws(&decode_entities(s)))
        .unwrap_or_default();
    let id_url = first_capture(&ID_RE, entry_xml)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let published = first_capture(&PUBLISHED_RE, entry_xml)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let updated = first_capture(&UPDATED_RE, entry_xml)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let authors: Vec<String> = AUTHOR_NAME_RE
        .captures_iter(entry_xml)
        .map(|c| collapse_ws(&decode_entities(&c[1])))
        .collect();
    let primary_cat = first_capture(&PRIMARY_CAT_RE, entry_xml)
        .map(str::to_string)
        .unwrap_or_default();
    let categories: Vec<String> = CATEGORY_RE
        .captures_iter(entry_xml)
        .map(|c| c[1].to_string())
        .collect();
    let pdf_url = find_link_href(entry_xml, |a| {
        a.get("title").map(|s| s.as_str()) == Some("pdf")
            || a.get("type").map(|s| s.as_str()) == Some("application/pdf")
    });
    let abs_url = find_link_href(entry_xml, |a| {
        a.get("rel").map(|s| s.as_str()) == Some("alternate")
    });

    let arxiv_id = arxiv_id_from_url(&id_url);

    let mut d = Document::new(summary);
    if !arxiv_id.is_empty() {
        d = d.with_id(&arxiv_id);
    }
    let mut put = |k: &str, v: Value| {
        d.metadata.insert(k.into(), v);
    };
    put("title", Value::String(title));
    put("authors", json!(authors));
    if !published.is_empty() {
        put("published", Value::String(published));
    }
    if !updated.is_empty() {
        put("updated", Value::String(updated));
    }
    if !primary_cat.is_empty() {
        put("primary_category", Value::String(primary_cat));
    }
    if !categories.is_empty() {
        put("categories", json!(categories));
    }
    put(
        "pdf_url",
        pdf_url.map(Value::String).unwrap_or(Value::Null),
    );
    put(
        "abs_url",
        abs_url.map(Value::String).unwrap_or(Value::Null),
    );
    if !id_url.is_empty() {
        put("arxiv_url", Value::String(id_url));
    }
    put("source", Value::String("arxiv".into()));
    d
}

/// `http://arxiv.org/abs/2104.12345v2` → `2104.12345v2`.
fn arxiv_id_from_url(url: &str) -> String {
    url.rsplit('/').next().unwrap_or("").trim().to_string()
}

/// Cheap entity decode for the handful of escapes arXiv emits. We don't
/// bring in `htmlescape` for five chars.
fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
}

/// Collapse internal whitespace (newlines, tabs, multiple spaces) and trim.
/// Atom titles/abstracts come pretty-printed across multiple lines.
fn collapse_ws(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for ch in s.trim().chars() {
        if ch.is_whitespace() {
            if !prev_space {
                out.push(' ');
                prev_space = true;
            }
        } else {
            out.push(ch);
            prev_space = false;
        }
    }
    out
}

/// Minimal URL-encoder for the query string. Only encodes the chars the
/// arXiv API can choke on; lets ASCII-safe ones (incl. `:`) pass through
/// to keep the URL readable in logs.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b':' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real arXiv response shape, trimmed to two entries. Captured from a
    /// live API call so test catches drift in their Atom format.
    const FIXTURE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <updated>2017-12-06T18:50:38Z</updated>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <summary>  The dominant sequence transduction models are based on
      complex recurrent or convolutional neural networks.  </summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <link href="http://arxiv.org/abs/1706.03762v5" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/1706.03762v5" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2005.14165v4</id>
    <updated>2020-07-22T17:42:30Z</updated>
    <published>2020-05-28T17:29:14Z</published>
    <title>Language Models are Few-Shot Learners</title>
    <summary>Recent work has demonstrated substantial gains.</summary>
    <author><name>Tom B. Brown</name></author>
    <link href="http://arxiv.org/abs/2005.14165v4" rel="alternate" type="text/html"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>"#;

    #[test]
    fn parses_two_entries() {
        let docs = parse_atom_feed(FIXTURE);
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn first_entry_has_correct_id_title_authors() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[0];
        assert_eq!(d.id.as_deref(), Some("1706.03762v5"));
        assert_eq!(
            d.metadata.get("title").and_then(|v| v.as_str()),
            Some("Attention Is All You Need")
        );
        let authors = d.metadata.get("authors").and_then(|v| v.as_array()).unwrap();
        assert_eq!(authors.len(), 2);
        assert_eq!(authors[0].as_str(), Some("Ashish Vaswani"));
        assert_eq!(authors[1].as_str(), Some("Noam Shazeer"));
    }

    #[test]
    fn abstract_lands_in_content_not_metadata() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[0];
        // Whitespace collapsed.
        assert!(d.content.starts_with("The dominant sequence"));
        assert!(!d.content.contains('\n'));
        assert!(!d.metadata.contains_key("summary"));
    }

    #[test]
    fn primary_category_and_categories_extracted() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[0];
        assert_eq!(
            d.metadata.get("primary_category").and_then(|v| v.as_str()),
            Some("cs.CL")
        );
        let cats = d.metadata.get("categories").and_then(|v| v.as_array()).unwrap();
        let cats_str: Vec<&str> = cats.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(cats_str, vec!["cs.CL", "cs.LG"]);
    }

    #[test]
    fn pdf_and_abs_links_extracted() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[0];
        assert_eq!(
            d.metadata.get("pdf_url").and_then(|v| v.as_str()),
            Some("http://arxiv.org/pdf/1706.03762v5")
        );
        assert_eq!(
            d.metadata.get("abs_url").and_then(|v| v.as_str()),
            Some("http://arxiv.org/abs/1706.03762v5")
        );
    }

    #[test]
    fn entry_without_pdf_link_yields_null() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[1];
        assert!(d.metadata.get("pdf_url").map(|v| v.is_null()).unwrap_or(false));
        // abs_url still present.
        assert_eq!(
            d.metadata.get("abs_url").and_then(|v| v.as_str()),
            Some("http://arxiv.org/abs/2005.14165v4")
        );
    }

    #[test]
    fn empty_feed_yields_empty_vec() {
        let xml = r#"<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom"></feed>"#;
        let docs = parse_atom_feed(xml);
        assert!(docs.is_empty());
    }

    #[test]
    fn timestamps_round_trip() {
        let docs = parse_atom_feed(FIXTURE);
        let d = &docs[0];
        assert_eq!(
            d.metadata.get("published").and_then(|v| v.as_str()),
            Some("2017-06-12T17:57:34Z")
        );
        assert_eq!(
            d.metadata.get("updated").and_then(|v| v.as_str()),
            Some("2017-12-06T18:50:38Z")
        );
    }

    #[test]
    fn html_entities_in_titles_decoded() {
        let xml = r#"<feed xmlns="http://www.w3.org/2005/Atom">
            <entry><id>http://arxiv.org/abs/9999.00001v1</id>
            <title>A &amp; B: &lt;tag&gt; with &quot;quotes&quot;</title>
            <summary>x</summary></entry></feed>"#;
        let docs = parse_atom_feed(xml);
        assert_eq!(
            docs[0].metadata.get("title").and_then(|v| v.as_str()),
            Some(r#"A & B: <tag> with "quotes""#)
        );
    }

    #[test]
    fn build_url_default_wraps_query_with_all() {
        let l = ArxivLoader::new("transformer attention");
        let url = l.build_url();
        assert!(url.contains("search_query=all:transformer+attention"), "{url}");
        assert!(url.contains("start=0"));
        assert!(url.contains("max_results=10"));
    }

    #[test]
    fn build_url_passes_field_selector_through() {
        let l = ArxivLoader::new("au:vaswani");
        let url = l.build_url();
        assert!(url.contains("search_query=au:vaswani"), "{url}");
    }

    #[test]
    fn build_url_includes_sort_when_set() {
        let l = ArxivLoader::new("x")
            .with_sort_by("submittedDate")
            .with_sort_order("descending");
        let url = l.build_url();
        assert!(url.contains("sortBy=submittedDate"), "{url}");
        assert!(url.contains("sortOrder=descending"), "{url}");
    }

    #[test]
    fn invalid_sort_value_silently_ignored() {
        let l = ArxivLoader::new("x").with_sort_by("invalid");
        assert!(l.sort_by.is_none());
    }

    #[test]
    fn id_extraction_strips_url_prefix() {
        assert_eq!(arxiv_id_from_url("http://arxiv.org/abs/2104.12345v2"), "2104.12345v2");
        assert_eq!(arxiv_id_from_url("https://arxiv.org/abs/cs.LG/0123456v1"), "0123456v1");
        assert_eq!(arxiv_id_from_url(""), "");
    }

    #[test]
    fn live_endpoint_override_via_env() {
        // Don't actually fetch — just confirm the env var is consulted
        // by build_url.
        std::env::set_var("LITGRAPH_ARXIV_ENDPOINT", "http://localhost:9/api");
        let l = ArxivLoader::new("test");
        let url = l.build_url();
        assert!(url.starts_with("http://localhost:9/api"), "{url}");
        std::env::remove_var("LITGRAPH_ARXIV_ENDPOINT");
    }
}

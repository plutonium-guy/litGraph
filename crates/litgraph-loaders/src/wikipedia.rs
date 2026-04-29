//! Wikipedia loader. Pulls the cleaned plain-text extract of one or more
//! pages via the MediaWiki Action API. Useful as a "ground truth" RAG
//! source — short, dense, license-friendly (CC BY-SA).
//!
//! # Two modes
//!
//! ## 1. Lookup by exact title
//!
//! `ArxivLoader::titles(["Transformer (machine learning model)", ...])`
//! → one Document per resolved title. Disambiguation pages are returned
//! as-is (caller's job to resolve); redirects are followed automatically
//! by the API and the final title lands in metadata.
//!
//! ## 2. Search-then-fetch
//!
//! `WikipediaLoader::search("attention mechanism").with_max_results(5)`
//! → top-N search hits, each fetched as a full extract. Useful for
//! "find me Wikipedia's take on X" style queries.
//!
//! # Why MediaWiki Action API and not REST `/page/summary`
//!
//! The summary endpoint truncates at ~one paragraph. For RAG you want
//! the full lead section minimum, often the whole article. Action API
//! `prop=extracts&explaintext` returns the full plaintext-converted body,
//! which is what we want. The cost: one extra parameter to set
//! (`exintro=false` for full body, `exintro=true` for lead only).
//!
//! # Document shape
//!
//! - `content` = plaintext extract (full body or intro depending on
//!   `intro_only`).
//! - `id` = pageid (stable across renames).
//! - `metadata`:
//!   - `title`: canonical (post-redirect) title
//!   - `pageid`: integer page id (also exposed as `id`)
//!   - `url`: human-readable wiki URL
//!   - `language`: language code (`"en"`, `"de"`, …)
//!   - `source`: `"wikipedia"`
//!
//! # Example
//!
//! ```no_run
//! use litgraph_loaders::{Loader, WikipediaLoader};
//!
//! // Direct lookup.
//! let docs = WikipediaLoader::titles(vec!["Rust (programming language)".into()])
//!     .with_intro_only(true)
//!     .load()
//!     .unwrap();
//!
//! // Search + fetch top 5.
//! let docs = WikipediaLoader::search("transformer model")
//!     .with_max_results(5)
//!     .load()
//!     .unwrap();
//! ```

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const DEFAULT_LANGUAGE: &str = "en";
const DEFAULT_MAX_RESULTS: usize = 5;

/// What the loader is supposed to do — pull explicit titles or run a
/// search first.
#[derive(Clone)]
enum Mode {
    Titles(Vec<String>),
    Search(String),
}

pub struct WikipediaLoader {
    mode: Mode,
    pub language: String,
    pub max_results: usize,
    pub intro_only: bool,
    pub timeout: Duration,
    pub user_agent: String,
}

impl WikipediaLoader {
    /// Look up the given titles directly. Order preserved in output.
    pub fn titles(titles: Vec<String>) -> Self {
        Self {
            mode: Mode::Titles(titles),
            language: DEFAULT_LANGUAGE.into(),
            // Caller decided exactly which titles — cap doesn't apply
            // unless they ask. Initialised generously.
            max_results: usize::MAX,
            intro_only: false,
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Run a search, then fetch each hit. Default cap: 5 hits — going
    /// higher fan-outs many fetches.
    pub fn search(query: impl Into<String>) -> Self {
        Self {
            mode: Mode::Search(query.into()),
            language: DEFAULT_LANGUAGE.into(),
            max_results: DEFAULT_MAX_RESULTS,
            intro_only: false,
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
        self
    }

    pub fn with_max_results(mut self, n: usize) -> Self {
        self.max_results = n;
        self
    }

    /// `true` → only the lead section; `false` (default) → full article.
    pub fn with_intro_only(mut self, b: bool) -> Self {
        self.intro_only = b;
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

    fn endpoint(&self) -> String {
        if let Ok(s) = std::env::var("LITGRAPH_WIKIPEDIA_ENDPOINT") {
            return s;
        }
        format!("https://{}.wikipedia.org/w/api.php", self.language)
    }

    fn http_client(&self) -> LoaderResult<reqwest::blocking::Client> {
        Ok(reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?)
    }

    /// Resolve the search query into a list of titles.
    fn run_search(&self, client: &reqwest::blocking::Client, q: &str) -> LoaderResult<Vec<String>> {
        let url = self.endpoint();
        let resp = client
            .get(&url)
            .query(&[
                ("action", "query"),
                ("format", "json"),
                ("list", "search"),
                ("srsearch", q),
                ("srlimit", &self.max_results.to_string()),
            ])
            .send()?;
        if !resp.status().is_success() {
            return Err(LoaderError::Other(format!(
                "wikipedia search {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }
        let body: Value = resp.json()?;
        Ok(parse_search_titles(&body))
    }

    fn fetch_extracts(
        &self,
        client: &reqwest::blocking::Client,
        titles: &[String],
    ) -> LoaderResult<Vec<Document>> {
        if titles.is_empty() {
            return Ok(Vec::new());
        }
        // MediaWiki accepts up to 50 titles per call. Chunk to be safe.
        const BATCH: usize = 50;
        let url = self.endpoint();
        let exintro = if self.intro_only { "1" } else { "0" };
        let mut out: Vec<Document> = Vec::with_capacity(titles.len());
        for chunk in titles.chunks(BATCH) {
            let titles_param = chunk.join("|");
            let resp = client
                .get(&url)
                .query(&[
                    ("action", "query"),
                    ("format", "json"),
                    ("prop", "extracts|info"),
                    ("inprop", "url"),
                    ("explaintext", "1"),
                    ("exintro", exintro),
                    ("redirects", "1"),
                    ("titles", &titles_param),
                ])
                .send()?;
            if !resp.status().is_success() {
                return Err(LoaderError::Other(format!(
                    "wikipedia extract {}: {}",
                    resp.status(),
                    resp.text().unwrap_or_default()
                )));
            }
            let body: Value = resp.json()?;
            let mut docs = parse_extracts(&body, &self.language);
            // Preserve caller-supplied title order. Pages dict from API
            // is keyed by pageid (string) — order is undefined.
            docs.sort_by_key(|d| {
                let title = d
                    .metadata
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                chunk
                    .iter()
                    .position(|t| {
                        t == title
                            || canonical_title(t).eq_ignore_ascii_case(&canonical_title(title))
                    })
                    .unwrap_or(usize::MAX)
            });
            out.extend(docs);
        }
        Ok(out)
    }
}

impl Loader for WikipediaLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.http_client()?;
        let titles = match &self.mode {
            Mode::Titles(t) => {
                if t.len() > self.max_results && self.max_results != usize::MAX {
                    t.iter().take(self.max_results).cloned().collect()
                } else {
                    t.clone()
                }
            }
            Mode::Search(q) => self.run_search(&client, q)?,
        };
        self.fetch_extracts(&client, &titles)
    }
}

// -- Parsing -----------------------------------------------------------------

pub(crate) fn parse_search_titles(body: &Value) -> Vec<String> {
    body.get("query")
        .and_then(|q| q.get("search"))
        .and_then(|s| s.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|h| h.get("title").and_then(|v| v.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

pub(crate) fn parse_extracts(body: &Value, language: &str) -> Vec<Document> {
    let pages = match body
        .get("query")
        .and_then(|q| q.get("pages"))
        .and_then(|p| p.as_object())
    {
        Some(p) => p,
        None => return Vec::new(),
    };
    let mut out = Vec::with_capacity(pages.len());
    for (_pageid_key, page) in pages {
        // Skip "missing" page entries — API returns a stub when a title
        // doesn't exist.
        if page.get("missing").is_some() {
            continue;
        }
        let title = page
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let extract = page
            .get("extract")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let pageid = page.get("pageid").and_then(|v| v.as_i64()).unwrap_or(0);
        let url = page
            .get("fullurl")
            .and_then(|v| v.as_str())
            .map(String::from)
            .unwrap_or_else(|| {
                format!(
                    "https://{language}.wikipedia.org/wiki/{}",
                    title.replace(' ', "_")
                )
            });

        let mut d = Document::new(extract);
        if pageid > 0 {
            d = d.with_id(pageid.to_string());
        }
        d.metadata.insert("title".into(), Value::String(title));
        if pageid > 0 {
            d.metadata
                .insert("pageid".into(), Value::Number(pageid.into()));
        }
        d.metadata.insert("url".into(), Value::String(url));
        d.metadata
            .insert("language".into(), Value::String(language.into()));
        d.metadata
            .insert("source".into(), Value::String("wikipedia".into()));
        out.push(d);
    }
    out
}

/// Normalise a title for fuzzy compare: replace `_` with space, trim.
/// Lowercasing is left to the caller (eq_ignore_ascii_case handles it).
fn canonical_title(t: &str) -> String {
    t.replace('_', " ").trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Real-shape extract response (one page).
    fn extracts_fixture() -> Value {
        json!({
            "batchcomplete": "",
            "query": {
                "pages": {
                    "1234": {
                        "pageid": 1234,
                        "ns": 0,
                        "title": "Rust (programming language)",
                        "fullurl": "https://en.wikipedia.org/wiki/Rust_(programming_language)",
                        "extract": "Rust is a multi-paradigm, general-purpose programming language. Rust emphasizes performance, type safety, and concurrency."
                    }
                }
            }
        })
    }

    /// Real-shape search response.
    fn search_fixture() -> Value {
        json!({
            "batchcomplete": "",
            "continue": {"sroffset": 5, "continue": "-||"},
            "query": {
                "searchinfo": {"totalhits": 9999},
                "search": [
                    {"ns": 0, "title": "Transformer (deep learning architecture)", "snippet": "..."},
                    {"ns": 0, "title": "Attention (machine learning)", "snippet": "..."},
                    {"ns": 0, "title": "Large language model", "snippet": "..."}
                ]
            }
        })
    }

    #[test]
    fn parses_extract_into_document() {
        let docs = parse_extracts(&extracts_fixture(), "en");
        assert_eq!(docs.len(), 1);
        let d = &docs[0];
        assert_eq!(d.id.as_deref(), Some("1234"));
        assert!(d.content.starts_with("Rust is a multi-paradigm"));
        assert_eq!(
            d.metadata.get("title").and_then(|v| v.as_str()),
            Some("Rust (programming language)")
        );
        assert_eq!(d.metadata.get("pageid").and_then(|v| v.as_i64()), Some(1234));
        assert_eq!(
            d.metadata.get("language").and_then(|v| v.as_str()),
            Some("en")
        );
        assert_eq!(
            d.metadata.get("source").and_then(|v| v.as_str()),
            Some("wikipedia")
        );
        assert!(d
            .metadata
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap()
            .contains("Rust_(programming_language)"));
    }

    #[test]
    fn missing_page_is_skipped() {
        let body = json!({
            "query": {
                "pages": {
                    "-1": {"ns": 0, "title": "Nonexistent_Page", "missing": ""}
                }
            }
        });
        let docs = parse_extracts(&body, "en");
        assert!(docs.is_empty());
    }

    #[test]
    fn url_synthesised_when_fullurl_absent() {
        // Older API responses don't always include fullurl; we synthesize.
        let body = json!({
            "query": {
                "pages": {
                    "5": {
                        "pageid": 5, "ns": 0, "title": "Foo Bar",
                        "extract": "x"
                    }
                }
            }
        });
        let docs = parse_extracts(&body, "en");
        assert_eq!(
            docs[0].metadata.get("url").and_then(|v| v.as_str()),
            Some("https://en.wikipedia.org/wiki/Foo_Bar")
        );
    }

    #[test]
    fn search_titles_extracted_in_order() {
        let titles = parse_search_titles(&search_fixture());
        assert_eq!(
            titles,
            vec![
                "Transformer (deep learning architecture)".to_string(),
                "Attention (machine learning)".to_string(),
                "Large language model".to_string(),
            ]
        );
    }

    #[test]
    fn empty_pages_yields_empty_vec() {
        let body = json!({"query": {"pages": {}}});
        assert!(parse_extracts(&body, "en").is_empty());
    }

    #[test]
    fn empty_search_yields_empty_vec() {
        let body = json!({"query": {"search": []}});
        assert!(parse_search_titles(&body).is_empty());
    }

    #[test]
    fn malformed_response_does_not_panic() {
        let body = json!({"foo": "bar"});
        assert!(parse_search_titles(&body).is_empty());
        assert!(parse_extracts(&body, "en").is_empty());
    }

    #[test]
    fn endpoint_uses_language_code() {
        let l = WikipediaLoader::titles(vec!["X".into()]).with_language("de");
        assert_eq!(l.endpoint(), "https://de.wikipedia.org/w/api.php");
    }

    #[test]
    fn endpoint_env_override_works() {
        std::env::set_var("LITGRAPH_WIKIPEDIA_ENDPOINT", "http://localhost:9/api");
        let l = WikipediaLoader::titles(vec!["X".into()]);
        assert_eq!(l.endpoint(), "http://localhost:9/api");
        std::env::remove_var("LITGRAPH_WIKIPEDIA_ENDPOINT");
    }

    #[test]
    fn intro_only_setter_persists() {
        let l = WikipediaLoader::titles(vec!["X".into()]).with_intro_only(true);
        assert!(l.intro_only);
    }

    #[test]
    fn search_constructor_caps_default_to_5() {
        let l = WikipediaLoader::search("x");
        assert_eq!(l.max_results, 5);
    }

    #[test]
    fn titles_constructor_uses_unbounded_cap() {
        let l = WikipediaLoader::titles(vec!["a".into(), "b".into()]);
        assert_eq!(l.max_results, usize::MAX);
    }
}

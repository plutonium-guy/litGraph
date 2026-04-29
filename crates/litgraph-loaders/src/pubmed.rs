//! PubMed loader — closes the research-RAG triple alongside
//! [`crate::ArxivLoader`] and [`crate::WikipediaLoader`].
//!
//! Backed by the NCBI E-utilities (`eutils.ncbi.nlm.nih.gov/entrez/eutils`).
//! Two-step protocol:
//!
//! 1. **ESearch** (`esearch.fcgi`, JSON) — query → list of PMIDs.
//! 2. **EFetch** (`efetch.fcgi`, XML) — PMIDs → full citation +
//!    abstract + MeSH terms.
//!
//! Returns one [`Document`] per article. Abstract goes to `content`
//! (the embedding-relevant field); title, authors, journal, DOI,
//! publication date, MeSH descriptors land in metadata.
//!
//! # API key
//!
//! NCBI rate-limits anonymous traffic at 3 req/s, registered keys at
//! 10 req/s. Pass an API key via [`PubMedLoader::with_api_key`] to ride
//! the higher limit. We do NOT enforce the limit client-side — the
//! caller's job to space out batches if scraping at scale.
//!
//! # Tool / email politeness
//!
//! NCBI asks API consumers to identify themselves via `tool=` and
//! `email=` parameters so they can contact you before banning. We send
//! `tool=litgraph-loaders` by default; supply your own contact via
//! [`with_email`] in production.
//!
//! # Document shape
//!
//! - `content` = abstract text (concatenated across structured sections
//!   like `BACKGROUND:` / `METHODS:` / `RESULTS:`).
//! - `id` = PMID (string).
//! - `metadata`:
//!   - `title`: article title (whitespace-collapsed)
//!   - `authors`: JSON array of `"FirstName LastName"` strings
//!   - `journal`: journal title (full, not abbreviation)
//!   - `pub_date`: best-effort `YYYY-MM-DD` / `YYYY-MM` / `YYYY`
//!   - `doi`: DOI string if present, else null
//!   - `pmcid`: PubMed Central id if present, else null
//!   - `mesh`: JSON array of MeSH descriptor strings
//!   - `pmid`: same as `id`, exposed for convenience
//!   - `source`: `"pubmed"`
//!
//! # Example
//!
//! ```no_run
//! use litgraph_loaders::{Loader, PubMedLoader};
//! let docs = PubMedLoader::search("transformer attention mechanism")
//!     .with_max_results(20)
//!     .with_email("you@example.com")
//!     .load()
//!     .unwrap();
//! ```

use std::time::Duration;

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const ESEARCH: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const EFETCH: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";
const DEFAULT_MAX_RESULTS: usize = 10;

#[derive(Clone)]
enum Mode {
    /// Free-form search → ESearch → EFetch
    Search(String),
    /// Caller supplies PMIDs directly → just EFetch
    Pmids(Vec<String>),
}

pub struct PubMedLoader {
    mode: Mode,
    pub max_results: usize,
    pub start: usize,
    pub api_key: Option<String>,
    pub email: Option<String>,
    pub tool: String,
    pub timeout: Duration,
    pub user_agent: String,
}

impl PubMedLoader {
    pub fn search(query: impl Into<String>) -> Self {
        Self {
            mode: Mode::Search(query.into()),
            max_results: DEFAULT_MAX_RESULTS,
            start: 0,
            api_key: None,
            email: None,
            tool: "litgraph-loaders".to_string(),
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Skip the search step — fetch these specific PMIDs directly.
    pub fn pmids(pmids: Vec<String>) -> Self {
        Self {
            mode: Mode::Pmids(pmids),
            max_results: usize::MAX,
            start: 0,
            api_key: None,
            email: None,
            tool: "litgraph-loaders".to_string(),
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
    pub fn with_api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
    /// NCBI asks API users to send a contact email so they can warn
    /// before banning. Strongly recommended for production traffic.
    pub fn with_email(mut self, email: impl Into<String>) -> Self {
        self.email = Some(email.into());
        self
    }
    pub fn with_tool(mut self, tool: impl Into<String>) -> Self {
        self.tool = tool.into();
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

    fn esearch_endpoint(&self) -> String {
        std::env::var("LITGRAPH_PUBMED_ESEARCH").unwrap_or_else(|_| ESEARCH.into())
    }
    fn efetch_endpoint(&self) -> String {
        std::env::var("LITGRAPH_PUBMED_EFETCH").unwrap_or_else(|_| EFETCH.into())
    }

    fn http_client(&self) -> LoaderResult<reqwest::blocking::Client> {
        Ok(reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?)
    }

    /// Build the common-to-both ESearch/EFetch param tuple. We use Vec
    /// rather than &[(&str,&str)] so optional auth fields slot in cleanly.
    fn common_params(&self) -> Vec<(&'static str, String)> {
        let mut p: Vec<(&'static str, String)> = vec![
            ("db", "pubmed".to_string()),
            ("tool", self.tool.clone()),
        ];
        if let Some(k) = &self.api_key {
            p.push(("api_key", k.clone()));
        }
        if let Some(e) = &self.email {
            p.push(("email", e.clone()));
        }
        p
    }

    fn run_esearch(&self, client: &reqwest::blocking::Client, q: &str) -> LoaderResult<Vec<String>> {
        let mut params = self.common_params();
        params.push(("retmode", "json".to_string()));
        params.push(("term", q.to_string()));
        params.push(("retmax", self.max_results.to_string()));
        params.push(("retstart", self.start.to_string()));
        let resp = client.get(self.esearch_endpoint()).query(&params).send()?;
        if !resp.status().is_success() {
            return Err(LoaderError::Other(format!(
                "pubmed esearch {}: {}",
                resp.status(),
                resp.text().unwrap_or_default()
            )));
        }
        let body: Value = resp.json()?;
        Ok(parse_esearch_ids(&body))
    }

    fn run_efetch(
        &self,
        client: &reqwest::blocking::Client,
        pmids: &[String],
    ) -> LoaderResult<Vec<Document>> {
        if pmids.is_empty() {
            return Ok(Vec::new());
        }
        // EFetch supports up to ~200 PMIDs per call; we batch at 100 to
        // keep URL length sane and let very-large bulk loads still work.
        const BATCH: usize = 100;
        let mut out: Vec<Document> = Vec::with_capacity(pmids.len());
        for chunk in pmids.chunks(BATCH) {
            let mut params = self.common_params();
            params.push(("retmode", "xml".to_string()));
            params.push(("rettype", "abstract".to_string()));
            params.push(("id", chunk.join(",")));
            let resp = client.get(self.efetch_endpoint()).query(&params).send()?;
            if !resp.status().is_success() {
                return Err(LoaderError::Other(format!(
                    "pubmed efetch {}: {}",
                    resp.status(),
                    resp.text().unwrap_or_default()
                )));
            }
            let xml = resp.text()?;
            out.extend(parse_efetch_xml(&xml));
        }
        Ok(out)
    }
}

impl Loader for PubMedLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.http_client()?;
        let pmids: Vec<String> = match &self.mode {
            Mode::Search(q) => self.run_esearch(&client, q)?,
            Mode::Pmids(p) => p.clone(),
        };
        self.run_efetch(&client, &pmids)
    }
}

// -- Parsing -----------------------------------------------------------------

pub(crate) fn parse_esearch_ids(body: &Value) -> Vec<String> {
    body.get("esearchresult")
        .and_then(|r| r.get("idlist"))
        .and_then(|i| i.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default()
}

static ARTICLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<PubmedArticle\b[^>]*>(.*?)</PubmedArticle>").expect("article regex")
});
static PMID_RE: Lazy<Regex> = Lazy::new(|| {
    // First MedlineCitation/PMID — there can be a CommentsCorrections PMID later.
    Regex::new(r"(?is)<MedlineCitation\b[^>]*>.*?<PMID\b[^>]*>(.*?)</PMID>").expect("pmid regex")
});
static TITLE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<ArticleTitle\b[^>]*>(.*?)</ArticleTitle>").expect("title re"));
static JOURNAL_TITLE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<Journal\b[^>]*>.*?<Title\b[^>]*>(.*?)</Title>").expect("journal re")
});
static ABSTRACT_TEXT_RE: Lazy<Regex> = Lazy::new(|| {
    // Captures the *contents* of every <AbstractText ...> tag inside <Abstract>.
    Regex::new(r"(?is)<AbstractText\b([^>]*)>(.*?)</AbstractText>").expect("abstract re")
});
static AUTHOR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<Author\b[^>]*>(.*?)</Author>").expect("author re")
});
static LAST_NAME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<LastName\b[^>]*>(.*?)</LastName>").expect("last re"));
static FORE_NAME_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<ForeName\b[^>]*>(.*?)</ForeName>").expect("fore re"));
static COLLECTIVE_NAME_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<CollectiveName\b[^>]*>(.*?)</CollectiveName>").expect("collective re")
});
static PUBDATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<PubDate\b[^>]*>(.*?)</PubDate>").expect("pubdate re")
});
static PUBDATE_YEAR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<Year\b[^>]*>(.*?)</Year>").expect("year re"));
static PUBDATE_MONTH_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<Month\b[^>]*>(.*?)</Month>").expect("month re"));
static PUBDATE_DAY_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?is)<Day\b[^>]*>(.*?)</Day>").expect("day re"));
static MEDLINE_DATE_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<MedlineDate\b[^>]*>(.*?)</MedlineDate>").expect("medline date re")
});
static ARTICLE_ID_RE: Lazy<Regex> = Lazy::new(|| {
    // Use attribute capture so id-type order doesn't matter.
    Regex::new(r#"(?is)<ArticleId\b([^>]*)>(.*?)</ArticleId>"#).expect("articleid re")
});
static ID_TYPE_ATTR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)\bIdType\s*=\s*"([^"]+)""#).expect("idtype re"));
static MESH_DESC_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?is)<DescriptorName\b[^>]*>(.*?)</DescriptorName>").expect("mesh re")
});

fn first_capture<'a>(re: &Regex, hay: &'a str) -> Option<&'a str> {
    re.captures(hay).and_then(|c| c.get(1).map(|m| m.as_str()))
}

/// Parse a full EFetch XML (one or many `<PubmedArticle>`) into Documents.
pub(crate) fn parse_efetch_xml(xml: &str) -> Vec<Document> {
    ARTICLE_RE
        .captures_iter(xml)
        .map(|c| parse_article(&c[1]))
        .collect()
}

fn parse_article(article_xml: &str) -> Document {
    let pmid = first_capture(&PMID_RE, article_xml)
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    let title = first_capture(&TITLE_RE, article_xml)
        .map(|s| collapse_ws(&strip_inline_tags(&decode_entities(s))))
        .unwrap_or_default();
    let journal = first_capture(&JOURNAL_TITLE_RE, article_xml)
        .map(|s| collapse_ws(&decode_entities(s)))
        .unwrap_or_default();
    let abstract_text = collect_abstract_sections(article_xml);
    let authors = collect_authors(article_xml);
    let pub_date = parse_pub_date(article_xml);
    let mesh = collect_mesh(article_xml);
    let (doi, pmcid) = collect_article_ids(article_xml);

    let mut d = Document::new(abstract_text);
    if !pmid.is_empty() {
        d = d.with_id(pmid.clone());
    }
    let mut put = |k: &str, v: Value| {
        d.metadata.insert(k.into(), v);
    };
    put("title", Value::String(title));
    put("authors", json!(authors));
    if !journal.is_empty() {
        put("journal", Value::String(journal));
    }
    if !pub_date.is_empty() {
        put("pub_date", Value::String(pub_date));
    }
    put(
        "doi",
        doi.map(Value::String).unwrap_or(Value::Null),
    );
    put(
        "pmcid",
        pmcid.map(Value::String).unwrap_or(Value::Null),
    );
    if !mesh.is_empty() {
        put("mesh", json!(mesh));
    }
    if !pmid.is_empty() {
        put("pmid", Value::String(pmid));
    }
    put("source", Value::String("pubmed".into()));
    d
}

/// PubMed abstracts are often structured: multiple `<AbstractText
/// Label="BACKGROUND">...</AbstractText>` segments. We concatenate with
/// the label as a heading so the embedder sees the section structure.
fn collect_abstract_sections(article_xml: &str) -> String {
    let mut out = String::new();
    for c in ABSTRACT_TEXT_RE.captures_iter(article_xml) {
        let attrs = &c[1];
        let body = collapse_ws(&strip_inline_tags(&decode_entities(&c[2])));
        if body.is_empty() {
            continue;
        }
        let label = extract_attr(attrs, "Label");
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        if let Some(l) = label {
            out.push_str(&l);
            out.push_str(": ");
        }
        out.push_str(&body);
    }
    out
}

fn collect_authors(article_xml: &str) -> Vec<String> {
    let mut authors = Vec::new();
    for c in AUTHOR_RE.captures_iter(article_xml) {
        let body = &c[1];
        if let Some(coll) = first_capture(&COLLECTIVE_NAME_RE, body) {
            authors.push(collapse_ws(&decode_entities(coll)));
            continue;
        }
        let last = first_capture(&LAST_NAME_RE, body)
            .map(|s| collapse_ws(&decode_entities(s)))
            .unwrap_or_default();
        let fore = first_capture(&FORE_NAME_RE, body)
            .map(|s| collapse_ws(&decode_entities(s)))
            .unwrap_or_default();
        let full = match (fore.is_empty(), last.is_empty()) {
            (false, false) => format!("{fore} {last}"),
            (true, false) => last,
            (false, true) => fore,
            _ => continue,
        };
        if !full.is_empty() {
            authors.push(full);
        }
    }
    authors
}

fn parse_pub_date(article_xml: &str) -> String {
    let block = match first_capture(&PUBDATE_RE, article_xml) {
        Some(b) => b,
        None => return String::new(),
    };
    // Some records use a freeform <MedlineDate> like "2020 Jan-Feb"; pass
    // it through verbatim — caller can parse if they really need.
    if let Some(m) = first_capture(&MEDLINE_DATE_RE, block) {
        return m.trim().to_string();
    }
    let year = first_capture(&PUBDATE_YEAR_RE, block).unwrap_or("").trim();
    let month_raw = first_capture(&PUBDATE_MONTH_RE, block).unwrap_or("").trim();
    let day = first_capture(&PUBDATE_DAY_RE, block).unwrap_or("").trim();
    if year.is_empty() {
        return String::new();
    }
    let month_num = normalize_month(month_raw);
    match (month_num, day.is_empty()) {
        (Some(m), false) => format!("{year}-{m:02}-{:02}", day.parse::<u32>().unwrap_or(1)),
        (Some(m), true) => format!("{year}-{m:02}"),
        (None, _) => year.to_string(),
    }
}

/// PubMed `Month` is sometimes `01`, sometimes `Jan`. Normalise to 1-12.
fn normalize_month(s: &str) -> Option<u32> {
    if s.is_empty() {
        return None;
    }
    if let Ok(n) = s.parse::<u32>() {
        if (1..=12).contains(&n) {
            return Some(n);
        }
    }
    let lower = s.to_lowercase();
    let table = [
        ("jan", 1), ("feb", 2), ("mar", 3), ("apr", 4),
        ("may", 5), ("jun", 6), ("jul", 7), ("aug", 8),
        ("sep", 9), ("oct", 10), ("nov", 11), ("dec", 12),
    ];
    for (prefix, n) in table {
        if lower.starts_with(prefix) {
            return Some(n);
        }
    }
    None
}

fn collect_article_ids(article_xml: &str) -> (Option<String>, Option<String>) {
    let mut doi = None;
    let mut pmcid = None;
    for c in ARTICLE_ID_RE.captures_iter(article_xml) {
        let attrs = &c[1];
        let value = collapse_ws(&decode_entities(&c[2]));
        let id_type = first_capture(&ID_TYPE_ATTR_RE, attrs)
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        match id_type.as_str() {
            "doi" => doi = Some(value),
            "pmc" | "pmcid" => pmcid = Some(value),
            _ => {}
        }
    }
    (doi, pmcid)
}

fn collect_mesh(article_xml: &str) -> Vec<String> {
    MESH_DESC_RE
        .captures_iter(article_xml)
        .map(|c| collapse_ws(&decode_entities(&c[1])))
        .filter(|s| !s.is_empty())
        .collect()
}

fn extract_attr(attrs: &str, name: &str) -> Option<String> {
    let pat = format!(r#"(?is)\b{}\s*=\s*"([^"]*)""#, regex::escape(name));
    let re = Regex::new(&pat).ok()?;
    re.captures(attrs).map(|c| c[1].to_string())
}

/// Decode the handful of XML entities NCBI emits. (Same five as arXiv.)
fn decode_entities(s: &str) -> String {
    s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
}

/// PubMed titles + abstracts can contain inline markup like `<i>...</i>`,
/// `<sub>...</sub>`, `<sup>...</sup>`, `<b>...</b>`. Strip them so
/// embeddings don't index angle-bracket noise.
fn strip_inline_tags(s: &str) -> String {
    static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<[^>]+>").expect("tag re"));
    TAG_RE.replace_all(s, "").into_owned()
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One real-shape article. Covers structured abstract, multiple
    /// authors, DOI + PMC, MeSH, mixed numeric/text month.
    const ARTICLE_FIXTURE: &str = r#"<?xml version="1.0"?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE" Owner="NLM">
      <PMID Version="1">12345678</PMID>
      <Article PubModel="Print">
        <Journal>
          <ISSN>1234-5678</ISSN>
          <Title>Journal of Foo</Title>
        </Journal>
        <ArticleTitle>A study of <i>Foo</i> &amp; Bar in <sub>2</sub>D.</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND" NlmCategory="BACKGROUND">Foo is hard.</AbstractText>
          <AbstractText Label="METHODS">We tried things.</AbstractText>
          <AbstractText Label="RESULTS">It worked.</AbstractText>
        </Abstract>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y">
            <LastName>Smith</LastName>
            <ForeName>Jane A</ForeName>
          </Author>
          <Author ValidYN="Y">
            <LastName>Doe</LastName>
            <ForeName>John</ForeName>
          </Author>
          <Author ValidYN="Y">
            <CollectiveName>The Foo Consortium</CollectiveName>
          </Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">12345678</ArticleId>
        <ArticleId IdType="doi">10.1234/foo.2024.001</ArticleId>
        <ArticleId IdType="pmc">PMC9999999</ArticleId>
      </ArticleIdList>
      <History>
        <PubMedPubDate PubStatus="received"><Year>2024</Year><Month>01</Month><Day>15</Day></PubMedPubDate>
      </History>
    </PubmedData>
    <MedlineCitation>
      <Article>
        <Journal><PubDate><Year>2024</Year><Month>Mar</Month><Day>5</Day></PubDate></Journal>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName UI="D000001">Foo</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName UI="D000002">Bar</DescriptorName></MeshHeading>
      </MeshHeadingList>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>"#;

    #[test]
    fn parses_one_article() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn pmid_lands_in_id_and_metadata() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let d = &docs[0];
        assert_eq!(d.id.as_deref(), Some("12345678"));
        assert_eq!(d.metadata.get("pmid").and_then(|v| v.as_str()), Some("12345678"));
    }

    #[test]
    fn title_strips_inline_markup_and_decodes_entities() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let title = docs[0].metadata.get("title").and_then(|v| v.as_str()).unwrap();
        assert_eq!(title, "A study of Foo & Bar in 2D.");
    }

    #[test]
    fn structured_abstract_concatenates_with_section_labels() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let body = &docs[0].content;
        assert!(body.contains("BACKGROUND: Foo is hard."), "got: {body}");
        assert!(body.contains("METHODS: We tried things."));
        assert!(body.contains("RESULTS: It worked."));
        // Sections joined by blank line.
        assert!(body.contains("\n\n"));
    }

    #[test]
    fn authors_handle_individuals_and_collectives() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let authors = docs[0]
            .metadata
            .get("authors")
            .and_then(|v| v.as_array())
            .unwrap();
        let names: Vec<&str> = authors.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(names, vec!["Jane A Smith", "John Doe", "The Foo Consortium"]);
    }

    #[test]
    fn doi_and_pmcid_extracted_via_attribute_match() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        assert_eq!(
            docs[0].metadata.get("doi").and_then(|v| v.as_str()),
            Some("10.1234/foo.2024.001")
        );
        assert_eq!(
            docs[0].metadata.get("pmcid").and_then(|v| v.as_str()),
            Some("PMC9999999")
        );
    }

    #[test]
    fn pub_date_normalises_text_month() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let date = docs[0].metadata.get("pub_date").and_then(|v| v.as_str()).unwrap();
        assert_eq!(date, "2024-03-05");
    }

    #[test]
    fn medline_date_passes_through() {
        let xml = r#"<PubmedArticle><MedlineCitation><PMID>1</PMID>
            <Article><PubDate><MedlineDate>2020 Jan-Feb</MedlineDate></PubDate>
            <ArticleTitle>x</ArticleTitle></Article></MedlineCitation></PubmedArticle>"#;
        let docs = parse_efetch_xml(xml);
        assert_eq!(
            docs[0].metadata.get("pub_date").and_then(|v| v.as_str()),
            Some("2020 Jan-Feb")
        );
    }

    #[test]
    fn pub_date_year_only_when_month_missing() {
        let xml = r#"<PubmedArticle><MedlineCitation><PMID>2</PMID>
            <Article><PubDate><Year>1999</Year></PubDate>
            <ArticleTitle>x</ArticleTitle></Article></MedlineCitation></PubmedArticle>"#;
        let docs = parse_efetch_xml(xml);
        assert_eq!(
            docs[0].metadata.get("pub_date").and_then(|v| v.as_str()),
            Some("1999")
        );
    }

    #[test]
    fn pub_date_year_month_when_day_missing() {
        let xml = r#"<PubmedArticle><MedlineCitation><PMID>3</PMID>
            <Article><PubDate><Year>2010</Year><Month>07</Month></PubDate>
            <ArticleTitle>x</ArticleTitle></Article></MedlineCitation></PubmedArticle>"#;
        let docs = parse_efetch_xml(xml);
        assert_eq!(
            docs[0].metadata.get("pub_date").and_then(|v| v.as_str()),
            Some("2010-07")
        );
    }

    #[test]
    fn mesh_descriptors_extracted() {
        let docs = parse_efetch_xml(ARTICLE_FIXTURE);
        let mesh = docs[0]
            .metadata
            .get("mesh")
            .and_then(|v| v.as_array())
            .unwrap();
        let strs: Vec<&str> = mesh.iter().filter_map(|v| v.as_str()).collect();
        assert_eq!(strs, vec!["Foo", "Bar"]);
    }

    #[test]
    fn esearch_id_list_extracted() {
        let body = json!({
            "header": {"type": "esearch", "version": "0.3"},
            "esearchresult": {
                "count": "2",
                "retmax": "2",
                "retstart": "0",
                "idlist": ["111", "222"],
                "translationset": [],
                "querytranslation": "x"
            }
        });
        assert_eq!(parse_esearch_ids(&body), vec!["111".to_string(), "222".to_string()]);
    }

    #[test]
    fn esearch_empty_results_yields_empty_vec() {
        let body = json!({"esearchresult": {"idlist": []}});
        assert!(parse_esearch_ids(&body).is_empty());
    }

    #[test]
    fn esearch_malformed_yields_empty_vec() {
        let body = json!({"foo": "bar"});
        assert!(parse_esearch_ids(&body).is_empty());
    }

    #[test]
    fn empty_efetch_yields_empty_vec() {
        let xml = r#"<?xml version="1.0"?><PubmedArticleSet></PubmedArticleSet>"#;
        assert!(parse_efetch_xml(xml).is_empty());
    }

    #[test]
    fn no_abstract_yields_empty_content_not_panic() {
        let xml = r#"<PubmedArticle><MedlineCitation><PMID>9</PMID>
            <Article><ArticleTitle>Letter</ArticleTitle></Article>
            </MedlineCitation></PubmedArticle>"#;
        let docs = parse_efetch_xml(xml);
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.is_empty());
    }

    #[test]
    fn missing_doi_pmcid_yields_null() {
        let xml = r#"<PubmedArticle><MedlineCitation><PMID>10</PMID>
            <Article><ArticleTitle>x</ArticleTitle></Article></MedlineCitation>
            </PubmedArticle>"#;
        let docs = parse_efetch_xml(xml);
        assert!(docs[0].metadata.get("doi").map(|v| v.is_null()).unwrap_or(false));
        assert!(docs[0].metadata.get("pmcid").map(|v| v.is_null()).unwrap_or(false));
    }

    #[test]
    fn normalize_month_handles_numeric_and_text() {
        assert_eq!(normalize_month("01"), Some(1));
        assert_eq!(normalize_month("12"), Some(12));
        assert_eq!(normalize_month("Jan"), Some(1));
        assert_eq!(normalize_month("dec"), Some(12));
        assert_eq!(normalize_month("September"), Some(9));
        assert_eq!(normalize_month(""), None);
        assert_eq!(normalize_month("13"), None);
        assert_eq!(normalize_month("foo"), None);
    }

    #[test]
    fn search_constructor_defaults_max_results_to_10() {
        let l = PubMedLoader::search("x");
        assert_eq!(l.max_results, 10);
    }

    #[test]
    fn pmids_constructor_uses_unbounded_cap() {
        let l = PubMedLoader::pmids(vec!["1".into()]);
        assert_eq!(l.max_results, usize::MAX);
    }

    #[test]
    fn api_key_and_email_persist() {
        let l = PubMedLoader::search("x")
            .with_api_key("KEY")
            .with_email("me@example.com");
        assert_eq!(l.api_key.as_deref(), Some("KEY"));
        assert_eq!(l.email.as_deref(), Some("me@example.com"));
    }
}

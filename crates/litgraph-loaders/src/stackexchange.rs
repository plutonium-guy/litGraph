//! `StackExchangeLoader` — fetch questions + answers from any
//! Stack Exchange site via the public API at
//! `api.stackexchange.com/2.3/`.
//!
//! Real prod scenarios:
//! - **Technical agents grounded on community Q&A**: ingest a
//!   tag's recent questions on Stack Overflow / Server Fault /
//!   Database Administrators / Math.SE / 100+ other sites.
//! - **Trend monitoring**: pull "questions tagged `rust`" daily
//!   to spot emerging issues users are running into.
//! - **Documentation supplementation**: pair authoritative docs
//!   (loaded via the right loader) with community Q&A for richer
//!   RAG context.
//!
//! # API key
//!
//! Optional. Without a key, the API allows ~300 req/day per IP.
//! With a key (free, register at stackapps.com), ~10k req/day.
//! Set via `with_key`.
//!
//! # Filter knobs
//!
//! - `with_site("stackoverflow")` — required-ish; defaults to
//!   `stackoverflow`. Other examples: `serverfault`, `superuser`,
//!   `unix`, `dba`, `math`, `physics`.
//! - `with_tags(["rust", "tokio"])` — restrict to questions
//!   carrying any of these tags.
//! - `with_max_questions(n)` — total cap (default 30).
//! - `with_include_answers(true)` — fetch each question's
//!   accepted answer body and concat into the document.
//!
//! # Output
//!
//! Each question → one Document with:
//! - `content`: title + question_body (+ optionally accepted
//!   answer body).
//! - `id`: `stackexchange:{site}#{question_id}`.
//! - `metadata`: `{site, question_id, title, tags, score, view_count,
//!   answer_count, is_answered, link, owner}`.

use std::time::Duration;

use litgraph_core::Document;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;

use crate::{Loader, LoaderError, LoaderResult};

const BASE: &str = "https://api.stackexchange.com/2.3";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_QUESTIONS: usize = 30;

#[derive(Debug, Clone)]
pub struct StackExchangeLoader {
    pub site: String,
    pub tags: Vec<String>,
    pub api_key: Option<String>,
    pub max_questions: usize,
    pub include_answers: bool,
    pub timeout: Duration,
    pub user_agent: String,
    /// Filter to use for question body. `withbody` is the
    /// Stack Exchange API filter that includes the rendered HTML
    /// `body` field in question results.
    pub filter: String,
    pub base_url: String,
}

impl StackExchangeLoader {
    pub fn new() -> Self {
        Self {
            site: "stackoverflow".into(),
            tags: Vec::new(),
            api_key: None,
            max_questions: DEFAULT_MAX_QUESTIONS,
            include_answers: false,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
            filter: "withbody".into(),
            base_url: BASE.into(),
        }
    }

    pub fn with_site(mut self, site: impl Into<String>) -> Self {
        self.site = site.into();
        self
    }
    pub fn with_tags<I, S>(mut self, tags: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.tags = tags.into_iter().map(Into::into).collect();
        self
    }
    pub fn with_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }
    pub fn with_max_questions(mut self, n: usize) -> Self {
        self.max_questions = n;
        self
    }
    pub fn with_include_answers(mut self, b: bool) -> Self {
        self.include_answers = b;
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
    pub fn with_filter(mut self, f: impl Into<String>) -> Self {
        self.filter = f.into();
        self
    }
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    fn http(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .user_agent(&self.user_agent)
            .timeout(self.timeout)
            .build()
            .map_err(|e| LoaderError::Other(format!("client build: {e}")))
    }

    fn questions_url(&self, page: u32) -> String {
        let mut url = format!(
            "{}/questions?site={}&order=desc&sort=activity&pagesize=100&page={}&filter={}",
            self.base_url.trim_end_matches('/'),
            self.site,
            page,
            self.filter,
        );
        if !self.tags.is_empty() {
            url.push_str("&tagged=");
            url.push_str(&self.tags.join(";"));
        }
        if let Some(k) = &self.api_key {
            url.push_str("&key=");
            url.push_str(k);
        }
        url
    }

    fn answers_url(&self, ids: &[u64]) -> String {
        let csv = ids
            .iter()
            .map(|i| i.to_string())
            .collect::<Vec<_>>()
            .join(";");
        let mut url = format!(
            "{}/questions/{}/answers?site={}&order=desc&sort=votes&pagesize=10&filter={}",
            self.base_url.trim_end_matches('/'),
            csv,
            self.site,
            self.filter,
        );
        if let Some(k) = &self.api_key {
            url.push_str("&key=");
            url.push_str(k);
        }
        url
    }

    fn fetch_questions(
        &self,
        client: &reqwest::blocking::Client,
    ) -> LoaderResult<Vec<Question>> {
        let mut all: Vec<Question> = Vec::new();
        let mut page = 1u32;
        while all.len() < self.max_questions {
            let url = self.questions_url(page);
            let resp = client
                .get(&url)
                .send()
                .map_err(|e| LoaderError::Other(format!("fetch {url}: {e}")))?;
            let status = resp.status();
            let body = resp
                .text()
                .map_err(|e| LoaderError::Other(format!("read body: {e}")))?;
            if !status.is_success() {
                return Err(LoaderError::Other(format!(
                    "stackexchange {status}: {}",
                    body.chars().take(200).collect::<String>(),
                )));
            }
            let parsed: QuestionsResponse =
                serde_json::from_str(&body).map_err(|e| {
                    LoaderError::Other(format!("parse questions: {e}"))
                })?;
            let got = parsed.items.len();
            for q in parsed.items {
                if all.len() >= self.max_questions {
                    break;
                }
                all.push(q);
            }
            if !parsed.has_more.unwrap_or(false) || got == 0 {
                break;
            }
            page += 1;
        }
        Ok(all)
    }

    /// Convert a question (and its accepted answer text, if any)
    /// into a Document. Public for offline tests.
    pub fn question_to_document(
        &self,
        q: &Question,
        accepted_answer_body: Option<&str>,
    ) -> Document {
        let mut content = String::new();
        content.push_str(&q.title);
        content.push_str("\n\n");
        if let Some(body) = &q.body {
            content.push_str(body);
        }
        if let Some(ans) = accepted_answer_body {
            content.push_str("\n\n--- Accepted Answer ---\n");
            content.push_str(ans);
        }
        let mut doc = Document::new(content).with_id(format!(
            "stackexchange:{}#{}",
            self.site, q.question_id,
        ));
        doc.metadata.insert("site".into(), json!(self.site));
        doc.metadata.insert("question_id".into(), json!(q.question_id));
        doc.metadata.insert("title".into(), json!(q.title));
        if !q.tags.is_empty() {
            doc.metadata.insert("tags".into(), json!(q.tags));
        }
        if let Some(s) = q.score {
            doc.metadata.insert("score".into(), json!(s));
        }
        if let Some(v) = q.view_count {
            doc.metadata.insert("view_count".into(), json!(v));
        }
        if let Some(a) = q.answer_count {
            doc.metadata.insert("answer_count".into(), json!(a));
        }
        if let Some(b) = q.is_answered {
            doc.metadata.insert("is_answered".into(), json!(b));
        }
        if let Some(l) = &q.link {
            doc.metadata.insert("link".into(), json!(l));
        }
        if let Some(o) = q.owner.as_ref().and_then(|o| o.display_name.clone()) {
            doc.metadata.insert("owner".into(), json!(o));
        }
        doc
    }
}

impl Default for StackExchangeLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl Loader for StackExchangeLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.http()?;
        let questions = self.fetch_questions(&client)?;

        // Optionally fetch accepted answers for each question.
        let answers_by_qid: std::collections::HashMap<u64, String> = if self.include_answers {
            let ids: Vec<u64> = questions.iter().map(|q| q.question_id).collect();
            // Fetch in batches of up to 100 (API cap).
            let mut accepted: std::collections::HashMap<u64, String> =
                std::collections::HashMap::new();
            let chunks: Vec<&[u64]> = ids.chunks(100).collect();
            // Rayon-parallel fetch across chunks.
            let user_agent = self.user_agent.clone();
            let timeout = self.timeout;
            let urls: Vec<String> = chunks.iter().map(|c| self.answers_url(c)).collect();
            let bodies: Vec<Option<String>> = urls
                .par_iter()
                .map(|url| {
                    let client = reqwest::blocking::Client::builder()
                        .user_agent(&user_agent)
                        .timeout(timeout)
                        .build()
                        .ok()?;
                    let resp = client.get(url).send().ok()?;
                    if !resp.status().is_success() {
                        return None;
                    }
                    resp.text().ok()
                })
                .collect();
            for body in bodies.into_iter().flatten() {
                let parsed: Result<AnswersResponse, _> = serde_json::from_str(&body);
                let Ok(parsed) = parsed else { continue };
                for a in parsed.items {
                    let entry = accepted.entry(a.question_id).or_default();
                    // Pick the highest-scoring answer body. The API
                    // sort=votes returns highest-first; first wins.
                    if entry.is_empty() {
                        if let Some(b) = a.body {
                            *entry = b;
                        }
                    }
                }
            }
            accepted
        } else {
            std::collections::HashMap::new()
        };

        let docs = questions
            .iter()
            .map(|q| {
                self.question_to_document(
                    q,
                    answers_by_qid.get(&q.question_id).map(|s| s.as_str()),
                )
            })
            .collect();
        Ok(docs)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct Question {
    pub question_id: u64,
    pub title: String,
    pub body: Option<String>,
    #[serde(default)]
    pub tags: Vec<String>,
    pub score: Option<i64>,
    pub view_count: Option<i64>,
    pub answer_count: Option<i64>,
    pub is_answered: Option<bool>,
    pub link: Option<String>,
    pub owner: Option<Owner>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Owner {
    pub display_name: Option<String>,
}

#[derive(Debug, Deserialize)]
struct QuestionsResponse {
    items: Vec<Question>,
    has_more: Option<bool>,
}

#[derive(Debug, Deserialize)]
struct Answer {
    question_id: u64,
    body: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnswersResponse {
    items: Vec<Answer>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_question() -> Question {
        Question {
            question_id: 12345,
            title: "How do I do X in Rust?".into(),
            body: Some("<p>I want to do X but the borrow checker complains.</p>".into()),
            tags: vec!["rust".into(), "borrow-checker".into()],
            score: Some(42),
            view_count: Some(1000),
            answer_count: Some(3),
            is_answered: Some(true),
            link: Some("https://stackoverflow.com/q/12345".into()),
            owner: Some(Owner {
                display_name: Some("alice".into()),
            }),
        }
    }

    #[test]
    fn question_to_document_basic() {
        let l = StackExchangeLoader::new();
        let q = fixture_question();
        let doc = l.question_to_document(&q, None);
        assert!(doc.content.contains("How do I do X in Rust?"));
        assert!(doc.content.contains("borrow checker complains"));
        assert_eq!(doc.id.as_deref(), Some("stackexchange:stackoverflow#12345"));
        assert_eq!(
            doc.metadata.get("question_id").and_then(|v| v.as_i64()),
            Some(12345),
        );
        assert_eq!(
            doc.metadata.get("score").and_then(|v| v.as_i64()),
            Some(42),
        );
        assert_eq!(
            doc.metadata.get("owner").and_then(|v| v.as_str()),
            Some("alice"),
        );
        let tags = doc.metadata.get("tags").and_then(|v| v.as_array()).unwrap();
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn question_with_accepted_answer_concatenated() {
        let l = StackExchangeLoader::new();
        let q = fixture_question();
        let doc = l.question_to_document(&q, Some("<p>Use Arc&lt;Mutex&gt;.</p>"));
        assert!(doc.content.contains("--- Accepted Answer ---"));
        assert!(doc.content.contains("Use Arc"));
    }

    #[test]
    fn question_without_body_falls_back_to_title() {
        let l = StackExchangeLoader::new();
        let q = Question {
            question_id: 7,
            title: "title only".into(),
            body: None,
            ..fixture_question()
        };
        let doc = l.question_to_document(&q, None);
        assert!(doc.content.contains("title only"));
    }

    #[test]
    fn questions_url_includes_tags_and_key() {
        let l = StackExchangeLoader::new()
            .with_site("serverfault")
            .with_tags(["nginx", "https"])
            .with_key("test_key_xyz");
        let url = l.questions_url(2);
        assert!(url.contains("/questions"));
        assert!(url.contains("site=serverfault"));
        assert!(url.contains("tagged=nginx;https"));
        assert!(url.contains("page=2"));
        assert!(url.contains("key=test_key_xyz"));
        assert!(url.contains("filter=withbody"));
    }

    #[test]
    fn questions_url_no_tags_no_key() {
        let l = StackExchangeLoader::new();
        let url = l.questions_url(1);
        assert!(!url.contains("tagged="));
        assert!(!url.contains("key="));
        assert!(url.contains("site=stackoverflow"));
    }

    #[test]
    fn answers_url_includes_csv_ids() {
        let l = StackExchangeLoader::new();
        let url = l.answers_url(&[1, 2, 3]);
        assert!(url.contains("/questions/1;2;3/answers"));
    }

    #[test]
    fn with_base_url_overrides_default() {
        let l = StackExchangeLoader::new().with_base_url("http://localhost:8080");
        assert!(l.questions_url(1).starts_with("http://localhost:8080"));
    }

    #[test]
    fn site_metadata_propagates_to_id_format() {
        let l = StackExchangeLoader::new().with_site("math");
        let q = fixture_question();
        let doc = l.question_to_document(&q, None);
        assert_eq!(doc.id.as_deref(), Some("stackexchange:math#12345"));
        assert_eq!(
            doc.metadata.get("site").and_then(|v| v.as_str()),
            Some("math"),
        );
    }
}

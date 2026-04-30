//! `HackerNewsLoader` — fetch stories from the public HN
//! Firebase API.
//!
//! No auth, no rate-limit headers; the API at
//! `https://hacker-news.firebaseio.com/v0/` is meant to be hit
//! freely. We still default to a small `max_items` and include
//! User-Agent + timeout knobs.
//!
//! # Feed sources
//!
//! - `Top` — `topstories.json` (front page)
//! - `New` — `newstories.json`
//! - `Best` — `beststories.json`
//! - `Ask` — `askstories.json`
//! - `Show` — `showstories.json`
//! - `Job` — `jobstories.json`
//!
//! Each endpoint returns up to ~500 item IDs as a JSON array;
//! the loader fetches the first `max_items` IDs and pulls each
//! item's metadata in parallel via Rayon.
//!
//! # Real prod use
//!
//! - **Tech-news brief**: agent summarizes the day's top 30 HN
//!   stories.
//! - **Title-only zero-shot embedding corpus**: cheap to build,
//!   useful for testing semantic search.
//! - **Trend monitoring**: ingest "Show HN" or "Ask HN" feeds
//!   over time to spot themes.

use std::time::Duration;

use litgraph_core::Document;
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;

use crate::{Loader, LoaderError, LoaderResult};

const BASE: &str = "https://hacker-news.firebaseio.com/v0";
const DEFAULT_TIMEOUT_SECS: u64 = 30;
const DEFAULT_MAX_ITEMS: usize = 30;

/// Which HN feed to fetch.
#[derive(Debug, Clone, Copy)]
pub enum HnFeed {
    Top,
    New,
    Best,
    Ask,
    Show,
    Job,
}

impl HnFeed {
    fn endpoint(&self) -> &'static str {
        match self {
            HnFeed::Top => "topstories.json",
            HnFeed::New => "newstories.json",
            HnFeed::Best => "beststories.json",
            HnFeed::Ask => "askstories.json",
            HnFeed::Show => "showstories.json",
            HnFeed::Job => "jobstories.json",
        }
    }

    fn name(&self) -> &'static str {
        match self {
            HnFeed::Top => "top",
            HnFeed::New => "new",
            HnFeed::Best => "best",
            HnFeed::Ask => "ask",
            HnFeed::Show => "show",
            HnFeed::Job => "job",
        }
    }
}

#[derive(Debug, Clone)]
pub struct HackerNewsLoader {
    pub feed: HnFeed,
    pub timeout: Duration,
    pub user_agent: String,
    pub max_items: usize,
    /// Optional override of the API base URL — primarily for
    /// testing against a local fake server.
    pub base_url: String,
}

impl HackerNewsLoader {
    pub fn new(feed: HnFeed) -> Self {
        Self {
            feed,
            timeout: Duration::from_secs(DEFAULT_TIMEOUT_SECS),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
            max_items: DEFAULT_MAX_ITEMS,
            base_url: BASE.to_string(),
        }
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }
    pub fn with_max_items(mut self, n: usize) -> Self {
        self.max_items = n;
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

    fn fetch_ids(&self, client: &reqwest::blocking::Client) -> LoaderResult<Vec<u64>> {
        let url = format!("{}/{}", self.base_url, self.feed.endpoint());
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
                "{url} returned {status}: {}",
                body.chars().take(200).collect::<String>(),
            )));
        }
        let ids: Vec<u64> = serde_json::from_str(&body)
            .map_err(|e| LoaderError::Other(format!("parse ids json: {e}")))?;
        Ok(ids)
    }

    fn fetch_item(
        client: &reqwest::blocking::Client,
        base: &str,
        id: u64,
    ) -> LoaderResult<HnItem> {
        let url = format!("{base}/item/{id}.json");
        let resp = client
            .get(&url)
            .send()
            .map_err(|e| LoaderError::Other(format!("fetch {url}: {e}")))?;
        let body = resp
            .text()
            .map_err(|e| LoaderError::Other(format!("read body: {e}")))?;
        let item: HnItem = serde_json::from_str(&body)
            .map_err(|e| LoaderError::Other(format!("parse item {id}: {e}")))?;
        Ok(item)
    }
}

impl Loader for HackerNewsLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.http()?;
        let mut ids = self.fetch_ids(&client)?;
        ids.truncate(self.max_items);

        let base = self.base_url.clone();
        let user_agent = self.user_agent.clone();
        let timeout = self.timeout;
        let feed_name = self.feed.name();

        // Rayon-parallel item fetch — the per-item HTTP calls are
        // independent and HN's API is happy to be hammered. Errors
        // for individual items are ignored (story may be deleted).
        let items: Vec<HnItem> = ids
            .par_iter()
            .filter_map(|id| {
                let c = reqwest::blocking::Client::builder()
                    .user_agent(&user_agent)
                    .timeout(timeout)
                    .build()
                    .ok()?;
                Self::fetch_item(&c, &base, *id).ok()
            })
            .collect();

        Ok(items
            .into_iter()
            .filter_map(|it| it.into_document(feed_name))
            .collect())
    }
}

/// One HN item as returned by the API. Public for callers that
/// want to use the loader's parser without going through
/// `Loader::load`.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct HnItem {
    pub id: Option<u64>,
    pub title: Option<String>,
    pub url: Option<String>,
    pub by: Option<String>,
    pub score: Option<i64>,
    pub time: Option<i64>,
    pub r#type: Option<String>,
    /// `text` is HN-rendered HTML for `Ask`/`Show`/`comment` items.
    pub text: Option<String>,
    pub descendants: Option<i64>,
}

impl HnItem {
    /// Convert to a Document. Returns None if the item was deleted
    /// / dead / lacks both title and text (nothing to index).
    pub fn into_document(self, feed_name: &str) -> Option<Document> {
        let id = self.id?;
        let title = self.title.unwrap_or_default();
        let body = self.text.clone().unwrap_or_default();
        if title.is_empty() && body.is_empty() {
            return None;
        }
        let content = if body.is_empty() {
            title.clone()
        } else if title.is_empty() {
            body.clone()
        } else {
            format!("{title}\n\n{body}")
        };
        let mut doc = Document::new(content)
            .with_id(format!("hn:{id}"));
        doc.metadata.insert("hn_id".into(), json!(id));
        doc.metadata.insert("title".into(), json!(title));
        if let Some(u) = &self.url {
            doc.metadata.insert("url".into(), json!(u));
        }
        if let Some(by) = &self.by {
            doc.metadata.insert("by".into(), json!(by));
        }
        if let Some(s) = self.score {
            doc.metadata.insert("score".into(), json!(s));
        }
        if let Some(t) = self.time {
            doc.metadata.insert("time".into(), json!(t));
        }
        if let Some(t) = &self.r#type {
            doc.metadata.insert("type".into(), json!(t));
        }
        if let Some(n) = self.descendants {
            doc.metadata.insert("descendants".into(), json!(n));
        }
        doc.metadata.insert("feed".into(), json!(feed_name));
        Some(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn item_into_document_combines_title_and_text() {
        let it = HnItem {
            id: Some(1),
            title: Some("Hello HN".into()),
            text: Some("<p>Body text</p>".into()),
            url: Some("https://example.com".into()),
            score: Some(42),
            by: Some("alice".into()),
            time: Some(1700000000),
            r#type: Some("story".into()),
            descendants: Some(5),
        };
        let doc = it.into_document("top").unwrap();
        assert!(doc.content.contains("Hello HN"));
        assert!(doc.content.contains("Body text"));
        assert_eq!(doc.id.as_deref(), Some("hn:1"));
        assert_eq!(
            doc.metadata.get("hn_id").and_then(|v| v.as_u64()),
            Some(1),
        );
        assert_eq!(
            doc.metadata.get("score").and_then(|v| v.as_i64()),
            Some(42),
        );
        assert_eq!(
            doc.metadata.get("feed").and_then(|v| v.as_str()),
            Some("top"),
        );
    }

    #[test]
    fn item_into_document_title_only() {
        let it = HnItem {
            id: Some(2),
            title: Some("Just a title".into()),
            ..Default::default()
        };
        let doc = it.into_document("new").unwrap();
        assert_eq!(doc.content, "Just a title");
    }

    #[test]
    fn item_into_document_drops_when_no_id() {
        let it = HnItem {
            id: None,
            title: Some("orphan".into()),
            ..Default::default()
        };
        assert!(it.into_document("top").is_none());
    }

    #[test]
    fn item_into_document_drops_when_empty() {
        let it = HnItem {
            id: Some(3),
            ..Default::default()
        };
        assert!(it.into_document("top").is_none());
    }

    #[test]
    fn feed_endpoints_are_correct() {
        assert_eq!(HnFeed::Top.endpoint(), "topstories.json");
        assert_eq!(HnFeed::New.endpoint(), "newstories.json");
        assert_eq!(HnFeed::Best.endpoint(), "beststories.json");
        assert_eq!(HnFeed::Ask.endpoint(), "askstories.json");
        assert_eq!(HnFeed::Show.endpoint(), "showstories.json");
        assert_eq!(HnFeed::Job.endpoint(), "jobstories.json");
    }

    #[test]
    fn loader_default_max_items_is_30() {
        let l = HackerNewsLoader::new(HnFeed::Top);
        assert_eq!(l.max_items, 30);
    }

    #[test]
    fn with_max_items_caps() {
        let l = HackerNewsLoader::new(HnFeed::Top).with_max_items(5);
        assert_eq!(l.max_items, 5);
    }

    #[test]
    fn with_base_url_overrides_default() {
        let l = HackerNewsLoader::new(HnFeed::Top).with_base_url("http://localhost:1234");
        assert_eq!(l.base_url, "http://localhost:1234");
    }
}

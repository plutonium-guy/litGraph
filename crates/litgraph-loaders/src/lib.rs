//! Document loaders. The trait is sync — loaders are filesystem-bound, not async.
//! Batch helpers parallelize across files with Rayon.
//!
//! For a Python-facing async loader, wrap the sync calls in `tokio::task::spawn_blocking`
//! at the PyO3 boundary.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Duration;

use litgraph_core::Document;
use rayon::prelude::*;
use serde_json::Value;

pub mod html;
pub use html::HtmlLoader;

pub mod html_to_markdown;
pub use html_to_markdown::{html_to_markdown, HtmlToMarkdownTransformer};

#[cfg(feature = "pdf")]
pub mod pdf;
#[cfg(feature = "pdf")]
pub use pdf::PdfLoader;

#[cfg(feature = "docx")]
pub mod docx;
#[cfg(feature = "docx")]
pub use docx::DocxLoader;

pub mod notion;
pub use notion::NotionLoader;

pub mod slack;
pub use slack::SlackLoader;

pub mod confluence;
pub use confluence::ConfluenceLoader;

pub mod github;
pub use github::GithubIssuesLoader;

pub mod github_files;
pub use github_files::GithubFilesLoader;

pub mod gmail;
pub use gmail::GmailLoader;

pub mod gdrive;
pub use gdrive::GoogleDriveLoader;

pub mod linear;
pub use linear::LinearIssuesLoader;

pub mod jira;
pub use jira::JiraIssuesLoader;

pub mod s3;
pub mod jupyter;
pub use jupyter::JupyterNotebookLoader;
pub mod gitlab;
pub use gitlab::GitLabIssuesLoader;
pub mod gitlab_files;
pub use gitlab_files::GitLabFilesLoader;
pub mod sitemap;
pub use sitemap::SitemapLoader;
pub use s3::S3Loader;
pub mod arxiv;
pub use arxiv::ArxivLoader;
pub mod wikipedia;
pub use wikipedia::WikipediaLoader;
pub mod pubmed;
pub use pubmed::PubMedLoader;
pub mod http_prompt_hub;
pub use http_prompt_hub::HttpPromptHub;
pub mod youtube;
pub use youtube::YouTubeTranscriptLoader;
pub mod discord;
pub use discord::DiscordChannelLoader;
pub mod outlook;
pub use outlook::OutlookMessagesLoader;
#[cfg(feature = "rss")]
pub mod rss;
#[cfg(feature = "rss")]
pub use rss::{FeedItem, RssAtomLoader};
pub mod hackernews;
pub use hackernews::{HackerNewsLoader, HnFeed, HnItem};
pub mod bitbucket;
pub use bitbucket::BitbucketIssuesLoader;
pub mod bitbucket_files;
pub use bitbucket_files::BitbucketFilesLoader;
pub mod concurrent;
pub use concurrent::{
    load_concurrent, load_concurrent_flat, load_concurrent_stream,
    load_concurrent_stream_with_progress, load_concurrent_stream_with_shutdown,
    load_concurrent_with_progress, load_concurrent_with_shutdown, LoadProgress,
    LoadStreamItem, DEFAULT_LOAD_CONCURRENCY,
};
pub mod ingest;
pub use ingest::{
    ingest_to_stream, ingest_to_stream_with_progress, IngestBatch, IngestConfig, IngestProgress,
};

#[derive(Debug, thiserror::Error)]
pub enum LoaderError {
    #[error("io: {0}")]
    Io(#[from] io::Error),
    #[error("json: {0}")]
    Json(#[from] serde_json::Error),
    #[error("glob: {0}")]
    Pattern(#[from] glob::PatternError),
    #[error("utf-8: {0}")]
    Utf8(#[from] std::string::FromUtf8Error),
    #[error("http: {0}")]
    Http(#[from] reqwest::Error),
    #[error("csv: {0}")]
    Csv(#[from] csv::Error),
    #[error("{0}")]
    Other(String),
}

pub type LoaderResult<T> = Result<T, LoaderError>;

pub trait Loader: Send + Sync {
    fn load(&self) -> LoaderResult<Vec<Document>>;
}

/// Plain UTF-8 text file — one file → one document.
#[derive(Clone)]
pub struct TextLoader {
    pub path: PathBuf,
}

impl TextLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self { Self { path: p.as_ref().to_path_buf() } }
}

impl Loader for TextLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let content = fs::read_to_string(&self.path)?;
        let mut d = Document::new(content).with_id(self.path.to_string_lossy());
        d.metadata.insert("source".into(), Value::String(self.path.to_string_lossy().into()));
        Ok(vec![d])
    }
}

/// JSONL — one line per document. `content_field` selects which JSON key becomes
/// `Document::content`; every other field is attached as metadata.
#[derive(Clone)]
pub struct JsonLinesLoader {
    pub path: PathBuf,
    pub content_field: String,
}

impl JsonLinesLoader {
    pub fn new<P: AsRef<Path>>(p: P, content_field: impl Into<String>) -> Self {
        Self { path: p.as_ref().to_path_buf(), content_field: content_field.into() }
    }
}

impl Loader for JsonLinesLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let raw = fs::read_to_string(&self.path)?;
        let mut out = Vec::new();
        for (i, line) in raw.lines().enumerate() {
            if line.trim().is_empty() { continue; }
            let v: Value = serde_json::from_str(line)?;
            let Some(obj) = v.as_object() else {
                return Err(LoaderError::Other(format!(
                    "{}: line {} is not a JSON object", self.path.display(), i
                )));
            };
            let content = obj
                .get(&self.content_field)
                .and_then(|v| v.as_str())
                .ok_or_else(|| LoaderError::Other(format!(
                    "{}: line {} missing string field `{}`",
                    self.path.display(), i, self.content_field
                )))?
                .to_string();
            let mut d = Document::new(content);
            d.id = Some(format!("{}#{}", self.path.display(), i));
            for (k, v) in obj {
                if k != &self.content_field {
                    d.metadata.insert(k.clone(), v.clone());
                }
            }
            d.metadata.insert("source".into(), Value::String(self.path.to_string_lossy().into()));
            out.push(d);
        }
        Ok(out)
    }
}

/// Single-file JSON loader with optional dot-path selector.
///
/// Three input shapes are supported:
///
///   1. **Bare object** (`{"k": "v"}`)
///      → 1 Document; content = the JSON serialized as compact text. Useful
///        when the agent should reason over the raw structure.
///   2. **Top-level array of objects** (`[{...}, {...}]`)
///      → N Documents. `content_field` (if set) selects a string key; if not
///        set, content = the serialized JSON of the row. Other fields → metadata.
///   3. **Nested array via `pointer`** (e.g. `"results.items"`)
///      → drill into the JSON tree and treat the array at that path as case 2.
///        Dot-separated keys; numeric components index into arrays.
///
/// Errors if the path doesn't resolve, or if a non-array sits at the pointer.
#[derive(Clone)]
pub struct JsonLoader {
    pub path: PathBuf,
    pub pointer: Option<String>,
    pub content_field: Option<String>,
}

impl JsonLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self { path: p.as_ref().to_path_buf(), pointer: None, content_field: None }
    }
    pub fn with_pointer(mut self, p: impl Into<String>) -> Self {
        self.pointer = Some(p.into());
        self
    }
    pub fn with_content_field(mut self, f: impl Into<String>) -> Self {
        self.content_field = Some(f.into());
        self
    }
}

/// Walk dot-separated keys / numeric indices into a JSON tree.
fn json_pointer<'a>(root: &'a Value, dotted: &str) -> Option<&'a Value> {
    let mut cur = root;
    for part in dotted.split('.') {
        cur = if let Ok(idx) = part.parse::<usize>() {
            cur.as_array()?.get(idx)?
        } else {
            cur.as_object()?.get(part)?
        };
    }
    Some(cur)
}

impl Loader for JsonLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let raw = fs::read_to_string(&self.path)?;
        let root: Value = serde_json::from_str(&raw)?;

        let target: &Value = if let Some(ref ptr) = self.pointer {
            json_pointer(&root, ptr).ok_or_else(|| LoaderError::Other(format!(
                "{}: pointer '{}' did not resolve", self.path.display(), ptr
            )))?
        } else {
            &root
        };

        let path_str = self.path.to_string_lossy().into_owned();

        // Branch on shape.
        if let Some(arr) = target.as_array() {
            let mut out = Vec::with_capacity(arr.len());
            for (i, item) in arr.iter().enumerate() {
                let (content, metadata_obj): (String, Option<&serde_json::Map<String, Value>>) =
                    if let (Some(field), Some(obj)) = (&self.content_field, item.as_object()) {
                        let content = obj
                            .get(field)
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| LoaderError::Other(format!(
                                "{}: element {} missing string field `{}`",
                                self.path.display(), i, field
                            )))?
                            .to_string();
                        (content, Some(obj))
                    } else {
                        (serde_json::to_string(item)?, item.as_object())
                    };
                let mut d = Document::new(content)
                    .with_id(format!("{path_str}#{i}"));
                if let Some(obj) = metadata_obj {
                    for (k, v) in obj {
                        if Some(k.as_str()) == self.content_field.as_deref() { continue; }
                        d.metadata.insert(k.clone(), v.clone());
                    }
                }
                d.metadata.insert("source".into(), Value::String(path_str.clone()));
                d.metadata.insert("index".into(), Value::from(i as u64));
                out.push(d);
            }
            Ok(out)
        } else if target.is_object() || target.is_string() || target.is_number() || target.is_boolean() {
            let content = if let (Some(field), Some(obj)) = (&self.content_field, target.as_object()) {
                obj.get(field)
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| LoaderError::Other(format!(
                        "{}: missing string field `{}`", self.path.display(), field
                    )))?
                    .to_string()
            } else {
                serde_json::to_string(target)?
            };
            let mut d = Document::new(content).with_id(path_str.clone());
            if let Some(obj) = target.as_object() {
                for (k, v) in obj {
                    if Some(k.as_str()) == self.content_field.as_deref() { continue; }
                    d.metadata.insert(k.clone(), v.clone());
                }
            }
            d.metadata.insert("source".into(), Value::String(path_str));
            Ok(vec![d])
        } else {
            // null
            Ok(vec![])
        }
    }
}

/// CSV (or any delimited text) — one row per `Document`. Two content modes:
///
/// - `Some(col)`: `Document::content` = that column's value; every other
///   column becomes metadata.
/// - `None`: every column becomes metadata AND `Document::content` is the
///   `key=value\n…` joined view (useful for embedding the full row).
///
/// First row is treated as a header. Customize the delimiter for TSV / pipe
/// files via `with_delimiter`.
#[derive(Clone)]
pub struct CsvLoader {
    pub path: PathBuf,
    pub content_column: Option<String>,
    pub delimiter: u8,
    pub max_rows: Option<usize>,
}

impl CsvLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self {
            path: p.as_ref().to_path_buf(),
            content_column: None,
            delimiter: b',',
            max_rows: None,
        }
    }
    pub fn with_content_column(mut self, col: impl Into<String>) -> Self {
        self.content_column = Some(col.into());
        self
    }
    pub fn with_delimiter(mut self, d: u8) -> Self {
        self.delimiter = d;
        self
    }
    pub fn with_max_rows(mut self, n: usize) -> Self {
        self.max_rows = Some(n);
        self
    }
}

impl Loader for CsvLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .delimiter(self.delimiter)
            .from_path(&self.path)?;
        let headers = rdr.headers()?.clone();
        if let Some(ref col) = self.content_column {
            if !headers.iter().any(|h| h == col) {
                return Err(LoaderError::Other(format!(
                    "{}: content_column '{col}' not found in headers {:?}",
                    self.path.display(),
                    headers.iter().collect::<Vec<_>>(),
                )));
            }
        }
        let path_str = self.path.to_string_lossy().to_string();
        let mut out = Vec::new();
        for (i, rec) in rdr.records().enumerate() {
            if let Some(max) = self.max_rows {
                if i >= max { break; }
            }
            let rec = rec?;
            // Build content string + metadata simultaneously.
            let mut content = String::new();
            let mut content_taken = false;
            let mut meta_pairs: Vec<(String, String)> = Vec::with_capacity(headers.len());
            for (h, v) in headers.iter().zip(rec.iter()) {
                if let Some(ref col) = self.content_column {
                    if h == col {
                        content.push_str(v);
                        content_taken = true;
                        continue;
                    }
                }
                meta_pairs.push((h.to_string(), v.to_string()));
            }
            if !content_taken {
                // Fallback OR no content_column set: join key=value lines.
                content = meta_pairs
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            let mut d = Document::new(content)
                .with_id(format!("{}#{i}", path_str));
            for (k, v) in meta_pairs {
                d.metadata.insert(k, Value::String(v));
            }
            d.metadata.insert("source".into(), Value::String(path_str.clone()));
            d.metadata.insert("row".into(), Value::from(i as u64));
            out.push(d);
        }
        Ok(out)
    }
}

/// Markdown file (no frontmatter parsing here — header-splitting happens in
/// `litgraph-splitters::MarkdownHeaderSplitter`).
#[derive(Clone)]
pub struct MarkdownLoader {
    pub path: PathBuf,
}

impl MarkdownLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self { Self { path: p.as_ref().to_path_buf() } }
}

impl Loader for MarkdownLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let content = fs::read_to_string(&self.path)?;
        let mut d = Document::new(content).with_id(self.path.to_string_lossy());
        d.metadata.insert("source".into(), Value::String(self.path.to_string_lossy().into()));
        d.metadata.insert("format".into(), Value::String("markdown".into()));
        Ok(vec![d])
    }
}

/// Directory loader — traverse a root, match files against a glob pattern, dispatch
/// each file to the loader produced by `make_loader`. Ingestion parallelizes across
/// Rayon; this is a free win over LangChain's Python loop.
pub struct DirectoryLoader<F>
where
    F: Fn(&Path) -> Option<Box<dyn Loader>> + Send + Sync,
{
    pub root: PathBuf,
    pub glob: String,
    pub make_loader: F,
    pub follow_symlinks: bool,
}

impl<F> DirectoryLoader<F>
where
    F: Fn(&Path) -> Option<Box<dyn Loader>> + Send + Sync,
{
    pub fn new<P: AsRef<Path>>(root: P, glob: impl Into<String>, make_loader: F) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            glob: glob.into(),
            make_loader,
            follow_symlinks: false,
        }
    }

    pub fn follow_symlinks(mut self, on: bool) -> Self { self.follow_symlinks = on; self }
}

impl<F> Loader for DirectoryLoader<F>
where
    F: Fn(&Path) -> Option<Box<dyn Loader>> + Send + Sync,
{
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let matcher = glob::Pattern::new(&self.glob)?;
        let mut paths: Vec<PathBuf> = Vec::new();
        for entry in walkdir::WalkDir::new(&self.root)
            .follow_links(self.follow_symlinks)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if !entry.file_type().is_file() { continue; }
            let p = entry.path();
            let rel = p.strip_prefix(&self.root).unwrap_or(p);
            if matcher.matches_path(rel) {
                paths.push(p.to_path_buf());
            }
        }

        // Parallel ingest across files. Errors collapse into LoaderError::Other with context.
        let results: Vec<LoaderResult<Vec<Document>>> = paths
            .par_iter()
            .map(|p| match (self.make_loader)(p) {
                Some(loader) => loader.load(),
                None => Ok(vec![]),
            })
            .collect();

        let mut out = Vec::new();
        for r in results {
            out.extend(r?);
        }
        Ok(out)
    }
}

/// HTTP loader — fetch a single URL, return one Document with the response body
/// as content and `source` + `status_code` + `content_type` in metadata.
///
/// Uses `reqwest`'s blocking client because the `Loader` trait is sync. For
/// fan-out (many URLs in parallel) wrap calls in `rayon` at the call site, or
/// build a `DirectoryLoader`-shaped multi-URL loader on top.
pub struct WebLoader {
    pub url: String,
    pub timeout: Duration,
    pub user_agent: String,
}

impl WebLoader {
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn with_timeout(mut self, t: Duration) -> Self { self.timeout = t; self }
    pub fn with_user_agent(mut self, ua: impl Into<String>) -> Self {
        self.user_agent = ua.into();
        self
    }
}

impl Loader for WebLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?;
        let resp = client.get(&self.url).send()?;
        let status = resp.status().as_u16();
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(String::from)
            .unwrap_or_default();
        let body = resp.text()?;
        let mut doc = Document::new(body).with_id(self.url.clone());
        doc.metadata.insert("source".into(), Value::String(self.url.clone()));
        doc.metadata.insert("status_code".into(), Value::Number(status.into()));
        doc.metadata.insert("content_type".into(), Value::String(content_type));
        Ok(vec![doc])
    }
}

/// Convenience: default file dispatcher — `.txt` → Text, `.md` → Markdown, `.jsonl` → JSONL
/// with content_field="content". Users override per loader if needed.
pub fn default_dispatcher(path: &Path) -> Option<Box<dyn Loader>> {
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    match ext.as_str() {
        "txt" | "text" => Some(Box::new(TextLoader::new(path))),
        "md" | "markdown" => Some(Box::new(MarkdownLoader::new(path))),
        "jsonl" | "ndjson" => Some(Box::new(JsonLinesLoader::new(path, "content"))),
        "json" => Some(Box::new(JsonLoader::new(path))),
        "csv" => Some(Box::new(CsvLoader::new(path))),
        "tsv" => Some(Box::new(CsvLoader::new(path).with_delimiter(b'\t'))),
        "html" | "htm" => Some(Box::new(HtmlLoader::new(path))),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp_file(name: &str, contents: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::Builder::new()
            .prefix(name)
            .tempfile()
            .unwrap();
        f.write_all(contents.as_bytes()).unwrap();
        f
    }

    #[test]
    fn text_loader_reads_file() {
        let f = tmp_file("txt-", "hello world");
        let docs = TextLoader::new(f.path()).load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "hello world");
    }

    #[test]
    fn jsonl_loader_parses_lines() {
        let payload = "{\"content\":\"one\",\"tag\":\"a\"}\n{\"content\":\"two\",\"tag\":\"b\"}\n";
        let f = tmp_file("jl-", payload);
        let docs = JsonLinesLoader::new(f.path(), "content").load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "one");
        assert_eq!(docs[0].metadata.get("tag").unwrap().as_str(), Some("a"));
    }

    #[test]
    fn directory_loader_ingests_in_parallel() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..12 {
            let p = dir.path().join(format!("f{i}.txt"));
            std::fs::write(&p, format!("doc {i}")).unwrap();
        }
        let loader = DirectoryLoader::new(dir.path(), "*.txt", default_dispatcher);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 12);
    }

    #[test]
    fn csv_loader_with_content_column_uses_that_value_and_others_become_metadata() {
        let f = tmp_file("rows-",
            "id,title,body,author\n\
             1,Hello,This is the first doc,alice\n\
             2,World,Second one,bob\n");
        let docs = CsvLoader::new(f.path())
            .with_content_column("body")
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "This is the first doc");
        assert_eq!(docs[1].content, "Second one");
        assert_eq!(docs[0].metadata.get("title").unwrap().as_str().unwrap(), "Hello");
        assert_eq!(docs[0].metadata.get("author").unwrap().as_str().unwrap(), "alice");
        assert!(docs[0].metadata.get("body").is_none(), "content column not in metadata");
        assert_eq!(docs[0].metadata.get("row").unwrap().as_u64().unwrap(), 0);
        assert!(docs[0].id.as_deref().unwrap().ends_with("#0"));
    }

    #[test]
    fn csv_loader_without_content_column_joins_all_fields() {
        let f = tmp_file("rows-", "k,v\nfoo,1\nbar,2\n");
        let docs = CsvLoader::new(f.path()).load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "k=foo\nv=1");
        assert_eq!(docs[1].content, "k=bar\nv=2");
    }

    #[test]
    fn csv_loader_max_rows_caps_output() {
        let f = tmp_file("rows-",
            "n\n1\n2\n3\n4\n5\n");
        let docs = CsvLoader::new(f.path())
            .with_max_rows(2)
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn csv_loader_tsv_delimiter() {
        let f = tmp_file("rows-", "a\tb\n1\t2\n");
        let docs = CsvLoader::new(f.path())
            .with_delimiter(b'\t')
            .with_content_column("b")
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "2");
        assert_eq!(docs[0].metadata.get("a").unwrap().as_str().unwrap(), "1");
    }

    #[test]
    fn csv_loader_unknown_content_column_errors() {
        let f = tmp_file("rows-", "a,b\n1,2\n");
        let err = CsvLoader::new(f.path())
            .with_content_column("nope")
            .load()
            .unwrap_err();
        assert!(format!("{err}").contains("'nope' not found"));
    }

    #[test]
    fn default_dispatcher_picks_csv_and_tsv() {
        // Existence not required — dispatcher only inspects extension.
        let p = std::path::Path::new("/tmp/x.csv");
        assert!(default_dispatcher(p).is_some());
        let p = std::path::Path::new("/tmp/x.tsv");
        assert!(default_dispatcher(p).is_some());
    }

    #[test]
    fn json_loader_array_yields_one_doc_per_element() {
        let f = tmp_file("data-",
            r#"[{"title":"a","body":"alpha"},{"title":"b","body":"beta"}]"#);
        let docs = JsonLoader::new(f.path())
            .with_content_field("body")
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "alpha");
        assert_eq!(docs[1].content, "beta");
        assert_eq!(docs[0].metadata.get("title").unwrap(), &serde_json::json!("a"));
        assert!(docs[0].metadata.get("body").is_none(), "content field excluded");
        assert_eq!(docs[0].metadata.get("index").unwrap().as_u64().unwrap(), 0);
    }

    #[test]
    fn json_loader_array_without_content_field_serializes_each_row() {
        let f = tmp_file("data-",
            r#"[{"k":1},{"k":2}]"#);
        let docs = JsonLoader::new(f.path()).load().unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, r#"{"k":1}"#);
        assert_eq!(docs[1].content, r#"{"k":2}"#);
    }

    #[test]
    fn json_loader_pointer_drills_into_nested_array() {
        let f = tmp_file("data-",
            r#"{"meta":{"v":1},"results":{"items":[{"x":"first"},{"x":"second"}]}}"#);
        let docs = JsonLoader::new(f.path())
            .with_pointer("results.items")
            .with_content_field("x")
            .load()
            .unwrap();
        assert_eq!(docs.len(), 2);
        assert_eq!(docs[0].content, "first");
        assert_eq!(docs[1].content, "second");
    }

    #[test]
    fn json_loader_pointer_supports_array_index() {
        let f = tmp_file("data-",
            r#"{"pages":[{"items":["a","b","c"]},{"items":["d"]}]}"#);
        // pages[1].items → ["d"]
        let docs = JsonLoader::new(f.path())
            .with_pointer("pages.1.items")
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, r#""d""#);
    }

    #[test]
    fn json_loader_unknown_pointer_errors() {
        let f = tmp_file("data-", r#"{"a":1}"#);
        let err = JsonLoader::new(f.path())
            .with_pointer("nope.deeper")
            .load()
            .unwrap_err();
        assert!(format!("{err}").contains("did not resolve"));
    }

    #[test]
    fn json_loader_bare_object_yields_one_doc() {
        let f = tmp_file("data-", r#"{"name":"X","weight":1.5}"#);
        let docs = JsonLoader::new(f.path()).load().unwrap();
        assert_eq!(docs.len(), 1);
        // Content = serialized JSON; metadata carries fields.
        assert!(docs[0].content.contains("\"name\":\"X\""));
        assert_eq!(docs[0].metadata.get("name").unwrap(), &serde_json::json!("X"));
    }

    #[test]
    fn default_dispatcher_picks_json() {
        let p = std::path::Path::new("/tmp/x.json");
        assert!(default_dispatcher(p).is_some());
    }
}

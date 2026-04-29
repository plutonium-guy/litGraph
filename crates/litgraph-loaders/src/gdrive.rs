//! Google Drive loader — pull files via the Drive REST API.
//!
//! Direct LangChain `GoogleDriveLoader` parity-in-spirit. Pure-Rust via
//! existing `reqwest::blocking`. Auth is OAuth2 Bearer — caller obtains the
//! access token externally (same pattern as iter-101 Vertex / iter-103 Gmail).
//!
//! # Why this matters
//!
//! Drive holds Google's native editor formats (Docs, Sheets, Slides) in
//! binary proto; they're NOT usable via the plain `files/{id}?alt=media`
//! download endpoint. The `/export` endpoint converts them to
//! human/LLM-readable formats (`text/plain` for Docs/Slides, `text/csv` for
//! Sheets). Non-native plain files (text/markdown) use `?alt=media`.
//!
//! # Flow
//!
//! 1. `GET /drive/v3/files?q=...&fields=...&pageSize=100` — paginated list.
//! 2. Per file, branch on `mimeType`:
//!    - Google native (`application/vnd.google-apps.*`) → `files/{id}/export`
//!      with an appropriate target `mimeType`.
//!    - Plain text/markdown → `files/{id}?alt=media` straight download.
//!    - Binary (PDF, DOCX, PNG, …) → skip unless `include_binaries=true`,
//!      in which case we add a metadata-only Document so callers can still
//!      enumerate them.
//!
//! # Metadata per document
//!
//! - `file_id`, `name`, `mime_type`
//! - `modified_time` (ISO-8601)
//! - `parents` — comma-joined folder ids
//! - `web_view_link` — user-facing URL (useful for LLM citations)
//! - `source` — `"gdrive:{file_id}"`

use std::time::Duration;

use litgraph_core::Document;
use serde_json::Value;

use crate::{Loader, LoaderError, LoaderResult};

const DRIVE_API: &str = "https://www.googleapis.com/drive/v3";

pub struct GoogleDriveLoader {
    pub access_token: String,
    pub base_url: String,
    pub timeout: Duration,
    pub query: Option<String>,
    pub folder_id: Option<String>,
    /// Optional allowlist of MIME types to include. Empty = include everything
    /// that has a readable export path.
    pub mime_types: Vec<String>,
    pub max_files: Option<usize>,
    /// When true, binary files (PDF, images, unknown formats) produce a
    /// metadata-only Document with empty content. Useful to enumerate a
    /// Drive folder without downloading gigabytes of blobs.
    pub include_binaries: bool,
}

impl GoogleDriveLoader {
    pub fn new(access_token: impl Into<String>) -> Self {
        Self {
            access_token: access_token.into(),
            base_url: DRIVE_API.into(),
            timeout: Duration::from_secs(30),
            query: None,
            folder_id: None,
            mime_types: Vec::new(),
            max_files: Some(500),
            include_binaries: false,
        }
    }

    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }

    /// Drive search syntax — see
    /// <https://developers.google.com/drive/api/guides/search-files>.
    /// Forwarded verbatim as `?q=` parameter. Combines with `folder_id`
    /// via AND if both are set.
    pub fn with_query(mut self, q: impl Into<String>) -> Self {
        self.query = Some(q.into());
        self
    }

    /// Convenience for the most common case: "all files in folder X".
    /// Compiles to `"'X' in parents and trashed = false"`.
    pub fn with_folder_id(mut self, id: impl Into<String>) -> Self {
        self.folder_id = Some(id.into());
        self
    }

    pub fn with_mime_types<I, S>(mut self, ts: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.mime_types = ts.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_max_files(mut self, n: Option<usize>) -> Self {
        self.max_files = n;
        self
    }

    pub fn with_include_binaries(mut self, b: bool) -> Self {
        self.include_binaries = b;
        self
    }

    fn client(&self) -> LoaderResult<reqwest::blocking::Client> {
        reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .build()
            .map_err(LoaderError::from)
    }

    fn authed(
        &self,
        b: reqwest::blocking::RequestBuilder,
    ) -> reqwest::blocking::RequestBuilder {
        b.bearer_auth(&self.access_token)
    }

    /// Build the full Drive `q=` parameter from folder_id + caller-supplied
    /// query. Both combine with AND.
    fn effective_query(&self) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();
        if let Some(fid) = &self.folder_id {
            parts.push(format!("'{fid}' in parents and trashed = false"));
        }
        if let Some(q) = &self.query {
            parts.push(format!("({q})"));
        }
        if parts.is_empty() {
            None
        } else {
            Some(parts.join(" and "))
        }
    }

    fn fetch_file_list_page(
        &self,
        client: &reqwest::blocking::Client,
        page_token: Option<&str>,
    ) -> LoaderResult<(Vec<Value>, Option<String>)> {
        // `fields` parameter selects exactly the columns we need — Drive
        // returns a trimmed response instead of the default 10+ fields per
        // file. Costs one fewer billable API unit per file on big corpora.
        let fields =
            "nextPageToken,files(id,name,mimeType,modifiedTime,parents,webViewLink,size)";
        let mut url = format!(
            "{}/files?pageSize=100&fields={}",
            self.base_url.trim_end_matches('/'),
            urlencode(fields),
        );
        if let Some(q) = self.effective_query() {
            url.push_str(&format!("&q={}", urlencode(&q)));
        }
        if let Some(pt) = page_token {
            url.push_str(&format!("&pageToken={}", urlencode(pt)));
        }
        let resp = self.authed(client.get(&url)).send()?;
        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().unwrap_or_default();
            return Err(LoaderError::Other(format!(
                "gdrive files.list {status}: {body}"
            )));
        }
        let v: Value = resp.json()?;
        let files: Vec<Value> = v
            .get("files")
            .and_then(|f| f.as_array())
            .cloned()
            .unwrap_or_default();
        let next = v
            .get("nextPageToken")
            .and_then(|s| s.as_str())
            .filter(|s| !s.is_empty())
            .map(String::from);
        Ok((files, next))
    }

    /// Decide which format to ask `/export` for, given a Google native
    /// `mime_type`. Returns None for non-native files (use alt=media).
    fn export_target(mime: &str) -> Option<&'static str> {
        match mime {
            "application/vnd.google-apps.document" => Some("text/plain"),
            "application/vnd.google-apps.spreadsheet" => Some("text/csv"),
            "application/vnd.google-apps.presentation" => Some("text/plain"),
            "application/vnd.google-apps.script+json" => None, // not text
            _ => None,
        }
    }

    /// Textual MIME types suitable for direct `alt=media` download. Covers
    /// the common plain-text formats Drive users upload.
    fn is_text_mime(mime: &str) -> bool {
        mime == "text/plain"
            || mime == "text/markdown"
            || mime == "text/html"
            || mime == "text/csv"
            || mime == "text/tab-separated-values"
            || mime == "application/json"
            || mime == "application/x-ndjson"
    }

    fn fetch_content(
        &self,
        client: &reqwest::blocking::Client,
        file_id: &str,
        mime: &str,
    ) -> LoaderResult<Option<String>> {
        if let Some(target) = Self::export_target(mime) {
            // Google-native editor format → /export
            let url = format!(
                "{}/files/{}/export?mimeType={}",
                self.base_url.trim_end_matches('/'),
                urlencode(file_id),
                urlencode(target),
            );
            let resp = self.authed(client.get(&url)).send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!(
                    "gdrive export {file_id} {status}: {body}"
                )));
            }
            return Ok(Some(resp.text()?));
        }
        if Self::is_text_mime(mime) {
            // Plain textual upload → alt=media
            let url = format!(
                "{}/files/{}?alt=media",
                self.base_url.trim_end_matches('/'),
                urlencode(file_id),
            );
            let resp = self.authed(client.get(&url)).send()?;
            if !resp.status().is_success() {
                let status = resp.status();
                let body = resp.text().unwrap_or_default();
                return Err(LoaderError::Other(format!(
                    "gdrive get {file_id} {status}: {body}"
                )));
            }
            return Ok(Some(resp.text()?));
        }
        // Binary file — can't extract text.
        Ok(None)
    }

    fn should_load(&self, mime: &str) -> bool {
        // Respect user's allowlist if provided.
        if !self.mime_types.is_empty() && !self.mime_types.iter().any(|m| m == mime) {
            return false;
        }
        // Otherwise, keep only files we know how to read (export or alt=media).
        Self::export_target(mime).is_some() || Self::is_text_mime(mime)
    }

    fn file_to_document(
        &self,
        file: &Value,
        content: Option<String>,
    ) -> Option<Document> {
        let id = file.get("id").and_then(|s| s.as_str())?.to_string();
        let name = file
            .get("name")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let mime = file
            .get("mimeType")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let modified = file
            .get("modifiedTime")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let web_link = file
            .get("webViewLink")
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_string();
        let parents: Vec<String> = file
            .get("parents")
            .and_then(|p| p.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|s| s.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();

        // Prefix filename as H1 so LLM sees the most-salient field first
        // (matches Gmail subject + GitHub issue title convention).
        let final_content = match content {
            Some(body) if !name.is_empty() => format!("# {name}\n\n{body}"),
            Some(body) => body,
            None if !name.is_empty() => format!("# {name}\n"),
            None => String::new(),
        };

        let mut d = Document::new(final_content);
        d.id = Some(format!("gdrive:{id}"));
        d.metadata.insert("file_id".into(), Value::String(id.clone()));
        if !name.is_empty() {
            d.metadata.insert("name".into(), Value::String(name));
        }
        if !mime.is_empty() {
            d.metadata.insert("mime_type".into(), Value::String(mime));
        }
        if !modified.is_empty() {
            d.metadata.insert("modified_time".into(), Value::String(modified));
        }
        if !parents.is_empty() {
            d.metadata
                .insert("parents".into(), Value::String(parents.join(",")));
        }
        if !web_link.is_empty() {
            d.metadata
                .insert("web_view_link".into(), Value::String(web_link));
        }
        d.metadata.insert(
            "source".into(),
            Value::String(format!("gdrive:{id}")),
        );
        Some(d)
    }
}

impl Loader for GoogleDriveLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let client = self.client()?;
        let mut docs: Vec<Document> = Vec::new();
        let mut page_token: Option<String> = None;
        loop {
            let (files, next) = self.fetch_file_list_page(&client, page_token.as_deref())?;
            for file in &files {
                let id = match file.get("id").and_then(|s| s.as_str()) {
                    Some(i) => i,
                    None => continue,
                };
                let mime = file
                    .get("mimeType")
                    .and_then(|s| s.as_str())
                    .unwrap_or("");
                let readable = self.should_load(mime);
                if !readable && !self.include_binaries {
                    continue;
                }
                let content = if readable {
                    self.fetch_content(&client, id, mime)?
                } else {
                    None // binary, but include_binaries enabled → metadata-only doc
                };
                if let Some(d) = self.file_to_document(file, content) {
                    docs.push(d);
                    if let Some(cap) = self.max_files {
                        if docs.len() >= cap {
                            return Ok(docs);
                        }
                    }
                }
            }
            match next {
                Some(t) => page_token = Some(t),
                None => break,
            }
        }
        Ok(docs)
    }
}

/// URL-encoder — preserves unreserved chars, encodes everything else.
/// Shared pattern with other SaaS loaders (kept local to avoid cross-module
/// coupling for a trivial helper).
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
            out.push(c);
        } else {
            for b in c.to_string().bytes() {
                out.push_str(&format!("%{:02X}", b));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::{Arc, Mutex};
    use std::thread;

    struct FakeServer {
        url: String,
        seen_paths: Arc<Mutex<Vec<String>>>,
        seen_auth: Arc<Mutex<Vec<Option<String>>>>,
        _shutdown: std::sync::mpsc::Sender<()>,
    }

    fn list_page_1() -> Value {
        json!({
            "nextPageToken": "NEXT",
            "files": [
                {"id": "doc1", "name": "Design Doc",
                 "mimeType": "application/vnd.google-apps.document",
                 "modifiedTime": "2024-01-01T12:00:00Z",
                 "parents": ["folder_x"],
                 "webViewLink": "https://docs.google.com/document/d/doc1/edit"},
                {"id": "sheet1", "name": "Budget",
                 "mimeType": "application/vnd.google-apps.spreadsheet",
                 "modifiedTime": "2024-01-02T12:00:00Z",
                 "webViewLink": "https://docs.google.com/spreadsheets/d/sheet1"},
                {"id": "md1", "name": "notes.md",
                 "mimeType": "text/markdown",
                 "modifiedTime": "2024-01-03T12:00:00Z",
                 "webViewLink": "https://drive.google.com/file/d/md1/view"},
                {"id": "png1", "name": "chart.png",
                 "mimeType": "image/png",
                 "modifiedTime": "2024-01-04T12:00:00Z"}
            ]
        })
    }

    fn list_page_2() -> Value {
        json!({
            "files": [
                {"id": "txt1", "name": "readme.txt",
                 "mimeType": "text/plain",
                 "modifiedTime": "2024-01-05T12:00:00Z"}
            ]
        })
    }

    fn spawn_fake() -> FakeServer {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let seen_paths: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let seen_auth: Arc<Mutex<Vec<Option<String>>>> = Arc::new(Mutex::new(Vec::new()));
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        listener.set_nonblocking(true).unwrap();
        let pw = seen_paths.clone();
        let aw = seen_auth.clone();
        thread::spawn(move || {
            use std::io::{Read, Write};
            loop {
                if rx.try_recv().is_ok() { break; }
                match listener.accept() {
                    Ok((mut s, _)) => {
                        s.set_nonblocking(false).ok();
                        let mut buf = [0u8; 8192];
                        let n = s.read(&mut buf).unwrap_or(0);
                        if n == 0 { continue; }
                        let req = String::from_utf8_lossy(&buf[..n]).to_string();
                        let first_line = req.lines().next().unwrap_or("");
                        let path = first_line
                            .split_whitespace()
                            .nth(1)
                            .unwrap_or("/")
                            .to_string();
                        pw.lock().unwrap().push(path.clone());
                        let mut auth = None;
                        for line in req.lines().skip(1) {
                            if let Some((k, v)) = line.split_once(':') {
                                if k.trim().eq_ignore_ascii_case("authorization") {
                                    auth = Some(v.trim().to_string());
                                }
                            }
                        }
                        aw.lock().unwrap().push(auth);

                        // Dispatch + content.
                        let (body, ctype) = if path.contains("/files/") && path.contains("/export") {
                            // /files/{id}/export?mimeType=...
                            // Extract target mimeType to vary response accordingly.
                            let text = if path.contains("mimeType=text%2Fcsv") {
                                "col1,col2\n1,2\n3,4".to_string()
                            } else {
                                // text/plain for Docs + Slides.
                                format!("Exported plain text for file {}",
                                    path.split("/files/").nth(1).unwrap_or("")
                                        .split('/').next().unwrap_or(""))
                            };
                            (text, "text/plain")
                        } else if path.contains("/files/") && path.contains("alt=media") {
                            // Plain download.
                            let id = path.split("/files/").nth(1).unwrap_or("")
                                .split('?').next().unwrap_or("");
                            (format!("Raw media content for {id}"), "text/plain")
                        } else if path.starts_with("/files?") || path.starts_with("/files&") {
                            // files.list
                            let body = if path.contains("pageToken=NEXT") {
                                list_page_2().to_string()
                            } else {
                                list_page_1().to_string()
                            };
                            (body, "application/json")
                        } else {
                            (json!({"error": "unknown"}).to_string(), "application/json")
                        };
                        let resp = format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\r\n{}",
                            ctype, body.len(), body
                        );
                        let _ = s.write_all(resp.as_bytes());
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(_) => break,
                }
            }
        });
        FakeServer {
            url: format!("http://127.0.0.1:{port}"),
            seen_paths,
            seen_auth,
            _shutdown: tx,
        }
    }

    #[test]
    fn paginates_and_loads_readable_files_only_by_default() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("ya29.test").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        // Page 1 has doc1 + sheet1 + md1 (3 readable) + png1 (binary, skipped).
        // Page 2 has txt1 (readable).
        // Default include_binaries=false → binary skipped.
        // Result: 4 docs total.
        assert_eq!(docs.len(), 4);
        let names: Vec<&str> = docs
            .iter()
            .filter_map(|d| d.metadata.get("name").and_then(|n| n.as_str()))
            .collect();
        assert!(names.contains(&"Design Doc"));
        assert!(names.contains(&"Budget"));
        assert!(names.contains(&"notes.md"));
        assert!(names.contains(&"readme.txt"));
        // PNG NOT present (binary, no include_binaries).
        assert!(!names.contains(&"chart.png"));
    }

    #[test]
    fn google_native_formats_use_export_endpoint_with_correct_target_mime() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t").with_base_url(&srv.url);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Doc exports as text/plain.
        assert!(paths.iter().any(|p|
            p.contains("/files/doc1/export") && p.contains("mimeType=text%2Fplain")
        ), "paths: {paths:#?}");
        // Sheet exports as text/csv.
        assert!(paths.iter().any(|p|
            p.contains("/files/sheet1/export") && p.contains("mimeType=text%2Fcsv")
        ), "paths: {paths:#?}");
    }

    #[test]
    fn plain_textual_files_use_alt_media_not_export() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t").with_base_url(&srv.url);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // notes.md (text/markdown) → alt=media, NOT export.
        assert!(paths.iter().any(|p|
            p.contains("/files/md1") && p.contains("alt=media")
        ));
        assert!(!paths.iter().any(|p|
            p.contains("/files/md1/export")
        ));
    }

    #[test]
    fn binary_files_skipped_by_default_fetched_metadata_only_when_opted_in() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t")
            .with_base_url(&srv.url)
            .with_include_binaries(true);
        let docs = loader.load().unwrap();
        // Now 5 docs total including the PNG.
        assert_eq!(docs.len(), 5);
        let png_doc = docs
            .iter()
            .find(|d| d.metadata.get("name").and_then(|n| n.as_str()) == Some("chart.png"))
            .expect("PNG should appear with include_binaries=true");
        // Content is metadata-only (just the H1-formatted name).
        assert_eq!(png_doc.content, "# chart.png\n");
        assert_eq!(png_doc.metadata["mime_type"].as_str(), Some("image/png"));

        // Assert no /files/png1?alt=media request was ever made (binary
        // content intentionally NOT downloaded).
        let paths = srv.seen_paths.lock().unwrap().clone();
        assert!(!paths.iter().any(|p|
            p.contains("/files/png1") && (p.contains("alt=media") || p.contains("export"))
        ), "paths: {paths:#?}");
    }

    #[test]
    fn metadata_captures_name_mime_modified_parents_web_view_link() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t").with_base_url(&srv.url);
        let docs = loader.load().unwrap();
        let doc1 = docs
            .iter()
            .find(|d| d.metadata.get("file_id").and_then(|v| v.as_str()) == Some("doc1"))
            .unwrap();
        assert_eq!(doc1.metadata["name"].as_str(), Some("Design Doc"));
        assert_eq!(
            doc1.metadata["mime_type"].as_str(),
            Some("application/vnd.google-apps.document")
        );
        assert_eq!(
            doc1.metadata["modified_time"].as_str(),
            Some("2024-01-01T12:00:00Z")
        );
        assert_eq!(doc1.metadata["parents"].as_str(), Some("folder_x"));
        assert!(doc1.metadata["web_view_link"].as_str().unwrap().starts_with("https://docs"));
        assert_eq!(doc1.metadata["source"].as_str(), Some("gdrive:doc1"));
        assert_eq!(doc1.id.as_deref(), Some("gdrive:doc1"));
        // Filename prefixed as markdown H1.
        assert!(doc1.content.starts_with("# Design Doc"));
    }

    #[test]
    fn auth_bearer_set_on_list_export_and_media_requests() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("ya29.SECRET").with_base_url(&srv.url);
        loader.load().unwrap();
        for a in srv.seen_auth.lock().unwrap().iter() {
            assert_eq!(a.as_deref(), Some("Bearer ya29.SECRET"));
        }
    }

    #[test]
    fn effective_query_combines_folder_id_and_caller_query_with_and() {
        let loader = GoogleDriveLoader::new("t")
            .with_folder_id("FOLDER_ABC")
            .with_query("name contains 'design'");
        let q = loader.effective_query().unwrap();
        assert!(q.contains("'FOLDER_ABC' in parents"));
        assert!(q.contains("trashed = false"));
        assert!(q.contains("name contains 'design'"));
        assert!(q.contains(" and "));
    }

    #[test]
    fn effective_query_returns_none_when_no_filters_set() {
        let loader = GoogleDriveLoader::new("t");
        assert!(loader.effective_query().is_none());
    }

    #[test]
    fn mime_types_allowlist_filters_out_other_readable_files() {
        let srv = spawn_fake();
        // Only want Docs — sheets + markdown + txt should be skipped.
        let loader = GoogleDriveLoader::new("t")
            .with_base_url(&srv.url)
            .with_mime_types(["application/vnd.google-apps.document"]);
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(
            docs[0].metadata["mime_type"].as_str(),
            Some("application/vnd.google-apps.document"),
        );
    }

    #[test]
    fn max_files_cap_truncates_result_short_of_pagination() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t")
            .with_base_url(&srv.url)
            .with_max_files(Some(2));
        let docs = loader.load().unwrap();
        assert_eq!(docs.len(), 2);
    }

    #[test]
    fn pagination_token_forwarded_to_second_list_request() {
        let srv = spawn_fake();
        let loader = GoogleDriveLoader::new("t").with_base_url(&srv.url);
        loader.load().unwrap();
        let paths = srv.seen_paths.lock().unwrap().clone();
        // Page 2 hit with pageToken=NEXT (URL-encoded or raw).
        assert!(paths.iter().any(|p| p.contains("pageToken=NEXT")));
    }

    #[test]
    fn export_target_maps_native_mime_types_correctly() {
        assert_eq!(
            GoogleDriveLoader::export_target("application/vnd.google-apps.document"),
            Some("text/plain"),
        );
        assert_eq!(
            GoogleDriveLoader::export_target("application/vnd.google-apps.spreadsheet"),
            Some("text/csv"),
        );
        assert_eq!(
            GoogleDriveLoader::export_target("application/vnd.google-apps.presentation"),
            Some("text/plain"),
        );
        // Non-native → None (caller uses alt=media).
        assert_eq!(GoogleDriveLoader::export_target("text/plain"), None);
        assert_eq!(GoogleDriveLoader::export_target("image/png"), None);
    }

    #[test]
    fn is_text_mime_covers_common_textual_types() {
        assert!(GoogleDriveLoader::is_text_mime("text/plain"));
        assert!(GoogleDriveLoader::is_text_mime("text/markdown"));
        assert!(GoogleDriveLoader::is_text_mime("text/csv"));
        assert!(GoogleDriveLoader::is_text_mime("application/json"));
        // Non-textual.
        assert!(!GoogleDriveLoader::is_text_mime("image/png"));
        assert!(!GoogleDriveLoader::is_text_mime("application/pdf"));
        assert!(!GoogleDriveLoader::is_text_mime("application/vnd.google-apps.document"));
    }
}

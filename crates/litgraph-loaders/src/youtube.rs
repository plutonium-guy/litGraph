//! YouTube transcript loader. Pulls captions from the legacy public
//! `timedtext` endpoint (no API key, no auth) and emits one
//! [`Document`] per video.
//!
//! # Realistic expectations
//!
//! YouTube's transcript story is a moving target. We hit the
//! `https://video.google.com/timedtext` endpoint, which has been stable
//! for ~15 years and returns the auto-generated or uploaded captions
//! as XML. It works for the majority of public videos but fails when:
//!
//! - the uploader disabled captions entirely,
//! - the video is age-restricted or member-only,
//! - the requested language doesn't exist (you can list available
//!   languages by passing no `v` arg, but that requires a second
//!   round-trip — leave to the caller).
//!
//! When transcripts aren't available, the loader returns an empty
//! `Vec<Document>` rather than erroring — this is the right behaviour
//! for batch ingestion (one missing video shouldn't kill the run).
//!
//! # Document shape
//!
//! - `content` = full transcript text, lines joined with spaces and
//!   whitespace collapsed. The embedder sees prose, not timestamped
//!   fragments.
//! - `id` = video id (the `?v=` parameter — stable across renames).
//! - `metadata`:
//!   - `video_id`: the canonical id
//!   - `language`: language code requested (`"en"`, `"de"`, …)
//!   - `cues`: JSON array of `{start_ms, dur_ms, text}` for callers
//!     who need timestamp-aligned access (highlight tracking, video
//!     editors, etc.).
//!   - `source`: `"youtube"`
//!   - `url`: the public watch URL (synthesised from `video_id`)
//!
//! # Example
//!
//! ```no_run
//! use litgraph_loaders::{Loader, YouTubeTranscriptLoader};
//! let docs = YouTubeTranscriptLoader::new("https://youtu.be/dQw4w9WgXcQ")
//!     .with_language("en")
//!     .load()
//!     .unwrap();
//! ```

use std::time::Duration;

use litgraph_core::Document;
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

const ENDPOINT: &str = "https://video.google.com/timedtext";

pub struct YouTubeTranscriptLoader {
    /// Caller-supplied input — extracted to a video id at fetch time
    /// rather than parse time so we keep the original error context.
    pub input: String,
    pub language: String,
    pub timeout: Duration,
    pub user_agent: String,
}

impl YouTubeTranscriptLoader {
    /// Accepts a bare video id or any common URL form
    /// (`youtu.be/X`, `youtube.com/watch?v=X`, `youtube.com/embed/X`,
    /// `youtube.com/shorts/X`). The id is resolved at `load()` time.
    pub fn new(input: impl Into<String>) -> Self {
        Self {
            input: input.into(),
            language: "en".to_string(),
            timeout: Duration::from_secs(30),
            user_agent: format!("litgraph-loaders/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    pub fn with_language(mut self, lang: impl Into<String>) -> Self {
        self.language = lang.into();
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
        std::env::var("LITGRAPH_YOUTUBE_TIMEDTEXT").unwrap_or_else(|_| ENDPOINT.into())
    }

    fn fetch(&self, video_id: &str) -> LoaderResult<Option<String>> {
        let client = reqwest::blocking::Client::builder()
            .timeout(self.timeout)
            .user_agent(&self.user_agent)
            .build()?;
        let url = format!(
            "{}?lang={}&v={}",
            self.endpoint(),
            urlencode(&self.language),
            urlencode(video_id),
        );
        let resp = client.get(&url).send()?;
        let status = resp.status();
        if !status.is_success() {
            return Err(LoaderError::Other(format!(
                "youtube timedtext {status} {url}: {}",
                resp.text().unwrap_or_default()
            )));
        }
        let body = resp.text()?;
        // Empty body = "no captions for this video/language". Surface
        // as `None` rather than erroring — batch ingestion friendliness.
        if body.trim().is_empty() {
            return Ok(None);
        }
        Ok(Some(body))
    }
}

impl Loader for YouTubeTranscriptLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let video_id = match extract_video_id(&self.input) {
            Some(v) => v,
            None => {
                return Err(LoaderError::Other(format!(
                    "youtube: couldn't extract video id from {:?}",
                    self.input
                )))
            }
        };
        let xml = match self.fetch(&video_id)? {
            Some(x) => x,
            None => return Ok(Vec::new()),
        };
        Ok(vec![transcript_to_document(&xml, &video_id, &self.language)])
    }
}

// ---- video-id extraction ---------------------------------------------------

/// Pull the canonical 11-character video id out of a URL or a bare id.
/// Returns `None` if no id is recognisable.
pub fn extract_video_id(input: &str) -> Option<String> {
    let trimmed = input.trim();

    // Bare id check first — short-circuit before regex.
    if is_likely_video_id(trimmed) {
        return Some(trimmed.to_string());
    }

    // youtu.be/<id>
    static YOUTU_BE_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"youtu\.be/([A-Za-z0-9_-]{11})").unwrap());
    if let Some(c) = YOUTU_BE_RE.captures(trimmed) {
        return Some(c[1].to_string());
    }

    // youtube.com/watch?v=<id>
    static WATCH_V_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"[?&]v=([A-Za-z0-9_-]{11})").unwrap());
    if let Some(c) = WATCH_V_RE.captures(trimmed) {
        return Some(c[1].to_string());
    }

    // youtube.com/embed/<id> or youtube.com/shorts/<id>
    static EMBED_RE: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"youtube\.com/(?:embed|shorts|live)/([A-Za-z0-9_-]{11})").unwrap()
    });
    if let Some(c) = EMBED_RE.captures(trimmed) {
        return Some(c[1].to_string());
    }

    None
}

fn is_likely_video_id(s: &str) -> bool {
    s.len() == 11
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

// ---- transcript XML → Document --------------------------------------------

/// Internal alias for one parsed cue.
#[derive(Debug, Clone)]
struct Cue {
    start_ms: u64,
    dur_ms: u64,
    text: String,
}

/// Parse the timedtext XML body and produce a Document. Public-crate
/// so tests can exercise without hitting the network.
pub(crate) fn transcript_to_document(xml: &str, video_id: &str, language: &str) -> Document {
    let cues = parse_cues(xml);

    let mut full = String::new();
    let mut cues_json: Vec<Value> = Vec::with_capacity(cues.len());
    for c in &cues {
        if !full.is_empty() {
            full.push(' ');
        }
        full.push_str(&c.text);
        cues_json.push(json!({
            "start_ms": c.start_ms,
            "dur_ms": c.dur_ms,
            "text": c.text,
        }));
    }

    let mut d = Document::new(full).with_id(video_id);
    let mut put = |k: &str, v: Value| {
        d.metadata.insert(k.into(), v);
    };
    put("video_id", Value::String(video_id.to_string()));
    put("language", Value::String(language.to_string()));
    put("cues", Value::Array(cues_json));
    put(
        "url",
        Value::String(format!("https://www.youtube.com/watch?v={video_id}")),
    );
    put("source", Value::String("youtube".into()));
    d
}

static TEXT_TAG_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#"(?is)<text\b([^>]*)>(.*?)</text>"#).expect("text tag regex")
});
static ATTR_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r#"(?is)\b([a-zA-Z_:][\w:.-]*)\s*=\s*"([^"]*)""#).expect("attr"));

fn parse_cues(xml: &str) -> Vec<Cue> {
    let mut out = Vec::new();
    for c in TEXT_TAG_RE.captures_iter(xml) {
        let attrs = parse_attrs(&c[1]);
        let body = collapse_ws(&decode_entities(&strip_inline_tags(&c[2])));
        if body.is_empty() {
            continue;
        }
        let start_secs = attrs
            .get("start")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        let dur_secs = attrs
            .get("dur")
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(0.0);
        out.push(Cue {
            start_ms: (start_secs * 1000.0) as u64,
            dur_ms: (dur_secs * 1000.0) as u64,
            text: body,
        });
    }
    out
}

fn parse_attrs(inside: &str) -> std::collections::HashMap<String, String> {
    ATTR_RE
        .captures_iter(inside)
        .map(|c| (c[1].to_string(), c[2].to_string()))
        .collect()
}

/// Strip the handful of inline tags YouTube can emit inside `<text>`
/// (e.g. `<br/>` line breaks, occasional `<i>` italics in uploaded
/// captions). Without this the embedder indexes angle-bracket noise.
fn strip_inline_tags(s: &str) -> String {
    static TAG_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?is)<[^>]+>").unwrap());
    TAG_RE.replace_all(s, " ").into_owned()
}

fn decode_entities(s: &str) -> String {
    // Order matters: `&amp;` last so we don't double-decode `&amp;lt;`.
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&#39;", "'")
        .replace("&#10;", "\n")
        .replace("&amp;", "&")
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

/// Minimal URL encoder for the timedtext query string. Only encodes
/// characters the endpoint chokes on.
fn urlencode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- video-id extraction ----

    #[test]
    fn extract_bare_id() {
        assert_eq!(
            extract_video_id("dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_from_youtu_be_short_url() {
        assert_eq!(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        // Trailing junk OK.
        assert_eq!(
            extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=42s"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_from_watch_url() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
        assert_eq!(
            extract_video_id("https://www.youtube.com/watch?feature=share&v=dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_from_embed_url() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_from_shorts_url() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_from_live_url() {
        assert_eq!(
            extract_video_id("https://www.youtube.com/live/dQw4w9WgXcQ"),
            Some("dQw4w9WgXcQ".to_string())
        );
    }

    #[test]
    fn extract_returns_none_for_garbage() {
        assert_eq!(extract_video_id(""), None);
        assert_eq!(extract_video_id("not a url"), None);
        assert_eq!(extract_video_id("https://example.com/foo"), None);
        // Wrong length.
        assert_eq!(extract_video_id("dQw4w9W"), None);
        assert_eq!(extract_video_id("dQw4w9WgXcQTOOLONG"), None);
    }

    #[test]
    fn extract_rejects_invalid_chars_in_bare_id() {
        // Spaces aren't valid in a bare id, but the URL regexes wouldn't
        // match either — must return None.
        assert_eq!(extract_video_id("dQw4w 9WgXcQ"), None);
    }

    // ---- transcript XML parsing ----

    const SAMPLE_XML: &str = r#"<?xml version="1.0" encoding="utf-8" ?>
<transcript>
<text start="0" dur="2.5">Hello and welcome.</text>
<text start="2.5" dur="3">Today we&#39;re talking about Rust.</text>
<text start="5.5" dur="2">It&apos;s &quot;fast&quot;.</text>
<text start="7.5" dur="1.2"><i>(intro music)</i></text>
<text start="8.7" dur="0.5"></text>
</transcript>"#;

    #[test]
    fn parses_all_non_empty_cues() {
        let cues = parse_cues(SAMPLE_XML);
        // Empty <text> is dropped.
        assert_eq!(cues.len(), 4);
    }

    #[test]
    fn cue_timing_converted_to_ms() {
        let cues = parse_cues(SAMPLE_XML);
        assert_eq!(cues[0].start_ms, 0);
        assert_eq!(cues[0].dur_ms, 2500);
        assert_eq!(cues[1].start_ms, 2500);
        assert_eq!(cues[1].dur_ms, 3000);
    }

    #[test]
    fn cue_text_decodes_html_entities() {
        let cues = parse_cues(SAMPLE_XML);
        assert_eq!(cues[1].text, "Today we're talking about Rust.");
    }

    #[test]
    fn cue_text_decodes_named_entities() {
        // Spec-correct single-encoding: `&apos;` and `&quot;` resolve
        // in one pass. Real YouTube responses sometimes double-encode
        // (`&amp;quot;`); callers needing that case can post-process.
        let cues = parse_cues(SAMPLE_XML);
        assert_eq!(cues[2].text, r#"It's "fast"."#);
    }

    #[test]
    fn cue_text_strips_inline_markup() {
        let cues = parse_cues(SAMPLE_XML);
        // <i>...</i> wrapper gone, content collapsed.
        assert_eq!(cues[3].text, "(intro music)");
    }

    #[test]
    fn transcript_to_document_round_trip() {
        let d = transcript_to_document(SAMPLE_XML, "abc1234defg", "en");
        // Note: id length must be 11 for this branch.
        assert_eq!(d.id.as_deref(), Some("abc1234defg"));
        assert!(d.content.starts_with("Hello and welcome."));
        // Cues joined with single spaces — no double spaces.
        assert!(!d.content.contains("  "));
        assert_eq!(
            d.metadata.get("video_id").and_then(|v| v.as_str()),
            Some("abc1234defg")
        );
        assert_eq!(
            d.metadata.get("language").and_then(|v| v.as_str()),
            Some("en")
        );
        assert_eq!(
            d.metadata.get("source").and_then(|v| v.as_str()),
            Some("youtube")
        );
        let url = d.metadata.get("url").and_then(|v| v.as_str()).unwrap();
        assert!(url.contains("watch?v=abc1234defg"));
        let cues = d.metadata.get("cues").and_then(|v| v.as_array()).unwrap();
        assert_eq!(cues.len(), 4);
        assert_eq!(cues[0]["start_ms"].as_u64().unwrap(), 0);
        assert_eq!(cues[0]["dur_ms"].as_u64().unwrap(), 2500);
    }

    #[test]
    fn empty_transcript_yields_empty_document() {
        let xml = r#"<transcript></transcript>"#;
        let d = transcript_to_document(xml, "abc1234defg", "en");
        assert!(d.content.is_empty());
        let cues = d.metadata.get("cues").and_then(|v| v.as_array()).unwrap();
        assert!(cues.is_empty());
    }

    #[test]
    fn cue_with_missing_dur_defaults_to_zero() {
        let xml = r#"<transcript><text start="1.5">no dur</text></transcript>"#;
        let cues = parse_cues(xml);
        assert_eq!(cues.len(), 1);
        assert_eq!(cues[0].start_ms, 1500);
        assert_eq!(cues[0].dur_ms, 0);
        assert_eq!(cues[0].text, "no dur");
    }

    #[test]
    fn cue_with_missing_start_defaults_to_zero() {
        let xml = r#"<transcript><text dur="1">x</text></transcript>"#;
        let cues = parse_cues(xml);
        assert_eq!(cues[0].start_ms, 0);
    }

    #[test]
    fn fractional_seconds_preserved_to_ms() {
        let xml = r#"<transcript><text start="1.234" dur="0.567">hi</text></transcript>"#;
        let cues = parse_cues(xml);
        assert_eq!(cues[0].start_ms, 1234);
        assert_eq!(cues[0].dur_ms, 567);
    }

    #[test]
    fn collapse_ws_handles_multiline_input() {
        // Real YouTube emits multi-line text inside <text> for long
        // captions. The collapse step joins them without trailing
        // whitespace.
        let xml = r#"<transcript><text start="0" dur="1">line one
line two
   line three   </text></transcript>"#;
        let cues = parse_cues(xml);
        assert_eq!(cues[0].text, "line one line two line three");
    }

    // ---- urlencode ----

    #[test]
    fn urlencode_passes_safe_chars() {
        assert_eq!(urlencode("dQw4w9WgXcQ"), "dQw4w9WgXcQ");
        assert_eq!(urlencode("en-US.foo_bar"), "en-US.foo_bar");
    }

    #[test]
    fn urlencode_percent_escapes_unsafe_chars() {
        assert_eq!(urlencode("a b"), "a%20b");
        assert_eq!(urlencode("a&b"), "a%26b");
        assert_eq!(urlencode("a/b"), "a%2Fb");
    }

    // ---- endpoint env override ----

    #[test]
    fn endpoint_uses_env_override() {
        // Single test covers both branches under a serial guard so
        // parallel test ordering can't make us race against another
        // test that touches the same env var.
        use std::sync::Mutex;
        static GUARD: Mutex<()> = Mutex::new(());
        let _g = GUARD.lock().unwrap();
        std::env::remove_var("LITGRAPH_YOUTUBE_TIMEDTEXT");
        let l = YouTubeTranscriptLoader::new("dQw4w9WgXcQ");
        assert_eq!(l.endpoint(), ENDPOINT);
        std::env::set_var("LITGRAPH_YOUTUBE_TIMEDTEXT", "http://localhost:9/timedtext");
        assert_eq!(l.endpoint(), "http://localhost:9/timedtext");
        std::env::remove_var("LITGRAPH_YOUTUBE_TIMEDTEXT");
    }

    #[test]
    fn language_setter_persists() {
        let l = YouTubeTranscriptLoader::new("x").with_language("de");
        assert_eq!(l.language, "de");
    }
}

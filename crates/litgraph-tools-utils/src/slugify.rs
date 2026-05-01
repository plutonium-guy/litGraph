//! `SlugifyTool` — convert arbitrary text to URL-safe slugs.
//!
//! # Why a dedicated tool
//!
//! Content-management agents constantly need to derive a URL slug
//! from a user-provided title: blog posts ("My First Post!" → `my-first-post`),
//! Git branch names ("Bug fix: nullptr in auth" → `bug-fix-nullptr-in-auth`),
//! folder paths from headings, file names from search queries.
//! Without a tool, the LLM has to do per-character cleanup in its
//! head, which it does inconsistently — sometimes drops emojis,
//! sometimes preserves them, sometimes uses underscores when the
//! caller wanted dashes.
//!
//! `SlugifyTool` does the canonical cleanup in one call:
//! 1. Lowercase.
//! 2. Strip diacritics (`Café` → `cafe`, `Münster` → `munster`,
//!    `Núñez` → `nunez`) via an inline Latin-1 mapping table —
//!    no `unicode-normalization` dep.
//! 3. Replace runs of non-alphanumeric characters with a single
//!    separator (default `-`).
//! 4. Trim leading / trailing separators.
//! 5. Optional `max_length` truncation at a word boundary.
//!
//! # What this is NOT
//!
//! This is NOT a full Unicode-aware slugifier. The diacritic table
//! covers Latin-1 + a handful of common extended-Latin characters.
//! For non-Latin scripts (Cyrillic, Arabic, CJK), this tool drops
//! all non-ASCII characters — which is the right behavior when the
//! target system requires ASCII URLs but obviously loses content
//! for Unicode-URL targets. Agents working with non-ASCII content
//! that must round-trip should use percent-encoding via
//! `UrlParseTool` / their own logic instead.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct SlugifyTool;

impl SlugifyTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for SlugifyTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "slugify".into(),
            description: "Convert text to a URL-safe slug. Lowercases, strips Latin diacritics \
                (à/á/â → a, é/è/ê → e, ñ → n, ß → ss, æ → ae, etc), replaces runs of non-\
                alphanumeric characters with a single separator (default `-`), trims leading/\
                trailing separators. Optional `max_length` truncates at a word boundary. \
                Optional `separator` overrides the default dash. Use for blog/article slugs, \
                Git branch names, filesystem-safe filenames, URL path components."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The input text to slugify."
                    },
                    "separator": {
                        "type": "string",
                        "description": "Single character to use between words. Default: '-'."
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Optional max output length. Truncated at a word boundary if possible."
                    }
                },
                "required": ["text"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let text = args
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("slugify: missing `text`"))?;
        let separator: char = args
            .get("separator")
            .and_then(|v| v.as_str())
            .and_then(|s| s.chars().next())
            .unwrap_or('-');
        let max_length = args.get("max_length").and_then(|v| v.as_u64()).map(|n| n as usize);
        let slug = slugify_with(text, separator, max_length);
        Ok(json!({ "slug": slug }))
    }
}

/// Slugify with default separator `-` and no length cap.
pub fn slugify(text: &str) -> String {
    slugify_with(text, '-', None)
}

/// Slugify with explicit separator + optional max length.
///
/// Length truncation prefers a word boundary: if the truncation
/// point falls mid-word, the slug is cut back to the most recent
/// separator. If the entire prefix has no separator, it's truncated
/// at the byte boundary (no word-aware option). Trailing separators
/// are always stripped after truncation.
pub fn slugify_with(text: &str, separator: char, max_length: Option<usize>) -> String {
    // Phase 1: lowercase + diacritic strip + char-level normalization.
    let mut buf = String::with_capacity(text.len());
    let mut last_was_sep = true; // suppresses leading separators
    for c in text.chars() {
        let lower = c.to_lowercase().collect::<String>();
        for c2 in lower.chars() {
            let mapped = strip_diacritic(c2);
            for ch in mapped.chars() {
                if ch.is_ascii_alphanumeric() {
                    buf.push(ch);
                    last_was_sep = false;
                } else if !last_was_sep {
                    buf.push(separator);
                    last_was_sep = true;
                }
            }
        }
    }
    // Strip trailing separator.
    while buf.ends_with(separator) {
        buf.pop();
    }
    // Phase 2: optional length cap with word-boundary preference.
    if let Some(max) = max_length {
        if buf.len() > max {
            // Truncate at a separator boundary if one exists in the prefix.
            let prefix = &buf[..max];
            if let Some(sep_pos) = prefix.rfind(separator) {
                buf.truncate(sep_pos);
            } else {
                buf.truncate(max);
            }
            // Strip any trailing separator that landed at the cut.
            while buf.ends_with(separator) {
                buf.pop();
            }
        }
    }
    buf
}

/// Map a single character to its ASCII-foldable form. Handles common
/// Latin-1 diacritics + a few extended-Latin compounds. Returns the
/// input as a 1-char string if no mapping applies — caller filters
/// out non-alphanumerics in the next phase.
///
/// This is deliberately a small inline table rather than a full
/// `unicode-normalization` dep. Covers the realistic agent-input
/// space (Latin-script user-typed titles); non-Latin scripts get
/// dropped in the alphanumeric filter, which is the right behavior
/// for ASCII-URL targets.
fn strip_diacritic(c: char) -> String {
    match c {
        // a-family
        'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' | 'ā' | 'ă' | 'ą' => "a".into(),
        'æ' => "ae".into(),
        // c-family
        'ç' | 'ć' | 'č' | 'ĉ' | 'ċ' => "c".into(),
        // d-family
        'ð' | 'đ' | 'ď' => "d".into(),
        // e-family
        'è' | 'é' | 'ê' | 'ë' | 'ē' | 'ė' | 'ę' | 'ě' => "e".into(),
        // g-family
        'ĝ' | 'ğ' | 'ġ' | 'ģ' => "g".into(),
        // h-family
        'ĥ' | 'ħ' => "h".into(),
        // i-family
        'ì' | 'í' | 'î' | 'ï' | 'ī' | 'į' | 'ı' => "i".into(),
        // j-family
        'ĵ' => "j".into(),
        // k-family
        'ķ' => "k".into(),
        // l-family
        'ĺ' | 'ļ' | 'ľ' | 'ŀ' | 'ł' => "l".into(),
        // n-family
        'ñ' | 'ń' | 'ņ' | 'ň' | 'ŋ' => "n".into(),
        // o-family
        'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' | 'ō' | 'ő' => "o".into(),
        'œ' => "oe".into(),
        // r-family
        'ŕ' | 'ŗ' | 'ř' => "r".into(),
        // s-family
        'ś' | 'ŝ' | 'ş' | 'š' => "s".into(),
        'ß' => "ss".into(),
        // t-family
        'ţ' | 'ť' | 'ŧ' => "t".into(),
        // u-family
        'ù' | 'ú' | 'û' | 'ü' | 'ū' | 'ů' | 'ű' | 'ų' => "u".into(),
        // w-family
        'ŵ' => "w".into(),
        // y-family
        'ý' | 'ÿ' | 'ŷ' => "y".into(),
        // z-family
        'ź' | 'ż' | 'ž' => "z".into(),
        // þ → th (Old English / Icelandic)
        'þ' => "th".into(),
        // Pass through unchanged (next phase filters non-alphanumerics).
        _ => c.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_lowercase_and_dash() {
        assert_eq!(slugify("Hello World"), "hello-world");
    }

    #[test]
    fn punctuation_collapses_to_separator() {
        assert_eq!(slugify("Hello!! World??"), "hello-world");
        assert_eq!(slugify("foo, bar; baz."), "foo-bar-baz");
    }

    #[test]
    fn leading_and_trailing_punctuation_stripped() {
        assert_eq!(slugify("---hello---"), "hello");
        assert_eq!(slugify("!!!hello!!!"), "hello");
        assert_eq!(slugify("   spaces   "), "spaces");
    }

    #[test]
    fn diacritic_stripping() {
        assert_eq!(slugify("Café Münster"), "cafe-munster");
        assert_eq!(slugify("Núñez García"), "nunez-garcia");
        assert_eq!(slugify("naïve"), "naive");
        assert_eq!(slugify("résumé"), "resume");
    }

    #[test]
    fn ligature_expansions() {
        // æ → ae, œ → oe, ß → ss
        assert_eq!(slugify("Encyclopædia"), "encyclopaedia");
        assert_eq!(slugify("Œuvre"), "oeuvre");
        assert_eq!(slugify("Straße"), "strasse");
    }

    #[test]
    fn nordic_chars() {
        // ø → o, å → a
        assert_eq!(slugify("Bjørn Borg"), "bjorn-borg");
        assert_eq!(slugify("Malmö"), "malmo");
        assert_eq!(slugify("Ålesund"), "alesund");
    }

    #[test]
    fn non_latin_dropped() {
        // CJK / Cyrillic / Arabic don't appear in the diacritic table
        // — they pass through `strip_diacritic` unchanged but get
        // dropped by the ascii-alphanumeric filter. With ONLY non-
        // Latin content the output is empty.
        assert_eq!(slugify("こんにちは"), "");
        assert_eq!(slugify("Здравствуйте"), "");
        // Mixed: ASCII parts survive; non-ASCII becomes a separator boundary.
        assert_eq!(slugify("Hello こんにちは World"), "hello-world");
    }

    #[test]
    fn numbers_preserved() {
        assert_eq!(slugify("Top 10 Tips for 2024"), "top-10-tips-for-2024");
    }

    #[test]
    fn underscores_treated_as_separator() {
        // Underscores aren't ASCII alphanumeric → become separator.
        assert_eq!(slugify("hello_world"), "hello-world");
    }

    #[test]
    fn custom_separator() {
        assert_eq!(slugify_with("hello world", '_', None), "hello_world");
        assert_eq!(slugify_with("foo bar baz", '.', None), "foo.bar.baz");
    }

    #[test]
    fn max_length_truncates_at_word_boundary() {
        // 30-char input; cap at 18. The hyphen at position 14 is the
        // last word boundary in the prefix → truncate to "the-quick-brown".
        let s = slugify_with("the quick brown fox jumps over", '-', Some(18));
        assert_eq!(s, "the-quick-brown");
        assert!(s.len() <= 18);
    }

    #[test]
    fn max_length_truncates_at_byte_when_no_separator_in_prefix() {
        // Single long word with no separator in the prefix → truncate
        // at byte boundary (the whole word, then trim trailing
        // separator if any landed at the cut).
        let s = slugify_with("supercalifragilisticexpialidocious", '-', Some(10));
        assert_eq!(s.len(), 10);
        assert_eq!(s, "supercalif");
    }

    #[test]
    fn max_length_above_input_no_op() {
        let s = slugify_with("hello", '-', Some(100));
        assert_eq!(s, "hello");
    }

    #[test]
    fn empty_input_yields_empty() {
        assert_eq!(slugify(""), "");
        assert_eq!(slugify("   "), "");
        assert_eq!(slugify("---"), "");
    }

    #[test]
    fn case_with_only_non_alphanumeric() {
        assert_eq!(slugify("!@#$%^&*()"), "");
    }

    #[test]
    fn realistic_blog_title() {
        assert_eq!(
            slugify("10 Things You Didn't Know About Rust's Ownership Model"),
            "10-things-you-didn-t-know-about-rust-s-ownership-model"
        );
    }

    #[test]
    fn realistic_git_branch_name() {
        assert_eq!(
            slugify("Bug fix: nullptr in auth middleware (#1234)"),
            "bug-fix-nullptr-in-auth-middleware-1234"
        );
    }

    #[tokio::test]
    async fn tool_basic_invoke() {
        let t = SlugifyTool::new();
        let v = t.run(json!({"text": "Hello, World!"})).await.unwrap();
        assert_eq!(v.get("slug").unwrap(), "hello-world");
    }

    #[tokio::test]
    async fn tool_custom_separator() {
        let t = SlugifyTool::new();
        let v = t
            .run(json!({"text": "Hello World", "separator": "_"}))
            .await
            .unwrap();
        assert_eq!(v.get("slug").unwrap(), "hello_world");
    }

    #[tokio::test]
    async fn tool_max_length() {
        let t = SlugifyTool::new();
        let v = t
            .run(json!({"text": "the quick brown fox jumps", "max_length": 15}))
            .await
            .unwrap();
        let slug = v.get("slug").unwrap().as_str().unwrap();
        assert!(slug.len() <= 15);
        assert!(slug.starts_with("the-"));
    }

    #[tokio::test]
    async fn tool_missing_text_errors() {
        let t = SlugifyTool::new();
        let r = t.run(json!({})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }
}

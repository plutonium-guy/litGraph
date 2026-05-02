//! List + boolean output parsers — LangChain parity for the "ask LLM for
//! a list" family. Complements `xml_parser` (iter 105) and `structured`
//! (iter 89) to finish the output-parser surface.
//!
//! # Why each one exists
//!
//! - `parse_comma_list` — #1 LangChain tutorial example. "Give me 5 X
//!   separated by commas." Loose-tolerant: trailing commas, surrounding
//!   prose, extra whitespace all accepted.
//! - `parse_numbered_list` — "Give me 5 items, numbered." Handles `1.`,
//!   `1)`, `1:` delimiters. Real LLMs mix these freely.
//! - `parse_markdown_list` — "Give me bullet points." Accepts `-`, `*`,
//!   `+` bullets plus any indent.
//! - `parse_boolean` — "Answer yes or no." Accepts yes/no/true/false in
//!   any case. Treats missing/ambiguous as `Err` (not a silent false —
//!   the caller needs to distinguish).
//!
//! All four tolerate surrounding prose because real LLM output looks like
//! "Sure! Here are 5 fruits: apple, banana, cherry, date, elderberry."

use std::collections::HashSet;

use crate::{Error, Result};

/// Parse a comma-separated list from LLM output. Extracts the first
/// run of comma-joined items; strips surrounding prose.
///
/// Rules:
/// - Splits on `,`. Whitespace around items trimmed.
/// - Empty items removed (`"a,,b"` → `["a", "b"]`).
/// - Surrounding quotes stripped from each item (`'a', "b"` → `a, b`).
/// - Leading/trailing prose ("Here are 5: a, b, c. Hope this helps!")
///   stripped by locating the largest comma-joined span.
///
/// ```ignore
/// let items = parse_comma_list("Here you go: apple, banana, cherry.");
/// assert_eq!(items, vec!["apple", "banana", "cherry"]);
/// ```
pub fn parse_comma_list(text: &str) -> Vec<String> {
    // Strategy: look at every line; the line with the most commas is the
    // list. If no commas anywhere, treat the whole text as one item
    // (LLM returned a single-item "list").
    let line = text
        .lines()
        .max_by_key(|l| l.matches(',').count())
        .unwrap_or("")
        .trim();

    if !line.contains(',') {
        let t = strip_quotes(line.trim_end_matches(['.', '!']).trim());
        if t.is_empty() {
            return vec![];
        }
        return vec![t];
    }

    // Split on commas; then strip intro prose from the first item (anything
    // up to and including the last `:`) and trailing prose from the last
    // item (anything after the first sentence-ending punctuation).
    let mut parts: Vec<&str> = line.split(',').collect();
    if let Some(first) = parts.first_mut() {
        if let Some(idx) = first.rfind(':') {
            *first = &first[idx + 1..];
        }
    }
    if let Some(last) = parts.last_mut() {
        if let Some(idx) = last.find(['.', '!', '?']) {
            *last = &last[..idx];
        }
    }
    parts
        .into_iter()
        .map(|s| strip_quotes(s.trim()))
        .filter(|s| !s.is_empty())
        .collect()
}

/// Parse a numbered list. Accepts `1. item`, `1) item`, `1: item`,
/// `1 - item` with any indent. Returns items in source order.
///
/// Numbering doesn't have to be dense (`1. a\n3. b` → `["a", "b"]`) —
/// LLMs sometimes skip. We don't validate sequence.
///
/// ```ignore
/// let text = "Here's the list:\n1. apple\n2. banana\n3. cherry";
/// let items = parse_numbered_list(text);
/// assert_eq!(items, vec!["apple", "banana", "cherry"]);
/// ```
pub fn parse_numbered_list(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_start();
        // Find the digit run.
        let mut bytes = line.bytes();
        let first = match bytes.next() {
            Some(b) if b.is_ascii_digit() => b,
            _ => continue,
        };
        let _ = first;
        let digits_end = line
            .bytes()
            .take_while(|b| b.is_ascii_digit())
            .count();
        if digits_end == 0 {
            continue;
        }
        // Require a delimiter after the digits.
        let rest = &line[digits_end..];
        let after_delim = if let Some(s) = rest.strip_prefix('.') {
            s
        } else if let Some(s) = rest.strip_prefix(')') {
            s
        } else if let Some(s) = rest.strip_prefix(':') {
            s
        } else if let Some(s) = rest.strip_prefix(" -") {
            s
        } else if let Some(s) = rest.strip_prefix(" –") {
            // en-dash — some LLMs emit this
            s
        } else {
            continue;
        };
        let item = after_delim.trim();
        if !item.is_empty() {
            out.push(strip_quotes(item));
        }
    }
    out
}

/// Parse a markdown-style bulleted list. Accepts `-`, `*`, `+` bullets
/// with any leading indent. Nested sub-bullets keep their raw text (we
/// don't build a tree — real use is flat).
///
/// ```ignore
/// let text = "Sure:\n- apple\n- banana\n* cherry";
/// let items = parse_markdown_list(text);
/// assert_eq!(items, vec!["apple", "banana", "cherry"]);
/// ```
pub fn parse_markdown_list(text: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for raw in text.lines() {
        let line = raw.trim_start();
        let after = if let Some(s) = line.strip_prefix("- ") {
            s
        } else if let Some(s) = line.strip_prefix("* ") {
            s
        } else if let Some(s) = line.strip_prefix("+ ") {
            s
        } else if let Some(s) = line.strip_prefix("• ") {
            // Sometimes LLMs use the unicode bullet
            s
        } else {
            continue;
        };
        let item = after.trim();
        if !item.is_empty() {
            out.push(strip_quotes(item));
        }
    }
    out
}

/// Parse a yes/no answer. Returns `Ok(true)` / `Ok(false)` or `Err` if
/// ambiguous or missing. Case-insensitive.
///
/// Accepted affirmative forms: yes / y / true / 1 / affirmative / correct.
/// Accepted negative forms: no / n / false / 0 / negative / incorrect.
///
/// Strategy: scan tokens in order; first matching token wins. Prose
/// around it ignored.
///
/// ```ignore
/// assert_eq!(parse_boolean("Yes, that's right.").unwrap(), true);
/// assert_eq!(parse_boolean("no way").unwrap(), false);
/// ```
pub fn parse_boolean(text: &str) -> Result<bool> {
    let affirm: HashSet<&str> = [
        "yes",
        "y",
        "true",
        "t",
        "1",
        "affirmative",
        "correct",
        "right",
    ]
    .into_iter()
    .collect();
    let negate: HashSet<&str> = [
        "no",
        "n",
        "false",
        "f",
        "0",
        "negative",
        "incorrect",
        "wrong",
    ]
    .into_iter()
    .collect();

    for token in tokenize_words(text) {
        let lower = token.to_ascii_lowercase();
        if affirm.contains(lower.as_str()) {
            return Ok(true);
        }
        if negate.contains(lower.as_str()) {
            return Ok(false);
        }
    }
    Err(Error::parse(format!(
        "parse_boolean: no yes/no token found in {:?}",
        text
    )))
}

fn strip_quotes(s: &str) -> String {
    let s = s.trim();
    if (s.starts_with('"') && s.ends_with('"') && s.len() >= 2)
        || (s.starts_with('\'') && s.ends_with('\'') && s.len() >= 2)
    {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

/// Split `text` into words, treating any non-alphanumeric char (except `_`)
/// as a separator. Used by `parse_boolean` to find a yes/no token inside
/// prose without matching substrings inside longer words.
fn tokenize_words(text: &str) -> Vec<&str> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    for (i, &b) in bytes.iter().enumerate() {
        let is_word = b.is_ascii_alphanumeric() || b == b'_';
        match (start, is_word) {
            (None, true) => start = Some(i),
            (Some(s), false) => {
                out.push(&text[s..i]);
                start = None;
            }
            _ => {}
        }
    }
    if let Some(s) = start {
        out.push(&text[s..]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comma_list_basic() {
        assert_eq!(
            parse_comma_list("apple, banana, cherry"),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn comma_list_strips_surrounding_prose() {
        assert_eq!(
            parse_comma_list("Here you go: apple, banana, cherry. Hope this helps!"),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn comma_list_strips_trailing_comma_and_empty_items() {
        assert_eq!(
            parse_comma_list("a, b, , c,"),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn comma_list_strips_quotes_around_items() {
        assert_eq!(
            parse_comma_list(r#""apple", 'banana', "cherry""#),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn comma_list_single_item_no_commas() {
        assert_eq!(parse_comma_list("just one"), vec!["just one"]);
    }

    #[test]
    fn comma_list_empty_input() {
        assert_eq!(parse_comma_list(""), Vec::<String>::new());
    }

    #[test]
    fn comma_list_picks_the_line_with_most_commas() {
        // Intro line has zero commas; list line has many.
        let text = "Sure, I can do that for you.\napple, banana, cherry, date";
        assert_eq!(
            parse_comma_list(text),
            vec!["apple", "banana", "cherry", "date"]
        );
    }

    #[test]
    fn numbered_list_dot_delimiter() {
        let text = "1. apple\n2. banana\n3. cherry";
        assert_eq!(
            parse_numbered_list(text),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn numbered_list_accepts_paren_and_colon() {
        let text = "1) apple\n2: banana\n3. cherry";
        assert_eq!(
            parse_numbered_list(text),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn numbered_list_accepts_hyphen_delim() {
        let text = "1 - apple\n2 - banana";
        assert_eq!(parse_numbered_list(text), vec!["apple", "banana"]);
    }

    #[test]
    fn numbered_list_handles_intro_prose() {
        let text = "Here's the list you asked for:\n1. apple\n2. banana\n\nLet me know if you need more.";
        assert_eq!(parse_numbered_list(text), vec!["apple", "banana"]);
    }

    #[test]
    fn numbered_list_tolerates_non_dense_numbering() {
        // LLMs sometimes skip numbers; we don't validate sequence.
        let text = "1. apple\n3. banana\n5. cherry";
        assert_eq!(
            parse_numbered_list(text),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn numbered_list_skips_lines_without_a_number_prefix() {
        let text = "Intro.\n1. apple\nMiddle prose.\n2. banana";
        assert_eq!(parse_numbered_list(text), vec!["apple", "banana"]);
    }

    #[test]
    fn numbered_list_strips_quotes() {
        let text = "1. \"apple\"\n2. 'banana'";
        assert_eq!(parse_numbered_list(text), vec!["apple", "banana"]);
    }

    #[test]
    fn markdown_list_dash_bullets() {
        let text = "- apple\n- banana\n- cherry";
        assert_eq!(
            parse_markdown_list(text),
            vec!["apple", "banana", "cherry"]
        );
    }

    #[test]
    fn markdown_list_mixed_bullets() {
        let text = "- apple\n* banana\n+ cherry\n• date";
        assert_eq!(
            parse_markdown_list(text),
            vec!["apple", "banana", "cherry", "date"]
        );
    }

    #[test]
    fn markdown_list_indented_bullets() {
        let text = "Here:\n  - apple\n    - banana (nested)\n  - cherry";
        assert_eq!(
            parse_markdown_list(text),
            vec!["apple", "banana (nested)", "cherry"]
        );
    }

    #[test]
    fn markdown_list_ignores_non_bullet_lines() {
        let text = "Intro.\n- apple\nMore prose.\n- banana\nClosing.";
        assert_eq!(parse_markdown_list(text), vec!["apple", "banana"]);
    }

    #[test]
    fn boolean_yes_no_case_insensitive() {
        assert_eq!(parse_boolean("Yes").unwrap(), true);
        assert_eq!(parse_boolean("NO").unwrap(), false);
        assert_eq!(parse_boolean("YeS").unwrap(), true);
    }

    #[test]
    fn boolean_true_false() {
        assert_eq!(parse_boolean("true").unwrap(), true);
        assert_eq!(parse_boolean("False").unwrap(), false);
    }

    #[test]
    fn boolean_inside_prose() {
        assert_eq!(
            parse_boolean("Sure, yes that's right.").unwrap(),
            true
        );
        assert_eq!(parse_boolean("No way, that's wrong.").unwrap(), false);
    }

    #[test]
    fn boolean_first_match_wins() {
        // LLM said "yes" first, then added a "no" caveat — take the first.
        assert_eq!(
            parse_boolean("Yes but actually no under certain conditions").unwrap(),
            true
        );
    }

    #[test]
    fn boolean_ambiguous_text_errors() {
        assert!(parse_boolean("Maybe? I'm not sure.").is_err());
        assert!(parse_boolean("").is_err());
    }

    #[test]
    fn boolean_does_not_match_substring_inside_word() {
        // "yesterday" contains "yes" — must NOT be parsed as yes.
        assert!(parse_boolean("Yesterday was fine.").is_err());
    }

    #[test]
    fn boolean_affirm_alt_forms() {
        assert_eq!(parse_boolean("affirmative").unwrap(), true);
        assert_eq!(parse_boolean("1").unwrap(), true);
        assert_eq!(parse_boolean("correct").unwrap(), true);
    }

    #[test]
    fn boolean_negate_alt_forms() {
        assert_eq!(parse_boolean("negative").unwrap(), false);
        assert_eq!(parse_boolean("0").unwrap(), false);
        assert_eq!(parse_boolean("incorrect").unwrap(), false);
    }
}

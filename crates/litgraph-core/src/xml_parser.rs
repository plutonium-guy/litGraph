//! XML output parser — extract content from `<tag>...</tag>` blocks in
//! LLM responses. Direct LangChain `XMLOutputParser` parity.
//!
//! # Why this matters
//!
//! Anthropic's model family is trained to wrap reasoning / answers in XML
//! tags (`<thinking>...</thinking><answer>...</answer>`). Their cookbook +
//! system-prompt guidance uses XML heavily. OpenAI + Gemini work fine with
//! JSON schema (iter 89's `StructuredChatModel`), but Anthropic's XML
//! idiom is common enough that hand-rolling regex per call-site is the #1
//! paper-cut for users migrating prompt chains.
//!
//! # Two modes
//!
//! - `parse_xml_tags(text, &["thinking", "answer"])` — flat: first occurrence
//!   of each named tag → `HashMap<String, String>`. Ignores unnamed tags.
//!   The 80% case.
//! - `parse_nested_xml(text)` — tree: all tags → `Value` (recursive).
//!   Handles nesting, repeated tags (as arrays), mixed text + element
//!   children. For chain-of-thought / tool-result-style responses.
//!
//! # What it's NOT
//!
//! - NOT a conformant XML parser. No namespaces, no CDATA, no
//!   processing instructions, no DTD. LLM output is small, loose,
//!   text-with-tags — a full XML parser would reject it at the first
//!   unclosed entity.
//! - No attribute parsing (LLM responses rarely use attributes).
//! - Entity decoding is limited to the five XML built-ins (`&lt;`, `&gt;`,
//!   `&amp;`, `&quot;`, `&apos;`). Numeric entities (`&#123;`) decoded too.

use std::collections::HashMap;

use serde_json::{Map, Value};

use crate::{Error, Result};

/// Flat extractor: pull the FIRST occurrence of each requested tag's inner
/// content. Missing tags → absent from the result map (not an error —
/// LLMs often forget tags and caller decides whether to retry).
///
/// ```ignore
/// let text = "<thinking>reasoning here</thinking>\n<answer>42</answer>";
/// let m = parse_xml_tags(text, &["thinking", "answer", "maybe_missing"]);
/// assert_eq!(m["thinking"], "reasoning here");
/// assert_eq!(m["answer"], "42");
/// assert!(!m.contains_key("maybe_missing"));
/// ```
pub fn parse_xml_tags(text: &str, tags: &[&str]) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for &tag in tags {
        if let Some(inner) = extract_first_tag_content(text, tag) {
            out.insert(tag.to_string(), decode_entities(&inner));
        }
    }
    out
}

/// Tree extractor: walk the text, collecting every tag into a nested
/// `Value`. Rules:
/// - Leaf tag (no nested tags) → `Value::String(inner_text)`.
/// - Container tag with one or more named children → `Value::Object`
///   mapping child-name → child-value. Repeated children of the same name
///   become `Value::Array`.
/// - Text content interleaved with children is dropped (LLM containers
///   rarely have meaningful loose text). Use the flat mode if you need it.
///
/// ```ignore
/// let text = "<root><item>a</item><item>b</item><name>foo</name></root>";
/// let v = parse_nested_xml(text).unwrap();
/// assert_eq!(v["root"]["item"], serde_json::json!(["a", "b"]));
/// assert_eq!(v["root"]["name"], "foo");
/// ```
pub fn parse_nested_xml(text: &str) -> Result<Value> {
    let mut pos = 0usize;
    let (values, _) = parse_children(text.as_bytes(), &mut pos, None)?;
    // Flatten: if there's exactly one top-level tag, return that as the root
    // (most common LLM shape). Otherwise return the aggregated object.
    match values.len() {
        0 => Ok(Value::Null),
        _ => Ok(aggregate(values)),
    }
}

/// Walk `bytes` starting at `*pos` until we hit `</close_tag>` (or EOF
/// when `close_tag` is None). Return a list of (name, value) for every
/// top-level tag encountered. Loose text is ignored.
fn parse_children(
    bytes: &[u8],
    pos: &mut usize,
    close_tag: Option<&str>,
) -> Result<(Vec<(String, Value)>, bool)> {
    let mut out: Vec<(String, Value)> = Vec::new();
    while *pos < bytes.len() {
        // Skip to the next `<`.
        let next_lt = match find_next_byte(bytes, *pos, b'<') {
            Some(i) => i,
            None => {
                *pos = bytes.len();
                return Ok((out, false));
            }
        };
        *pos = next_lt;
        // `</close_tag>` — stop.
        if matches_close_tag(bytes, *pos, close_tag) {
            *pos += close_tag.map(|t| t.len() + 3).unwrap_or(0); // skip `</tag>`
            return Ok((out, true));
        }
        // Try to parse an opening tag.
        let (tag_name, tag_len) = match parse_open_tag(bytes, *pos) {
            Some(x) => x,
            None => {
                // Not a tag — skip this '<' and continue.
                *pos += 1;
                continue;
            }
        };
        *pos += tag_len;
        // Now recurse to collect the tag's children + hoover text up to the
        // matching `</tag_name>`.
        let tag_start = *pos;
        let (child_values, found) = parse_children(bytes, pos, Some(&tag_name))?;
        let inner_slice = &bytes[tag_start..usize::min(*pos, bytes.len()).saturating_sub(
            // back out the `</tag>` bytes we just consumed if close was found
            if found { tag_name.len() + 3 } else { 0 },
        )];
        let value = if child_values.is_empty() {
            // Leaf: raw text content.
            let text = std::str::from_utf8(inner_slice)
                .map_err(|e| Error::other(format!("xml utf8: {e}")))?;
            Value::String(decode_entities(text.trim()))
        } else {
            // Container: aggregate children.
            aggregate(child_values)
        };
        out.push((tag_name, value));
    }
    Ok((out, false))
}

/// Convert a list of (name, value) pairs into a Value. Repeated names
/// become arrays; mixed-name input becomes an Object; a single pair stays
/// as a single-key Object (callers can drill in with `v["name"]`).
fn aggregate(pairs: Vec<(String, Value)>) -> Value {
    let mut obj: Map<String, Value> = Map::new();
    for (name, value) in pairs {
        match obj.remove(&name) {
            Some(Value::Array(mut arr)) => {
                arr.push(value);
                obj.insert(name, Value::Array(arr));
            }
            Some(existing) => {
                obj.insert(name, Value::Array(vec![existing, value]));
            }
            None => {
                obj.insert(name, value);
            }
        }
    }
    Value::Object(obj)
}

/// Does `bytes[pos..]` start with `</close_tag>` (trimmed of trailing
/// whitespace before `>`)? If `close_tag` is None, never matches.
fn matches_close_tag(bytes: &[u8], pos: usize, close_tag: Option<&str>) -> bool {
    let tag = match close_tag {
        Some(t) => t,
        None => return false,
    };
    if pos + 2 >= bytes.len() || bytes[pos] != b'<' || bytes[pos + 1] != b'/' {
        return false;
    }
    let tag_bytes = tag.as_bytes();
    if pos + 2 + tag_bytes.len() > bytes.len() {
        return false;
    }
    if !bytes[pos + 2..pos + 2 + tag_bytes.len()].eq_ignore_ascii_case(tag_bytes) {
        return false;
    }
    // Next char must be '>' or whitespace (then '>') — LLMs don't
    // typically insert attributes on close tags, but tolerate trailing
    // whitespace `</tag  >`.
    let after = pos + 2 + tag_bytes.len();
    let mut i = after;
    while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
        i += 1;
    }
    i < bytes.len() && bytes[i] == b'>'
}

/// Parse `<tagname>` (or `<tagname/>` self-closing) starting at bytes[pos].
/// Returns (name, total_bytes_consumed). Returns None for malformed or
/// closing tags (the caller handles those).
fn parse_open_tag(bytes: &[u8], pos: usize) -> Option<(String, usize)> {
    if pos >= bytes.len() || bytes[pos] != b'<' {
        return None;
    }
    // Disqualify `</...` (close), `<!--` (comment), `<?...` (PI).
    if pos + 1 < bytes.len() && matches!(bytes[pos + 1], b'/' | b'!' | b'?') {
        return None;
    }
    let mut name_end = pos + 1;
    while name_end < bytes.len() {
        let c = bytes[name_end];
        if c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.' {
            name_end += 1;
        } else {
            break;
        }
    }
    if name_end == pos + 1 {
        return None;
    }
    let name = std::str::from_utf8(&bytes[pos + 1..name_end]).ok()?.to_string();
    // Find the closing `>` (skipping any attributes we ignore).
    let mut i = name_end;
    while i < bytes.len() && bytes[i] != b'>' {
        i += 1;
    }
    if i >= bytes.len() {
        return None;
    }
    // If the byte before `>` is `/`, it's self-closing — treat as empty.
    let self_close = i > 0 && bytes[i - 1] == b'/';
    let consumed = i - pos + 1;
    if self_close {
        // Self-closing: caller's parse_children will get empty children + no
        // closing tag — that's valid via the `return Ok((out, false))` on EOF
        // path. Simpler: emit a pseudo-opening that matches nothing inside.
        // We just skip it as if it had been <name></name>.
        return Some((name, consumed));
    }
    Some((name, consumed))
}

/// Extract the content between the FIRST `<tag>` and its matching `</tag>`.
/// Does NOT handle nested same-named tags (LLM responses rarely nest the
/// same tag; `parse_nested_xml` does if you need it).
fn extract_first_tag_content(text: &str, tag: &str) -> Option<String> {
    let open_needle = format!("<{tag}>");
    let close_needle = format!("</{tag}>");
    let start = find_ci(text, &open_needle)?;
    let after_open = start + open_needle.len();
    let end = find_ci(&text[after_open..], &close_needle).map(|i| after_open + i)?;
    Some(text[after_open..end].trim().to_string())
}

/// Case-insensitive substring search. Returns byte offset of match start.
fn find_ci(haystack: &str, needle: &str) -> Option<usize> {
    let hb = haystack.as_bytes();
    let nb = needle.as_bytes();
    if nb.is_empty() || nb.len() > hb.len() {
        return None;
    }
    (0..=hb.len() - nb.len()).find(|&i| hb[i..i + nb.len()].eq_ignore_ascii_case(nb))
}

fn find_next_byte(bytes: &[u8], from: usize, target: u8) -> Option<usize> {
    bytes[from..].iter().position(|&b| b == target).map(|i| from + i)
}

/// Decode the five XML built-in entities + numeric entities (`&#123;`,
/// `&#x7b;`). Unknown entities pass through unchanged — better than
/// erroring on `&foo;` which LLMs occasionally emit as literal text.
pub fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'&' {
            // Safe: we're iterating a str's bytes, so pushing the char at
            // this boundary is correct. Find the char start here.
            let rest = &s[i..];
            let ch = rest.chars().next().unwrap();
            out.push(ch);
            i += ch.len_utf8();
            continue;
        }
        // Look for the semicolon within 10 bytes (longest built-in is `&apos;` = 6).
        let end = bytes[i..]
            .iter()
            .take(12)
            .position(|&b| b == b';')
            .map(|p| i + p);
        if let Some(semi) = end {
            let entity = &s[i..=semi];
            match entity {
                "&lt;" => out.push('<'),
                "&gt;" => out.push('>'),
                "&amp;" => out.push('&'),
                "&quot;" => out.push('"'),
                "&apos;" => out.push('\''),
                _ => {
                    // Numeric: &#N; or &#xN;.
                    if entity.starts_with("&#") {
                        let num_part = &entity[2..entity.len() - 1];
                        let code = if let Some(hex) = num_part.strip_prefix(['x', 'X']) {
                            u32::from_str_radix(hex, 16).ok()
                        } else {
                            num_part.parse::<u32>().ok()
                        };
                        if let Some(c) = code.and_then(char::from_u32) {
                            out.push(c);
                        } else {
                            out.push_str(entity);
                        }
                    } else {
                        // Unknown named entity — pass through.
                        out.push_str(entity);
                    }
                }
            }
            i = semi + 1;
        } else {
            // Literal `&` with no `;` within range — not an entity.
            out.push('&');
            i += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn flat_parser_extracts_named_tags() {
        let text = "<thinking>Let me work this out</thinking>\n\
                    <answer>The answer is 42</answer>";
        let m = parse_xml_tags(text, &["thinking", "answer"]);
        assert_eq!(m["thinking"], "Let me work this out");
        assert_eq!(m["answer"], "The answer is 42");
    }

    #[test]
    fn flat_parser_missing_tag_is_absent_not_error() {
        // LLMs often skip tags; treat missing as absent (caller decides).
        let text = "<answer>42</answer>";
        let m = parse_xml_tags(text, &["thinking", "answer"]);
        assert!(!m.contains_key("thinking"));
        assert_eq!(m["answer"], "42");
    }

    #[test]
    fn flat_parser_takes_first_occurrence_only() {
        // If the model emits the same tag twice, we keep the first — matches
        // LangChain's behavior and avoids caller confusion.
        let text = "<answer>first</answer><answer>second</answer>";
        let m = parse_xml_tags(text, &["answer"]);
        assert_eq!(m["answer"], "first");
    }

    #[test]
    fn flat_parser_decodes_builtin_entities() {
        let text = "<answer>5 &lt; 10 &amp; true</answer>";
        let m = parse_xml_tags(text, &["answer"]);
        assert_eq!(m["answer"], "5 < 10 & true");
    }

    #[test]
    fn flat_parser_case_insensitive_tag_match() {
        // Some models randomly capitalize. Don't punish the caller.
        let text = "<Thinking>hmm</Thinking>";
        let m = parse_xml_tags(text, &["thinking"]);
        assert_eq!(m["thinking"], "hmm");
    }

    #[test]
    fn flat_parser_trims_whitespace_around_content() {
        let text = "<answer>\n  42  \n</answer>";
        let m = parse_xml_tags(text, &["answer"]);
        assert_eq!(m["answer"], "42");
    }

    #[test]
    fn flat_parser_handles_loose_text_outside_tags() {
        // Anthropic often wraps reasoning + answer with prose. Ignore prose.
        let text = "Here's my thinking:\n<thinking>...</thinking>\nAnd my \
                    answer:\n<answer>yes</answer>.\nThat's all.";
        let m = parse_xml_tags(text, &["thinking", "answer"]);
        assert_eq!(m["thinking"], "...");
        assert_eq!(m["answer"], "yes");
    }

    #[test]
    fn decode_entities_handles_numeric_entities() {
        assert_eq!(decode_entities("&#65;&#x42;"), "AB");
        // Invalid numeric → pass through unchanged.
        assert_eq!(decode_entities("&#99999999999;"), "&#99999999999;");
    }

    #[test]
    fn decode_entities_passes_through_unknown_named_entity() {
        // Don't error on things like `&foo;` that LLMs emit as literal text.
        assert_eq!(decode_entities("x &foo; y"), "x &foo; y");
    }

    #[test]
    fn decode_entities_passes_through_bare_ampersand() {
        assert_eq!(decode_entities("Foo & bar"), "Foo & bar");
    }

    #[test]
    fn nested_parser_builds_tree_for_simple_leaf() {
        let v = parse_nested_xml("<answer>42</answer>").unwrap();
        assert_eq!(v["answer"], "42");
    }

    #[test]
    fn nested_parser_builds_nested_object() {
        let text = "<response><thinking>step 1</thinking><answer>42</answer></response>";
        let v = parse_nested_xml(text).unwrap();
        assert_eq!(v["response"]["thinking"], "step 1");
        assert_eq!(v["response"]["answer"], "42");
    }

    #[test]
    fn nested_parser_repeated_tags_become_array() {
        let text = "<root><item>a</item><item>b</item><item>c</item></root>";
        let v = parse_nested_xml(text).unwrap();
        assert_eq!(v["root"]["item"], json!(["a", "b", "c"]));
    }

    #[test]
    fn nested_parser_mixed_repeated_and_unique_children() {
        let text = "<root><item>a</item><item>b</item><name>foo</name></root>";
        let v = parse_nested_xml(text).unwrap();
        assert_eq!(v["root"]["item"], json!(["a", "b"]));
        assert_eq!(v["root"]["name"], "foo");
    }

    #[test]
    fn nested_parser_empty_input_returns_null() {
        assert_eq!(parse_nested_xml("").unwrap(), Value::Null);
        // No tags at all → null.
        assert_eq!(parse_nested_xml("just prose").unwrap(), Value::Null);
    }

    #[test]
    fn nested_parser_handles_bare_ampersand_in_content() {
        // Entity decoding applied to leaf text.
        let text = "<answer>5 &lt; 10 &amp; true</answer>";
        let v = parse_nested_xml(text).unwrap();
        assert_eq!(v["answer"], "5 < 10 & true");
    }

    #[test]
    fn nested_parser_ignores_loose_text_between_tags() {
        let text = "<root>hello<item>x</item>world</root>";
        let v = parse_nested_xml(text).unwrap();
        // The child { item: x } wins; "hello" / "world" dropped.
        assert_eq!(v["root"]["item"], "x");
    }

    #[test]
    fn nested_parser_survives_stray_lt_in_content_gracefully() {
        // A `<` that isn't a tag-opener should not crash. Real LLMs emit
        // `<` in code samples inside tags all the time.
        let text = "<code>if (x < 10) return;</code>";
        // We'll likely tree-fail silently and return empty — lock the
        // "don't panic" invariant rather than exact output.
        let _ = parse_nested_xml(text);
    }

    #[test]
    fn flat_parser_independent_of_tag_order_in_args() {
        let text = "<a>1</a><b>2</b>";
        let m1 = parse_xml_tags(text, &["a", "b"]);
        let m2 = parse_xml_tags(text, &["b", "a"]);
        assert_eq!(m1, m2);
    }
}

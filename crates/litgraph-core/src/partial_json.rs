//! Partial / streaming JSON parser. FEATURES.md line 75.
//!
//! Closes the iter-121 audit gap #5: "parse structured output as tokens
//! arrive instead of waiting for full response." LangChain's cookbook
//! calls this `partial_json` or "streaming JSON".
//!
//! # Use case
//!
//! OpenAI / Anthropic / Gemini structured output streams byte-by-byte.
//! The user wants to see the object building up in real time — UI
//! progressively renders fields as they arrive. This needs a parser
//! that accepts partial input and returns a best-effort `Value` snapshot.
//!
//! ```ignore
//! let mut acc = String::new();
//! while let Some(token) = stream.next_token().await {
//!     acc.push_str(&token);
//!     if let Some(partial) = parse_partial_json(&acc) {
//!         ui.render(&partial);  // may be missing trailing fields, etc
//!     }
//! }
//! let final: Value = serde_json::from_str(&acc)?;  // full parse at end
//! ```
//!
//! # Strategy
//!
//! `serde_json` refuses partial input. We instead:
//! 1. Locate the first `{` or `[` — the root container start.
//! 2. Walk forward, tracking:
//!    - bracket / brace depth
//!    - whether currently inside a string literal (and whether the next
//!      char is an escape)
//!    - whether the last non-whitespace token was `:` or `,` (so we
//!      know if we need to drop a dangling key-with-no-value)
//! 3. At the end of input, close any open string with `"`, drop any
//!    trailing comma/colon, and close any open containers with `}` / `]`
//!    in reverse nesting order.
//! 4. Parse the repaired string with `serde_json`. Return `None` if that
//!    fails (buffer not yet parseable — common when the stream is
//!    mid-number like `"n": 4` where the 4 could be the start of 42).
//!
//! # Invariant: monotonic completeness
//!
//! If `parse_partial_json(s1)` returns `Some(v1)` and `s2` is `s1` with
//! MORE tokens appended, then `parse_partial_json(s2)` returns `Some(v2)`
//! where `v2` has all the keys of `v1` (values may be refined — e.g.,
//! string grows from "hel" to "hello"). Caller-visible behavior: no UI
//! flicker from keys disappearing.
//!
//! # What it's NOT
//!
//! - NOT a replacement for `serde_json::from_str` on complete JSON.
//!   Always use the real parser once the stream closes.
//! - NOT an event-stream parser. See `parse_partial_json_events` for a
//!   token-emitting variant — out of scope for iter 122.

use serde_json::Value;

/// Best-effort parse of a partial JSON string. Returns `Some(value)`
/// if the buffer repairs into valid JSON after auto-closing; `None`
/// if the buffer is empty, has no recognizable root, or repair
/// still produces an invalid shape.
///
/// Safe to call on every token — amortized cost is linear in buffer
/// length (single forward walk + one parse call).
pub fn parse_partial_json(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    let start = find_root_start(bytes)?;
    let repaired = repair(&text[start..]);
    serde_json::from_str(&repaired).ok()
}

/// Like `parse_partial_json` but returns the REPAIRED string instead of
/// the parsed `Value`. Useful for callers that want to forward to a
/// different JSON library or inspect what the repair looked like.
pub fn repair_partial_json(text: &str) -> Option<String> {
    let bytes = text.as_bytes();
    let start = find_root_start(bytes)?;
    Some(repair(&text[start..]))
}

fn find_root_start(bytes: &[u8]) -> Option<usize> {
    bytes.iter().position(|&b| b == b'{' || b == b'[')
}

fn repair(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut stack: Vec<u8> = Vec::new(); // 'o' for object, 'a' for array
    let mut in_string = false;
    let mut escape = false;
    // Track last non-whitespace, non-string byte outside string context.
    // Used to detect dangling `,` or `:` at EOF.
    let mut last_sig: Option<u8> = None;
    // Track truncation position for dangling construct removal.
    let mut i = 0usize;
    while i < bytes.len() {
        let b = bytes[i];
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
                last_sig = Some(b'"');
            }
            i += 1;
            continue;
        }
        match b {
            b'"' => {
                in_string = true;
                last_sig = Some(b'"');
            }
            b'{' => {
                stack.push(b'o');
                last_sig = Some(b'{');
            }
            b'[' => {
                stack.push(b'a');
                last_sig = Some(b'[');
            }
            b'}' | b']' => {
                stack.pop();
                last_sig = Some(b);
            }
            b' ' | b'\t' | b'\n' | b'\r' => {}
            _ => {
                last_sig = Some(b);
            }
        }
        i += 1;
    }

    // Start from the full input; chop dangling constructs.
    let mut out = s.to_string();

    // If currently in string, close it first.
    // Handle orphan trailing backslash inside an incomplete escape:
    // `"\\` at end of input — strip the lone `\` before adding `"` so we
    // don't end up with `"\""` (which means literal `"`), which changes
    // the semantic of what the user was typing. Simplest: strip trailing
    // `\` if it would form a lone escape.
    if in_string {
        if out.ends_with('\\') && !out.ends_with("\\\\") {
            out.pop();
        }
        out.push('"');
    }

    // Drop dangling `,` or `:` that would fail the parse.
    // Whitespace + `,`/`:` at the end → trim whitespace first, then
    // one trailing comma/colon.
    while out
        .chars()
        .last()
        .map(|c| c.is_whitespace())
        .unwrap_or(false)
    {
        out.pop();
    }
    if out.ends_with(',') || out.ends_with(':') {
        out.pop();
    }
    // After dropping `:`, there may be a dangling key — a string right
    // before the colon without a value. `{"a":` → repaired to `{"a"`.
    // That fails to parse. Strip it back to the last `,` or `{`.
    //
    // IMPORTANT: only fire when the innermost open container is an
    // object. Inside an array, a comma-preceded string is a VALUE, not
    // a dangling key — stripping it deletes real data.
    let innermost = stack.last().copied();
    if innermost == Some(b'o') {
        out = strip_dangling_key_if_any(&out);
    }

    // Suppress final placeholder value: if the buffer ends with `[` or `{`
    // that's fine — empty container. If it ends with `:`, we already
    // stripped. If it ends with `,`, stripped.
    let _ = last_sig;

    // Close any still-open containers.
    while let Some(top) = stack.pop() {
        out.push(if top == b'o' { '}' } else { ']' });
    }

    out
}

fn strip_dangling_key_if_any(s: &str) -> String {
    let bytes = s.as_bytes();
    if bytes.is_empty() {
        return s.to_string();
    }
    // Must end with a closing `"`.
    if *bytes.last().unwrap() != b'"' {
        return s.to_string();
    }
    // Walk backward to the matching open quote.
    let i = bytes.len() - 1; // points at closing "
    if i == 0 {
        return s.to_string();
    }
    let mut j = i;
    j -= 1;
    while j > 0 {
        if bytes[j] == b'"' && bytes[j - 1] != b'\\' {
            break;
        }
        j -= 1;
    }
    if bytes[j] != b'"' {
        return s.to_string();
    }
    // j = open quote. Look at char before j (skipping whitespace).
    let mut k = j;
    while k > 0 {
        k -= 1;
        if !(bytes[k] as char).is_whitespace() {
            break;
        }
    }
    // If k precedes j with `{` or `,`, it's a dangling key.
    let prev = bytes[k];
    if prev == b'{' || prev == b',' {
        // Strip from position k+1 to end, then trim any trailing `,` left
        // on the char before k (since dropping the first key in `{`
        // leaves `{` — fine. Dropping `"foo","` leaves `{"other":"x",`
        // — we'd then need to drop the trailing comma too).
        let cutoff = if prev == b',' { k } else { k + 1 };
        let mut out = s[..cutoff].to_string();
        // Trim whitespace that might be at the end now.
        while out
            .chars()
            .last()
            .map(|c| c.is_whitespace())
            .unwrap_or(false)
        {
            out.pop();
        }
        return out;
    }
    // Not dangling (preceded by `:` or similar — it's a value, not a key).
    // `i` variable unused in this branch.
    let _ = i;
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn complete_object_parses_identity() {
        let v = parse_partial_json(r#"{"a": 1, "b": "hi"}"#).unwrap();
        assert_eq!(v, json!({"a": 1, "b": "hi"}));
    }

    #[test]
    fn unclosed_object_auto_closes() {
        let v = parse_partial_json(r#"{"a": 1, "b": "hi""#).unwrap();
        assert_eq!(v, json!({"a": 1, "b": "hi"}));
    }

    #[test]
    fn unclosed_string_value_auto_closes() {
        // `"hi` → `"hi"` → completes.
        let v = parse_partial_json(r#"{"a": "hi"#).unwrap();
        assert_eq!(v, json!({"a": "hi"}));
    }

    #[test]
    fn unclosed_nested_containers_auto_close_in_reverse_order() {
        let v = parse_partial_json(r#"{"outer": {"inner": [1, 2, 3"#).unwrap();
        assert_eq!(v, json!({"outer": {"inner": [1, 2, 3]}}));
    }

    #[test]
    fn dangling_colon_strips_the_key() {
        // `{"a":` — incomplete key-value. Parser should strip `"a":` and
        // parse as empty object `{}`.
        let v = parse_partial_json(r#"{"a":"#).unwrap();
        assert_eq!(v, json!({}));
    }

    #[test]
    fn dangling_comma_is_dropped() {
        let v = parse_partial_json(r#"{"a": 1,"#).unwrap();
        assert_eq!(v, json!({"a": 1}));
    }

    #[test]
    fn dangling_key_without_colon_is_stripped() {
        // `{"ke` → unclosed string → close to `"ke"` → then dangling
        // key detected (preceded by `{`) → strip → `{}`.
        let v = parse_partial_json(r#"{"ke"#).unwrap();
        assert_eq!(v, json!({}));
    }

    #[test]
    fn arrays_auto_close_too() {
        let v = parse_partial_json("[1, 2, 3").unwrap();
        assert_eq!(v, json!([1, 2, 3]));
    }

    #[test]
    fn array_with_partial_string_element_completes() {
        let v = parse_partial_json(r#"["hi", "wor"#).unwrap();
        assert_eq!(v, json!(["hi", "wor"]));
    }

    #[test]
    fn empty_input_returns_none() {
        assert!(parse_partial_json("").is_none());
    }

    #[test]
    fn whitespace_only_input_returns_none() {
        assert!(parse_partial_json("   \n\t  ").is_none());
    }

    #[test]
    fn no_root_container_returns_none() {
        // Plain prose with no `{` or `[` → we don't even try.
        assert!(parse_partial_json("just words").is_none());
    }

    #[test]
    fn leading_prose_before_root_is_skipped() {
        // LLM-style "Sure! Here's the JSON: {..."
        let v = parse_partial_json(r#"Here you go: {"a": 1"#).unwrap();
        assert_eq!(v, json!({"a": 1}));
    }

    #[test]
    fn escape_sequences_preserved() {
        let v = parse_partial_json(r#"{"msg": "he said \"hi\"""#).unwrap();
        assert_eq!(v, json!({"msg": "he said \"hi\""}));
    }

    #[test]
    fn trailing_backslash_in_partial_string_stripped() {
        // `"\` at end → don't emit `"\"` (parses as literal `"`) → strip
        // the `\` first, then close the string.
        let v = parse_partial_json(r#"{"a": "hello\"#).unwrap();
        assert_eq!(v, json!({"a": "hello"}));
    }

    #[test]
    fn monotonic_growth_invariant_holds() {
        // As the buffer grows, the parse result keeps accumulating keys.
        let progressions = vec![
            r#"{"name": "Bob","#,
            r#"{"name": "Bob", "age":"#,
            r#"{"name": "Bob", "age": 3"#,
            r#"{"name": "Bob", "age": 30"#,
            r#"{"name": "Bob", "age": 30, "city": "NYC"}"#,
        ];
        let mut prior_keys: Vec<String> = Vec::new();
        for p in &progressions {
            let v = parse_partial_json(p).unwrap();
            let keys: Vec<String> = v
                .as_object()
                .unwrap()
                .keys()
                .cloned()
                .collect();
            // Every prior key survives.
            for pk in &prior_keys {
                assert!(
                    keys.contains(pk),
                    "snapshot {p:?} dropped prior key {pk:?}"
                );
            }
            prior_keys = keys;
        }
    }

    #[test]
    fn repair_exposes_repaired_text() {
        let r = repair_partial_json(r#"{"a": "hi"#).unwrap();
        assert_eq!(r, r#"{"a": "hi"}"#);
    }

    #[test]
    fn deeply_nested_four_levels_closes_correctly() {
        let v = parse_partial_json(r#"{"a": {"b": {"c": [1, {"d": 42"#).unwrap();
        assert_eq!(v, json!({"a": {"b": {"c": [1, {"d": 42}]}}}));
    }

    #[test]
    fn brackets_inside_string_dont_affect_stack() {
        let v = parse_partial_json(r#"{"x": "a}[{b"#).unwrap();
        assert_eq!(v, json!({"x": "a}[{b"}));
    }

    #[test]
    fn mid_number_returns_number_when_parseable() {
        // `"n": 42` is terminated by `}`. Partial `"n": 4` auto-closes
        // to `{"n": 4}` — a valid int 4 (we accept the risk of showing
        // 4 briefly before 42 arrives; caller sees monotonic refinement
        // on strings but NOT numbers).
        let v = parse_partial_json(r#"{"n": 4"#).unwrap();
        assert_eq!(v, json!({"n": 4}));
    }
}

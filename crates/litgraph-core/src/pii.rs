//! PII (Personally Identifiable Information) scrubber for logs, traces,
//! and LLM inputs. Detects common sensitive patterns + replaces with
//! typed tokens (`<EMAIL>`, `<PHONE>`, etc).
//!
//! # Why
//!
//! - **GDPR / CCPA** — don't send real user emails to third-party LLMs
//!   or log sinks unless you've got the lawful basis sorted.
//! - **SOC 2** — secrets in logs is an audit finding every time.
//! - **Prompt injection hygiene** — redacting AWS keys / JWTs before a
//!   prompt goes to an LLM reduces the blast radius of a rogue model
//!   response.
//!
//! # What it detects
//!
//! - **Email**: RFC-5322-ish simple form (`[local]@[domain]`)
//! - **Phone**: US-shaped + international `+CC` variants (not exhaustive)
//! - **SSN**: `NNN-NN-NNNN`
//! - **Credit card**: 13–19 digit sequences that pass the Luhn check
//!   (prevents flagging random numeric IDs)
//! - **AWS access key**: `AKIA[0-9A-Z]{16}` / `ASIA[0-9A-Z]{16}` / etc
//! - **JWT**: three base64url segments separated by `.`
//! - **IPv4 / IPv6**: dotted-quad + colon-hex forms
//!
//! # What it WON'T catch
//!
//! Names, physical addresses, free-form sensitive text. For those you
//! need a full NER/NLP model — out of scope here. This is the high-
//! recall, low-false-positive "strings that LOOK like secrets" layer.
//!
//! # Custom patterns
//!
//! `PiiScrubber::with_patterns(extra)` takes a `Vec<(label, regex)>` so
//! ops teams can add internal ID formats (customer IDs, tenant IDs,
//! etc) to the scrubber without forking the crate.

use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// One detection + its replacement.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Replacement {
    /// Kind label — e.g. `"EMAIL"`, `"PHONE"`. Uppercase for the
    /// token form `<EMAIL>`.
    pub kind: String,
    /// The original string that was matched.
    pub original: String,
    /// The token that replaced it in the scrubbed text.
    pub redacted: String,
    /// Byte offset of the match in the ORIGINAL (pre-scrub) text.
    pub start: usize,
}

/// Scrubber output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScrubResult {
    pub scrubbed: String,
    pub replacements: Vec<Replacement>,
}

/// Default detector patterns. Ordering matters: more-specific patterns
/// come first so a string that matches multiple kinds (e.g., a JWT that
/// contains a URL-safe segment resembling an email) gets tagged by the
/// more specific one.
static DEFAULT_PATTERNS: Lazy<Vec<(&'static str, Regex)>> = Lazy::new(|| {
    vec![
        // AWS access keys — very specific prefix, check first.
        (
            "AWS_ACCESS_KEY",
            Regex::new(r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASCA)[0-9A-Z]{16}\b").unwrap(),
        ),
        // JWT: three base64url segments separated by dots. Header +
        // payload + signature. Require min 10 chars per segment to cut
        // false positives on short domain-looking strings.
        (
            "JWT",
            Regex::new(r"\beyJ[0-9A-Za-z_\-]{10,}\.[0-9A-Za-z_\-]{10,}\.[0-9A-Za-z_\-]{10,}\b")
                .unwrap(),
        ),
        // Email — simple RFC-5322-ish. Intentionally narrow TLD to
        // reduce false positives on file paths with `@` (e.g. npm scopes).
        (
            "EMAIL",
            Regex::new(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b").unwrap(),
        ),
        // SSN: NNN-NN-NNNN. Require dashes to reduce false positives on
        // 9-digit numbers that are actually IDs.
        ("SSN", Regex::new(r"\b\d{3}-\d{2}-\d{4}\b").unwrap()),
        // Credit card: 13–19 digits, optionally separated by dashes or
        // spaces. MUST come before PHONE — a 16-digit card looks like
        // a phone number to the phone regex (it matches its first 10
        // digits greedily) but is authoritatively a card. Luhn-validated
        // downstream to cut false positives on random long numeric IDs.
        (
            "CREDIT_CARD",
            Regex::new(r"\b(?:\d[ \-]?){12,18}\d\b").unwrap(),
        ),
        // Phone: US + international. Accepts `(123) 456-7890`,
        // `+1-234-567-8900`, `+44 20 7946 0958`, etc. 7–15 digits total
        // after stripping separators (E.164 range).
        (
            "PHONE",
            Regex::new(
                r"(?x)
                \b
                (?: \+ \d{1,3} [\s\-]? )?      # optional country code
                \(? \d{2,4} \)? [\s\-]?        # area
                \d{3,4} [\s\-]? \d{4}          # local
                \b",
            )
            .unwrap(),
        ),
        // IPv4 dotted quad.
        (
            "IPV4",
            Regex::new(r"\b(?:\d{1,3}\.){3}\d{1,3}\b").unwrap(),
        ),
        // IPv6 (simplified — full spec is nasty; this catches canonical
        // forms + `::`-compressed).
        (
            "IPV6",
            Regex::new(r"\b(?:[0-9a-fA-F]{1,4}:){2,7}[0-9a-fA-F]{0,4}\b").unwrap(),
        ),
    ]
});

#[derive(Clone)]
pub struct PiiScrubber {
    patterns: Vec<(String, Regex)>,
    /// Apply Luhn check on CREDIT_CARD matches — true by default. Turn
    /// off for internal test IDs that happen to be 16 digits.
    pub validate_luhn: bool,
}

impl Default for PiiScrubber {
    fn default() -> Self {
        Self::new()
    }
}

impl PiiScrubber {
    /// Scrubber with the default pattern set.
    pub fn new() -> Self {
        let patterns: Vec<(String, Regex)> = DEFAULT_PATTERNS
            .iter()
            .map(|pair: &(&'static str, Regex)| (pair.0.to_string(), pair.1.clone()))
            .collect();
        Self {
            patterns,
            validate_luhn: true,
        }
    }

    /// Append additional patterns. Each `(label, regex)` tuple adds a
    /// kind that renders as `<LABEL>` in the scrubbed output.
    pub fn with_patterns<I>(mut self, extras: I) -> Self
    where
        I: IntoIterator<Item = (String, Regex)>,
    {
        for (label, re) in extras {
            self.patterns.push((label, re));
        }
        self
    }

    /// Replace an empty pattern set (drop defaults). Use to build a
    /// scrubber that ONLY matches operator-provided custom patterns.
    pub fn only_custom(mut self) -> Self {
        self.patterns.clear();
        self
    }

    pub fn without_luhn(mut self) -> Self {
        self.validate_luhn = false;
        self
    }

    /// Scrub `text`. Returns the redacted string plus a list of
    /// `Replacement`s for audit logging.
    pub fn scrub(&self, text: &str) -> ScrubResult {
        // Multi-pattern overlap resolution: walk each pattern, collect
        // its match spans, then greedily accept non-overlapping matches
        // in pattern-priority order (first-declared wins on conflict).
        #[derive(Debug, Clone)]
        struct Match {
            start: usize,
            end: usize,
            kind: String,
            captured: String,
        }

        let mut accepted: Vec<Match> = Vec::new();
        for (kind, re) in &self.patterns {
            for cap in re.find_iter(text) {
                let m = Match {
                    start: cap.start(),
                    end: cap.end(),
                    kind: kind.clone(),
                    captured: cap.as_str().to_string(),
                };

                // Luhn check for credit cards.
                if kind == "CREDIT_CARD" && self.validate_luhn {
                    let digits: String = m.captured.chars().filter(|c| c.is_ascii_digit()).collect();
                    if !luhn_valid(&digits) {
                        continue;
                    }
                }

                // Reject if it overlaps an earlier (higher-priority) accepted span.
                if accepted
                    .iter()
                    .any(|a| !(m.end <= a.start || m.start >= a.end))
                {
                    continue;
                }
                accepted.push(m);
            }
        }
        // Sort accepted matches by start offset for linear rebuild.
        accepted.sort_by_key(|m| m.start);

        // Build the scrubbed string in one pass.
        let mut out = String::with_capacity(text.len());
        let mut cursor = 0;
        let mut replacements = Vec::with_capacity(accepted.len());
        for m in &accepted {
            if m.start >= cursor {
                out.push_str(&text[cursor..m.start]);
                let token = format!("<{}>", m.kind);
                out.push_str(&token);
                replacements.push(Replacement {
                    kind: m.kind.clone(),
                    original: m.captured.clone(),
                    redacted: token,
                    start: m.start,
                });
                cursor = m.end;
            }
        }
        out.push_str(&text[cursor..]);

        ScrubResult {
            scrubbed: out,
            replacements,
        }
    }
}

/// Standard Luhn check (mod-10). Used to avoid flagging random 16-digit
/// IDs as credit cards.
pub fn luhn_valid(digits: &str) -> bool {
    let ds: Vec<u32> = digits
        .chars()
        .filter_map(|c| c.to_digit(10))
        .collect();
    if ds.len() < 13 || ds.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    let mut alt = false;
    for &d in ds.iter().rev() {
        let mut v = d;
        if alt {
            v *= 2;
            if v > 9 {
                v -= 9;
            }
        }
        sum += v;
        alt = !alt;
    }
    sum % 10 == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_redacted() {
        let s = PiiScrubber::new();
        let r = s.scrub("Contact me at alice@example.com for details.");
        assert!(r.scrubbed.contains("<EMAIL>"));
        assert!(!r.scrubbed.contains("alice"));
        assert_eq!(r.replacements.len(), 1);
        assert_eq!(r.replacements[0].kind, "EMAIL");
        assert_eq!(r.replacements[0].original, "alice@example.com");
    }

    #[test]
    fn ssn_redacted() {
        let s = PiiScrubber::new();
        let r = s.scrub("My SSN is 123-45-6789.");
        assert!(r.scrubbed.contains("<SSN>"));
        assert!(!r.scrubbed.contains("123-45-6789"));
    }

    #[test]
    fn aws_access_key_redacted() {
        let s = PiiScrubber::new();
        let r = s.scrub("key=AKIAIOSFODNN7EXAMPLE in config");
        assert!(r.scrubbed.contains("<AWS_ACCESS_KEY>"));
    }

    #[test]
    fn jwt_redacted() {
        let s = PiiScrubber::new();
        // A plausible-looking JWT. 3 segments, all min-10 chars.
        let jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c";
        let r = s.scrub(&format!("Authorization: Bearer {jwt}"));
        assert!(r.scrubbed.contains("<JWT>"));
    }

    #[test]
    fn credit_card_must_pass_luhn() {
        let s = PiiScrubber::new();
        // Valid Visa test number (passes Luhn).
        let r = s.scrub("CC: 4111 1111 1111 1111");
        assert!(r.scrubbed.contains("<CREDIT_CARD>"));

        // 16 digits that FAIL Luhn (e.g. all-ones) should not be redacted.
        let r2 = s.scrub("Customer ID: 1234 5678 9012 3456");
        assert!(!r2.scrubbed.contains("<CREDIT_CARD>"));
    }

    #[test]
    fn luhn_can_be_disabled() {
        let s = PiiScrubber::new().without_luhn();
        // Any 16-digit sequence now flags as credit card.
        let r = s.scrub("1234 5678 9012 3456");
        assert!(r.scrubbed.contains("<CREDIT_CARD>"));
    }

    #[test]
    fn ipv4_redacted() {
        let s = PiiScrubber::new();
        let r = s.scrub("Client IP: 192.168.1.1 connected");
        assert!(r.scrubbed.contains("<IPV4>"));
    }

    #[test]
    fn phone_number_redacted() {
        let s = PiiScrubber::new();
        let r = s.scrub("Call +1-555-867-5309 anytime.");
        assert!(r.scrubbed.contains("<PHONE>"));
    }

    #[test]
    fn multiple_pii_in_one_string() {
        let s = PiiScrubber::new();
        let r = s.scrub("Contact alice@example.com or call 555-867-5309 (AKIAIOSFODNN7EXAMPLE)");
        // EMAIL + PHONE + AWS_ACCESS_KEY all redacted.
        let kinds: std::collections::HashSet<String> = r
            .replacements
            .iter()
            .map(|rep| rep.kind.clone())
            .collect();
        assert!(kinds.contains("EMAIL"));
        assert!(kinds.contains("PHONE"));
        assert!(kinds.contains("AWS_ACCESS_KEY"));
    }

    #[test]
    fn overlapping_matches_higher_priority_wins() {
        // AWS key pattern is declared before any generic numeric pattern;
        // an AKIA key that partially overlaps with something should keep
        // its AWS_ACCESS_KEY label.
        let s = PiiScrubber::new();
        let r = s.scrub("AKIAIOSFODNN7EXAMPLE");
        assert_eq!(r.replacements.len(), 1);
        assert_eq!(r.replacements[0].kind, "AWS_ACCESS_KEY");
    }

    #[test]
    fn no_match_returns_original_text_unchanged() {
        let s = PiiScrubber::new();
        let r = s.scrub("Just some regular prose, nothing to hide.");
        assert_eq!(r.scrubbed, "Just some regular prose, nothing to hide.");
        assert!(r.replacements.is_empty());
    }

    #[test]
    fn replacement_carries_original_offset_for_audit() {
        let s = PiiScrubber::new();
        let text = "Email: alice@example.com is done";
        let r = s.scrub(text);
        assert_eq!(r.replacements[0].start, 7);
        // Caller can recover: &text[start..start+original.len()] == original
        let rep = &r.replacements[0];
        assert_eq!(
            &text[rep.start..rep.start + rep.original.len()],
            &rep.original
        );
    }

    #[test]
    fn custom_patterns_append_to_defaults() {
        let s = PiiScrubber::new().with_patterns(vec![(
            "ORDER_ID".to_string(),
            Regex::new(r"\bORD-\d{6}\b").unwrap(),
        )]);
        let r = s.scrub("Order ORD-123456 from alice@example.com");
        assert!(r.scrubbed.contains("<ORDER_ID>"));
        assert!(r.scrubbed.contains("<EMAIL>"));
    }

    #[test]
    fn only_custom_drops_defaults() {
        let s = PiiScrubber::new()
            .only_custom()
            .with_patterns(vec![(
                "INTERNAL_ID".to_string(),
                Regex::new(r"\bID-\d+\b").unwrap(),
            )]);
        let r = s.scrub("alice@example.com contacted about ID-99");
        // Email NOT scrubbed because we dropped defaults.
        assert!(r.scrubbed.contains("alice@example.com"));
        assert!(r.scrubbed.contains("<INTERNAL_ID>"));
    }

    #[test]
    fn luhn_validates_known_valid_cards() {
        assert!(luhn_valid("4111111111111111")); // Visa test
        assert!(luhn_valid("5500000000000004")); // MC test
        assert!(luhn_valid("340000000000009")); // Amex test (15 digits)
        assert!(!luhn_valid("4111111111111112")); // off by one
    }

    #[test]
    fn luhn_rejects_too_short_or_too_long() {
        assert!(!luhn_valid("411")); // too short
        assert!(!luhn_valid(&"4".repeat(20))); // too long
    }
}

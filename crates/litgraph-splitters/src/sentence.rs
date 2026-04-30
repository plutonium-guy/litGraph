//! `SentenceSplitter` — split text on sentence boundaries.
//!
//! # Why a separate splitter
//!
//! `RecursiveCharacterSplitter` chops by character count. For
//! some prod use cases the unit of meaning is the sentence:
//!
//! - **Sentence-level RAG**: retrieve the most relevant sentence
//!   rather than a 1000-char chunk.
//! - **Fine-grained QA citations**: answer cites a specific
//!   sentence by index.
//! - **Translation pipelines**: translate sentence-by-sentence
//!   to keep semantic units intact.
//! - **Summarization**: reduce a doc to its top-N sentences.
//!
//! # Algorithm
//!
//! Rule-based, no NLP-model dependency. Handles the common false
//! positives that defeat naive `text.split(".")` approaches:
//!
//! - **Abbreviations**: Dr., Mr., Mrs., Ms., Prof., e.g., i.e.,
//!   etc., vs., Inc., Ltd., No., etc. Don't split after these.
//! - **Acronyms**: `U.S.A.`, `U.K.`, `e.g.`, `i.e.`. Multi-letter
//!   patterns with internal periods are NOT sentence terminators.
//! - **Decimal numbers**: `3.14`, `1,234.56`. Period between
//!   digits is NOT a terminator.
//! - **Ellipses** (`...`): treated as a single terminator (no
//!   spurious empty splits).
//! - **Trailing quote/paren after terminator**: `"Hello."` or
//!   `(see above.)` — terminator is at the period, splitter
//!   includes the closing punctuation in the prior sentence.
//!
//! Coverage is "good enough for ~95% of clean prose." For
//! adversarial input or specialized corpora (clinical notes,
//! legal docs), use a Punkt-style ML splitter via a separate
//! crate.

use std::collections::HashSet;

use once_cell::sync::Lazy;

use crate::Splitter;

/// Common English abbreviations whose trailing `.` should NOT
/// terminate a sentence. Lowercased for case-insensitive matching.
static ABBREVIATIONS: Lazy<HashSet<&'static str>> = Lazy::new(|| {
    [
        // Titles
        "mr", "mrs", "ms", "dr", "prof", "rev", "fr", "sr", "jr",
        "st", "hon", "gen", "col", "lt", "cmdr", "capt", "sgt",
        // Latin abbreviations
        "e.g", "i.e", "etc", "vs", "viz", "cf", "et", "al",
        // Business / common
        "inc", "ltd", "co", "corp", "llc", "plc",
        // Months (abbrev)
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep",
        "sept", "oct", "nov", "dec",
        // Days
        "mon", "tue", "wed", "thu", "fri", "sat", "sun",
        // Geography
        "u.s", "u.s.a", "u.k", "e.u", "n.y", "l.a", "d.c",
        // Other
        "no", "vol", "fig", "ed", "eds", "p", "pp", "ch",
        "approx", "est",
    ]
    .iter()
    .copied()
    .collect()
});

#[derive(Debug, Clone)]
pub struct SentenceSplitter {
    /// Minimum chars in a sentence to count as one. Shorter
    /// "sentences" (just punctuation, fragments) get glued onto
    /// the previous one. Default 1 (no minimum).
    pub min_length: usize,
}

impl Default for SentenceSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceSplitter {
    pub fn new() -> Self {
        Self { min_length: 1 }
    }

    pub fn with_min_length(mut self, n: usize) -> Self {
        self.min_length = n.max(1);
        self
    }
}

impl Splitter for SentenceSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let chars: Vec<char> = text.chars().collect();
        let mut sentences: Vec<String> = Vec::new();
        let mut buf = String::new();
        let mut i = 0;
        while i < chars.len() {
            let c = chars[i];
            buf.push(c);
            if matches!(c, '.' | '!' | '?') {
                // Collapse runs of `.` (ellipsis) into one terminator.
                while i + 1 < chars.len() && chars[i + 1] == c {
                    i += 1;
                    buf.push(chars[i]);
                }
                // Optional trailing close-quote or paren.
                while i + 1 < chars.len()
                    && matches!(chars[i + 1], '"' | '\'' | ')' | ']' | '”' | '’')
                {
                    i += 1;
                    buf.push(chars[i]);
                }

                // Look ahead: must be whitespace then uppercase /
                // digit / opening-quote to count as a boundary.
                let look = i + 1;
                let has_ws = look < chars.len() && chars[look].is_whitespace();
                if !has_ws {
                    i += 1;
                    continue;
                }
                // Skip whitespace to peek the next non-ws char.
                let mut j = look;
                while j < chars.len() && chars[j].is_whitespace() {
                    j += 1;
                }
                if j >= chars.len() {
                    // Trailing whitespace + EOF → end of last sentence.
                    flush(&mut buf, &mut sentences, self.min_length);
                    i += 1;
                    continue;
                }
                let next_starts_sentence = chars[j].is_uppercase()
                    || chars[j].is_ascii_digit()
                    || matches!(chars[j], '"' | '\'' | '(' | '“' | '‘');
                if !next_starts_sentence {
                    // Followed by lowercase (likely abbrev or run-on) — don't split.
                    i += 1;
                    continue;
                }

                // False-positive checks before committing the boundary.
                if c == '.' {
                    // Decimal digit context: digit . digit (e.g. 3.14).
                    if i >= 1
                        && i + 1 < chars.len()
                        && chars[i - 1].is_ascii_digit()
                        && chars[look].is_ascii_digit()
                    {
                        // Not a sentence boundary — but we've already
                        // consumed whitespace check; the look check above
                        // requires whitespace, so this case shouldn't
                        // hit. Belt-and-suspenders.
                    }
                    // Abbreviation context: word ending right before `.`
                    // matches a known abbreviation (case-insensitive).
                    if is_abbreviation_before(&chars, i) {
                        i += 1;
                        continue;
                    }
                }

                flush(&mut buf, &mut sentences, self.min_length);
            }
            i += 1;
        }
        if !buf.trim().is_empty() {
            sentences.push(buf.trim().to_string());
        }
        sentences
    }
}

/// Push `buf` (trimmed) as a new sentence, OR if it's shorter
/// than `min_length`, glue it onto the previous sentence so the
/// caller doesn't get fragments like ".)" or "1." as separate.
fn flush(buf: &mut String, sentences: &mut Vec<String>, min_length: usize) {
    let trimmed = buf.trim().to_string();
    buf.clear();
    if trimmed.is_empty() {
        return;
    }
    if trimmed.chars().count() < min_length {
        if let Some(last) = sentences.last_mut() {
            last.push(' ');
            last.push_str(&trimmed);
            return;
        }
    }
    sentences.push(trimmed);
}

/// Look at the characters immediately before position `dot_pos`
/// and return true if the token (alphanumeric run, possibly with
/// internal `.`s for acronym-style abbrevs) matches a known
/// abbreviation.
fn is_abbreviation_before(chars: &[char], dot_pos: usize) -> bool {
    if dot_pos == 0 {
        return false;
    }
    // Walk backward over alphanum + `.` to capture forms like
    // `U.S.A` (where the final `.` is at dot_pos).
    let mut start = dot_pos;
    while start > 0 {
        let prev = chars[start - 1];
        if prev.is_alphanumeric() || prev == '.' {
            start -= 1;
        } else {
            break;
        }
    }
    let token: String = chars[start..dot_pos].iter().collect();
    let token_lower = token.to_lowercase();
    ABBREVIATIONS.contains(token_lower.as_str())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_simple_sentences() {
        let s = SentenceSplitter::new();
        let out = s.split_text("Hello world. This is a test. Goodbye.");
        assert_eq!(
            out,
            vec!["Hello world.", "This is a test.", "Goodbye."]
        );
    }

    #[test]
    fn handles_question_and_exclamation() {
        let s = SentenceSplitter::new();
        let out = s.split_text("Are you sure? Yes! I am ready.");
        assert_eq!(out.len(), 3);
        assert!(out[0].ends_with('?'));
        assert!(out[1].ends_with('!'));
        assert!(out[2].ends_with('.'));
    }

    #[test]
    fn does_not_split_after_titles() {
        let s = SentenceSplitter::new();
        let out = s.split_text("Dr. Smith and Mr. Jones met. They talked.");
        // 2 sentences, not 4.
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("Dr. Smith"));
        assert!(out[0].contains("Mr. Jones"));
    }

    #[test]
    fn does_not_split_e_g_i_e() {
        let s = SentenceSplitter::new();
        let out = s.split_text("Many fruits, e.g. apples and oranges, are tasty. End.");
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("e.g."));
    }

    #[test]
    fn does_not_split_us_acronym() {
        let s = SentenceSplitter::new();
        let out = s.split_text("She lives in the U.S. She likes it.");
        // The space + capital after "U.S." LOOKS like a boundary, but the
        // splitter recognizes "u.s" as an abbreviation token.
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn does_not_split_decimals() {
        let s = SentenceSplitter::new();
        // "3.14 is" — the `.` is between digits → not a boundary.
        let out = s.split_text("Pi is 3.14 approximately. End.");
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn ellipsis_treated_as_one_terminator() {
        let s = SentenceSplitter::new();
        let out = s.split_text("Wait... What was that? Hmm.");
        // 3 sentences: "Wait...", "What was that?", "Hmm."
        assert_eq!(out.len(), 3);
        assert!(out[0].ends_with("..."));
    }

    #[test]
    fn closing_quote_after_period_attached_to_prior() {
        let s = SentenceSplitter::new();
        let out = s.split_text(r#""Hello." She smiled. "Goodbye.""#);
        assert_eq!(out.len(), 3);
        assert!(out[0].contains(r#"Hello.""#));
    }

    #[test]
    fn preserves_no_leading_or_trailing_whitespace() {
        let s = SentenceSplitter::new();
        let out = s.split_text("  Sentence one.   Sentence two.  ");
        assert_eq!(out, vec!["Sentence one.", "Sentence two."]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = SentenceSplitter::new();
        assert!(s.split_text("").is_empty());
        assert!(s.split_text("   \n\t  ").is_empty());
    }

    #[test]
    fn handles_unterminated_final_sentence() {
        let s = SentenceSplitter::new();
        let out = s.split_text("First sentence. Second sentence without period");
        assert_eq!(out.len(), 2);
        assert_eq!(out[1], "Second sentence without period");
    }

    #[test]
    fn min_length_glues_short_fragments_to_prior() {
        let s = SentenceSplitter::new().with_min_length(5);
        // "Hi." is 3 chars; glued onto previous if shorter than min.
        let out = s.split_text("This is one. Hi. That was a fragment.");
        // After gluing: "This is one. Hi.", "That was a fragment."
        assert_eq!(out.len(), 2);
        assert!(out[0].contains("Hi."));
    }
}

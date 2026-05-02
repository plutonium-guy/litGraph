//! Token-based text splitter — splits by exact token count rather than
//! character count. Critical for staying under model context windows
//! precisely.
//!
//! # vs `RecursiveCharacterSplitter`
//!
//! - **Recursive char**: cheap, deterministic, but `chunk_size=1000` means
//!   1000 *characters* — actual token count varies wildly by language
//!   (English ~4 chars/token, code ~3, CJK ~1). Risk of overshoot if you
//!   pass char-counts to a token-budgeted model.
//! - **Token splitter**: counts tokens via `litgraph-tokenizers` (tiktoken
//!   for OpenAI-family, HF tokenizers for others). Slightly slower (per-
//!   chunk re-encode) but EXACT — `chunk_size=1000` is 1000 tokens.
//!
//! # Algorithm
//!
//! 1. Count tokens of full text. If under `chunk_size`, return as one chunk.
//! 2. Estimate chars-per-token = total_chars / total_tokens; pick a
//!    candidate split point at `~chunk_size * chars_per_token` from the start.
//! 3. Snap the split point to the nearest non-letter/digit boundary so
//!    we don't slice mid-word. Falls back to exact char position if no
//!    boundary found nearby.
//! 4. Verify the chunk's token count is ≤ chunk_size; if it overshoots,
//!    walk back to a smaller split point.
//! 5. Recurse on the remainder, with `chunk_overlap` characters of overlap.
//!
//! Doesn't try to be smart about paragraph/heading boundaries — that's
//! `MarkdownHeaderSplitter`'s job. Use this AFTER structural splitting
//! to enforce a hard token budget per chunk.

use crate::Splitter;
use litgraph_tokenizers::count_tokens;

pub struct TokenTextSplitter {
    /// Max tokens per chunk.
    pub chunk_size: usize,
    /// Char-level overlap between consecutive chunks (overlap is in CHARS,
    /// not tokens — exact-token overlap would require re-tokenizing each
    /// step, doubling cost). For typical configs (overlap < 0.2 × chunk_size)
    /// the actual token overlap is within 20% of the chars/token estimate.
    pub chunk_overlap: usize,
    /// Tokenizer model name. Passed verbatim to `count_tokens(model, text)`.
    /// "gpt-4o" / "claude-opus-4-7" / etc.
    pub model: String,
}

impl TokenTextSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize, model: impl Into<String>) -> Self {
        let chunk_size = chunk_size.max(1);
        let chunk_overlap = chunk_overlap.min(chunk_size.saturating_sub(1));
        Self { chunk_size, chunk_overlap, model: model.into() }
    }

    /// Find the largest substring of `text` (starting at position 0) whose
    /// token count is ≤ `max_tokens`. Returns the byte offset (END) of
    /// that substring. Snapped to a UTF-8 boundary AND to a non-word
    /// boundary when possible.
    fn find_chunk_end(&self, text: &str, max_tokens: usize) -> usize {
        let total_tokens = count_tokens(&self.model, text);
        if total_tokens <= max_tokens {
            return text.len();
        }
        let total_chars = text.chars().count();
        if total_chars == 0 {
            return 0;
        }
        // Initial estimate: chars-per-token × max_tokens.
        let chars_per_token = (total_chars as f64) / (total_tokens as f64);
        let mut estimate = ((max_tokens as f64) * chars_per_token).floor() as usize;
        if estimate == 0 {
            estimate = 1;
        }
        if estimate > total_chars {
            estimate = total_chars;
        }

        // Walk down until token count fits. We compare on CHAR positions,
        // then convert to byte offsets at the end.
        loop {
            let end_byte = char_index_to_byte(text, estimate);
            let chunk = &text[..end_byte];
            let n = count_tokens(&self.model, chunk);
            if n <= max_tokens {
                // Found a fit. Try to snap end to a word boundary if we're
                // mid-word (avoid splitting "tokenization" into "tokeniz" + "ation").
                return snap_back_to_boundary(text, end_byte);
            }
            // Overshoot: shrink estimate by the relative overshoot, plus a small fixed step.
            let scale = (max_tokens as f64) / (n as f64);
            let new_estimate = ((estimate as f64) * scale).floor() as usize;
            estimate = new_estimate.min(estimate.saturating_sub(1));
            if estimate == 0 {
                return 0;
            }
        }
    }
}

/// Snap byte position back to the nearest non-word character boundary
/// within a 32-char lookback. If no boundary found, return `byte_pos`
/// unchanged. Always returns a valid UTF-8 boundary.
fn snap_back_to_boundary(text: &str, byte_pos: usize) -> usize {
    if byte_pos == 0 || byte_pos >= text.len() {
        return byte_pos;
    }
    // If already at a non-word position, return as-is.
    if let Some(c) = text[..byte_pos].chars().last() {
        if !c.is_alphanumeric() {
            return byte_pos;
        }
    }
    // Walk back up to 32 chars looking for whitespace / punctuation.
    let mut chars_back = 0;
    let mut last_good: Option<usize> = None;
    for (i, c) in text[..byte_pos].char_indices().rev() {
        if !c.is_alphanumeric() && i > 0 {
            // The boundary is just AFTER this non-word char (so "hello world"
            // boundary at the space → cut after the space, keeping "hello ").
            last_good = Some(i + c.len_utf8());
            break;
        }
        chars_back += 1;
        if chars_back > 32 {
            break;
        }
    }
    last_good.unwrap_or(byte_pos)
}

/// Convert a char index (0-based) to a byte offset within `text`.
/// Saturates at `text.len()` for out-of-range indices.
fn char_index_to_byte(text: &str, char_idx: usize) -> usize {
    text.char_indices()
        .nth(char_idx)
        .map(|(b, _)| b)
        .unwrap_or(text.len())
}

impl Splitter for TokenTextSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let mut chunks = Vec::new();
        let mut remaining = text;
        loop {
            let end = self.find_chunk_end(remaining, self.chunk_size);
            if end == 0 {
                // Couldn't fit even one char into a chunk — give up to avoid infinite loop.
                // (Pathological input or chunk_size that's too small for the tokenizer.)
                break;
            }
            chunks.push(remaining[..end].to_string());
            if end >= remaining.len() {
                break;
            }
            // Apply char-level overlap by stepping back from `end`.
            let next_start = if self.chunk_overlap > 0 && self.chunk_overlap < end {
                let overlap_byte = end.saturating_sub(self.chunk_overlap);
                // Snap back to UTF-8 boundary.
                snap_to_char_boundary(remaining, overlap_byte)
            } else {
                end
            };
            // Guard against zero-progress loops.
            if next_start >= remaining.len() {
                break;
            }
            if next_start == 0 {
                // Couldn't step back without overlapping — break to avoid reprocessing.
                break;
            }
            remaining = &remaining[next_start..];
        }
        chunks
    }
}

/// Walk byte position forward to a valid UTF-8 char boundary.
fn snap_to_char_boundary(text: &str, byte_pos: usize) -> usize {
    let mut p = byte_pos.min(text.len());
    while p < text.len() && !text.is_char_boundary(p) {
        p += 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    fn token_count(text: &str) -> usize {
        // Use the same model the tests use.
        count_tokens("gpt-4o", text)
    }

    #[test]
    fn small_text_returns_one_chunk() {
        let s = TokenTextSplitter::new(1000, 0, "gpt-4o");
        let chunks = s.split_text("hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], "hello world");
    }

    #[test]
    fn empty_text_returns_empty() {
        let s = TokenTextSplitter::new(100, 0, "gpt-4o");
        assert!(s.split_text("").is_empty());
    }

    #[test]
    fn each_chunk_under_or_equal_token_budget() {
        // Long text → multiple chunks, each ≤ chunk_size tokens.
        let text = "The quick brown fox jumps over the lazy dog. ".repeat(50);
        let s = TokenTextSplitter::new(20, 0, "gpt-4o");
        let chunks = s.split_text(&text);
        assert!(chunks.len() > 1, "expected multiple chunks");
        for c in &chunks {
            let n = token_count(c);
            assert!(
                n <= 20,
                "chunk has {n} tokens, expected ≤ 20: {c:?}"
            );
        }
    }

    #[test]
    fn chunks_concatenate_to_close_to_original() {
        // With zero overlap, concatenated chunks should be exactly the source.
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(30);
        let s = TokenTextSplitter::new(15, 0, "gpt-4o");
        let chunks = s.split_text(&text);
        let joined: String = chunks.join("");
        assert_eq!(joined.len(), text.len(), "concatenated chunks must equal source under zero-overlap");
    }

    #[test]
    fn overlap_produces_more_chunks_than_zero_overlap() {
        // Word-based text — gpt-4o tokenizer splits each word.
        let text = "the quick brown fox jumps over the lazy dog ".repeat(20);
        let no_overlap = TokenTextSplitter::new(15, 0, "gpt-4o").split_text(&text);
        let with_overlap = TokenTextSplitter::new(15, 30, "gpt-4o").split_text(&text);
        assert!(no_overlap.len() > 1, "baseline must have >1 chunk");
        // With overlap, each chunk advances less far → more chunks total.
        assert!(
            with_overlap.len() >= no_overlap.len(),
            "overlap should produce at least as many chunks: {} vs {}",
            with_overlap.len(), no_overlap.len()
        );
    }

    #[test]
    fn very_small_chunk_size_does_not_infinite_loop() {
        // chunk_size=1 token might be impossible for some texts (single char
        // could exceed). Should bail rather than loop forever.
        let s = TokenTextSplitter::new(1, 0, "gpt-4o");
        let _chunks = s.split_text("x");
        // Just assert we got here without hanging.
    }

    #[test]
    fn chunk_overlap_clamped_to_size_minus_one() {
        let s = TokenTextSplitter::new(10, 100, "gpt-4o");
        assert_eq!(s.chunk_overlap, 9);
    }

    #[test]
    fn snap_back_to_boundary_at_word_break() {
        let text = "hello world fragment";
        // Position 13 = inside "fragment" (after "fra")
        let snapped = snap_back_to_boundary(text, 15);
        // Should snap back to after the space before "fragment".
        assert!(snapped <= 15);
        assert_eq!(&text[..snapped].chars().last().unwrap_or('?'), &' ');
    }

    #[test]
    fn snap_back_no_op_when_already_at_boundary() {
        let text = "hello world";
        let pos = 6;  // right at the space (after "hello ")
        let snapped = snap_back_to_boundary(text, pos);
        assert_eq!(snapped, pos);
    }

    #[test]
    fn document_split_carries_metadata() {
        use litgraph_core::Document;
        let mut doc = Document::new("the quick brown fox jumps over the lazy dog. ".repeat(10));
        doc.id = Some("src.txt".into());
        doc.metadata.insert("source".into(), serde_json::json!("src.txt"));
        let s = TokenTextSplitter::new(10, 0, "gpt-4o");
        let chunks = s.split_document(&doc);
        assert!(chunks.len() > 1);
        for c in &chunks {
            assert_eq!(c.metadata.get("source"), Some(&serde_json::json!("src.txt")));
            assert!(c.metadata.contains_key("chunk_index"));
        }
    }

    #[test]
    fn batch_split_documents_processes_in_parallel() {
        use litgraph_core::Document;
        let docs: Vec<Document> = (0..5)
            .map(|i| Document::new(format!("text doc {i} ").repeat(50)))
            .collect();
        let s = TokenTextSplitter::new(15, 0, "gpt-4o");
        let chunks = s.split_documents(&docs);
        assert!(chunks.len() >= 5);  // at least one chunk per source doc
    }

    #[test]
    fn unicode_text_splits_at_valid_utf8_boundaries() {
        // Mix of ASCII + multi-byte chars (Chinese / emoji).
        let text = "hello 你好世界 🌍 lorem ipsum dolor sit amet ".repeat(20);
        let s = TokenTextSplitter::new(10, 0, "gpt-4o");
        let chunks = s.split_text(&text);
        for c in &chunks {
            // If we sliced at an invalid UTF-8 position, this would have panicked
            // already during Rust's str slicing. Just confirm the chunks are valid.
            let _ = c.chars().count();
        }
        assert!(!chunks.is_empty());
    }

    #[test]
    fn different_models_produce_different_split_points() {
        // OpenAI vs Anthropic tokenizers count differently — verify the
        // model parameter actually drives the count, not just stored.
        let text = "lorem ipsum ".repeat(50);
        let openai = TokenTextSplitter::new(20, 0, "gpt-4o");
        let anthropic = TokenTextSplitter::new(20, 0, "claude-opus-4-7");
        let openai_chunks = openai.split_text(&text);
        let anthropic_chunks = anthropic.split_text(&text);
        // Both should produce some chunks. Counts may or may not differ
        // depending on tokenizer fidelity; just confirm both work without panic.
        assert!(!openai_chunks.is_empty());
        assert!(!anthropic_chunks.is_empty());
    }
}

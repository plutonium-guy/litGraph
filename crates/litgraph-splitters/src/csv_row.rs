//! `CsvRowSplitter` — split CSV text into chunks by row count
//! while preserving the header row on every chunk.
//!
//! # The problem
//!
//! Recursive char / token splitters break CSV rows mid-cell —
//! retrieval over chunks like `Alice,2024-01-15,$10` followed
//! by `0,paid,Bob,2024-01-16,$200,unpaid` is meaningless. The
//! commas and column positions only carry meaning relative to
//! the header.
//!
//! # The fix
//!
//! Walk the CSV by line. Treat the first non-empty line as the
//! header (or an empty header if `with_header(false)`). Emit
//! chunks of `rows_per_chunk` data rows, each chunk prefixed
//! with the header.
//!
//! Quoted-field-with-newline handling: a CSV cell wrapped in
//! double quotes can contain literal `\n`. The walker tracks an
//! "in-quote" flag and only ends a row at a newline that lands
//! outside any quoted field. This is sufficient for RFC-4180
//! quoting (the standard CSV format used by Python `csv`,
//! Postgres `COPY`, etc).

use crate::Splitter;

const DEFAULT_ROWS_PER_CHUNK: usize = 100;

pub struct CsvRowSplitter {
    pub rows_per_chunk: usize,
    pub include_header: bool,
}

impl Default for CsvRowSplitter {
    fn default() -> Self {
        Self::new(DEFAULT_ROWS_PER_CHUNK)
    }
}

impl CsvRowSplitter {
    pub fn new(rows_per_chunk: usize) -> Self {
        Self {
            rows_per_chunk: rows_per_chunk.max(1),
            include_header: true,
        }
    }

    /// If `false`, treats the entire input as data rows (no
    /// header is prefixed to each chunk). Defaults to `true`.
    pub fn with_header(mut self, b: bool) -> Self {
        self.include_header = b;
        self
    }
}

impl Splitter for CsvRowSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let rows = parse_rows(text);
        if rows.is_empty() {
            return Vec::new();
        }

        let (header, data): (&str, &[String]) = if self.include_header {
            (rows[0].as_str(), &rows[1..])
        } else {
            ("", &rows[..])
        };

        if data.is_empty() {
            // Header-only file: emit a single chunk with just the header
            // (or empty if !include_header — caught upstream).
            if self.include_header && !header.is_empty() {
                return vec![header.to_string()];
            }
            return Vec::new();
        }

        let mut chunks: Vec<String> = Vec::new();
        for batch in data.chunks(self.rows_per_chunk) {
            let mut chunk = String::new();
            if self.include_header && !header.is_empty() {
                chunk.push_str(header);
                chunk.push('\n');
            }
            for (i, row) in batch.iter().enumerate() {
                if i > 0 {
                    chunk.push('\n');
                }
                chunk.push_str(row);
            }
            chunks.push(chunk);
        }
        chunks
    }
}

/// Walk `text` and return one entry per logical CSV row. Handles
/// RFC-4180 quoted fields containing literal newlines. Strips
/// trailing `\r` from rows so CRLF input round-trips cleanly.
fn parse_rows(text: &str) -> Vec<String> {
    let mut rows: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_quote = false;
    for ch in text.chars() {
        match ch {
            '"' => {
                in_quote = !in_quote;
                cur.push(ch);
            }
            '\n' if !in_quote => {
                if cur.ends_with('\r') {
                    cur.pop();
                }
                if !cur.is_empty() {
                    rows.push(std::mem::take(&mut cur));
                } else {
                    cur.clear();
                }
            }
            _ => cur.push(ch),
        }
    }
    if !cur.is_empty() {
        if cur.ends_with('\r') {
            cur.pop();
        }
        rows.push(cur);
    }
    rows
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_into_chunks_of_n_rows_with_header() {
        let csv = "name,age\nalice,30\nbob,25\ncarol,40\ndan,35\nev,22";
        let s = CsvRowSplitter::new(2);
        let chunks = s.split_text(csv);
        // 5 data rows / 2 per chunk → 3 chunks (2 + 2 + 1).
        assert_eq!(chunks.len(), 3);
        for c in &chunks {
            assert!(c.starts_with("name,age\n"), "header missing in chunk:\n{c}");
        }
        assert!(chunks[0].contains("alice,30"));
        assert!(chunks[0].contains("bob,25"));
        assert!(chunks[1].contains("carol,40"));
        assert!(chunks[1].contains("dan,35"));
        assert!(chunks[2].contains("ev,22"));
    }

    #[test]
    fn header_only_file_returns_single_header_chunk() {
        let csv = "name,age";
        let s = CsvRowSplitter::new(10);
        let chunks = s.split_text(csv);
        assert_eq!(chunks, vec!["name,age".to_string()]);
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = CsvRowSplitter::new(10);
        assert!(s.split_text("").is_empty());
    }

    #[test]
    fn with_header_false_skips_header_prefix() {
        let csv = "a,b\nc,d\ne,f";
        let s = CsvRowSplitter::new(2).with_header(false);
        let chunks = s.split_text(csv);
        // 3 rows / 2 per chunk → 2 chunks. No header prefix.
        assert_eq!(chunks.len(), 2);
        assert!(!chunks[0].contains("a,b\nc,d") || chunks[0] == "a,b\nc,d");
        // First chunk has the first 2 logical rows.
        assert_eq!(chunks[0], "a,b\nc,d");
        assert_eq!(chunks[1], "e,f");
    }

    #[test]
    fn quoted_field_with_newline_kept_intact() {
        // RFC-4180: cell wrapped in double quotes may contain a newline.
        let csv = "name,note\nalice,\"hello\nworld\"\nbob,plain";
        let s = CsvRowSplitter::new(2);
        let chunks = s.split_text(csv);
        // 2 logical data rows → 1 chunk.
        assert_eq!(chunks.len(), 1);
        let c = &chunks[0];
        assert!(c.starts_with("name,note\n"));
        assert!(c.contains("alice,\"hello\nworld\""));
        assert!(c.contains("bob,plain"));
    }

    #[test]
    fn crlf_line_endings_handled() {
        let csv = "a,b\r\nc,d\r\ne,f";
        let s = CsvRowSplitter::new(1);
        let chunks = s.split_text(csv);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], "a,b\nc,d");
        assert_eq!(chunks[1], "a,b\ne,f");
    }

    #[test]
    fn rows_per_chunk_zero_normalised_to_one() {
        let s = CsvRowSplitter::new(0);
        assert_eq!(s.rows_per_chunk, 1);
    }

    #[test]
    fn last_chunk_smaller_when_rows_dont_divide_evenly() {
        let csv = "h\n1\n2\n3\n4\n5\n6\n7";
        let s = CsvRowSplitter::new(3);
        let chunks = s.split_text(csv);
        // 7 rows / 3 per chunk → 3, 3, 1.
        assert_eq!(chunks.len(), 3);
        assert!(chunks[2].starts_with("h\n7") && !chunks[2].contains('6'));
    }
}

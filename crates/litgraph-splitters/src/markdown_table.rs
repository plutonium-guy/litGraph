//! `MarkdownTableSplitter` — preserve markdown tables as atomic
//! chunks.
//!
//! # The problem
//!
//! Existing splitters (recursive char, markdown header, token)
//! split by character / token count without table awareness. A
//! table in the middle of a doc gets fragmented mid-row, which
//! destroys downstream comprehension: the embedding model sees
//! a header row in chunk N, a separator row in chunk N+1, and
//! orphan data rows in chunk N+2. None of those individually
//! convey the table's meaning.
//!
//! # The fix
//!
//! Walk the markdown line-by-line. Detect table blocks by the
//! GFM/CommonMark pattern:
//!
//! ```text
//! | header1 | header2 |
//! | ------- | ------- |
//! | cell    | cell    |
//! ```
//!
//! Emit each table block as a SINGLE chunk regardless of size.
//! Emit non-table prose via an inner splitter (default:
//! `RecursiveCharacterSplitter` with chunk_size 1000, overlap
//! 200 — same defaults as LangChain `MarkdownTextSplitter`).
//!
//! # Caveats
//!
//! - Tables larger than the embedding context window will get
//!   truncated downstream regardless. Use `with_max_table_chars`
//!   to enable hard-cap fragmentation as a last resort —
//!   defaults disabled because losing table integrity is usually
//!   worse than emitting a long chunk.
//! - Borderless / "compact" tables without the `|---|` separator
//!   row are NOT recognized as tables (no reliable signal). Use
//!   [`crate::MarkdownHeaderSplitter`] for that flavor.

use crate::recursive::RecursiveCharacterSplitter;
use crate::Splitter;

pub struct MarkdownTableSplitter {
    /// Splitter applied to non-table prose blocks between
    /// tables.
    pub prose_splitter: Box<dyn Splitter>,
    /// Optional cap on a single table chunk's size (chars). When
    /// `Some(n)`, any table whose serialized form exceeds `n`
    /// chars is fragmented row-wise (each fragment includes the
    /// header + separator). Default `None` — tables are kept
    /// atomic.
    pub max_table_chars: Option<usize>,
}

impl Default for MarkdownTableSplitter {
    fn default() -> Self {
        Self::new()
    }
}

impl MarkdownTableSplitter {
    pub fn new() -> Self {
        Self {
            prose_splitter: Box::new(RecursiveCharacterSplitter::new(1000, 200)),
            max_table_chars: None,
        }
    }

    pub fn with_prose_splitter(mut self, s: Box<dyn Splitter>) -> Self {
        self.prose_splitter = s;
        self
    }

    pub fn with_max_table_chars(mut self, n: Option<usize>) -> Self {
        self.max_table_chars = n;
        self
    }
}

impl Splitter for MarkdownTableSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        let lines: Vec<&str> = text.lines().collect();
        let mut chunks: Vec<String> = Vec::new();
        let mut prose_buf = String::new();
        let mut i = 0;
        while i < lines.len() {
            if let Some(table_end) = detect_table(&lines, i) {
                // Flush prose buffer first.
                if !prose_buf.is_empty() {
                    for c in self.prose_splitter.split_text(&prose_buf) {
                        chunks.push(c);
                    }
                    prose_buf.clear();
                }
                // Push the full table as one chunk (or fragment if cap set).
                let table = lines[i..table_end].join("\n");
                if let Some(cap) = self.max_table_chars {
                    if table.chars().count() > cap {
                        let header_sep = lines[i..i + 2].join("\n");
                        chunks.extend(fragment_table(
                            &header_sep,
                            &lines[i + 2..table_end],
                            cap,
                        ));
                    } else {
                        chunks.push(table);
                    }
                } else {
                    chunks.push(table);
                }
                i = table_end;
            } else {
                if !prose_buf.is_empty() {
                    prose_buf.push('\n');
                }
                prose_buf.push_str(lines[i]);
                i += 1;
            }
        }
        // Final prose flush.
        if !prose_buf.is_empty() {
            for c in self.prose_splitter.split_text(&prose_buf) {
                chunks.push(c);
            }
        }
        chunks
    }
}

/// Detect a GFM table starting at `lines[i]`. Returns the
/// exclusive end index of the table block. Requires:
/// - Line `i` is a row (`| ... |`).
/// - Line `i+1` is a separator row (`| --- | --- |`).
/// - Following rows continue while they look like rows.
fn detect_table(lines: &[&str], i: usize) -> Option<usize> {
    let header = lines.get(i)?;
    let sep = lines.get(i + 1)?;
    if !is_table_row(header) || !is_separator_row(sep) {
        return None;
    }
    let mut j = i + 2;
    while j < lines.len() && is_table_row(lines[j]) {
        j += 1;
    }
    Some(j)
}

fn is_table_row(line: &str) -> bool {
    let t = line.trim();
    t.starts_with('|') && t.ends_with('|') && t.len() >= 3
}

fn is_separator_row(line: &str) -> bool {
    let t = line.trim();
    if !t.starts_with('|') || !t.ends_with('|') {
        return false;
    }
    let cells: Vec<&str> = t.split('|').filter(|c| !c.is_empty()).collect();
    if cells.is_empty() {
        return false;
    }
    cells.iter().all(|c| {
        let s = c.trim();
        !s.is_empty()
            && s.contains('-')
            && s.chars().all(|ch| ch == '-' || ch == ':' || ch == ' ')
    })
}

/// Fragment a too-large table while preserving the header +
/// separator row in each fragment. Each fragment is at most
/// `cap` chars; rows are added until adding one more would
/// exceed the cap.
fn fragment_table(header_sep: &str, data_rows: &[&str], cap: usize) -> Vec<String> {
    let mut frags: Vec<String> = Vec::new();
    let mut cur = String::from(header_sep);
    for row in data_rows {
        let candidate = cur.chars().count() + 1 + row.chars().count();
        if candidate > cap && cur != header_sep {
            frags.push(cur.clone());
            cur = format!("{header_sep}\n{row}");
        } else {
            cur.push('\n');
            cur.push_str(row);
        }
    }
    if !frags.last().map(|f| f == &cur).unwrap_or(false) && cur != header_sep {
        frags.push(cur);
    }
    frags
}

#[cfg(test)]
mod tests {
    use super::*;

    const FIXTURE: &str = "Some intro paragraph about widgets.\n\
This is a follow-up sentence.\n\
\n\
| Name | Price | In Stock |\n\
| ---- | ----- | -------- |\n\
| Widget A | $10 | yes |\n\
| Widget B | $20 | no |\n\
| Widget C | $30 | yes |\n\
\n\
After the table, more prose.\n\
And yet more prose to flesh out the section.";

    #[test]
    fn detects_simple_gfm_table() {
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(FIXTURE);
        // We should have at least 2 chunks: prose + table + trailing prose.
        assert!(chunks.len() >= 2);
        // Find the chunk that contains the table.
        let table_chunks: Vec<&String> = chunks
            .iter()
            .filter(|c| c.contains("Widget A") && c.contains("Widget B"))
            .collect();
        assert_eq!(table_chunks.len(), 1, "table fragmented");
        let table = table_chunks[0];
        // Must contain header, separator, and ALL data rows.
        assert!(table.contains("| Name"));
        assert!(table.contains("| ----"));
        assert!(table.contains("Widget A"));
        assert!(table.contains("Widget B"));
        assert!(table.contains("Widget C"));
    }

    #[test]
    fn prose_chunks_dont_contain_table_lines() {
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(FIXTURE);
        for c in &chunks {
            // No prose chunk should contain BOTH a header and any data row.
            let has_header = c.contains("| Name |");
            let has_data = c.contains("Widget A");
            // If it has a header, it must also have data (it's the table).
            // If it has data, it must have a header.
            assert!(
                has_header == has_data,
                "table fragmented across prose/table boundary: {c:?}",
            );
        }
    }

    #[test]
    fn prose_only_input_passes_through_inner_splitter() {
        let s = MarkdownTableSplitter::new();
        let text = "Just plain text, no tables here.\n\nA second paragraph.";
        let chunks = s.split_text(text);
        // No tables → all prose → at least one chunk, no fragmentation needed.
        assert!(!chunks.is_empty());
        assert!(chunks.iter().all(|c| !c.contains('|')));
    }

    #[test]
    fn no_separator_row_is_not_recognized_as_table() {
        // Pipe-bordered text without the separator row is just text.
        let text = "| Name | Price |\n| Widget A | $10 |\n";
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(text);
        // Should be treated as prose (no table detected).
        for c in &chunks {
            assert!(!c.contains("---"));
        }
    }

    #[test]
    fn table_at_start_of_document() {
        let text = "| H1 | H2 |\n| -- | -- |\n| a  | b  |\n\nFollow-up prose.";
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(text);
        let first = &chunks[0];
        assert!(first.contains("| H1 | H2 |"));
        assert!(first.contains("| a  | b  |"));
    }

    #[test]
    fn table_at_end_of_document() {
        let text = "Lead-in prose.\n\n| H1 | H2 |\n| -- | -- |\n| a  | b  |";
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(text);
        let last = chunks.last().unwrap();
        assert!(last.contains("| H1 | H2 |"));
    }

    #[test]
    fn alignment_separators_are_recognized() {
        let text =
            "| H1 | H2 | H3 |\n| :--- | :---: | ---: |\n| left | center | right |\n";
        let s = MarkdownTableSplitter::new();
        let chunks = s.split_text(text);
        let table = chunks
            .iter()
            .find(|c| c.contains("| H1"))
            .expect("table not detected");
        assert!(table.contains(":---:"));
        assert!(table.contains("left | center | right"));
    }

    #[test]
    fn max_table_chars_fragments_when_set() {
        // Large 50-row table, cap at 200 chars → multiple fragments.
        let mut text = String::from("| col |\n| --- |\n");
        for i in 0..50 {
            text.push_str(&format!("| row {i} with some extra text |\n"));
        }
        let s = MarkdownTableSplitter::new().with_max_table_chars(Some(200));
        let chunks = s.split_text(&text);
        let table_frags: Vec<&String> = chunks
            .iter()
            .filter(|c| c.contains("| col |") && c.contains("| --- |"))
            .collect();
        assert!(
            table_frags.len() > 1,
            "expected fragmentation, got {} frags",
            table_frags.len()
        );
        // Every fragment should retain the header + separator.
        for f in &table_frags {
            assert!(f.contains("| col |"));
            assert!(f.contains("| --- |"));
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = MarkdownTableSplitter::new();
        assert!(s.split_text("").is_empty());
    }
}

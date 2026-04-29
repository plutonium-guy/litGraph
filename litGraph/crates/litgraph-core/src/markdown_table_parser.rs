//! Markdown table parser. LLMs emit tabular data in markdown ("Here are
//! the results: | name | score | …"); downstream code that wants
//! row-by-row access needs a structured form.
//!
//! # What's parsed
//!
//! Standard GFM-style markdown tables:
//!
//! ```text
//! | header1 | header2 | header3 |
//! |---------|---------|---------|
//! | val1    | val2    | val3    |
//! | val4    | val5    | val6    |
//! ```
//!
//! Returns a `Vec<MarkdownTable>` (one per table found in the input).
//! Each table has `headers: Vec<String>` + `rows: Vec<HashMap<String, String>>`.
//! Cell values are trimmed; pipes inside backtick-quoted cells are preserved
//! (so `\`foo|bar\`` doesn't split on the inner `|`).
//!
//! # What's NOT parsed
//!
//! - Multi-line cells (rare in LLM output; not worth the complexity).
//! - Column-alignment markers (`|:---:|`) — recognized as the separator
//!   row + skipped, but not surfaced. Most consumers don't care.
//! - HTML tables (`<table>...</table>`) — use HtmlLoader for those.
//! - Tables without a separator row — invalid GFM; dropped.
//!
//! # Tolerance
//!
//! - Leading/trailing pipes optional: `a | b | c` works as well as `| a | b | c |`.
//! - Mismatched cell counts vs headers: extra cells dropped, missing cells
//!   become empty strings. Non-fatal — agents emit ragged tables sometimes.
//! - Multiple tables in one input → all returned, in document order.
//! - Markdown around tables ignored (paragraphs, lists, code blocks
//!   between tables don't break parsing).

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct MarkdownTable {
    pub headers: Vec<String>,
    /// Each row is a map from header name → cell value.
    pub rows: Vec<HashMap<String, String>>,
}

impl MarkdownTable {
    /// Convenience: get value at `(row_idx, header)`. Returns empty string
    /// for out-of-range row OR missing header (forgiving — agents reading
    /// LLM tables shouldn't crash on schema drift).
    pub fn get(&self, row_idx: usize, header: &str) -> &str {
        self.rows
            .get(row_idx)
            .and_then(|r| r.get(header))
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Number of data rows (excludes header + separator).
    pub fn len(&self) -> usize { self.rows.len() }
    pub fn is_empty(&self) -> bool { self.rows.is_empty() }
}

/// Parse all markdown tables in `text`, in document order. Returns an
/// empty vec if no tables found (not an error — text just had none).
pub fn parse_markdown_tables(text: &str) -> Vec<MarkdownTable> {
    let lines: Vec<&str> = text.lines().collect();
    let mut tables = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        // Look for a header row (must contain `|`) followed by a separator
        // row matching `|---|---|...` (with optional `:` for alignment).
        let header = lines[i].trim();
        if !looks_like_table_row(header) {
            i += 1;
            continue;
        }
        if i + 1 >= lines.len() {
            break;
        }
        let sep = lines[i + 1].trim();
        if !is_separator_row(sep) {
            i += 1;
            continue;
        }

        // We have a table; consume header + separator + data rows.
        let headers = split_row(header);
        let mut rows = Vec::new();
        let mut j = i + 2;
        while j < lines.len() {
            let row = lines[j].trim();
            if !looks_like_table_row(row) {
                break;
            }
            let cells = split_row(row);
            let mut map = HashMap::new();
            for (idx, h) in headers.iter().enumerate() {
                let v = cells.get(idx).cloned().unwrap_or_default();
                map.insert(h.clone(), v);
            }
            rows.push(map);
            j += 1;
        }

        tables.push(MarkdownTable { headers, rows });
        i = j;
    }
    tables
}

/// True if the line plausibly looks like a table row (contains at least
/// one un-quoted `|`).
fn looks_like_table_row(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return false;
    }
    // Must contain a pipe outside of backticks.
    let mut in_code = false;
    for ch in trimmed.chars() {
        if ch == '`' {
            in_code = !in_code;
        } else if ch == '|' && !in_code {
            return true;
        }
    }
    false
}

/// True if the line is a separator: `|---|---|` or `|:---:|---|`, etc.
/// Accepts optional `:` for alignment markers; cells must be at least
/// one `-` after stripping `:`.
fn is_separator_row(line: &str) -> bool {
    let cells = split_row(line);
    if cells.is_empty() {
        return false;
    }
    cells.iter().all(|c| {
        let stripped = c.trim().trim_matches(':');
        !stripped.is_empty() && stripped.chars().all(|ch| ch == '-')
    })
}

/// Split a `|`-delimited row into trimmed cells. Strips a leading and
/// trailing `|` if present (LLMs often include them; spec allows either).
/// Pipes inside `\`...\`` code spans are NOT split on.
fn split_row(line: &str) -> Vec<String> {
    let mut s = line.trim();
    if s.starts_with('|') {
        s = &s[1..];
    }
    if s.ends_with('|') {
        s = &s[..s.len() - 1];
    }
    let mut cells = Vec::new();
    let mut current = String::new();
    let mut in_code = false;
    for ch in s.chars() {
        match ch {
            '`' => {
                in_code = !in_code;
                current.push(ch);
            }
            '|' if !in_code => {
                cells.push(current.trim().to_string());
                current = String::new();
            }
            _ => current.push(ch),
        }
    }
    cells.push(current.trim().to_string());
    cells
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_three_column_table() {
        let md = "| name | score | rank |\n|------|-------|------|\n| alice | 95 | 1 |\n| bob | 80 | 2 |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
        let t = &tables[0];
        assert_eq!(t.headers, vec!["name", "score", "rank"]);
        assert_eq!(t.len(), 2);
        assert_eq!(t.get(0, "name"), "alice");
        assert_eq!(t.get(0, "score"), "95");
        assert_eq!(t.get(1, "rank"), "2");
    }

    #[test]
    fn parses_table_without_outer_pipes() {
        let md = "name | score\n-----|------\nalice | 95\nbob | 80\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].len(), 2);
        assert_eq!(tables[0].get(0, "name"), "alice");
    }

    #[test]
    fn parses_alignment_separator() {
        let md = "| h1 | h2 |\n|:---|:---:|\n| a | b |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].get(0, "h1"), "a");
        assert_eq!(tables[0].get(0, "h2"), "b");
    }

    #[test]
    fn empty_text_returns_empty_vec() {
        assert!(parse_markdown_tables("").is_empty());
        assert!(parse_markdown_tables("just some text").is_empty());
    }

    #[test]
    fn header_without_separator_not_a_table() {
        let md = "| a | b |\n| c | d |\n";  // no separator row
        let tables = parse_markdown_tables(md);
        assert!(tables.is_empty());
    }

    #[test]
    fn extra_cells_dropped_missing_become_empty() {
        let md = "| a | b |\n|---|---|\n| 1 | 2 | 3 |\n| only-one |\n";
        let tables = parse_markdown_tables(md);
        let t = &tables[0];
        assert_eq!(t.len(), 2);
        // Extra cell "3" dropped (no header).
        assert_eq!(t.get(0, "a"), "1");
        assert_eq!(t.get(0, "b"), "2");
        // Missing cell becomes empty.
        assert_eq!(t.get(1, "a"), "only-one");
        assert_eq!(t.get(1, "b"), "");
    }

    #[test]
    fn pipes_inside_backticks_not_split() {
        let md = "| code | desc |\n|------|------|\n| `a|b` | combined |\n";
        let tables = parse_markdown_tables(md);
        let t = &tables[0];
        assert_eq!(t.get(0, "code"), "`a|b`");
        assert_eq!(t.get(0, "desc"), "combined");
    }

    #[test]
    fn multiple_tables_separated_by_text() {
        let md = "First table:\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nSome paragraph.\n\n| x | y |\n|---|---|\n| u | v |\n| w | z |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].headers, vec!["a", "b"]);
        assert_eq!(tables[0].len(), 1);
        assert_eq!(tables[1].headers, vec!["x", "y"]);
        assert_eq!(tables[1].len(), 2);
    }

    #[test]
    fn cell_whitespace_trimmed() {
        let md = "|   col1   |   col2   |\n|---|---|\n|   alpha   |   beta   |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables[0].headers, vec!["col1", "col2"]);
        assert_eq!(tables[0].get(0, "col1"), "alpha");
        assert_eq!(tables[0].get(0, "col2"), "beta");
    }

    #[test]
    fn empty_data_rows_table_returns_zero_rows() {
        let md = "| h1 | h2 |\n|---|---|\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].headers, vec!["h1", "h2"]);
        assert!(tables[0].is_empty());
    }

    #[test]
    fn table_at_start_and_end_of_text() {
        let md = "| a |\n|---|\n| 1 |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
        assert_eq!(tables[0].get(0, "a"), "1");
    }

    #[test]
    fn get_returns_empty_string_for_missing_row_or_header() {
        let md = "| a |\n|---|\n| 1 |\n";
        let t = &parse_markdown_tables(md)[0];
        assert_eq!(t.get(99, "a"), "");
        assert_eq!(t.get(0, "nonexistent"), "");
    }

    #[test]
    fn separator_row_with_only_dashes_no_colons() {
        // Most common shape — pure dashes.
        let md = "| h |\n|---|\n| v |\n";
        let tables = parse_markdown_tables(md);
        assert_eq!(tables.len(), 1);
    }

    #[test]
    fn invalid_separator_with_letters_not_a_table() {
        // Separator row contains letters → not a separator → not a table.
        let md = "| a | b |\n| not | a sep |\n| 1 | 2 |\n";
        let tables = parse_markdown_tables(md);
        assert!(tables.is_empty());
    }

    #[test]
    fn numeric_string_values_preserved_as_strings() {
        // We don't try to parse types — agents/callers do that downstream.
        let md = "| n |\n|---|\n| 42 |\n| 3.14 |\n| true |\n";
        let t = &parse_markdown_tables(md)[0];
        assert_eq!(t.get(0, "n"), "42");
        assert_eq!(t.get(1, "n"), "3.14");
        assert_eq!(t.get(2, "n"), "true");
    }
}

//! Tabular output parser + query engine — LangChain
//! `PandasDataFrameOutputParser` parity, no Python dep.
//!
//! Two halves:
//!
//! 1. **Parse a [`Table`] from LLM output.** Three accepted formats so
//!    the prompt can pick whichever is cheapest for the task:
//!    - JSON object `{"columns": [...], "rows": [[...], ...]}`
//!    - JSON array-of-objects `[{col: val, ...}, ...]`
//!    - CSV (RFC-4180-ish; double-quote escapes, comma separator)
//! 2. **Execute structured queries** against a known `Table`. Operations:
//!    - `column:<name>` → all values in that column
//!    - `row:<index>` → all values in that row, keyed by column
//!    - `<column>:<row>` → single cell
//!    - `<op>:<column>` for `mean`, `sum`, `min`, `max`, `count`, `unique`
//!    Each op rejects type-incompatible columns cleanly (e.g. `mean` on
//!    a string column → `Error::invalid`).
//!
//! # Why bake this into core
//!
//! Pure-Rust, no new deps (CSV parser is ~30 lines and only handles the
//! shapes LLMs actually emit — no streaming, no fancy quoting). Pulling
//! in `polars` for a parser that runs on hundreds of cells would be
//! overkill.
//!
//! # Example flow
//!
//! ```no_run
//! use litgraph_core::table_parser::{Table, TableQuery, format_instructions};
//!
//! // 1. Show the LLM the format spec.
//! let instructions = format_instructions(&["price", "city"]);
//!
//! // 2. LLM emits something like: `column:price`
//! let query = TableQuery::parse("column:price").unwrap();
//!
//! // 3. Resolve against your data.
//! let table = Table::from_records(
//!     vec!["price".into(), "city".into()],
//!     vec![
//!         vec![serde_json::json!(10), serde_json::json!("NYC")],
//!         vec![serde_json::json!(20), serde_json::json!("LA")],
//!     ],
//! ).unwrap();
//! let result = table.execute(&query).unwrap();
//! ```

use std::collections::HashSet;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{Error, Result};

/// Tabular value carrier — column-aligned rows.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Table {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

impl Table {
    /// Build from explicit columns + row matrix. Errors if any row has
    /// the wrong number of cells or columns are duplicated.
    pub fn from_records(columns: Vec<String>, rows: Vec<Vec<Value>>) -> Result<Self> {
        let n = columns.len();
        if n == 0 {
            return Err(Error::invalid("Table: needs at least one column"));
        }
        let mut seen = HashSet::with_capacity(n);
        for c in &columns {
            if !seen.insert(c.clone()) {
                return Err(Error::invalid(format!("Table: duplicate column `{c}`")));
            }
        }
        for (i, r) in rows.iter().enumerate() {
            if r.len() != n {
                return Err(Error::invalid(format!(
                    "Table: row {i} has {} cells, expected {n}",
                    r.len()
                )));
            }
        }
        Ok(Self { columns, rows })
    }

    pub fn n_rows(&self) -> usize {
        self.rows.len()
    }
    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Index of a column name, or `Err` if absent.
    pub fn col_index(&self, name: &str) -> Result<usize> {
        self.columns
            .iter()
            .position(|c| c == name)
            .ok_or_else(|| Error::invalid(format!("column `{name}` not found")))
    }

    /// All values in `column`, in row order.
    pub fn column(&self, name: &str) -> Result<Vec<Value>> {
        let idx = self.col_index(name)?;
        Ok(self.rows.iter().map(|r| r[idx].clone()).collect())
    }

    /// All cells in `row`, keyed by column name. Errors if `row` is OOB.
    pub fn row_record(&self, row: usize) -> Result<serde_json::Map<String, Value>> {
        let r = self.rows.get(row).ok_or_else(|| {
            Error::invalid(format!("row {row} out of range (n_rows={})", self.n_rows()))
        })?;
        let mut m = serde_json::Map::with_capacity(self.columns.len());
        for (c, v) in self.columns.iter().zip(r.iter()) {
            m.insert(c.clone(), v.clone());
        }
        Ok(m)
    }

    /// Single cell at `(column, row)`.
    pub fn cell(&self, column: &str, row: usize) -> Result<Value> {
        let idx = self.col_index(column)?;
        let r = self.rows.get(row).ok_or_else(|| {
            Error::invalid(format!("row {row} out of range (n_rows={})", self.n_rows()))
        })?;
        Ok(r[idx].clone())
    }

    /// Run a parsed [`TableQuery`] and return the result as a JSON value.
    /// The shape varies by op — `column` returns an array, `row`
    /// returns an object, `cell` returns the raw value, `mean`/`sum`/
    /// `min`/`max` return numbers, `count` returns an integer, `unique`
    /// returns an array.
    pub fn execute(&self, q: &TableQuery) -> Result<Value> {
        match q {
            TableQuery::Column { name } => Ok(Value::Array(self.column(name)?)),
            TableQuery::Row { index } => Ok(Value::Object(self.row_record(*index)?)),
            TableQuery::Cell { column, row } => Ok(self.cell(column, *row)?),
            TableQuery::Mean { column } => {
                let nums = numeric_column(self, column)?;
                if nums.is_empty() {
                    return Err(Error::invalid(format!("mean: column `{column}` is empty")));
                }
                let m = nums.iter().sum::<f64>() / nums.len() as f64;
                Ok(json!(m))
            }
            TableQuery::Sum { column } => {
                let nums = numeric_column(self, column)?;
                Ok(json!(nums.iter().sum::<f64>()))
            }
            TableQuery::Min { column } => {
                let nums = numeric_column(self, column)?;
                let m = nums
                    .iter()
                    .copied()
                    .fold(f64::INFINITY, f64::min);
                if m.is_infinite() {
                    return Err(Error::invalid(format!("min: column `{column}` is empty")));
                }
                Ok(json!(m))
            }
            TableQuery::Max { column } => {
                let nums = numeric_column(self, column)?;
                let m = nums
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                if m.is_infinite() {
                    return Err(Error::invalid(format!("max: column `{column}` is empty")));
                }
                Ok(json!(m))
            }
            TableQuery::Count { column } => {
                let idx = self.col_index(column)?;
                let n = self
                    .rows
                    .iter()
                    .filter(|r| !r[idx].is_null())
                    .count();
                Ok(json!(n))
            }
            TableQuery::Unique { column } => {
                let idx = self.col_index(column)?;
                let mut seen = Vec::new();
                let mut keys: HashSet<String> = HashSet::new();
                for r in &self.rows {
                    let v = &r[idx];
                    let key = serde_json::to_string(v).unwrap_or_default();
                    if keys.insert(key) {
                        seen.push(v.clone());
                    }
                }
                Ok(Value::Array(seen))
            }
        }
    }
}

fn numeric_column(t: &Table, column: &str) -> Result<Vec<f64>> {
    let idx = t.col_index(column)?;
    let mut out = Vec::with_capacity(t.rows.len());
    for (row_i, r) in t.rows.iter().enumerate() {
        match r[idx].as_f64() {
            Some(f) => out.push(f),
            None if r[idx].is_null() => continue, // skip nulls
            None => {
                return Err(Error::invalid(format!(
                    "numeric op on column `{column}`: row {row_i} is non-numeric ({})",
                    r[idx]
                )))
            }
        }
    }
    Ok(out)
}

// ---- query language --------------------------------------------------------

/// Parsed LLM-issued table query. Each variant maps to one execute path.
#[derive(Debug, Clone, PartialEq)]
pub enum TableQuery {
    Column { name: String },
    Row { index: usize },
    Cell { column: String, row: usize },
    Mean { column: String },
    Sum { column: String },
    Min { column: String },
    Max { column: String },
    Count { column: String },
    Unique { column: String },
}

impl TableQuery {
    /// Parse a single-line query string. Tolerates surrounding whitespace
    /// and case-insensitive op names. Trailing punctuation (`.`, `;`)
    /// stripped — LLMs often emit a sentence-style answer.
    pub fn parse(s: &str) -> Result<Self> {
        let s = s.trim().trim_end_matches(['.', ';', ',']);
        if s.is_empty() {
            return Err(Error::invalid("table query: empty"));
        }

        // Split on the FIRST colon. After that, the remainder is either:
        // - a bare name (column / op)
        // - `name:row` (cell shortcut)
        let (head, rest) = s
            .split_once(':')
            .ok_or_else(|| Error::invalid(format!("table query: expected `:` in `{s}`")))?;
        let head_lc = head.trim().to_ascii_lowercase();
        let rest = rest.trim();

        // Reserved op names. Any op acting on a column.
        const REDUCTIONS: &[&str] = &["mean", "sum", "min", "max", "count", "unique"];
        if REDUCTIONS.contains(&head_lc.as_str()) {
            if rest.is_empty() {
                return Err(Error::invalid(format!(
                    "table query: `{head_lc}:` needs a column"
                )));
            }
            let column = rest.to_string();
            return Ok(match head_lc.as_str() {
                "mean" => Self::Mean { column },
                "sum" => Self::Sum { column },
                "min" => Self::Min { column },
                "max" => Self::Max { column },
                "count" => Self::Count { column },
                "unique" => Self::Unique { column },
                _ => unreachable!(),
            });
        }

        if head_lc == "column" {
            if rest.is_empty() {
                return Err(Error::invalid("table query: `column:` needs a name"));
            }
            return Ok(Self::Column {
                name: rest.to_string(),
            });
        }

        if head_lc == "row" {
            let idx = rest
                .parse::<usize>()
                .map_err(|_| Error::invalid(format!("table query: row index `{rest}` not int")))?;
            return Ok(Self::Row { index: idx });
        }

        // Cell shortcut: `<column>:<row>` — `head` is the column name
        // (case-preserved), `rest` is a usize.
        let row = rest
            .parse::<usize>()
            .map_err(|_| Error::invalid(format!(
                "table query: unknown op `{head}` (or non-int row `{rest}` for cell shortcut)"
            )))?;
        Ok(Self::Cell {
            column: head.trim().to_string(),
            row,
        })
    }
}

// ---- format instructions ---------------------------------------------------

/// Render a system-prompt-friendly description of the query language.
/// Pass `available_columns` so the LLM can reference real column names.
/// Empty slice → no example column listing.
pub fn format_instructions(available_columns: &[&str]) -> String {
    let mut s = String::new();
    s.push_str(
        "Output exactly one line in one of these forms:\n\
         - `column:<name>` — all values in a column\n\
         - `row:<index>` — all cells in a row (zero-indexed)\n\
         - `<column>:<index>` — a single cell\n\
         - `mean:<column>` / `sum:<column>` / `min:<column>` / `max:<column>` — \
           aggregate over a numeric column\n\
         - `count:<column>` — number of non-null cells\n\
         - `unique:<column>` — distinct values in order of first occurrence\n\n\
         Output ONLY the query line. No prose, no JSON wrapper.",
    );
    if !available_columns.is_empty() {
        s.push_str("\n\nAvailable columns: ");
        for (i, c) in available_columns.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(c);
        }
    }
    s
}

// ---- ingest parsers --------------------------------------------------------

/// Parse a table from LLM-emitted JSON. Accepts both:
///
/// 1. `{"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}` — column-major
///    schema explicit.
/// 2. `[{"a": 1, "b": 2}, {"a": 3, "b": 4}]` — record array. Columns
///    inferred from the first object; later objects must have the same
///    keys.
pub fn parse_table_json(input: &str) -> Result<Table> {
    let v: Value = serde_json::from_str(input.trim())
        .map_err(|e| Error::other(format!("table parse json: {e}")))?;
    parse_table_value(&v)
}

pub fn parse_table_value(v: &Value) -> Result<Table> {
    if let Some(obj) = v.as_object() {
        if let (Some(cols), Some(rows)) = (
            obj.get("columns").and_then(|x| x.as_array()),
            obj.get("rows").and_then(|x| x.as_array()),
        ) {
            let columns: Vec<String> = cols
                .iter()
                .map(|c| {
                    c.as_str()
                        .map(String::from)
                        .ok_or_else(|| Error::invalid("table parse: non-string column"))
                })
                .collect::<Result<Vec<_>>>()?;
            let row_vecs: Vec<Vec<Value>> = rows
                .iter()
                .enumerate()
                .map(|(i, r)| {
                    r.as_array().cloned().ok_or_else(|| {
                        Error::invalid(format!("table parse: row {i} is not an array"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            return Table::from_records(columns, row_vecs);
        }
    }
    if let Some(arr) = v.as_array() {
        return parse_records_array(arr);
    }
    Err(Error::invalid(
        "table parse: expected object {columns,rows} or array of objects",
    ))
}

fn parse_records_array(arr: &[Value]) -> Result<Table> {
    if arr.is_empty() {
        return Err(Error::invalid("table parse: empty record array"));
    }
    let first = arr[0]
        .as_object()
        .ok_or_else(|| Error::invalid("table parse: records must be objects"))?;
    // Preserve key order from the first object.
    let columns: Vec<String> = first.keys().cloned().collect();
    let mut rows = Vec::with_capacity(arr.len());
    for (i, rec) in arr.iter().enumerate() {
        let m = rec
            .as_object()
            .ok_or_else(|| Error::invalid(format!("table parse: record {i} not object")))?;
        let mut row = Vec::with_capacity(columns.len());
        for col in &columns {
            row.push(m.get(col).cloned().unwrap_or(Value::Null));
        }
        rows.push(row);
    }
    Table::from_records(columns, rows)
}

/// Parse RFC-4180-flavoured CSV. Limits matched to LLM output, not
/// production-grade CSV parsing — we don't need streaming, BOM
/// detection, or non-comma separators. First row is header.
pub fn parse_table_csv(input: &str) -> Result<Table> {
    let rows = csv_rows(input.trim())?;
    let mut iter = rows.into_iter();
    let header = iter
        .next()
        .ok_or_else(|| Error::invalid("table parse csv: empty"))?;
    let mut data = Vec::new();
    for row in iter {
        let mut padded = row;
        // Pad short rows with nulls so column count stays consistent — LLMs
        // sometimes drop trailing empty cells.
        while padded.len() < header.len() {
            padded.push(String::new());
        }
        let json_row: Vec<Value> = padded
            .into_iter()
            .take(header.len())
            .map(|cell| {
                // Best-effort numeric coercion: if the cell looks like
                // an int or float, store as JSON number. Otherwise string.
                if let Ok(n) = cell.parse::<i64>() {
                    Value::Number(n.into())
                } else if let Ok(f) = cell.parse::<f64>() {
                    serde_json::Number::from_f64(f)
                        .map(Value::Number)
                        .unwrap_or(Value::String(cell))
                } else {
                    Value::String(cell)
                }
            })
            .collect();
        data.push(json_row);
    }
    Table::from_records(header, data)
}

/// Tokenise CSV into rows-of-strings. Handles double-quoted fields
/// (including embedded commas + `""` escapes for literal `"`).
fn csv_rows(input: &str) -> Result<Vec<Vec<String>>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut chars = input.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            match c {
                '"' if chars.peek() == Some(&'"') => {
                    field.push('"');
                    chars.next();
                }
                '"' => in_quotes = false,
                _ => field.push(c),
            }
            continue;
        }
        match c {
            '"' => in_quotes = true,
            ',' => {
                row.push(std::mem::take(&mut field));
            }
            '\n' => {
                row.push(std::mem::take(&mut field));
                rows.push(std::mem::take(&mut row));
            }
            '\r' => {} // tolerate CRLF
            _ => field.push(c),
        }
    }
    if !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    if in_quotes {
        return Err(Error::invalid("table parse csv: unclosed quote"));
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Table {
        Table::from_records(
            vec!["price".into(), "city".into(), "qty".into()],
            vec![
                vec![json!(10.0), json!("NYC"), json!(2)],
                vec![json!(20.0), json!("LA"), json!(5)],
                vec![json!(15.0), json!("NYC"), json!(3)],
                vec![json!(30.0), json!("Boston"), json!(1)],
            ],
        )
        .unwrap()
    }

    // ---- Table construction ----

    #[test]
    fn from_records_rejects_zero_columns() {
        assert!(Table::from_records(Vec::new(), Vec::new()).is_err());
    }

    #[test]
    fn from_records_rejects_duplicate_columns() {
        let err = Table::from_records(
            vec!["a".into(), "a".into()],
            vec![vec![json!(1), json!(2)]],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("duplicate"));
    }

    #[test]
    fn from_records_rejects_ragged_rows() {
        let err = Table::from_records(
            vec!["a".into(), "b".into()],
            vec![vec![json!(1)]],
        )
        .unwrap_err();
        assert!(format!("{err}").contains("row 0"));
    }

    // ---- Query parsing ----

    #[test]
    fn parse_column_query() {
        let q = TableQuery::parse("column:price").unwrap();
        assert_eq!(q, TableQuery::Column { name: "price".into() });
    }

    #[test]
    fn parse_row_query() {
        let q = TableQuery::parse("row:2").unwrap();
        assert_eq!(q, TableQuery::Row { index: 2 });
    }

    #[test]
    fn parse_cell_shortcut() {
        let q = TableQuery::parse("price:1").unwrap();
        assert_eq!(
            q,
            TableQuery::Cell {
                column: "price".into(),
                row: 1
            }
        );
    }

    #[test]
    fn parse_reduction_ops_case_insensitive() {
        for (input, want) in [
            ("mean:price", TableQuery::Mean { column: "price".into() }),
            ("SUM:qty", TableQuery::Sum { column: "qty".into() }),
            ("Min:qty", TableQuery::Min { column: "qty".into() }),
            ("MAX:price", TableQuery::Max { column: "price".into() }),
            ("count:city", TableQuery::Count { column: "city".into() }),
            ("unique:city", TableQuery::Unique { column: "city".into() }),
        ] {
            assert_eq!(TableQuery::parse(input).unwrap(), want, "{input}");
        }
    }

    #[test]
    fn parse_strips_trailing_punctuation_and_whitespace() {
        let q = TableQuery::parse("  column:price.  ").unwrap();
        assert_eq!(q, TableQuery::Column { name: "price".into() });
    }

    #[test]
    fn parse_rejects_empty_input() {
        assert!(TableQuery::parse("").is_err());
        assert!(TableQuery::parse("   ").is_err());
    }

    #[test]
    fn parse_rejects_no_colon() {
        let err = TableQuery::parse("price").unwrap_err();
        assert!(format!("{err}").contains("expected `:`"));
    }

    #[test]
    fn parse_rejects_unknown_op_without_int_row() {
        let err = TableQuery::parse("plot:price").unwrap_err();
        assert!(format!("{err}").contains("unknown op"));
    }

    #[test]
    fn parse_rejects_empty_column_after_reduction() {
        let err = TableQuery::parse("mean:").unwrap_err();
        assert!(format!("{err}").contains("needs a column"));
    }

    // ---- Execute ----

    #[test]
    fn execute_column() {
        let t = sample();
        let out = t.execute(&TableQuery::Column { name: "city".into() }).unwrap();
        assert_eq!(out, json!(["NYC", "LA", "NYC", "Boston"]));
    }

    #[test]
    fn execute_row_returns_object_keyed_by_column() {
        let t = sample();
        let out = t.execute(&TableQuery::Row { index: 1 }).unwrap();
        assert_eq!(out["price"], 20.0);
        assert_eq!(out["city"], "LA");
        assert_eq!(out["qty"], 5);
    }

    #[test]
    fn execute_cell() {
        let t = sample();
        let out = t
            .execute(&TableQuery::Cell {
                column: "city".into(),
                row: 0,
            })
            .unwrap();
        assert_eq!(out, "NYC");
    }

    #[test]
    fn execute_mean_sum_min_max() {
        let t = sample();
        // qty: 2, 5, 3, 1 → sum 11, mean 2.75, min 1, max 5.
        assert_eq!(
            t.execute(&TableQuery::Sum { column: "qty".into() }).unwrap(),
            json!(11.0)
        );
        let mean = t
            .execute(&TableQuery::Mean { column: "qty".into() })
            .unwrap()
            .as_f64()
            .unwrap();
        assert!((mean - 2.75).abs() < 1e-9);
        assert_eq!(
            t.execute(&TableQuery::Min { column: "qty".into() }).unwrap(),
            json!(1.0)
        );
        assert_eq!(
            t.execute(&TableQuery::Max { column: "qty".into() }).unwrap(),
            json!(5.0)
        );
    }

    #[test]
    fn execute_count_skips_nulls() {
        let t = Table::from_records(
            vec!["x".into()],
            vec![
                vec![json!(1)],
                vec![json!(null)],
                vec![json!(3)],
                vec![json!(null)],
            ],
        )
        .unwrap();
        assert_eq!(
            t.execute(&TableQuery::Count { column: "x".into() }).unwrap(),
            json!(2)
        );
    }

    #[test]
    fn execute_unique_preserves_first_occurrence_order() {
        let t = sample();
        let out = t
            .execute(&TableQuery::Unique { column: "city".into() })
            .unwrap();
        assert_eq!(out, json!(["NYC", "LA", "Boston"]));
    }

    #[test]
    fn execute_mean_on_string_column_errors() {
        let t = sample();
        let err = t
            .execute(&TableQuery::Mean { column: "city".into() })
            .unwrap_err();
        assert!(format!("{err}").contains("non-numeric"));
    }

    #[test]
    fn execute_unknown_column_errors() {
        let t = sample();
        let err = t
            .execute(&TableQuery::Column { name: "nope".into() })
            .unwrap_err();
        assert!(format!("{err}").contains("not found"));
    }

    #[test]
    fn execute_oob_row_errors() {
        let t = sample();
        let err = t
            .execute(&TableQuery::Row { index: 99 })
            .unwrap_err();
        assert!(format!("{err}").contains("out of range"));
    }

    #[test]
    fn execute_min_max_on_empty_column_errors() {
        let t = Table::from_records(vec!["x".into()], Vec::new()).unwrap();
        assert!(t.execute(&TableQuery::Min { column: "x".into() }).is_err());
        assert!(t.execute(&TableQuery::Max { column: "x".into() }).is_err());
    }

    // ---- JSON ingest ----

    #[test]
    fn parse_json_columns_rows_shape() {
        let t = parse_table_json(
            r#"{"columns": ["a", "b"], "rows": [[1, 2], [3, 4]]}"#,
        )
        .unwrap();
        assert_eq!(t.n_rows(), 2);
        assert_eq!(t.columns, vec!["a", "b"]);
        assert_eq!(t.cell("b", 1).unwrap(), 4);
    }

    #[test]
    fn parse_json_records_shape() {
        let t = parse_table_json(r#"[{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]"#).unwrap();
        assert_eq!(t.columns, vec!["a", "b"]);
        assert_eq!(t.cell("b", 0).unwrap(), "x");
    }

    #[test]
    fn parse_json_records_pads_missing_keys_with_null() {
        let t = parse_table_json(r#"[{"a": 1, "b": 2}, {"a": 3}]"#).unwrap();
        assert_eq!(t.cell("b", 1).unwrap(), Value::Null);
    }

    #[test]
    fn parse_json_rejects_empty_record_array() {
        let err = parse_table_json("[]").unwrap_err();
        assert!(format!("{err}").contains("empty"));
    }

    #[test]
    fn parse_json_rejects_non_array_records() {
        let err = parse_table_json(r#"[1, 2]"#).unwrap_err();
        assert!(format!("{err}").contains("must be objects"));
    }

    #[test]
    fn parse_json_rejects_garbage() {
        assert!(parse_table_json("not json").is_err());
        assert!(parse_table_json(r#""string""#).is_err());
    }

    // ---- CSV ingest ----

    #[test]
    fn parse_csv_basic() {
        let t = parse_table_csv("price,city\n10,NYC\n20,LA").unwrap();
        assert_eq!(t.columns, vec!["price", "city"]);
        assert_eq!(t.cell("price", 0).unwrap(), 10);
        assert_eq!(t.cell("city", 1).unwrap(), "LA");
    }

    #[test]
    fn parse_csv_handles_quoted_commas() {
        let t = parse_table_csv("name,desc\n\"Smith, John\",hello\nDoe,\"with, comma\"")
            .unwrap();
        assert_eq!(t.cell("name", 0).unwrap(), "Smith, John");
        assert_eq!(t.cell("desc", 1).unwrap(), "with, comma");
    }

    #[test]
    fn parse_csv_handles_escaped_quotes() {
        let t = parse_table_csv("q\n\"she said \"\"hi\"\"\"").unwrap();
        assert_eq!(t.cell("q", 0).unwrap(), r#"she said "hi""#);
    }

    #[test]
    fn parse_csv_pads_short_rows_with_empty_strings() {
        let t = parse_table_csv("a,b,c\n1,2").unwrap();
        // Short row gets padded — third cell is "".
        assert_eq!(t.cell("c", 0).unwrap(), "");
    }

    #[test]
    fn parse_csv_coerces_numeric_strings() {
        let t = parse_table_csv("x\n42\n3.14\nhello").unwrap();
        assert_eq!(t.cell("x", 0).unwrap(), 42);
        assert!((t.cell("x", 1).unwrap().as_f64().unwrap() - 3.14).abs() < 1e-9);
        assert_eq!(t.cell("x", 2).unwrap(), "hello");
    }

    #[test]
    fn parse_csv_rejects_unclosed_quote() {
        let err = parse_table_csv("a\n\"open").unwrap_err();
        assert!(format!("{err}").contains("unclosed"));
    }

    #[test]
    fn parse_csv_handles_crlf() {
        let t = parse_table_csv("a,b\r\n1,2\r\n3,4").unwrap();
        assert_eq!(t.n_rows(), 2);
    }

    // ---- format_instructions ----

    #[test]
    fn format_instructions_lists_columns() {
        let s = format_instructions(&["price", "city"]);
        assert!(s.contains("column:<name>"));
        assert!(s.contains("Available columns: price, city"));
    }

    #[test]
    fn format_instructions_no_columns() {
        let s = format_instructions(&[]);
        assert!(!s.contains("Available columns"));
    }
}

//! Jupyter notebook (.ipynb) loader.
//!
//! Notebooks are JSON. Each cell is a code or markdown block; some have
//! attached outputs. By default, this loader emits ONE Document per cell
//! (cell ordering preserved via metadata `cell_index`). Each Document's
//! content is the cell's source text; metadata carries `cell_type`,
//! `cell_index`, and (optionally) the cell's text outputs.
//!
//! Why per-cell instead of per-notebook: notebooks are inherently chunked
//! by the author. Treating each cell as a Document preserves that
//! authorial structure for retrieval — a query about "how to load the
//! data" surfaces THE data-loading cell, not the entire notebook.
//!
//! Knobs:
//! - `include_outputs` (default false): include text/plain outputs in
//!   the cell's content. Off by default — outputs are noisy (warnings,
//!   stack traces, repr) and inflate token cost without adding signal.
//!   Turn on for analytical RAG (output values matter — error logs,
//!   model metrics, plot captions).
//! - `cell_types` (default `["code", "markdown"]`): which cell types
//!   to emit. `"raw"` cells (rare — Jinja blocks etc) skipped by default.
//! - `concat_into_one_doc` (default false): emit one Document per
//!   notebook (cells joined with double newlines + cell-type headers).
//!   Use when you want a single document per file (e.g. to align with a
//!   text-loader pipeline that expects one-doc-per-source).

use std::fs;
use std::path::{Path, PathBuf};

use litgraph_core::Document;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::{Loader, LoaderError, LoaderResult};

pub struct JupyterNotebookLoader {
    pub path: PathBuf,
    pub include_outputs: bool,
    pub cell_types: Vec<String>,
    pub concat_into_one_doc: bool,
}

impl JupyterNotebookLoader {
    pub fn new<P: AsRef<Path>>(p: P) -> Self {
        Self {
            path: p.as_ref().to_path_buf(),
            include_outputs: false,
            cell_types: vec!["code".into(), "markdown".into()],
            concat_into_one_doc: false,
        }
    }

    pub fn with_outputs(mut self, on: bool) -> Self {
        self.include_outputs = on;
        self
    }

    pub fn with_cell_types(mut self, types: Vec<String>) -> Self {
        self.cell_types = types;
        self
    }

    pub fn concat_into_one_doc(mut self, on: bool) -> Self {
        self.concat_into_one_doc = on;
        self
    }
}

impl Loader for JupyterNotebookLoader {
    fn load(&self) -> LoaderResult<Vec<Document>> {
        let raw = fs::read_to_string(&self.path)?;
        let nb: Notebook = serde_json::from_str(&raw)?;
        let source_str = self.path.to_string_lossy().to_string();

        let mut emitted = Vec::new();
        for (idx, cell) in nb.cells.iter().enumerate() {
            if !self.cell_types.iter().any(|t| t == &cell.cell_type) {
                continue;
            }
            let body = join_source(&cell.source);
            let outputs_text = if self.include_outputs {
                extract_text_outputs(&cell.outputs)
            } else {
                String::new()
            };
            let content = if outputs_text.is_empty() {
                body
            } else {
                format!("{body}\n\n--- output ---\n{outputs_text}")
            };
            emitted.push((idx, cell.cell_type.clone(), content));
        }

        if self.concat_into_one_doc {
            // Single Document — cells joined with a header per cell.
            let joined: String = emitted
                .iter()
                .map(|(idx, ctype, content)| {
                    format!("# [cell {idx} — {ctype}]\n{content}")
                })
                .collect::<Vec<_>>()
                .join("\n\n");
            let mut d = Document::new(joined).with_id(source_str.clone());
            d.metadata.insert("source".into(), Value::String(source_str));
            d.metadata
                .insert("n_cells".into(), json!(emitted.len()));
            if let Some(lang) = nb.language() {
                d.metadata.insert("language".into(), Value::String(lang));
            }
            return Ok(vec![d]);
        }

        // Per-cell: one Document per emitted cell.
        let lang = nb.language();
        let docs: Vec<Document> = emitted
            .into_iter()
            .map(|(idx, ctype, content)| {
                let id = format!("{source_str}#cell-{idx}");
                let mut d = Document::new(content).with_id(id);
                d.metadata
                    .insert("source".into(), Value::String(source_str.clone()));
                d.metadata.insert("cell_index".into(), json!(idx));
                d.metadata
                    .insert("cell_type".into(), Value::String(ctype));
                if let Some(l) = &lang {
                    d.metadata
                        .insert("language".into(), Value::String(l.clone()));
                }
                d
            })
            .collect();
        Ok(docs)
    }
}

#[derive(Debug, Deserialize)]
struct Notebook {
    #[serde(default)]
    cells: Vec<Cell>,
    #[serde(default)]
    metadata: Value,
}

impl Notebook {
    fn language(&self) -> Option<String> {
        // Standard nbformat path: metadata.kernelspec.language OR
        // metadata.language_info.name.
        self.metadata
            .get("kernelspec")
            .and_then(|k| k.get("language"))
            .and_then(|v| v.as_str())
            .or_else(|| {
                self.metadata
                    .get("language_info")
                    .and_then(|li| li.get("name"))
                    .and_then(|v| v.as_str())
            })
            .map(|s| s.to_string())
    }
}

#[derive(Debug, Deserialize)]
struct Cell {
    cell_type: String,
    /// `source` may be either a string OR an array of strings (nbformat 4).
    /// We accept both via untagged `SourceField`.
    #[serde(default)]
    source: SourceField,
    #[serde(default)]
    outputs: Vec<Value>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum SourceField {
    Lines(Vec<String>),
    Single(String),
    Empty(()),
}

impl Default for SourceField {
    fn default() -> Self { SourceField::Empty(()) }
}

fn join_source(src: &SourceField) -> String {
    match src {
        SourceField::Single(s) => s.clone(),
        SourceField::Lines(lines) => lines.concat(), // nbformat 4: lines already include "\n"
        SourceField::Empty(_) => String::new(),
    }
}

/// Pull text from cell outputs. Handles three nbformat output types:
/// - `stream` (stdout/stderr) → concat `text` lines
/// - `display_data` / `execute_result` → grab `data["text/plain"]`
/// - `error` → "<error: {ename}: {evalue}>" header, no traceback
///   (tracebacks are color-coded ANSI; useless in plain text)
fn extract_text_outputs(outputs: &[Value]) -> String {
    let mut buf = String::new();
    for out in outputs {
        let otype = out.get("output_type").and_then(|v| v.as_str()).unwrap_or("");
        match otype {
            "stream" => {
                if let Some(t) = out.get("text") {
                    buf.push_str(&join_text_field(t));
                }
            }
            "display_data" | "execute_result" => {
                if let Some(text) = out
                    .get("data")
                    .and_then(|d| d.get("text/plain"))
                {
                    buf.push_str(&join_text_field(text));
                    buf.push('\n');
                }
            }
            "error" => {
                let ename = out.get("ename").and_then(|v| v.as_str()).unwrap_or("Error");
                let evalue = out.get("evalue").and_then(|v| v.as_str()).unwrap_or("");
                buf.push_str(&format!("<error: {ename}: {evalue}>\n"));
            }
            _ => {} // unknown output type — skip
        }
    }
    buf.trim_end().to_string()
}

fn join_text_field(t: &Value) -> String {
    match t {
        Value::String(s) => s.clone(),
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .collect::<Vec<_>>()
            .concat(),
        _ => String::new(),
    }
}

#[allow(dead_code)]
impl From<&'static str> for LoaderError {
    fn from(s: &'static str) -> Self { LoaderError::Other(s.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_nb(content: &str) -> NamedTempFile {
        let mut f = tempfile::Builder::new()
            .suffix(".ipynb")
            .tempfile()
            .unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f
    }

    fn sample_nb() -> &'static str {
        // Double-`#` raw delimiter — the JSON content has `"#"` inside
        // (e.g. `"# Title"`), so the single-`#` raw form would terminate early.
        r##"{
            "metadata": {
                "kernelspec": {"name": "python3", "language": "python"}
            },
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "Some intro text."]},
                {"cell_type": "code",
                 "source": "import os\nprint(os.getcwd())\n",
                 "outputs": [
                    {"output_type": "stream", "text": "/Users/x\n"}
                 ]
                },
                {"cell_type": "code", "source": ["x = 1\n", "x + 1\n"],
                 "outputs": [
                    {"output_type": "execute_result",
                     "data": {"text/plain": "2"}}
                 ]
                },
                {"cell_type": "raw", "source": "raw cell skipped by default"}
            ]
        }"##
    }

    #[test]
    fn loads_per_cell_by_default() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        // markdown + 2 code cells = 3 (raw skipped).
        assert_eq!(docs.len(), 3);
        assert_eq!(docs[0].metadata["cell_type"], "markdown");
        assert_eq!(docs[1].metadata["cell_type"], "code");
        // Markdown cell content joined from line array.
        assert!(docs[0].content.contains("Title"));
        assert!(docs[0].content.contains("Some intro text."));
        // Code cell content joined.
        assert!(docs[1].content.contains("import os"));
    }

    #[test]
    fn outputs_excluded_by_default() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        for d in &docs {
            assert!(!d.content.contains("--- output ---"));
            assert!(!d.content.contains("/Users/x"));
        }
    }

    #[test]
    fn outputs_included_when_opted_in() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).with_outputs(true).load().unwrap();
        // First code cell: stream output "/Users/x".
        let code1 = docs.iter().find(|d| d.content.contains("import os")).unwrap();
        assert!(code1.content.contains("--- output ---"));
        assert!(code1.content.contains("/Users/x"));
        // Second code cell: execute_result "2".
        let code2 = docs.iter().find(|d| d.content.contains("x = 1")).unwrap();
        assert!(code2.content.contains("--- output ---"));
        assert!(code2.content.contains("2"));
    }

    #[test]
    fn cell_index_metadata_preserved() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        // After raw-skip: indices 0 (md), 1 (code), 2 (code) — the ORIGINAL
        // notebook indices are preserved (raw cell was at index 3, skipped).
        assert_eq!(docs[0].metadata["cell_index"], 0);
        assert_eq!(docs[1].metadata["cell_index"], 1);
        assert_eq!(docs[2].metadata["cell_index"], 2);
    }

    #[test]
    fn cell_types_filter_applied() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path())
            .with_cell_types(vec!["markdown".into()])
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata["cell_type"], "markdown");
    }

    #[test]
    fn concat_mode_emits_single_document() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path())
            .concat_into_one_doc(true)
            .load()
            .unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].metadata["n_cells"], 3);
        assert!(docs[0].content.contains("# [cell 0 — markdown]"));
        assert!(docs[0].content.contains("# [cell 1 — code]"));
    }

    #[test]
    fn language_metadata_extracted_from_kernelspec() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        for d in &docs {
            assert_eq!(d.metadata["language"], "python");
        }
    }

    #[test]
    fn document_id_includes_cell_index() {
        let f = write_nb(sample_nb());
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        for (i, d) in docs.iter().enumerate() {
            let id = d.id.as_ref().unwrap();
            assert!(id.contains(&format!("#cell-{}", d.metadata["cell_index"])),
                    "id {id} should reference cell_index for doc {i}");
        }
    }

    #[test]
    fn empty_notebook_returns_empty_doc_list() {
        let f = write_nb(r#"{"cells": [], "metadata": {}}"#);
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn invalid_json_surfaces_loader_error() {
        let f = write_nb("not valid json");
        let err = JupyterNotebookLoader::new(f.path()).load().unwrap_err();
        assert!(matches!(err, LoaderError::Json(_)));
    }

    #[test]
    fn error_output_surfaces_as_text_marker() {
        let nb = r#"{
            "metadata": {},
            "cells": [
                {"cell_type": "code", "source": "1/0",
                 "outputs": [
                    {"output_type": "error",
                     "ename": "ZeroDivisionError",
                     "evalue": "division by zero",
                     "traceback": ["...colored ANSI gibberish..."]}
                 ]
                }
            ]
        }"#;
        let f = write_nb(nb);
        let docs = JupyterNotebookLoader::new(f.path()).with_outputs(true).load().unwrap();
        assert_eq!(docs.len(), 1);
        assert!(docs[0].content.contains("<error: ZeroDivisionError: division by zero>"));
        // Traceback NOT included — ANSI gibberish would inflate token count.
        assert!(!docs[0].content.contains("ANSI"));
    }

    #[test]
    fn source_as_array_of_lines_joins_correctly() {
        // nbformat 4: source is array of lines, each WITH its trailing \n.
        let nb = r#"{
            "metadata": {},
            "cells": [
                {"cell_type": "code", "source": ["line1\n", "line2\n", "line3"]}
            ]
        }"#;
        let f = write_nb(nb);
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        assert_eq!(docs[0].content, "line1\nline2\nline3");
    }

    #[test]
    fn empty_source_does_not_panic() {
        let nb = r#"{"metadata": {}, "cells": [{"cell_type": "code"}]}"#;
        let f = write_nb(nb);
        let docs = JupyterNotebookLoader::new(f.path()).load().unwrap();
        assert_eq!(docs.len(), 1);
        assert_eq!(docs[0].content, "");
    }
}

//! Definition-boundary code splitter — keeps whole functions/classes
//! together when they fit, falls back to char-splitting only for oversize
//! single definitions.
//!
//! # vs `RecursiveCharacterSplitter::for_language`
//!
//! - **Recursive** uses separator-priority fallthrough: tries `\nclass `
//!   first, then `\ndef `, then `\n\n`, etc. Works well, but can break
//!   mid-function when forced to honor `chunk_size`.
//! - **`CodeSplitter`** explicitly identifies definition START lines and
//!   treats each definition (def-to-def-or-EOF span) as one atomic unit.
//!   Smaller definitions get GROUPED greedy-pack-style up to `chunk_size`;
//!   oversize definitions (a 3KB function with `chunk_size=1000`) get
//!   recursively split via the existing recursive splitter as fallback.
//!
//! Trade-off: heavier per-text scan (regex + line iteration), cleaner
//! semantic boundaries. Recommended for code RAG / code review where
//! function-level provenance matters more than uniform chunk sizes.
//!
//! Languages: Python, Rust, JavaScript, TypeScript, Go, Java, Cpp, Ruby,
//! PHP. Reuses the `Language` enum from `recursive.rs` for consistency.
//! No tree-sitter dep — regex-based; ~95% accurate on idiomatic code,
//! same blind spots as recursive (string literals containing `def `, etc).

use crate::recursive::{Language, RecursiveCharacterSplitter};
use crate::Splitter;
use regex::Regex;
use std::sync::OnceLock;

pub struct CodeSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    language: Language,
}

impl CodeSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize, language: Language) -> Self {
        // Same clamping as RecursiveCharacterSplitter — non-zero, overlap < size.
        let chunk_size = chunk_size.max(1);
        let chunk_overlap = chunk_overlap.min(chunk_size.saturating_sub(1));
        Self { chunk_size, chunk_overlap, language }
    }

    /// Regex matching the START of a definition for `language`. Built lazily
    /// per-language; cached forever (these patterns never change at runtime).
    fn def_start_re(language: Language) -> &'static Regex {
        // Cache per-variant. There are 11 Language variants; one OnceLock per
        // language keeps initialization simple.
        macro_rules! cached {
            ($lang:expr, $pat:expr) => {{
                static CELL: OnceLock<Regex> = OnceLock::new();
                CELL.get_or_init(|| Regex::new($pat).expect("valid regex"))
            }};
        }
        match language {
            // Python: def or class at line start (any indent), possibly preceded
            // by `async ` or `@decorator` lines (decorators handled by grouping
            // below — we still split AT the def, decorators come along by
            // sitting directly above).
            Language::Python => cached!(Language::Python, r"(?m)^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+\w"),
            Language::Rust => cached!(
                Language::Rust,
                r"(?m)^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?(?:async[ \t]+)?(?:unsafe[ \t]+)?(?:fn|struct|enum|trait|impl|mod|union|type|const|static)[ \t]+\w"
            ),
            Language::JavaScript | Language::TypeScript => cached!(
                Language::JavaScript,
                r"(?m)^[ \t]*(?:export[ \t]+)?(?:default[ \t]+)?(?:async[ \t]+)?(?:function[ \t]*\*?|class|const|let|var)[ \t]+\w"
            ),
            Language::Go => cached!(Language::Go, r"(?m)^[ \t]*(?:func|type|var|const)[ \t]+(?:\([^)]*\)[ \t]+)?\w"),
            Language::Java => cached!(
                Language::Java,
                r"(?m)^[ \t]*(?:public|protected|private|static|final|abstract)[ \t]+[\w<>\[\],?\s]+\w[ \t]*[\(\{]"
            ),
            Language::Cpp => cached!(
                Language::Cpp,
                r"(?m)^[ \t]*(?:class|struct|namespace|template|void|int|float|double|bool|char|auto|long|short|unsigned)[ \t]+\w"
            ),
            Language::Ruby => cached!(Language::Ruby, r"(?m)^[ \t]*(?:def|class|module)[ \t]+\w"),
            Language::Php => cached!(Language::Php, r"(?m)^[ \t]*(?:function|class|interface|trait)[ \t]+\w"),
            // Markdown / Html aren't really "code" in this sense — fall back
            // to recursive char splitting via the trait impl below. Use a
            // never-matching regex so all input lands in a single fallback
            // block (which itself gets recursive-split).
            Language::Markdown | Language::Html => cached!(Language::Markdown, r"(?-u)^\x00$NEVER^"),
        }
    }

    /// Return the byte offsets of every definition-start match. Always
    /// includes 0 as the implicit start (preamble before first def).
    fn definition_offsets(&self, text: &str) -> Vec<usize> {
        let re = Self::def_start_re(self.language);
        let mut offsets = vec![0usize];
        for m in re.find_iter(text) {
            if m.start() != 0 {
                offsets.push(m.start());
            }
        }
        offsets
    }

    /// Greedy-pack consecutive definitions into chunks under `chunk_size`.
    /// Oversize single definitions get split via the recursive char splitter.
    fn pack_definitions(&self, text: &str, offsets: &[usize]) -> Vec<String> {
        let mut chunks = Vec::new();
        let mut current = String::new();
        let bytes = text.as_bytes();

        // Build (start, end) spans: each definition spans from offsets[i] to
        // offsets[i+1]; the last span runs from offsets[N-1] to text.len().
        let mut spans: Vec<(usize, usize)> = offsets
            .windows(2)
            .map(|w| (w[0], w[1]))
            .collect();
        if let Some(&last) = offsets.last() {
            if last < text.len() {
                spans.push((last, text.len()));
            }
        }

        for (start, end) in spans {
            if start >= end || end > bytes.len() {
                continue;
            }
            let slice = &text[start..end];
            // If slice itself exceeds chunk_size, flush current + recursively
            // split this slice.
            if slice.len() > self.chunk_size {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                let recursive = RecursiveCharacterSplitter::for_language(
                    self.chunk_size,
                    self.chunk_overlap,
                    self.language,
                );
                chunks.extend(recursive.split_text(slice));
                continue;
            }
            // Greedy-pack: append if it fits, else flush + start fresh.
            if current.len() + slice.len() <= self.chunk_size {
                current.push_str(slice);
            } else {
                if !current.is_empty() {
                    chunks.push(std::mem::take(&mut current));
                }
                current = slice.to_string();
            }
        }
        if !current.is_empty() {
            chunks.push(current);
        }
        chunks
    }
}

impl Splitter for CodeSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        if text.is_empty() {
            return Vec::new();
        }
        let offsets = self.definition_offsets(text);
        // No definitions found (markdown/html, or edge code) → fall back
        // to recursive char splitting directly.
        if offsets.len() <= 1 {
            let recursive = RecursiveCharacterSplitter::for_language(
                self.chunk_size,
                self.chunk_overlap,
                self.language,
            );
            return recursive.split_text(text);
        }
        self.pack_definitions(text, &offsets)
            .into_iter()
            .filter(|c| !c.trim().is_empty())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_splits_at_def_and_class_boundaries() {
        let src = "def foo():\n    return 1\n\ndef bar():\n    return 2\n\nclass Baz:\n    def method(self):\n        return 3\n";
        let s = CodeSplitter::new(40, 0, Language::Python);
        let chunks = s.split_text(src);
        // Three definitions, none over 40 chars individually but their
        // concatenation is over → at least 2 chunks, each starting at a
        // def or class.
        assert!(chunks.len() >= 2, "expected >=2 chunks, got {}: {:?}", chunks.len(), chunks);
        for c in &chunks {
            assert!(c.contains("def ") || c.contains("class "), "chunk lacks def/class: {c}");
        }
    }

    #[test]
    fn rust_splits_at_fn_impl_struct_boundaries() {
        let src = "fn one() -> i32 { 1 }\n\nimpl Foo {\n    fn two(&self) {}\n}\n\nstruct Bar;\n\nenum E { A, B }\n";
        let s = CodeSplitter::new(50, 0, Language::Rust);
        let chunks = s.split_text(src);
        assert!(chunks.len() >= 2);
        // Each chunk should start at some Rust definition keyword (possibly
        // indented — the splitter treats nested fn-in-impl as its own def).
        for c in &chunks {
            let first = c.lines().next().unwrap_or("").trim_start();
            assert!(
                first.starts_with("fn ") || first.starts_with("impl ")
                    || first.starts_with("struct ") || first.starts_with("enum "),
                "chunk first line not a def: {first:?}"
            );
        }
    }

    #[test]
    fn small_definitions_get_grouped_under_chunk_size() {
        let src = "def a():\n    pass\n\ndef b():\n    pass\n\ndef c():\n    pass\n";
        let s = CodeSplitter::new(200, 0, Language::Python);
        let chunks = s.split_text(src);
        // All three small defs fit in 200 → one chunk.
        assert_eq!(chunks.len(), 1, "got {} chunks: {:?}", chunks.len(), chunks);
        assert!(chunks[0].contains("def a"));
        assert!(chunks[0].contains("def b"));
        assert!(chunks[0].contains("def c"));
    }

    #[test]
    fn oversize_definition_falls_back_to_recursive_split() {
        let body = "    print('x')\n".repeat(50);
        let src = format!("def huge():\n{body}\n");
        let s = CodeSplitter::new(80, 0, Language::Python);
        let chunks = s.split_text(&src);
        // The single huge def must split into multiple sub-chunks.
        assert!(chunks.len() > 1, "huge def must split: got {}", chunks.len());
    }

    #[test]
    fn js_splits_at_function_class_const() {
        let src = "function alpha() { return 1; }\n\nclass Beta {}\n\nconst gamma = () => 2;\n\nexport function delta() {}\n";
        let s = CodeSplitter::new(40, 0, Language::JavaScript);
        let chunks = s.split_text(src);
        // 4 defs, each under 40 chars; expect grouping but with bounded chunks.
        assert!(chunks.len() >= 2);
        let joined = chunks.join("\n");
        for keyword in ["function alpha", "class Beta", "const gamma", "delta"] {
            assert!(joined.contains(keyword), "lost {keyword}");
        }
    }

    #[test]
    fn empty_input_returns_empty() {
        let s = CodeSplitter::new(100, 0, Language::Python);
        assert_eq!(s.split_text(""), Vec::<String>::new());
    }

    #[test]
    fn no_definitions_falls_back_to_recursive() {
        // Pure expression script — no def/class.
        let src = "x = 1\ny = 2\nprint(x + y)\n";
        let s = CodeSplitter::new(100, 0, Language::Python);
        let chunks = s.split_text(src);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("print"));
    }

    #[test]
    fn preamble_before_first_def_preserved() {
        let src = "import os\nimport sys\n\ndef main():\n    pass\n";
        let s = CodeSplitter::new(200, 0, Language::Python);
        let chunks = s.split_text(src);
        // Imports come along with the def in one chunk.
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].contains("import os"));
        assert!(chunks[0].contains("def main"));
    }

    #[test]
    fn go_splits_at_func_type_var() {
        let src = "func one() {}\n\ntype X struct {}\n\nvar y = 1\n\nfunc two() {}\n";
        let s = CodeSplitter::new(20, 0, Language::Go);
        let chunks = s.split_text(src);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn ruby_splits_at_def_class_module() {
        let src = "def foo\n  1\nend\n\nclass Bar\nend\n\nmodule Baz\nend\n";
        let s = CodeSplitter::new(20, 0, Language::Ruby);
        let chunks = s.split_text(src);
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn document_split_preserves_metadata() {
        use litgraph_core::Document;
        let mut doc = Document::new("def a():\n    pass\n\ndef b():\n    pass\n");
        doc.id = Some("src.py".into());
        doc.metadata.insert("language".into(), serde_json::json!("python"));
        let s = CodeSplitter::new(15, 0, Language::Python);
        let chunks = s.split_document(&doc);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert_eq!(c.metadata.get("language"), Some(&serde_json::json!("python")));
            assert_eq!(c.metadata.get("source_id"), Some(&serde_json::json!("src.py")));
        }
    }

    #[test]
    fn chunk_overlap_clamped_to_size_minus_one() {
        let s = CodeSplitter::new(10, 100, Language::Python);
        assert_eq!(s.chunk_overlap, 9);
    }
}

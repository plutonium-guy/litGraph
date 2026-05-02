//! Recursive character splitter — splits on the most specific separator that keeps
//! chunks under the size budget, then falls back to coarser ones. Mirrors LangChain's
//! `RecursiveCharacterTextSplitter` semantics.

use crate::Splitter;

#[derive(Debug, Clone)]
pub struct RecursiveCharacterSplitter {
    pub chunk_size: usize,
    pub chunk_overlap: usize,
    pub separators: Vec<String>,
}

impl Default for RecursiveCharacterSplitter {
    fn default() -> Self {
        Self {
            chunk_size: 1000,
            chunk_overlap: 200,
            separators: vec!["\n\n".into(), "\n".into(), " ".into(), "".into()],
        }
    }
}

impl RecursiveCharacterSplitter {
    pub fn new(chunk_size: usize, chunk_overlap: usize) -> Self {
        Self {
            chunk_size: chunk_size.max(1),
            chunk_overlap: chunk_overlap.min(chunk_size.saturating_sub(1)),
            ..Default::default()
        }
    }

    pub fn with_separators<I, S>(mut self, seps: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.separators = seps.into_iter().map(Into::into).collect();
        self
    }

    /// Construct a splitter pre-configured with separator priorities tuned for
    /// a given source-code language. The separators run from coarsest semantic
    /// boundary (e.g. class definition) to finest (whitespace, then byte).
    /// Falls back to character-level slicing for files with no recognizable
    /// boundaries — just like the default text splitter.
    ///
    /// Mirrors LangChain's `RecursiveCharacterTextSplitter.from_language(...)`
    /// at a fraction of the LOC. Tree-sitter is overkill for "find a sensible
    /// place to break a 1000-char window" — string anchors win on simplicity
    /// AND latency for the chunk sizes RAG actually uses.
    pub fn for_language(chunk_size: usize, chunk_overlap: usize, lang: Language) -> Self {
        Self::new(chunk_size, chunk_overlap)
            .with_separators(lang.separators().iter().copied())
    }
}

/// Programming languages with curated separator priorities for the recursive
/// splitter. Add new ones by extending `separators()` — no other plumbing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Python,
    Rust,
    JavaScript,
    TypeScript,
    Go,
    Java,
    Cpp,
    Ruby,
    Php,
    /// Markdown — different from `MarkdownHeaderSplitter` (which is structural,
    /// returns one chunk per heading). Use this when you want size-bounded
    /// chunks that *prefer* breaking at headings/blocks but won't split mid-
    /// paragraph if it fits.
    Markdown,
    /// Plain HTML — split on tag boundaries (block elements first).
    Html,
}

impl Language {
    pub fn separators(self) -> &'static [&'static str] {
        match self {
            Language::Python => &[
                "\nclass ", "\ndef ", "\n\tdef ",   // top-level + nested defs
                "\n\n", "\n", " ", "",
            ],
            Language::Rust => &[
                "\nfn ", "\nimpl ", "\nstruct ", "\nenum ", "\ntrait ", "\nmod ",
                "\n\n", "\n", " ", "",
            ],
            Language::JavaScript | Language::TypeScript => &[
                "\nfunction ", "\nclass ", "\nconst ", "\nlet ", "\nvar ",
                "\nexport ", "\nimport ",
                "\n\n", "\n", " ", "",
            ],
            Language::Go => &[
                "\nfunc ", "\ntype ", "\nvar ", "\nconst ", "\npackage ",
                "\n\n", "\n", " ", "",
            ],
            Language::Java => &[
                "\nclass ", "\ninterface ", "\nenum ",
                "\npublic ", "\nprotected ", "\nprivate ", "\nstatic ",
                "\n\n", "\n", " ", "",
            ],
            Language::Cpp => &[
                "\nclass ", "\nstruct ", "\nnamespace ", "\ntemplate ",
                "\nvoid ", "\nint ", "\nfloat ", "\ndouble ", "\nbool ",
                "\n\n", "\n", " ", "",
            ],
            Language::Ruby => &[
                "\ndef ", "\nclass ", "\nmodule ",
                "\n\n", "\n", " ", "",
            ],
            Language::Php => &[
                "\nfunction ", "\nclass ", "\ninterface ", "\ntrait ",
                "\n\n", "\n", " ", "",
            ],
            Language::Markdown => &[
                "\n# ", "\n## ", "\n### ", "\n#### ", "\n##### ", "\n###### ",
                "\n```\n", "\n\n", "\n", " ", "",
            ],
            Language::Html => &[
                "<body", "<div", "<section", "<article", "<header", "<footer",
                "<nav", "<aside", "<table", "<ul", "<ol", "<p>", "<br",
                "\n\n", "\n", " ", "",
            ],
        }
    }
}

impl RecursiveCharacterSplitter {
    fn split_recursive(&self, text: &str, sep_idx: usize) -> Vec<String> {
        if text.len() <= self.chunk_size {
            return vec![text.to_string()];
        }

        // Find first separator that actually appears; skip separators not present.
        if sep_idx >= self.separators.len() {
            return hard_split(text, self.chunk_size, self.chunk_overlap);
        }
        let s = &self.separators[sep_idx];
        let sep = if s.is_empty() || text.contains(s.as_str()) {
            s.clone()
        } else {
            return self.split_recursive(text, sep_idx + 1);
        };

        if sep.is_empty() {
            return hard_split(text, self.chunk_size, self.chunk_overlap);
        }

        let parts: Vec<&str> = text.split(&sep).collect();
        let mut buf: Vec<String> = Vec::new();
        let mut cur = String::new();

        for (i, part) in parts.iter().enumerate() {
            let with_sep = if i == 0 { part.to_string() } else { format!("{sep}{part}") };
            if cur.len() + with_sep.len() > self.chunk_size && !cur.is_empty() {
                buf.push(std::mem::take(&mut cur));
            }
            if with_sep.len() > self.chunk_size {
                // Still too big at this separator level → recurse with next one.
                let mut deeper = self.split_recursive(&with_sep, sep_idx + 1);
                if let Some(first) = deeper.first().cloned() {
                    if cur.len() + first.len() <= self.chunk_size {
                        cur.push_str(&first);
                        deeper.remove(0);
                    } else if !cur.is_empty() {
                        buf.push(std::mem::take(&mut cur));
                    }
                }
                buf.extend(deeper);
            } else {
                cur.push_str(&with_sep);
            }
        }
        if !cur.is_empty() { buf.push(cur); }

        apply_overlap(buf, self.chunk_overlap)
    }
}

impl Splitter for RecursiveCharacterSplitter {
    fn split_text(&self, text: &str) -> Vec<String> {
        self.split_recursive(text, 0)
            .into_iter()
            .filter(|c| !c.trim().is_empty())
            .collect()
    }
}

/// Last-resort: slice by byte boundaries (respecting UTF-8 via `char_indices`).
fn hard_split(text: &str, size: usize, overlap: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    if chars.is_empty() { return chunks; }
    let len = text.len();
    let mut start = 0usize;
    let stride = size.saturating_sub(overlap).max(1);
    while start < len {
        let end = (start + size).min(len);
        // Align end to a char boundary going backwards.
        let end = text[..end].char_indices().last().map(|(i, c)| i + c.len_utf8()).unwrap_or(end);
        chunks.push(text[start..end].to_string());
        if end == len { break; }
        start += stride;
        // Align start to char boundary.
        while start < len && !text.is_char_boundary(start) { start += 1; }
    }
    chunks
}

fn apply_overlap(chunks: Vec<String>, overlap: usize) -> Vec<String> {
    if overlap == 0 || chunks.len() < 2 { return chunks; }
    let mut out = Vec::with_capacity(chunks.len());
    for (i, c) in chunks.iter().enumerate() {
        if i == 0 {
            out.push(c.clone());
        } else {
            let prev = &chunks[i - 1];
            let take = overlap.min(prev.len());
            // Align take to char boundary.
            let mut take = take;
            while take > 0 && !prev.is_char_boundary(prev.len() - take) { take -= 1; }
            let tail = &prev[prev.len() - take..];
            out.push(format!("{tail}{c}"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_text_is_one_chunk() {
        let s = RecursiveCharacterSplitter::new(100, 10);
        let r = s.split_text("hello world");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn splits_on_paragraphs_first() {
        let text = "para1\n\npara2\n\npara3";
        let s = RecursiveCharacterSplitter::new(8, 0);
        let r = s.split_text(text);
        assert!(r.len() >= 2);
    }

    #[test]
    fn handles_unicode() {
        let text = "日本語テキスト ".repeat(200);
        let s = RecursiveCharacterSplitter::new(100, 10);
        let r = s.split_text(&text);
        assert!(!r.is_empty());
        for c in &r { assert!(c.is_char_boundary(0)); assert!(c.is_char_boundary(c.len())); }
    }

    #[test]
    fn for_language_python_breaks_at_def_boundaries() {
        let src = r#"
def foo():
    return 1

def bar():
    return 2

def baz():
    return 3
"#;
        let s = RecursiveCharacterSplitter::for_language(40, 0, Language::Python);
        let chunks = s.split_text(src);
        assert!(chunks.len() >= 2, "got {} chunks: {:?}", chunks.len(), chunks);
        // At least one chunk should still contain a complete `def` block.
        assert!(chunks.iter().any(|c| c.contains("def foo")));
        assert!(chunks.iter().any(|c| c.contains("def baz")));
    }

    #[test]
    fn for_language_rust_breaks_at_fn_boundaries() {
        let src = "fn alpha() { 1 }\n\nfn beta() { 2 }\n\nfn gamma() { 3 }\n";
        let s = RecursiveCharacterSplitter::for_language(20, 0, Language::Rust);
        let chunks = s.split_text(src);
        assert!(chunks.len() >= 2);
        assert!(chunks.iter().any(|c| c.contains("fn alpha")));
        assert!(chunks.iter().any(|c| c.contains("fn gamma")));
    }

    #[test]
    fn for_language_javascript_separators_set_correctly() {
        // Don't depend on splitting behavior — just verify the separator
        // priority list got installed (the "function " anchor comes first).
        let s = RecursiveCharacterSplitter::for_language(100, 0, Language::JavaScript);
        assert_eq!(s.separators[0], "\nfunction ");
        assert!(s.separators.contains(&"\nimport ".to_string()));
    }

    #[test]
    fn each_language_has_separators_terminating_in_empty_string() {
        // Invariant: every language's separator list MUST end with ""
        // (the byte-fallback marker). Without it, hard_split never runs and
        // an extra-long token can never be broken.
        for lang in [
            Language::Python, Language::Rust, Language::JavaScript,
            Language::TypeScript, Language::Go, Language::Java, Language::Cpp,
            Language::Ruby, Language::Php, Language::Markdown, Language::Html,
        ] {
            let seps = lang.separators();
            assert!(!seps.is_empty(), "{lang:?} has no separators");
            assert_eq!(*seps.last().unwrap(), "",
                "{lang:?} last separator must be \"\" (byte-fallback)");
        }
    }
}

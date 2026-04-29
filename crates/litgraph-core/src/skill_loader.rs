//! AGENTS.md + skills directory loaders. Mirrors LangChain `deepagents`
//! memory-files / skills primitive: persistent system-prompt context that
//! lives on disk and is assembled into the agent's system message at start.
//!
//! We keep this dead simple — the value is in the *convention*, not the
//! parser. A `Skill` is `{name, description, content}` parsed from a Markdown
//! file. `name` defaults to the filename stem, `description` to the first
//! non-empty line of the body (typically a `# Heading`), `content` is the
//! full file. Optional YAML frontmatter (`---\nname:...\ndescription:...\n---`)
//! overrides those defaults.
//!
//! `SystemPromptBuilder` assembles a base prompt + AGENTS.md text + skills
//! into a single structured system message. The final message uses Markdown
//! headings so a reader (and the LLM) can navigate the sections, but no
//! Markdown is *required* — the assembler just concatenates with separators.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A single named skill loaded from disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Skill {
    pub name: String,
    pub description: String,
    pub content: String,
    /// Source path on disk — useful for surfacing reload semantics later.
    pub source: Option<PathBuf>,
}

/// Lightweight YAML-frontmatter parser. Recognises `---\nkey: value\n...\n---`
/// at the top of a file. We intentionally avoid pulling a YAML dep — only
/// the two well-known keys (`name`, `description`) need to be extracted, and
/// values are scalar strings. Anything more elaborate stays in the body.
fn split_frontmatter(text: &str) -> (Option<&str>, &str) {
    let trimmed = text.trim_start_matches(['\u{feff}', '\r', '\n']);
    if !trimmed.starts_with("---") {
        return (None, text);
    }
    let after_open = match trimmed.strip_prefix("---") {
        Some(rest) => rest.trim_start_matches('\r').trim_start_matches('\n'),
        None => return (None, text),
    };
    if let Some(end_idx) = after_open.find("\n---") {
        let fm = &after_open[..end_idx];
        let after_close = &after_open[end_idx + 4..];
        let body = after_close.trim_start_matches('\r').trim_start_matches('\n');
        (Some(fm), body)
    } else {
        (None, text)
    }
}

fn parse_frontmatter_kv(fm: &str, key: &str) -> Option<String> {
    for line in fm.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case(key) {
                let v = v.trim();
                let v = v.trim_matches(|c| c == '"' || c == '\'');
                if !v.is_empty() {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

fn first_nonempty_line(body: &str) -> String {
    for line in body.lines() {
        let l = line.trim_start_matches('#').trim();
        if !l.is_empty() {
            return l.to_string();
        }
    }
    String::new()
}

/// Read a single AGENTS.md / CLAUDE.md / system-prompt file. Returns `None`
/// if the file does not exist (non-fatal — agents may legitimately have no
/// memory file). Other I/O errors propagate.
pub fn load_agents_md(path: impl AsRef<Path>) -> io::Result<Option<String>> {
    let p = path.as_ref();
    match fs::read_to_string(p) {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e),
    }
}

/// Read every `*.md` file in `dir` as a `Skill`. Sorted by filename for
/// stable output. Files starting with `.` are skipped. If `dir` does not
/// exist, returns an empty list.
pub fn load_skills_dir(dir: impl AsRef<Path>) -> io::Result<Vec<Skill>> {
    let dir = dir.as_ref();
    if !dir.exists() {
        return Ok(Vec::new());
    }
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            let is_md = p
                .extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("md"))
                .unwrap_or(false);
            let is_hidden = p
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.starts_with('.'))
                .unwrap_or(false);
            is_md && !is_hidden
        })
        .collect();
    entries.sort();

    let mut out = Vec::with_capacity(entries.len());
    for path in entries {
        let raw = fs::read_to_string(&path)?;
        let (fm, body) = split_frontmatter(&raw);
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("skill")
            .to_string();
        let name = fm
            .and_then(|f| parse_frontmatter_kv(f, "name"))
            .unwrap_or(stem);
        let description = fm
            .and_then(|f| parse_frontmatter_kv(f, "description"))
            .unwrap_or_else(|| first_nonempty_line(body));
        out.push(Skill {
            name,
            description,
            content: body.to_string(),
            source: Some(path),
        });
    }
    Ok(out)
}

/// Builder for an assembled system prompt. Sections render in this order:
/// 1. `base` (the bare system prompt)
/// 2. AGENTS.md memory file contents
/// 3. Each skill (in the order added)
///
/// Sections are separated by a blank line plus a Markdown `## Section` header,
/// matching the convention used by deepagents and similar harnesses. Empty
/// inputs are skipped — no header without a body.
#[derive(Debug, Clone, Default)]
pub struct SystemPromptBuilder {
    base: String,
    agents_md: Option<String>,
    skills: Vec<Skill>,
    extra_sections: Vec<(String, String)>,
}

impl SystemPromptBuilder {
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            ..Self::default()
        }
    }

    pub fn with_agents_md(mut self, body: impl Into<String>) -> Self {
        let s: String = body.into();
        if !s.trim().is_empty() {
            self.agents_md = Some(s);
        }
        self
    }

    pub fn with_skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    pub fn with_skills(mut self, skills: impl IntoIterator<Item = Skill>) -> Self {
        self.skills.extend(skills);
        self
    }

    pub fn with_section(mut self, heading: impl Into<String>, body: impl Into<String>) -> Self {
        let h = heading.into();
        let b: String = body.into();
        if !b.trim().is_empty() {
            self.extra_sections.push((h, b));
        }
        self
    }

    pub fn build(&self) -> String {
        let mut out = String::new();
        if !self.base.trim().is_empty() {
            out.push_str(self.base.trim_end());
            out.push('\n');
        }
        if let Some(a) = &self.agents_md {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("## Memory (AGENTS.md)\n\n");
            out.push_str(a.trim());
            out.push('\n');
        }
        if !self.skills.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str("## Skills\n\n");
            out.push_str(
                "Available reusable skills — invoke when their description matches the task.\n\n",
            );
            for s in &self.skills {
                out.push_str(&format!(
                    "### {}\n\n{}\n\n{}\n\n",
                    s.name,
                    s.description.trim(),
                    s.content.trim()
                ));
            }
        }
        for (h, b) in &self.extra_sections {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(&format!("## {}\n\n{}\n", h, b.trim()));
        }
        out.trim_end().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn td() -> TempDir {
        TempDir::new().expect("tempdir")
    }

    #[test]
    fn split_frontmatter_handles_no_frontmatter() {
        let (fm, body) = split_frontmatter("hello\nworld");
        assert!(fm.is_none());
        assert_eq!(body, "hello\nworld");
    }

    #[test]
    fn split_frontmatter_extracts_yaml_block() {
        let raw = "---\nname: foo\ndescription: bar\n---\nbody here\n";
        let (fm, body) = split_frontmatter(raw);
        assert!(fm.is_some());
        assert!(fm.unwrap().contains("name: foo"));
        assert_eq!(body.trim(), "body here");
    }

    #[test]
    fn parse_frontmatter_kv_strips_quotes_and_whitespace() {
        let fm = "name: \"My Skill\"\ndescription: 'helps you'";
        assert_eq!(
            parse_frontmatter_kv(fm, "name").as_deref(),
            Some("My Skill")
        );
        assert_eq!(
            parse_frontmatter_kv(fm, "description").as_deref(),
            Some("helps you")
        );
    }

    #[test]
    fn parse_frontmatter_kv_is_case_insensitive_on_keys() {
        let fm = "Name: foo";
        assert_eq!(parse_frontmatter_kv(fm, "name").as_deref(), Some("foo"));
    }

    #[test]
    fn first_nonempty_line_strips_markdown_hash() {
        assert_eq!(first_nonempty_line("\n\n# Title\nbody"), "Title");
        assert_eq!(first_nonempty_line("plain"), "plain");
    }

    #[test]
    fn load_agents_md_returns_none_for_missing_file() {
        let d = td();
        let path = d.path().join("AGENTS.md");
        assert!(load_agents_md(&path).unwrap().is_none());
    }

    #[test]
    fn load_agents_md_reads_existing_file() {
        let d = td();
        let path = d.path().join("AGENTS.md");
        fs::write(&path, "remember: be terse").unwrap();
        let got = load_agents_md(&path).unwrap().unwrap();
        assert_eq!(got, "remember: be terse");
    }

    #[test]
    fn load_skills_dir_returns_empty_when_missing() {
        let d = td();
        let dir = d.path().join("skills");
        assert!(load_skills_dir(&dir).unwrap().is_empty());
    }

    #[test]
    fn load_skills_dir_filters_to_md_and_sorts() {
        let d = td();
        let dir = d.path().join("skills");
        fs::create_dir(&dir).unwrap();
        fs::write(dir.join("b.md"), "# Beta\nbeta body").unwrap();
        fs::write(dir.join("a.md"), "# Alpha\nalpha body").unwrap();
        fs::write(dir.join("c.txt"), "ignored").unwrap();
        fs::write(dir.join(".hidden.md"), "ignored").unwrap();
        let skills = load_skills_dir(&dir).unwrap();
        assert_eq!(skills.len(), 2);
        assert_eq!(skills[0].name, "a");
        assert_eq!(skills[0].description, "Alpha");
        assert_eq!(skills[1].name, "b");
    }

    #[test]
    fn frontmatter_overrides_filename_and_first_line() {
        let d = td();
        let dir = d.path().join("skills");
        fs::create_dir(&dir).unwrap();
        fs::write(
            dir.join("file.md"),
            "---\nname: Custom\ndescription: Custom desc\n---\n# Heading\nbody",
        )
        .unwrap();
        let skills = load_skills_dir(&dir).unwrap();
        assert_eq!(skills[0].name, "Custom");
        assert_eq!(skills[0].description, "Custom desc");
        assert!(skills[0].content.contains("body"));
    }

    #[test]
    fn builder_emits_sections_in_order() {
        let s = SystemPromptBuilder::new("base prompt")
            .with_agents_md("memory body")
            .with_skill(Skill {
                name: "s1".into(),
                description: "desc".into(),
                content: "skill body".into(),
                source: None,
            })
            .build();
        let base_pos = s.find("base prompt").unwrap();
        let mem_pos = s.find("## Memory (AGENTS.md)").unwrap();
        let skills_pos = s.find("## Skills").unwrap();
        assert!(base_pos < mem_pos);
        assert!(mem_pos < skills_pos);
        assert!(s.contains("### s1"));
    }

    #[test]
    fn builder_skips_empty_sections() {
        let s = SystemPromptBuilder::new("base").build();
        assert!(!s.contains("## Memory"));
        assert!(!s.contains("## Skills"));
    }

    #[test]
    fn builder_skips_whitespace_only_inputs() {
        let s = SystemPromptBuilder::new("base")
            .with_agents_md("   \n\n   ")
            .build();
        assert!(!s.contains("## Memory"));
    }

    #[test]
    fn builder_extra_section_appears_at_end() {
        let s = SystemPromptBuilder::new("base")
            .with_section("Custom", "extra body")
            .build();
        assert!(s.contains("## Custom"));
        assert!(s.contains("extra body"));
    }

    #[test]
    fn builder_output_is_deterministic() {
        let mk = || {
            SystemPromptBuilder::new("b")
                .with_agents_md("m")
                .with_skill(Skill {
                    name: "n".into(),
                    description: "d".into(),
                    content: "c".into(),
                    source: None,
                })
                .build()
        };
        assert_eq!(mk(), mk());
    }
}

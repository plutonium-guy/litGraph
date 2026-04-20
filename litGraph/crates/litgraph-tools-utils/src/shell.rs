//! `ShellTool` — execute commands from a strict allowlist with a working-dir
//! restriction, output size cap, and timeout.
//!
//! # Why allowlist-only
//!
//! There is no safe way to give an LLM-driven agent unrestricted shell access.
//! "Just ask the model nicely" doesn't work — every prompt-injection paper
//! ever written says it doesn't. So this tool requires the operator to declare,
//! at construction time, the exact set of programs the agent may invoke. No
//! shell expansion, no `sh -c`, no globbing — we call `Command::new(prog)`
//! directly and pass arg vectors verbatim, eliminating the standard injection
//! surface (`; rm -rf /` and friends are inert because no shell parses them).
//!
//! Even with the allowlist, the agent can still make a mess of things inside
//! the working directory or `cargo install` something nasty if `cargo` is
//! allowed. Operators should pair this with a chroot/container/jail in any
//! production deployment.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{Value, json};
use tokio::process::Command;

#[derive(Clone, Debug)]
pub struct ShellTool {
    /// Programs the agent is allowed to run. Compared against the literal
    /// `command` arg from the tool call — no PATH resolution, no symlink
    /// following at the comparison level (we leave that to the OS at exec
    /// time). Empty allowlist = nothing runs.
    pub allowed_commands: HashSet<String>,
    /// Working directory the child inherits. Mandatory — unset would let the
    /// agent's cwd be wherever this Rust process happens to be running, which
    /// is rarely what the operator wants.
    pub working_dir: PathBuf,
    /// Hard timeout per call. Default 30s.
    pub timeout: Duration,
    /// Hard cap on combined stdout+stderr bytes returned (each, separately).
    /// Default 64 KiB. Output above this is truncated with a marker line.
    pub max_output_bytes: usize,
}

impl ShellTool {
    /// Build a ShellTool. `working_dir` MUST exist and be a directory
    /// (validated immediately). `allowed_commands` is normalized into a set;
    /// duplicates are deduped.
    pub fn new<P, I, S>(working_dir: P, allowed_commands: I) -> Result<Self>
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let wd = working_dir.as_ref().to_path_buf();
        let meta = std::fs::metadata(&wd)
            .map_err(|e| Error::other(format!("ShellTool: cannot stat working_dir {}: {e}", wd.display())))?;
        if !meta.is_dir() {
            return Err(Error::other(format!("ShellTool: working_dir {} is not a directory", wd.display())));
        }
        Ok(Self {
            allowed_commands: allowed_commands.into_iter().map(Into::into).collect(),
            working_dir: wd,
            timeout: Duration::from_secs(30),
            max_output_bytes: 64 * 1024,
        })
    }

    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_max_output_bytes(mut self, n: usize) -> Self {
        self.max_output_bytes = n;
        self
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn schema(&self) -> ToolSchema {
        let allowed: Vec<&str> = self.allowed_commands.iter().map(|s| s.as_str()).collect();
        ToolSchema {
            name: "shell".into(),
            description: format!(
                "Execute a command from a fixed allowlist. Args are passed verbatim — no shell expansion, no globbing, no piping. Allowed commands: {}.",
                if allowed.is_empty() { "(none)".into() }
                else { let mut a = allowed; a.sort(); a.join(", ") }
            ),
            parameters: json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "description": "Program name (must match the allowlist)." },
                    "args": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Argument vector. Empty array = no args."
                    }
                },
                "required": ["command"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let cmd = args.get("command").and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("shell: missing `command`"))?;
        if !self.allowed_commands.contains(cmd) {
            return Err(Error::invalid(format!(
                "shell: '{cmd}' is not in the allowlist ({} entries)",
                self.allowed_commands.len()
            )));
        }
        let arg_vec: Vec<String> = args.get("args")
            .and_then(|v| v.as_array())
            .map(|a| a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect())
            .unwrap_or_default();

        let mut child = Command::new(cmd);
        child
            .args(&arg_vec)
            .current_dir(&self.working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let started = std::time::Instant::now();
        let output_fut = async {
            let child = child.spawn().map_err(|e| Error::other(format!("spawn {cmd}: {e}")))?;
            child.wait_with_output().await.map_err(|e| Error::other(format!("wait: {e}")))
        };
        let output = match tokio::time::timeout(self.timeout, output_fut).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(Error::other(format!(
                "shell: '{cmd}' timed out after {:?}", self.timeout
            ))),
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;
        let exit_code = output.status.code();

        let stdout = truncate_for_agent(&output.stdout, self.max_output_bytes);
        let stderr = truncate_for_agent(&output.stderr, self.max_output_bytes);

        Ok(json!({
            "command": cmd,
            "args": arg_vec,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "elapsed_ms": elapsed_ms,
        }))
    }
}

/// Lossy-UTF-8 decode + cap to N bytes. Truncation is loud (the agent needs
/// to know it didn't get the full picture) and the cap is byte-level so we
/// can't blow the agent's context budget on a noisy process.
fn truncate_for_agent(bytes: &[u8], max: usize) -> String {
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let mut head = String::from_utf8_lossy(&bytes[..max]).into_owned();
    head.push_str(&format!(
        "\n... [truncated {} bytes; raised cap = {} → use a more targeted command]",
        bytes.len() - max, max
    ));
    head
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn allowed_command_runs_and_captures_stdout() {
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["echo"]).unwrap();
        let out = t.run(json!({"command": "echo", "args": ["hello"]})).await.unwrap();
        assert_eq!(out["exit_code"].as_i64().unwrap(), 0);
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "hello");
        assert!(out["elapsed_ms"].as_u64().is_some());
    }

    #[tokio::test]
    async fn disallowed_command_is_refused() {
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["echo"]).unwrap();
        let err = t.run(json!({"command": "rm", "args": ["-rf", "/"]})).await.unwrap_err();
        assert!(format!("{err}").contains("not in the allowlist"));
    }

    #[tokio::test]
    async fn args_are_passed_verbatim_no_shell_expansion() {
        // Pass shell metacharacters as args — they must reach the program
        // literally, not get interpreted by a shell. We assert by checking
        // that `echo` echoes them back unchanged (no glob expansion).
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["echo"]).unwrap();
        let out = t.run(json!({"command": "echo", "args": ["foo;rm -rf /;bar"]}))
            .await.unwrap();
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "foo;rm -rf /;bar");
    }

    #[tokio::test]
    async fn timeout_kills_long_running_process() {
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["sleep"])
            .unwrap()
            .with_timeout(Duration::from_millis(100));
        let started = std::time::Instant::now();
        let err = t.run(json!({"command": "sleep", "args": ["5"]})).await.unwrap_err();
        // Must fail FAST — timeout is 100ms, sleep would otherwise be 5s.
        assert!(started.elapsed() < Duration::from_secs(2));
        assert!(format!("{err}").contains("timed out"));
    }

    #[tokio::test]
    async fn output_truncation_marks_truncation_loudly() {
        let td = tempfile::tempdir().unwrap();
        // `head -c 1000 /dev/zero` writes 1000 NUL bytes; cap to 50.
        let t = ShellTool::new(td.path(), ["head"]).unwrap()
            .with_max_output_bytes(50);
        let out = t.run(json!({"command": "head", "args": ["-c", "1000", "/dev/zero"]}))
            .await.unwrap();
        let s = out["stdout"].as_str().unwrap();
        assert!(s.contains("[truncated"), "no truncation marker: {s:?}");
    }

    #[tokio::test]
    async fn working_dir_is_inherited_by_child() {
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["pwd"]).unwrap();
        let out = t.run(json!({"command": "pwd", "args": []})).await.unwrap();
        // macOS `tempdir()` lives under /var → /private/var. Either resolution
        // is fine; just assert the basename matches.
        let stdout = out["stdout"].as_str().unwrap().trim();
        let our_basename = td.path().file_name().unwrap().to_string_lossy();
        assert!(stdout.contains(&*our_basename),
            "child cwd ({stdout}) doesn't reference the working_dir basename ({our_basename})");
    }

    #[test]
    fn shell_tool_rejects_non_directory_working_dir() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let err = ShellTool::new(f.path(), ["echo"]).unwrap_err();
        assert!(format!("{err}").contains("not a directory"));
    }

    #[test]
    fn schema_lists_allowlist_alphabetically() {
        let td = tempfile::tempdir().unwrap();
        let t = ShellTool::new(td.path(), ["zcat", "ls", "echo"]).unwrap();
        let s = t.schema();
        assert_eq!(s.name, "shell");
        // Allowlist appears in description, sorted.
        let pos_echo = s.description.find("echo").unwrap();
        let pos_ls = s.description.find("ls").unwrap();
        let pos_zcat = s.description.find("zcat").unwrap();
        assert!(pos_echo < pos_ls && pos_ls < pos_zcat);
    }
}

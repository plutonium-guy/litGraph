//! `PythonReplTool` — execute Python code in a subprocess sandbox.
//!
//! Direct LangChain `PythonREPLTool` parity (the agent's go-to for ad-hoc
//! computation: data analysis, math beyond Calculator's expression grammar,
//! string transforms, JSON wrangling).
//!
//! # Security model
//!
//! Code execution against an LLM-driven agent is intrinsically risky.
//! What we do:
//! - **Sandboxed via subprocess**, NOT `eval`. Failures stay in the child.
//! - **Restricted env**: only `PATH`, `HOME`, `LANG`, `LC_ALL`, `TMPDIR`
//!   pass through. AWS / OpenAI / etc credentials are stripped.
//! - **Working dir**: caller-mandated, child's CWD is set there. The agent
//!   can't `os.chdir("/")` — the child can `chdir`, but the parent won't
//!   re-use that.
//! - **stdin closed**: code can't `input()`-block.
//! - **Timeout**: tokio's `timeout` + `kill_on_drop`. SIGKILL on overrun.
//! - **Output cap**: stdout + stderr each capped (default 64 KiB) so a
//!   `print(loop)` can't blow the agent's context budget.
//!
//! What we DON'T do:
//! - No filesystem allowlist beyond `working_dir`. Code can read your `~`
//!   if `HOME` is in scope. Operators must run this inside a chroot/jail
//!   for adversarial input.
//! - No CPU/RAM limit (rlimit varies wildly across platforms; not portable
//!   in a maintainable way). Time-limit only.
//! - No network restrictions. Code can `urllib.request.urlopen("...")`.
//!   Operators concerned about exfiltration should run in a network-
//!   isolated container.
//!
//! # Tool args (LLM-facing)
//!
//! ```json
//! {
//!   "code":      "import math\nprint(math.sqrt(2))",   // required
//!   "timeout_s": 30                                     // optional override
//! }
//! ```
//!
//! Returns: `{exit_code, stdout, stderr, elapsed_ms}`.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::time::Duration;

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::process::Command;

#[derive(Debug, Clone)]
pub struct PythonReplConfig {
    /// Path to the Python interpreter. Default `python3`.
    pub python: String,
    /// Working directory for the child. Mandatory — operators choose where
    /// agent-generated code is allowed to read/write transient files.
    pub working_dir: PathBuf,
    /// Hard timeout. Default 30s. Per-call override allowed via tool args.
    pub timeout: Duration,
    /// Combined cap on stdout / stderr (each, separately). Default 64 KiB.
    pub max_output_bytes: usize,
    /// Env vars to pass to the child. Default keeps PATH / HOME / LANG /
    /// LC_ALL / TMPDIR; everything else (including secrets) is stripped.
    pub allowed_env: HashSet<String>,
}

impl PythonReplConfig {
    pub fn new(working_dir: impl AsRef<Path>) -> Result<Self> {
        let wd = working_dir.as_ref().to_path_buf();
        let meta = std::fs::metadata(&wd).map_err(|e| {
            Error::other(format!(
                "PythonReplTool: cannot stat working_dir {}: {e}",
                wd.display()
            ))
        })?;
        if !meta.is_dir() {
            return Err(Error::other(format!(
                "PythonReplTool: working_dir {} is not a directory",
                wd.display()
            )));
        }
        let mut allowed = HashSet::new();
        for k in &["PATH", "HOME", "LANG", "LC_ALL", "TMPDIR"] {
            allowed.insert(k.to_string());
        }
        Ok(Self {
            python: "python3".into(),
            working_dir: wd,
            timeout: Duration::from_secs(30),
            max_output_bytes: 64 * 1024,
            allowed_env: allowed,
        })
    }
    pub fn with_python(mut self, p: impl Into<String>) -> Self {
        self.python = p.into();
        self
    }
    pub fn with_timeout(mut self, t: Duration) -> Self {
        self.timeout = t;
        self
    }
    pub fn with_max_output_bytes(mut self, n: usize) -> Self {
        self.max_output_bytes = n;
        self
    }
    pub fn with_extra_env<I, S>(mut self, keys: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        for k in keys {
            self.allowed_env.insert(k.into());
        }
        self
    }
}

pub struct PythonReplTool {
    cfg: PythonReplConfig,
}

impl PythonReplTool {
    pub fn new(cfg: PythonReplConfig) -> Self {
        Self { cfg }
    }
}

#[derive(Debug, Deserialize)]
struct ReplArgs {
    code: String,
    #[serde(default)]
    timeout_s: Option<u64>,
}

#[async_trait]
impl Tool for PythonReplTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "python_repl".to_string(),
            description: "Execute a Python snippet in a sandboxed subprocess. Code is run with \
                          `python3 -c <code>` (no interactive REPL). Returns stdout, stderr, and \
                          exit code. Use for: math beyond Calculator, data wrangling, JSON / CSV \
                          transforms, regex extraction. Output is capped (~64 KiB per stream). \
                          Has a hard timeout. Does NOT have access to the parent process's \
                          environment variables (no leaked credentials). Side effects in the \
                          working directory ARE persisted across calls."
                .to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python source. Use `print(...)` to surface results — return values are not captured.",
                    },
                    "timeout_s": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Per-call timeout override. May not exceed the configured cap."
                    }
                },
                "required": ["code"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let parsed: ReplArgs = serde_json::from_value(args)
            .map_err(|e| Error::invalid(format!("python_repl args: {e}")))?;
        if parsed.code.is_empty() {
            return Err(Error::invalid("python_repl: code must be non-empty"));
        }

        // Per-call timeout, capped at the configured ceiling.
        let timeout = match parsed.timeout_s {
            Some(t) => Duration::from_secs(t).min(self.cfg.timeout),
            None => self.cfg.timeout,
        };

        // Build a clean env: pull through only the allowed keys.
        let env: Vec<(String, String)> = std::env::vars()
            .filter(|(k, _)| self.cfg.allowed_env.contains(k))
            .collect();

        let mut child = Command::new(&self.cfg.python);
        child
            .arg("-I") // isolated mode: ignore PYTHONPATH, no user-site, no env-driven paths
            .arg("-c")
            .arg(&parsed.code)
            .current_dir(&self.cfg.working_dir)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env_clear()
            .envs(env)
            .kill_on_drop(true);

        let started = std::time::Instant::now();
        let output_fut = async {
            let child = child
                .spawn()
                .map_err(|e| Error::other(format!("spawn {}: {e}", self.cfg.python)))?;
            child
                .wait_with_output()
                .await
                .map_err(|e| Error::other(format!("wait: {e}")))
        };
        let output = match tokio::time::timeout(timeout, output_fut).await {
            Ok(Ok(o)) => o,
            Ok(Err(e)) => return Err(e),
            Err(_) => {
                return Err(Error::other(format!(
                    "python_repl: timed out after {:?}",
                    timeout
                )))
            }
        };
        let elapsed_ms = started.elapsed().as_millis() as u64;

        Ok(json!({
            "exit_code": output.status.code(),
            "stdout": truncate_for_agent(&output.stdout, self.cfg.max_output_bytes),
            "stderr": truncate_for_agent(&output.stderr, self.cfg.max_output_bytes),
            "elapsed_ms": elapsed_ms,
        }))
    }
}

fn truncate_for_agent(bytes: &[u8], max: usize) -> String {
    if bytes.len() <= max {
        return String::from_utf8_lossy(bytes).into_owned();
    }
    let mut head = String::from_utf8_lossy(&bytes[..max]).into_owned();
    head.push_str(&format!(
        "\n... [truncated {} bytes; cap = {}]",
        bytes.len() - max,
        max
    ));
    head
}

#[cfg(test)]
mod tests {
    use super::*;

    fn skip_if_no_python() -> bool {
        std::process::Command::new("python3")
            .arg("--version")
            .output()
            .map(|o| !o.status.success())
            .unwrap_or(true)
    }

    #[tokio::test]
    async fn simple_print_round_trips_stdout() {
        if skip_if_no_python() {
            eprintln!("python3 not on PATH; skipping");
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let out = tool
            .run(json!({"code": "print('hello world')"}))
            .await
            .unwrap();
        assert_eq!(out["exit_code"].as_i64().unwrap(), 0);
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "hello world");
        assert_eq!(out["stderr"].as_str().unwrap(), "");
    }

    #[tokio::test]
    async fn nonzero_exit_captures_stderr() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let out = tool
            .run(json!({"code": "import sys; sys.exit(2)"}))
            .await
            .unwrap();
        assert_eq!(out["exit_code"].as_i64().unwrap(), 2);
    }

    #[tokio::test]
    async fn syntax_error_surfaces_in_stderr() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let out = tool
            .run(json!({"code": "def broken(:"}))
            .await
            .unwrap();
        assert_ne!(out["exit_code"].as_i64().unwrap(), 0);
        assert!(out["stderr"]
            .as_str()
            .unwrap()
            .contains("SyntaxError"));
    }

    #[tokio::test]
    async fn timeout_kills_runaway_loop() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path())
            .unwrap()
            .with_timeout(Duration::from_millis(500));
        let tool = PythonReplTool::new(cfg);
        let started = std::time::Instant::now();
        let err = tool
            .run(json!({"code": "while True: pass"}))
            .await
            .unwrap_err();
        let elapsed = started.elapsed();
        let msg = format!("{err}");
        assert!(msg.contains("timed out"), "got: {msg}");
        // Should have killed within ~1s, not waited for default 30s.
        assert!(elapsed < Duration::from_secs(2));
    }

    #[tokio::test]
    async fn empty_code_errors_invalid() {
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let err = tool.run(json!({"code": ""})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn parent_env_secrets_stripped_by_default() {
        if skip_if_no_python() {
            return;
        }
        // SAFETY: setting an env var in tests can race with concurrent
        // tests. Using a unique key.
        let key = "LITGRAPH_TEST_SECRET_42";
        std::env::set_var(key, "should-not-leak");
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let code = format!(
            "import os; print(os.environ.get('{}', 'MISSING'))",
            key
        );
        let out = tool.run(json!({"code": code})).await.unwrap();
        let stdout = out["stdout"].as_str().unwrap().trim();
        assert_eq!(stdout, "MISSING");
        std::env::remove_var(key);
    }

    #[tokio::test]
    async fn extra_env_can_be_passed_through_explicitly() {
        if skip_if_no_python() {
            return;
        }
        let key = "LITGRAPH_TEST_OPT_IN_43";
        std::env::set_var(key, "visible");
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path())
            .unwrap()
            .with_extra_env([key]);
        let tool = PythonReplTool::new(cfg);
        let code = format!(
            "import os; print(os.environ.get('{}', 'MISSING'))",
            key
        );
        let out = tool.run(json!({"code": code})).await.unwrap();
        assert_eq!(out["stdout"].as_str().unwrap().trim(), "visible");
        std::env::remove_var(key);
    }

    #[tokio::test]
    async fn working_dir_is_child_cwd() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let out = tool
            .run(json!({"code": "import os; print(os.getcwd())"}))
            .await
            .unwrap();
        // Compare canonical paths — macOS has /private/var → /var symlinks.
        let got = std::path::Path::new(out["stdout"].as_str().unwrap().trim())
            .canonicalize()
            .unwrap();
        let want = td.path().canonicalize().unwrap();
        assert_eq!(got, want);
    }

    #[tokio::test]
    async fn output_truncation_caps_long_stdout() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path())
            .unwrap()
            .with_max_output_bytes(100);
        let tool = PythonReplTool::new(cfg);
        let out = tool
            .run(json!({"code": "print('x' * 1000)"}))
            .await
            .unwrap();
        let stdout = out["stdout"].as_str().unwrap();
        assert!(stdout.len() < 1000);
        assert!(stdout.contains("[truncated"));
    }

    #[tokio::test]
    async fn schema_has_correct_name_and_required_field() {
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path()).unwrap();
        let tool = PythonReplTool::new(cfg);
        let s = tool.schema();
        assert_eq!(s.name, "python_repl");
        let required = s.parameters["required"].as_array().unwrap();
        assert_eq!(required.len(), 1);
        assert_eq!(required[0], "code");
    }

    #[tokio::test]
    async fn per_call_timeout_capped_at_config_ceiling() {
        if skip_if_no_python() {
            return;
        }
        let td = tempfile::tempdir().unwrap();
        let cfg = PythonReplConfig::new(td.path())
            .unwrap()
            .with_timeout(Duration::from_millis(500));
        let tool = PythonReplTool::new(cfg);
        let started = std::time::Instant::now();
        // LLM asks for 60s but config caps at 500ms.
        let err = tool
            .run(json!({"code": "while True: pass", "timeout_s": 60}))
            .await
            .unwrap_err();
        let elapsed = started.elapsed();
        assert!(format!("{err}").contains("timed out"));
        assert!(elapsed < Duration::from_secs(2));
    }
}

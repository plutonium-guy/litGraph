//! `JsonPatchTool` — RFC 6902 JSON Patch ops over JSON Pointer paths.
//!
//! # Why a dedicated tool
//!
//! Agents modifying structured documents (configs, workflow state,
//! API request bodies) currently have to either:
//! 1. Re-emit the entire modified document, which is wasteful for
//!    small changes and error-prone (the LLM might silently change
//!    a field it shouldn't have touched), OR
//! 2. Use shell tools to run `jq` / `sed`, which is fragile and
//!    introduces shell escaping concerns.
//!
//! `JsonPatchTool` provides a structured edit primitive: the agent
//! emits a list of `{op, path, value?}` operations and the tool
//! applies them atomically, returning the modified document. Pairs
//! with iter-281 `JsonExtractTool` — that one *reads* JSON via
//! JSONPath-lite, this one *writes* via JSON Pointer.
//!
//! # Operations (RFC 6902)
//!
//! - `add` — insert or replace at path. For arrays, inserts (shifts
//!   subsequent elements right); for `-`, appends. For objects,
//!   sets the field (creates if missing).
//! - `remove` — delete at path. For arrays, shifts subsequent left.
//! - `replace` — overwrite at path. Path MUST exist.
//! - `move` — atomic remove(from) + add(path).
//! - `copy` — get(from) + add(path), source unchanged.
//! - `test` — assert value at path equals expected. Acts as a
//!   precondition — fails the entire patch if mismatched.
//!
//! # JSON Pointer (RFC 6901) path syntax
//!
//! `""` — root. `"/foo"` — field "foo". `"/foo/0"` — array index 0
//! of field "foo". `"/-"` — "after the last element of an array"
//! (only valid as `path` of an `add` op). Two escape sequences:
//! `~0` decodes to `~`, `~1` decodes to `/` (so a field literally
//! named `a/b` is referenced as `/a~1b`).
//!
//! # Atomicity
//!
//! Operations are applied sequentially; if any op fails (path not
//! found for `remove`/`replace`/`test`, type mismatch, `test`
//! assertion fail), the ENTIRE patch is rejected and the original
//! document is returned unchanged. The implementation operates on
//! a deep clone of the input so no partial state leaks.

use async_trait::async_trait;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct JsonPatchTool;

impl JsonPatchTool {
    pub fn new() -> Self {
        Self
    }
}

#[async_trait]
impl Tool for JsonPatchTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "json_patch".into(),
            description: "Apply RFC 6902 JSON Patch ops to a document. `patch` is an array \
                of {op, path, value?, from?} operations. Ops: add (insert/replace), remove, \
                replace, move, copy, test (precondition assert). Paths use RFC 6901 JSON \
                Pointer syntax: '/foo/0/bar' navigates field 'foo' → array index 0 → field \
                'bar'. '/-' appends to an array (add op only). Escape '~0' for literal '~', \
                '~1' for literal '/'. Atomic: any op failure rolls back the whole patch."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "document": {
                        "description": "The JSON document to patch (any JSON value)."
                    },
                    "patch": {
                        "type": "array",
                        "description": "Array of patch operations.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "op": {
                                    "type": "string",
                                    "enum": ["add", "remove", "replace", "move", "copy", "test"]
                                },
                                "path": {"type": "string"},
                                "value": {},
                                "from": {"type": "string"}
                            },
                            "required": ["op", "path"]
                        }
                    }
                },
                "required": ["document", "patch"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let document = args
            .get("document")
            .ok_or_else(|| Error::invalid("json_patch: missing `document`"))?
            .clone();
        let patch = args
            .get("patch")
            .and_then(|v| v.as_array())
            .ok_or_else(|| Error::invalid("json_patch: missing or non-array `patch`"))?;
        let result = json_patch(document, patch)?;
        Ok(result)
    }
}

/// Apply a slice of patch operations to a document. Atomic — any op
/// failure rolls back to the original input.
///
/// Public so callers can invoke directly without the Tool trait.
pub fn json_patch(document: Value, patch: &[Value]) -> Result<Value> {
    let mut working = document;
    for (i, op_obj) in patch.iter().enumerate() {
        apply_op(&mut working, op_obj).map_err(|e| {
            Error::invalid(format!("json_patch: op {i} failed: {e}"))
        })?;
    }
    Ok(working)
}

fn apply_op(doc: &mut Value, op_obj: &Value) -> std::result::Result<(), String> {
    let op = op_obj
        .get("op")
        .and_then(|v| v.as_str())
        .ok_or("missing `op`")?;
    let path = op_obj
        .get("path")
        .and_then(|v| v.as_str())
        .ok_or("missing `path`")?;
    match op {
        "add" => {
            let value = op_obj
                .get("value")
                .ok_or("`add` requires `value`")?
                .clone();
            op_add(doc, path, value)
        }
        "remove" => op_remove(doc, path).map(|_| ()),
        "replace" => {
            let value = op_obj
                .get("value")
                .ok_or("`replace` requires `value`")?
                .clone();
            op_replace(doc, path, value)
        }
        "move" => {
            let from = op_obj
                .get("from")
                .and_then(|v| v.as_str())
                .ok_or("`move` requires `from`")?;
            let removed = op_remove(doc, from)?;
            op_add(doc, path, removed)
        }
        "copy" => {
            let from = op_obj
                .get("from")
                .and_then(|v| v.as_str())
                .ok_or("`copy` requires `from`")?;
            let value = pointer_get(doc, from)?.clone();
            op_add(doc, path, value)
        }
        "test" => {
            let expected = op_obj
                .get("value")
                .ok_or("`test` requires `value`")?;
            let actual = pointer_get(doc, path)?;
            if actual == expected {
                Ok(())
            } else {
                Err(format!(
                    "`test` assertion failed: path {path} got {actual}, expected {expected}"
                ))
            }
        }
        other => Err(format!("unknown op {other:?}")),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// JSON Pointer (RFC 6901) — parse + traverse
// ─────────────────────────────────────────────────────────────────────────────

/// Decode a JSON Pointer token: `~1` → `/`, `~0` → `~`, in that order
/// (the spec mandates ~1 before ~0 to handle `~01` → `~1` correctly).
fn decode_token(token: &str) -> String {
    token.replace("~1", "/").replace("~0", "~")
}

/// Parse a JSON Pointer into a sequence of decoded tokens.
/// `""` → empty vec. `"/"` → vec![""].`/foo/bar` → vec!["foo", "bar"].
fn parse_pointer(path: &str) -> std::result::Result<Vec<String>, String> {
    if path.is_empty() {
        return Ok(Vec::new());
    }
    if !path.starts_with('/') {
        return Err(format!("pointer must be empty or start with /: {path:?}"));
    }
    Ok(path[1..].split('/').map(decode_token).collect())
}

/// Get a reference to the value at `path`. Returns Err if any
/// intermediate token doesn't exist.
fn pointer_get<'a>(doc: &'a Value, path: &str) -> std::result::Result<&'a Value, String> {
    let tokens = parse_pointer(path)?;
    let mut cur = doc;
    for tok in &tokens {
        cur = traverse_one(cur, tok)?;
    }
    Ok(cur)
}

fn traverse_one<'a>(cur: &'a Value, tok: &str) -> std::result::Result<&'a Value, String> {
    match cur {
        Value::Object(map) => map
            .get(tok)
            .ok_or_else(|| format!("path token {tok:?} not found in object")),
        Value::Array(arr) => {
            let idx = parse_array_index(tok, arr.len(), false)?;
            arr.get(idx)
                .ok_or_else(|| format!("array index {idx} out of bounds (len={})", arr.len()))
        }
        _ => Err(format!(
            "cannot traverse into non-container at token {tok:?}"
        )),
    }
}

/// Parse an array-index token. `-` is allowed only when `allow_dash`
/// is true (the `add` op's "append" semantics) and translates to
/// `arr.len()` (one-past-the-end).
fn parse_array_index(
    tok: &str,
    arr_len: usize,
    allow_dash: bool,
) -> std::result::Result<usize, String> {
    if tok == "-" {
        if allow_dash {
            return Ok(arr_len);
        }
        return Err("`-` is only valid as the last token of an `add` op path".into());
    }
    tok.parse::<usize>()
        .map_err(|_| format!("invalid array index token {tok:?}"))
}

// ─────────────────────────────────────────────────────────────────────────────
// Op implementations — operate on a mutable Value.
// ─────────────────────────────────────────────────────────────────────────────

fn op_add(doc: &mut Value, path: &str, value: Value) -> std::result::Result<(), String> {
    let tokens = parse_pointer(path)?;
    if tokens.is_empty() {
        // Replace the whole document.
        *doc = value;
        return Ok(());
    }
    let (parent_path, last) = tokens.split_at(tokens.len() - 1);
    let parent = pointer_get_mut_path(doc, parent_path)?;
    let last = &last[0];
    match parent {
        Value::Object(map) => {
            map.insert(last.clone(), value);
            Ok(())
        }
        Value::Array(arr) => {
            // `add` to an array INSERTS (not replaces) — array indexes
            // shift right.
            let idx = parse_array_index(last, arr.len(), true)?;
            if idx > arr.len() {
                return Err(format!("array index {idx} out of bounds for add"));
            }
            arr.insert(idx, value);
            Ok(())
        }
        _ => Err(format!(
            "cannot `add` into non-container parent at path {path:?}"
        )),
    }
}

fn op_remove(doc: &mut Value, path: &str) -> std::result::Result<Value, String> {
    let tokens = parse_pointer(path)?;
    if tokens.is_empty() {
        return Err("cannot `remove` the root document".into());
    }
    let (parent_path, last) = tokens.split_at(tokens.len() - 1);
    let parent = pointer_get_mut_path(doc, parent_path)?;
    let last = &last[0];
    match parent {
        Value::Object(map) => map
            .remove(last)
            .ok_or_else(|| format!("path token {last:?} not found in object for remove")),
        Value::Array(arr) => {
            let idx = parse_array_index(last, arr.len(), false)?;
            if idx >= arr.len() {
                return Err(format!("array index {idx} out of bounds for remove"));
            }
            Ok(arr.remove(idx))
        }
        _ => Err("cannot `remove` from non-container parent".into()),
    }
}

fn op_replace(doc: &mut Value, path: &str, value: Value) -> std::result::Result<(), String> {
    let tokens = parse_pointer(path)?;
    if tokens.is_empty() {
        *doc = value;
        return Ok(());
    }
    let (parent_path, last) = tokens.split_at(tokens.len() - 1);
    let parent = pointer_get_mut_path(doc, parent_path)?;
    let last = &last[0];
    match parent {
        Value::Object(map) => {
            if !map.contains_key(last) {
                return Err(format!(
                    "`replace` target path {last:?} does not exist in object"
                ));
            }
            map.insert(last.clone(), value);
            Ok(())
        }
        Value::Array(arr) => {
            let idx = parse_array_index(last, arr.len(), false)?;
            if idx >= arr.len() {
                return Err(format!(
                    "`replace` array index {idx} out of bounds (len={})",
                    arr.len()
                ));
            }
            arr[idx] = value;
            Ok(())
        }
        _ => Err("cannot `replace` in non-container parent".into()),
    }
}

/// Walk to the parent at `tokens`, returning a mutable reference.
fn pointer_get_mut_path<'a>(
    doc: &'a mut Value,
    tokens: &[String],
) -> std::result::Result<&'a mut Value, String> {
    let mut cur = doc;
    for tok in tokens {
        cur = match cur {
            Value::Object(map) => map
                .get_mut(tok)
                .ok_or_else(|| format!("path token {tok:?} not found in object"))?,
            Value::Array(arr) => {
                let idx = parse_array_index(tok, arr.len(), false)?;
                arr.get_mut(idx)
                    .ok_or_else(|| format!("array index {idx} out of bounds"))?
            }
            _ => return Err(format!("cannot traverse non-container at {tok:?}")),
        };
    }
    Ok(cur)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn patch_doc(doc: Value, patch: Value) -> Result<Value> {
        let arr = patch.as_array().expect("patch must be array").clone();
        json_patch(doc, &arr)
    }

    // ─── Pointer parsing ────────────────────────────────────────────

    #[test]
    fn parse_pointer_root_is_empty() {
        assert_eq!(parse_pointer("").unwrap(), Vec::<String>::new());
    }

    #[test]
    fn parse_pointer_single_token() {
        assert_eq!(parse_pointer("/foo").unwrap(), vec!["foo"]);
    }

    #[test]
    fn parse_pointer_multi_token() {
        assert_eq!(
            parse_pointer("/foo/bar/0").unwrap(),
            vec!["foo", "bar", "0"]
        );
    }

    #[test]
    fn parse_pointer_escapes() {
        // ~1 → /, ~0 → ~. The spec mandates ~1 first to handle
        // ~01 → ~1 not /.
        assert_eq!(parse_pointer("/a~1b").unwrap(), vec!["a/b"]);
        assert_eq!(parse_pointer("/a~0b").unwrap(), vec!["a~b"]);
        assert_eq!(parse_pointer("/~01").unwrap(), vec!["~1"]);
    }

    #[test]
    fn parse_pointer_must_start_with_slash() {
        assert!(parse_pointer("foo").is_err());
    }

    // ─── add ────────────────────────────────────────────────────────

    #[test]
    fn add_to_object_creates_field() {
        let r = patch_doc(json!({"a": 1}), json!([{"op": "add", "path": "/b", "value": 2}]))
            .unwrap();
        assert_eq!(r, json!({"a": 1, "b": 2}));
    }

    #[test]
    fn add_to_object_overwrites_existing_field() {
        let r = patch_doc(
            json!({"a": 1}),
            json!([{"op": "add", "path": "/a", "value": 99}]),
        )
        .unwrap();
        assert_eq!(r, json!({"a": 99}));
    }

    #[test]
    fn add_to_array_inserts_shifts_right() {
        let r = patch_doc(
            json!([1, 2, 3]),
            json!([{"op": "add", "path": "/1", "value": 99}]),
        )
        .unwrap();
        assert_eq!(r, json!([1, 99, 2, 3]));
    }

    #[test]
    fn add_dash_appends_to_array() {
        let r = patch_doc(
            json!([1, 2, 3]),
            json!([{"op": "add", "path": "/-", "value": 4}]),
        )
        .unwrap();
        assert_eq!(r, json!([1, 2, 3, 4]));
    }

    #[test]
    fn add_to_root_replaces_whole_document() {
        let r = patch_doc(json!({"old": 1}), json!([{"op": "add", "path": "", "value": "new"}]))
            .unwrap();
        assert_eq!(r, json!("new"));
    }

    #[test]
    fn add_nested_path() {
        let r = patch_doc(
            json!({"a": {"b": [1, 2]}}),
            json!([{"op": "add", "path": "/a/b/-", "value": 3}]),
        )
        .unwrap();
        assert_eq!(r, json!({"a": {"b": [1, 2, 3]}}));
    }

    // ─── remove ─────────────────────────────────────────────────────

    #[test]
    fn remove_from_object() {
        let r = patch_doc(
            json!({"a": 1, "b": 2}),
            json!([{"op": "remove", "path": "/a"}]),
        )
        .unwrap();
        assert_eq!(r, json!({"b": 2}));
    }

    #[test]
    fn remove_from_array_shifts_left() {
        let r = patch_doc(
            json!([1, 2, 3, 4]),
            json!([{"op": "remove", "path": "/1"}]),
        )
        .unwrap();
        assert_eq!(r, json!([1, 3, 4]));
    }

    #[test]
    fn remove_missing_field_errors() {
        let r = patch_doc(
            json!({"a": 1}),
            json!([{"op": "remove", "path": "/nonexistent"}]),
        );
        assert!(r.is_err());
    }

    #[test]
    fn remove_root_errors() {
        let r = patch_doc(json!({"a": 1}), json!([{"op": "remove", "path": ""}]));
        assert!(r.is_err());
    }

    // ─── replace ────────────────────────────────────────────────────

    #[test]
    fn replace_existing_field() {
        let r = patch_doc(
            json!({"a": 1}),
            json!([{"op": "replace", "path": "/a", "value": 99}]),
        )
        .unwrap();
        assert_eq!(r, json!({"a": 99}));
    }

    #[test]
    fn replace_missing_field_errors() {
        // Distinct from `add` which would create — `replace` MUST hit existing.
        let r = patch_doc(
            json!({"a": 1}),
            json!([{"op": "replace", "path": "/missing", "value": 99}]),
        );
        assert!(r.is_err());
    }

    #[test]
    fn replace_array_element() {
        let r = patch_doc(
            json!([1, 2, 3]),
            json!([{"op": "replace", "path": "/1", "value": 99}]),
        )
        .unwrap();
        assert_eq!(r, json!([1, 99, 3]));
    }

    // ─── move ───────────────────────────────────────────────────────

    #[test]
    fn move_field_in_object() {
        let r = patch_doc(
            json!({"a": 1, "b": 2}),
            json!([{"op": "move", "from": "/a", "path": "/c"}]),
        )
        .unwrap();
        assert_eq!(r, json!({"b": 2, "c": 1}));
    }

    #[test]
    fn move_array_element() {
        let r = patch_doc(
            json!([1, 2, 3]),
            json!([{"op": "move", "from": "/0", "path": "/-"}]),
        )
        .unwrap();
        // 1 removed (shifts to [2,3]), then appended → [2, 3, 1].
        assert_eq!(r, json!([2, 3, 1]));
    }

    // ─── copy ───────────────────────────────────────────────────────

    #[test]
    fn copy_field() {
        let r = patch_doc(
            json!({"a": "x", "b": "y"}),
            json!([{"op": "copy", "from": "/a", "path": "/c"}]),
        )
        .unwrap();
        assert_eq!(r, json!({"a": "x", "b": "y", "c": "x"}));
    }

    // ─── test ───────────────────────────────────────────────────────

    #[test]
    fn test_pass_returns_unchanged() {
        let r = patch_doc(
            json!({"a": 1}),
            json!([{"op": "test", "path": "/a", "value": 1}]),
        )
        .unwrap();
        assert_eq!(r, json!({"a": 1}));
    }

    #[test]
    fn test_fail_aborts_whole_patch() {
        // test fails BEFORE the add. Whole patch should be rejected;
        // the add must NOT have been applied.
        let r = patch_doc(
            json!({"a": 1}),
            json!([
                {"op": "test", "path": "/a", "value": 99},
                {"op": "add", "path": "/b", "value": 2}
            ]),
        );
        assert!(r.is_err());
    }

    // ─── atomicity ──────────────────────────────────────────────────

    #[test]
    fn atomicity_failure_returns_err_no_partial() {
        // First op succeeds (add /b), second fails (replace nonexistent).
        // The Tool returns Err — but the original `document` arg is owned
        // (deep cloned in run()). We just verify the error surfaces.
        let r = patch_doc(
            json!({"a": 1}),
            json!([
                {"op": "add", "path": "/b", "value": 2},
                {"op": "replace", "path": "/missing", "value": 99}
            ]),
        );
        assert!(r.is_err());
    }

    // ─── Tool integration ──────────────────────────────────────────

    #[tokio::test]
    async fn tool_basic_patch() {
        let t = JsonPatchTool::new();
        let v = t
            .run(json!({
                "document": {"a": 1},
                "patch": [{"op": "add", "path": "/b", "value": 2}]
            }))
            .await
            .unwrap();
        assert_eq!(v, json!({"a": 1, "b": 2}));
    }

    #[tokio::test]
    async fn tool_missing_document_errors() {
        let t = JsonPatchTool::new();
        let r = t.run(json!({"patch": []})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn tool_missing_patch_errors() {
        let t = JsonPatchTool::new();
        let r = t.run(json!({"document": {}})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    #[tokio::test]
    async fn tool_non_array_patch_errors() {
        let t = JsonPatchTool::new();
        let r = t.run(json!({"document": {}, "patch": "not-array"})).await;
        assert!(matches!(r, Err(Error::InvalidInput(_))));
    }

    // ─── Realistic scenarios ────────────────────────────────────────

    #[test]
    fn realistic_config_update() {
        let config = json!({
            "database": {"host": "old.example.com", "port": 5432},
            "features": {"flag_a": true}
        });
        let patch = json!([
            {"op": "replace", "path": "/database/host", "value": "new.example.com"},
            {"op": "add", "path": "/features/flag_b", "value": false}
        ]);
        let r = patch_doc(config, patch).unwrap();
        assert_eq!(
            r,
            json!({
                "database": {"host": "new.example.com", "port": 5432},
                "features": {"flag_a": true, "flag_b": false}
            })
        );
    }

    #[test]
    fn realistic_workflow_state_with_test_precondition() {
        // Patch with a precondition: only update if status is currently "pending".
        let state = json!({"status": "pending", "result": null});
        let patch = json!([
            {"op": "test", "path": "/status", "value": "pending"},
            {"op": "replace", "path": "/status", "value": "completed"},
            {"op": "replace", "path": "/result", "value": 42}
        ]);
        let r = patch_doc(state, patch).unwrap();
        assert_eq!(r, json!({"status": "completed", "result": 42}));
    }
}

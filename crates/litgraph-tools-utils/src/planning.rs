//! `PlanningTool` — Deep-Agents-style todo list for long-horizon tasks.
//!
//! The agent sees a single tool with an `action` discriminator: `list`, `add`,
//! `set_status`, `update`, `clear`. State lives behind an `Arc<Mutex<…>>` so
//! multiple parallel tool calls don't tear it. The list is per-instance — give
//! each agent session its own `PlanningTool::new()` to scope state.
//!
//! Mirrors LangChain `deepagents` planning primitive (write_todos / read_todos)
//! but collapsed into one tool to fit our schema-per-tool model.

use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Done,
    Cancelled,
}

impl TodoStatus {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "in_progress" | "in-progress" | "doing" => Some(Self::InProgress),
            "done" | "completed" | "complete" => Some(Self::Done),
            "cancelled" | "canceled" => Some(Self::Cancelled),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TodoItem {
    pub id: u32,
    pub content: String,
    pub status: TodoStatus,
    pub created_at_ms: u64,
    pub updated_at_ms: u64,
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[derive(Debug, Default)]
struct PlanState {
    next_id: u32,
    items: Vec<TodoItem>,
}

#[derive(Debug, Clone, Default)]
pub struct PlanningTool {
    state: Arc<Mutex<PlanState>>,
}

impl PlanningTool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Snapshot the current todo list (clones).
    pub fn snapshot(&self) -> Vec<TodoItem> {
        self.state.lock().unwrap().items.clone()
    }

    /// Reset the list. Returns the count of removed items.
    pub fn clear(&self) -> usize {
        let mut g = self.state.lock().unwrap();
        let n = g.items.len();
        g.items.clear();
        n
    }
}

fn item_json(item: &TodoItem) -> Value {
    json!({
        "id": item.id,
        "content": item.content,
        "status": item.status,
        "created_at_ms": item.created_at_ms,
        "updated_at_ms": item.updated_at_ms,
    })
}

#[async_trait]
impl Tool for PlanningTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "planning".into(),
            description: "Maintain a todo list for the current task. Use action='list' to read \
                          the current plan, 'add' to append items (provide `items` as a list of \
                          strings, or a single `content` string), 'set_status' to mark progress \
                          (`id` + `status` in {pending, in_progress, done, cancelled}), \
                          'update' to rewrite an item's text (`id` + `content`), and 'clear' to \
                          reset. Returns the updated list."
                .into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["list", "add", "set_status", "update", "clear"]
                    },
                    "items": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "For action=add: list of todo strings to append."
                    },
                    "content": {
                        "type": "string",
                        "description": "For action=add (single) or update: todo text."
                    },
                    "id": {
                        "type": "integer",
                        "description": "For set_status / update: target todo id."
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending", "in_progress", "done", "cancelled"],
                        "description": "For set_status: new status."
                    }
                },
                "required": ["action"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let action = args
            .get("action")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("planning: missing `action`"))?;
        let mut state = self.state.lock().unwrap();
        match action {
            "list" => {}
            "clear" => {
                state.items.clear();
            }
            "add" => {
                let mut to_add: Vec<String> = Vec::new();
                if let Some(arr) = args.get("items").and_then(|v| v.as_array()) {
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            if !s.trim().is_empty() {
                                to_add.push(s.to_string());
                            }
                        }
                    }
                }
                if let Some(s) = args.get("content").and_then(|v| v.as_str()) {
                    if !s.trim().is_empty() {
                        to_add.push(s.to_string());
                    }
                }
                if to_add.is_empty() {
                    return Err(Error::invalid(
                        "planning(add): provide `items` (list) or `content` (string)",
                    ));
                }
                let now = now_ms();
                for content in to_add {
                    state.next_id += 1;
                    let id = state.next_id;
                    state.items.push(TodoItem {
                        id,
                        content,
                        status: TodoStatus::Pending,
                        created_at_ms: now,
                        updated_at_ms: now,
                    });
                }
            }
            "set_status" => {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| Error::invalid("planning(set_status): missing `id`"))?
                    as u32;
                let status_str = args
                    .get("status")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::invalid("planning(set_status): missing `status`"))?;
                let status = TodoStatus::parse(status_str)
                    .ok_or_else(|| Error::invalid(format!("planning: unknown status {status_str}")))?;
                let now = now_ms();
                let item = state
                    .items
                    .iter_mut()
                    .find(|i| i.id == id)
                    .ok_or_else(|| Error::invalid(format!("planning: no todo with id {id}")))?;
                item.status = status;
                item.updated_at_ms = now;
            }
            "update" => {
                let id = args
                    .get("id")
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| Error::invalid("planning(update): missing `id`"))?
                    as u32;
                let content = args
                    .get("content")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| Error::invalid("planning(update): missing `content`"))?;
                if content.trim().is_empty() {
                    return Err(Error::invalid("planning(update): content is empty"));
                }
                let now = now_ms();
                let item = state
                    .items
                    .iter_mut()
                    .find(|i| i.id == id)
                    .ok_or_else(|| Error::invalid(format!("planning: no todo with id {id}")))?;
                item.content = content.to_string();
                item.updated_at_ms = now;
            }
            other => {
                return Err(Error::invalid(format!("planning: unknown action `{other}`")));
            }
        }
        let view: Vec<Value> = state.items.iter().map(item_json).collect();
        let pending = state
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::Pending)
            .count();
        let in_progress = state
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::InProgress)
            .count();
        let done = state
            .items
            .iter()
            .filter(|i| i.status == TodoStatus::Done)
            .count();
        Ok(json!({
            "items": view,
            "summary": {
                "total": state.items.len(),
                "pending": pending,
                "in_progress": in_progress,
                "done": done,
            }
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn list_starts_empty() {
        let t = PlanningTool::new();
        let r = t.run(json!({"action": "list"})).await.unwrap();
        assert_eq!(r["summary"]["total"], 0);
        assert_eq!(r["items"].as_array().unwrap().len(), 0);
    }

    #[tokio::test]
    async fn add_items_assigns_monotonic_ids() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "items": ["a", "b", "c"]}))
            .await
            .unwrap();
        let r = t.run(json!({"action": "list"})).await.unwrap();
        let ids: Vec<u64> = r["items"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| i["id"].as_u64().unwrap())
            .collect();
        assert_eq!(ids, vec![1, 2, 3]);
    }

    #[tokio::test]
    async fn add_single_via_content_field() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "content": "hello"}))
            .await
            .unwrap();
        let snap = t.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].content, "hello");
        assert_eq!(snap[0].status, TodoStatus::Pending);
    }

    #[tokio::test]
    async fn add_strips_empty_strings() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "items": ["", "  ", "real"]}))
            .await
            .unwrap();
        assert_eq!(t.snapshot().len(), 1);
    }

    #[tokio::test]
    async fn add_with_no_content_errors() {
        let t = PlanningTool::new();
        let err = t.run(json!({"action": "add"})).await.unwrap_err();
        assert!(format!("{err}").contains("items"));
    }

    #[tokio::test]
    async fn set_status_transitions_item() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "content": "x"})).await.unwrap();
        t.run(json!({"action": "set_status", "id": 1, "status": "in_progress"}))
            .await
            .unwrap();
        let snap = t.snapshot();
        assert_eq!(snap[0].status, TodoStatus::InProgress);
        t.run(json!({"action": "set_status", "id": 1, "status": "done"}))
            .await
            .unwrap();
        assert_eq!(t.snapshot()[0].status, TodoStatus::Done);
    }

    #[tokio::test]
    async fn set_status_unknown_rejected() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "content": "x"})).await.unwrap();
        let err = t
            .run(json!({"action": "set_status", "id": 1, "status": "snorkel"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("snorkel"));
    }

    #[tokio::test]
    async fn set_status_missing_id_rejected() {
        let t = PlanningTool::new();
        let err = t
            .run(json!({"action": "set_status", "id": 99, "status": "done"}))
            .await
            .unwrap_err();
        assert!(format!("{err}").contains("99"));
    }

    #[tokio::test]
    async fn update_rewrites_content() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "content": "old"})).await.unwrap();
        t.run(json!({"action": "update", "id": 1, "content": "new"}))
            .await
            .unwrap();
        assert_eq!(t.snapshot()[0].content, "new");
    }

    #[tokio::test]
    async fn clear_removes_all() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "items": ["a", "b"]}))
            .await
            .unwrap();
        let r = t.run(json!({"action": "clear"})).await.unwrap();
        assert_eq!(r["summary"]["total"], 0);
        assert_eq!(t.snapshot().len(), 0);
    }

    #[tokio::test]
    async fn summary_buckets_by_status() {
        let t = PlanningTool::new();
        t.run(json!({"action": "add", "items": ["a", "b", "c"]}))
            .await
            .unwrap();
        t.run(json!({"action": "set_status", "id": 1, "status": "in_progress"}))
            .await
            .unwrap();
        t.run(json!({"action": "set_status", "id": 2, "status": "done"}))
            .await
            .unwrap();
        let r = t.run(json!({"action": "list"})).await.unwrap();
        assert_eq!(r["summary"]["pending"], 1);
        assert_eq!(r["summary"]["in_progress"], 1);
        assert_eq!(r["summary"]["done"], 1);
    }

    #[tokio::test]
    async fn unknown_action_rejected() {
        let t = PlanningTool::new();
        let err = t.run(json!({"action": "delete"})).await.unwrap_err();
        assert!(format!("{err}").contains("delete"));
    }
}

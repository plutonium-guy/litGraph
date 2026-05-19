//! Reducers for merging `NodeOutput` updates into the running state.
//!
//! A reducer takes the current state and a partial update (as `serde_json::Value`)
//! and produces a new state. Users typically derive reducers via a macro (future work);
//! for now they implement `Reducer<S>` manually or use the built-in helpers.

use serde::{Serialize, de::DeserializeOwned};
use serde_json::Value;

use crate::Result;

pub trait Reducer<S>: Send + Sync + 'static {
    fn apply(&self, current: S, update: Value) -> Result<S>;
}

/// Default reducer: shallow JSON merge where each top-level key is replaced by the
/// update's value. For `Vec<_>` fields that should concatenate, use [`merge_append`].
pub fn merge_replace<S: Serialize + DeserializeOwned>(current: S, update: Value) -> Result<S> {
    let mut base = serde_json::to_value(&current)?;
    if let (Value::Object(base_map), Value::Object(upd_map)) = (&mut base, update) {
        for (k, v) in upd_map {
            base_map.insert(k, v);
        }
    }
    Ok(serde_json::from_value(base)?)
}

/// Merge update into current; for keys whose value is an array in both, concatenate.
/// Non-array keys follow replace semantics.
pub fn merge_append<S: Serialize + DeserializeOwned>(current: S, update: Value) -> Result<S> {
    let mut base = serde_json::to_value(&current)?;
    if let (Value::Object(base_map), Value::Object(upd_map)) = (&mut base, update) {
        for (k, v) in upd_map {
            match (base_map.get_mut(&k), v) {
                (Some(Value::Array(a)), Value::Array(b)) => a.extend(b),
                (_, v) => { base_map.insert(k, v); }
            }
        }
    }
    Ok(serde_json::from_value(base)?)
}

/// Merge update into current; arrays concatenate AND get deduped by the
/// JSON value pointed to by `key` on each element. First occurrence wins
/// (preserves insertion order — important for `Send`-fan-out workloads
/// where determinism matters more than recency).
///
/// Non-array keys follow replace semantics (same as `merge_append`).
/// Array elements that aren't JSON objects, or that lack `key`, pass
/// through unchanged — partial dedup is a footgun nobody asked for.
///
/// Closes the "branch fan-in dedup-by-key reducer" Tier-2 graph gap.
/// Typical use after `add_parallel_for`:
///
/// ```text
/// each branch emits { "docs": [{ "id": "X", "text": ... }] }
///         ↓
/// reducer drops duplicate-id docs across branches
/// ```
///
/// The `key` arg is the JSON object-key on each array element — NOT a
/// JSON pointer. Stick to one-level scalar keys; nested dedup belongs
/// in user code.
pub fn merge_dedup_by_key<S: Serialize + DeserializeOwned>(
    current: S,
    update: Value,
    key: &str,
) -> Result<S> {
    let mut base = serde_json::to_value(&current)?;
    if let (Value::Object(base_map), Value::Object(upd_map)) = (&mut base, update) {
        for (k, v) in upd_map {
            match (base_map.get_mut(&k), v) {
                (Some(Value::Array(a)), Value::Array(b)) => {
                    // Build a set of seen keys from current array first
                    // so already-present items aren't re-appended.
                    use std::collections::HashSet;
                    let mut seen: HashSet<String> = HashSet::new();
                    for item in a.iter() {
                        if let Some(k_val) = item.get(key) {
                            seen.insert(k_val.to_string());
                        }
                    }
                    for item in b {
                        let dedup_key = item.get(key).map(|v| v.to_string());
                        match dedup_key {
                            Some(k_val) => {
                                if seen.insert(k_val) {
                                    a.push(item);
                                }
                                // else: already seen — drop.
                            }
                            None => a.push(item),
                        }
                    }
                }
                (_, v) => { base_map.insert(k, v); }
            }
        }
    }
    Ok(serde_json::from_value(base)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone, Default, Debug, PartialEq, Serialize, Deserialize)]
    struct Docs {
        docs: Vec<serde_json::Value>,
    }

    #[test]
    fn merge_dedup_by_key_drops_duplicate_ids() {
        let s = Docs {
            docs: vec![
                json!({"id": "A", "text": "alpha"}),
                json!({"id": "B", "text": "bravo"}),
            ],
        };
        let upd = json!({
            "docs": [
                {"id": "B", "text": "bravo-dup"},
                {"id": "C", "text": "charlie"},
            ]
        });
        let out = merge_dedup_by_key(s, upd, "id").unwrap();
        assert_eq!(out.docs.len(), 3, "B should not duplicate; got {:?}", out.docs);
        let ids: Vec<String> = out
            .docs
            .iter()
            .filter_map(|d| d.get("id").and_then(|v| v.as_str()).map(|s| s.to_string()))
            .collect();
        assert_eq!(ids, vec!["A", "B", "C"]);
        // First B wins.
        assert_eq!(out.docs[1]["text"], "bravo");
    }

    #[test]
    fn merge_dedup_by_key_passes_through_elements_without_key() {
        // Elements lacking the key are appended as-is — no partial-dedup
        // silent failure modes.
        let s = Docs { docs: vec![json!({"id": "A"})] };
        let upd = json!({
            "docs": [
                {"id": "A"},                  // dup → drop
                {"text": "no-id"},            // missing key → keep
                {"id": "B"},                  // new → keep
            ]
        });
        let out = merge_dedup_by_key(s, upd, "id").unwrap();
        assert_eq!(out.docs.len(), 3);
        assert_eq!(out.docs[0]["id"], "A");
        assert_eq!(out.docs[1]["text"], "no-id");
        assert_eq!(out.docs[2]["id"], "B");
    }

    #[test]
    fn merge_dedup_by_key_non_array_key_uses_replace() {
        #[derive(Clone, Default, Serialize, Deserialize)]
        struct Mixed {
            label: String,
            docs: Vec<serde_json::Value>,
        }
        let s = Mixed { label: "old".into(), docs: vec![] };
        let upd = json!({"label": "new", "docs": [{"id": "A"}]});
        let out = merge_dedup_by_key(s, upd, "id").unwrap();
        assert_eq!(out.label, "new");
        assert_eq!(out.docs.len(), 1);
    }

    #[test]
    fn merge_dedup_by_key_idempotent_across_repeat_merges() {
        let s = Docs { docs: vec![] };
        let upd = json!({"docs": [{"id": "X"}]});
        let mid = merge_dedup_by_key(s, upd.clone(), "id").unwrap();
        let after = merge_dedup_by_key(mid, upd, "id").unwrap();
        // Re-applying the same update must not duplicate.
        assert_eq!(after.docs.len(), 1);
    }
}


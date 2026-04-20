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


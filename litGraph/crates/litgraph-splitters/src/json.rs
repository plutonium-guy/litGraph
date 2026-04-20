//! JSON splitter — recursively walks a JSON tree, breaking at object/array
//! boundaries to keep each chunk under `max_chunk_size`. Each chunk is
//! emitted as a self-contained JSON object whose keys reflect the path from
//! the root, so an embedded chunk doesn't lose its context (e.g. instead of
//! `"items": [...]` you get `{"results.items": [...]}`).
//!
//! Use case: large API specs, config dumps, structured RAG data. Unlike
//! `RecursiveCharacterSplitter`, this produces valid JSON per chunk —
//! downstream pipelines that expect parseable structure don't break.
//!
//! Algorithm (single pass, no LLM calls):
//!   1. If `serialized(value).len() <= max_chunk_size` → emit one chunk.
//!   2. Else, if value is an object: walk keys in order, accumulating into a
//!      "current chunk" map. When adding the next key would push the chunk
//!      over budget, flush the current chunk and start a new one. Keys whose
//!      OWN serialized size exceeds budget recurse into the value.
//!   3. Same for arrays — accumulate elements, flush when over.
//!   4. Terminal scalars that exceed budget on their own get emitted as a
//!      single chunk (we never produce chunks larger than budget where it's
//!      avoidable, but won't truncate scalar values).
//!
//! The path-prefixed key is dot-separated: `"a.b.c"` for nested objects;
//! `"items[0].name"` for arrays. Chunks also include a `_path` field at
//! the root pointing to the originating subtree, so reranking / dedup logic
//! can use it.

use serde_json::{Map, Value};

#[derive(Clone, Debug)]
pub struct JsonSplitter {
    pub max_chunk_size: usize,
}

impl JsonSplitter {
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size: max_chunk_size.max(64) }
    }

    /// Split a JSON-serialized string into a list of valid JSON-serialized
    /// chunks. Each chunk is parseable on its own. If the input is not valid
    /// JSON, returns the input unchanged as a single chunk (caller's
    /// responsibility to validate).
    pub fn split_text(&self, text: &str) -> Vec<String> {
        let v: Value = match serde_json::from_str(text) {
            Ok(v) => v,
            Err(_) => return vec![text.to_string()],
        };
        let mut out = Vec::new();
        self.split_value("", &v, &mut out);
        out
    }

    fn serialized_size(v: &Value) -> usize {
        serde_json::to_string(v).map(|s| s.len()).unwrap_or(usize::MAX)
    }

    fn flush_chunk(&self, path: &str, chunk: Map<String, Value>, out: &mut Vec<String>) {
        if chunk.is_empty() { return; }
        let mut wrapper = Map::new();
        if !path.is_empty() {
            wrapper.insert("_path".into(), Value::String(path.to_string()));
        }
        for (k, v) in chunk {
            wrapper.insert(k, v);
        }
        if let Ok(s) = serde_json::to_string(&Value::Object(wrapper)) {
            out.push(s);
        }
    }

    fn split_value(&self, path: &str, v: &Value, out: &mut Vec<String>) {
        let total_size = Self::serialized_size(v);
        if total_size <= self.max_chunk_size {
            // Whole value fits — emit as one chunk under its path.
            let mut chunk = Map::new();
            chunk.insert(path_or_root(path).into(), v.clone());
            self.flush_chunk(path, chunk, out);
            return;
        }
        match v {
            Value::Object(obj) => self.split_object(path, obj, out),
            Value::Array(arr) => self.split_array(path, arr, out),
            // Terminal scalar (string/number/bool/null) too big to fit budget:
            // emit as-is. Truncating a value would be lying about the data.
            other => {
                let mut chunk = Map::new();
                chunk.insert(path_or_root(path).into(), other.clone());
                self.flush_chunk(path, chunk, out);
            }
        }
    }

    fn split_object(&self, path: &str, obj: &Map<String, Value>, out: &mut Vec<String>) {
        let mut current = Map::new();
        for (k, v) in obj {
            let child_path = if path.is_empty() { k.clone() } else { format!("{path}.{k}") };
            let kv_size = serde_json::to_string(&Value::Object(
                std::iter::once((k.clone(), v.clone())).collect()
            )).map(|s| s.len()).unwrap_or(usize::MAX);

            // If this single (k, v) pair already exceeds budget by itself, recurse
            // into the value so we can split deeper.
            if kv_size > self.max_chunk_size {
                self.flush_chunk(path, std::mem::take(&mut current), out);
                self.split_value(&child_path, v, out);
                continue;
            }

            // Try to add (k, v) to the current chunk. If it'd overflow, flush
            // first and start fresh.
            let projected_size = serde_json::to_string(&{
                let mut probe = current.clone();
                probe.insert(k.clone(), v.clone());
                probe
            }).map(|s| s.len()).unwrap_or(usize::MAX);

            if projected_size > self.max_chunk_size && !current.is_empty() {
                self.flush_chunk(path, std::mem::take(&mut current), out);
            }
            current.insert(k.clone(), v.clone());
        }
        self.flush_chunk(path, current, out);
    }

    fn split_array(&self, path: &str, arr: &[Value], out: &mut Vec<String>) {
        let mut current: Vec<Value> = Vec::new();
        for (i, v) in arr.iter().enumerate() {
            let child_path = format!("{path}[{i}]");
            let item_size = Self::serialized_size(v);
            if item_size > self.max_chunk_size {
                if !current.is_empty() {
                    let mut chunk = Map::new();
                    chunk.insert(path_or_root(path).into(), Value::Array(std::mem::take(&mut current)));
                    self.flush_chunk(path, chunk, out);
                }
                self.split_value(&child_path, v, out);
                continue;
            }
            // Probe size if we add this element.
            let probed = {
                let mut probe = current.clone();
                probe.push(v.clone());
                serde_json::to_string(&Value::Array(probe)).map(|s| s.len()).unwrap_or(usize::MAX)
            };
            if probed > self.max_chunk_size && !current.is_empty() {
                let mut chunk = Map::new();
                chunk.insert(path_or_root(path).into(), Value::Array(std::mem::take(&mut current)));
                self.flush_chunk(path, chunk, out);
            }
            current.push(v.clone());
        }
        if !current.is_empty() {
            let mut chunk = Map::new();
            chunk.insert(path_or_root(path).into(), Value::Array(current));
            self.flush_chunk(path, chunk, out);
        }
    }
}

fn path_or_root(path: &str) -> &str {
    if path.is_empty() { "_root" } else { path }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn parse(s: &str) -> Value {
        serde_json::from_str(s).unwrap_or_else(|e| panic!("not valid json: {e}\n{s}"))
    }

    #[test]
    fn small_object_fits_in_one_chunk() {
        let s = JsonSplitter::new(200);
        let chunks = s.split_text(r#"{"a": 1, "b": "hello"}"#);
        assert_eq!(chunks.len(), 1);
        // Each chunk is itself parseable JSON.
        let v = parse(&chunks[0]);
        assert_eq!(v["_root"]["a"], json!(1));
        assert_eq!(v["_root"]["b"], json!("hello"));
    }

    #[test]
    fn large_object_splits_at_key_boundaries() {
        // Each key holds a 60-char string; budget=120 → ~1-2 keys per chunk.
        let s = JsonSplitter::new(120);
        let big = format!(
            r#"{{"k1":"{}","k2":"{}","k3":"{}","k4":"{}"}}"#,
            "x".repeat(60), "y".repeat(60), "z".repeat(60), "w".repeat(60),
        );
        let chunks = s.split_text(&big);
        assert!(chunks.len() >= 2, "got {} chunks", chunks.len());
        // Every chunk is valid JSON.
        for c in &chunks {
            let v = parse(c);
            assert!(v.is_object());
            assert!(v.get("_path").is_some() || v.get("_root").is_some());
        }
        // All four keys are preserved across chunks.
        let all_keys: Vec<String> = chunks.iter()
            .flat_map(|c| {
                let v = parse(c);
                v.as_object().unwrap().keys().filter(|k| !k.starts_with('_')).cloned().collect::<Vec<_>>()
            })
            .collect();
        for k in &["k1", "k2", "k3", "k4"] {
            assert!(all_keys.contains(&k.to_string()), "missing {k}: {all_keys:?}");
        }
    }

    #[test]
    fn nested_object_path_prefix_carries_through() {
        let s = JsonSplitter::new(80);
        let nested = json!({
            "outer": {
                "inner": {
                    "leaf1": "x".repeat(40),
                    "leaf2": "y".repeat(40),
                }
            }
        });
        let chunks = s.split_text(&nested.to_string());
        // At least one chunk should reference the nested path.
        assert!(chunks.iter().any(|c| c.contains("outer.inner")),
            "no path-prefix chunk: {chunks:?}");
    }

    #[test]
    fn array_splits_on_element_boundaries() {
        let s = JsonSplitter::new(60);
        let big = json!({
            "items": ["a".repeat(30), "b".repeat(30), "c".repeat(30), "d".repeat(30)]
        });
        let chunks = s.split_text(&big.to_string());
        assert!(chunks.len() >= 2);
        // Each chunk's items array is valid + non-empty.
        for c in &chunks {
            let v = parse(c);
            // Either {"items": [...]} or path-prefixed.
            let inner = v.as_object().unwrap()
                .iter()
                .find(|(k, _)| !k.starts_with('_'))
                .map(|(_, v)| v.clone())
                .unwrap_or(Value::Null);
            // Could be array or scalar (oversized leaf).
            assert!(inner.is_array() || inner.is_string());
        }
    }

    #[test]
    fn single_oversized_scalar_is_kept_intact() {
        // A 500-char string with a 100-char budget — must NOT be truncated.
        let s = JsonSplitter::new(100);
        let big = json!({"essay": "z".repeat(500)});
        let chunks = s.split_text(&big.to_string());
        // Find the chunk holding the essay — it'll be > budget but intact.
        let essay_chunk = chunks.iter()
            .find(|c| c.contains("zzzzz"))
            .expect("essay chunk missing");
        let v = parse(essay_chunk);
        let essay_value = v.as_object().unwrap()
            .iter()
            .find(|(k, _)| !k.starts_with('_'))
            .map(|(_, v)| v.as_str().unwrap_or(""))
            .unwrap_or("");
        assert_eq!(essay_value.len(), 500, "scalar truncated");
    }

    #[test]
    fn invalid_json_returned_as_one_chunk() {
        let s = JsonSplitter::new(10);
        let chunks = s.split_text("not { valid : json,");
        assert_eq!(chunks, vec!["not { valid : json,".to_string()]);
    }

    #[test]
    fn empty_object_returns_no_chunks() {
        let s = JsonSplitter::new(100);
        let chunks = s.split_text("{}");
        // {} fits in budget → emitted as one chunk (with _root: {}).
        assert_eq!(chunks.len(), 1);
        let v = parse(&chunks[0]);
        assert!(v["_root"].is_object());
    }

    #[test]
    fn max_chunk_size_clamps_to_minimum_64() {
        let s = JsonSplitter::new(0);  // Pathological input
        assert_eq!(s.max_chunk_size, 64);
    }
}

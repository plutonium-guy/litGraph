//! Python bindings for the long-term memory `Store` (namespace+key JSON
//! document store, LangGraph-parity). One in-process `InMemoryStore` exposed
//! today; Postgres / Redis backends will register here when their crates land.

use std::sync::Arc;

use litgraph_core::semantic_store::SemanticStore;
use litgraph_core::store::{InMemoryStore, SearchFilter, Store, StoreItem};
use pyo3::exceptions::{PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::embeddings::extract_embeddings;
use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyInMemoryStore>()?;
    m.add_class::<PySemanticStore>()?;
    Ok(())
}

fn ns_from_py(value: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    if let Ok(seq) = value.downcast::<PyTuple>() {
        seq.iter().map(|s| s.extract::<String>()).collect()
    } else if let Ok(seq) = value.downcast::<PyList>() {
        seq.iter().map(|s| s.extract::<String>()).collect()
    } else if let Ok(s) = value.extract::<String>() {
        Ok(vec![s])
    } else {
        Err(PyRuntimeError::new_err(
            "namespace must be a tuple/list of strings or a single string",
        ))
    }
}

fn json_to_py<'py>(py: Python<'py>, value: &serde_json::Value) -> PyResult<Bound<'py, PyAny>> {
    let json_mod = py.import_bound("json")?;
    let s = serde_json::to_string(value)
        .map_err(|e| PyRuntimeError::new_err(format!("json encode: {e}")))?;
    json_mod.call_method1("loads", (s,))
}

fn py_to_json(value: &Bound<'_, PyAny>) -> PyResult<serde_json::Value> {
    let py = value.py();
    let json_mod = py.import_bound("json")?;
    let s: String = json_mod
        .call_method1("dumps", (value.clone(),))?
        .extract()?;
    serde_json::from_str(&s).map_err(|e| PyRuntimeError::new_err(format!("json decode: {e}")))
}

fn item_to_py<'py>(py: Python<'py>, item: &StoreItem) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new_bound(py);
    dict.set_item("namespace", item.namespace.clone())?;
    dict.set_item("key", item.key.clone())?;
    dict.set_item("value", json_to_py(py, &item.value)?)?;
    dict.set_item("expires_at_ms", item.expires_at_ms)?;
    dict.set_item("created_at_ms", item.created_at_ms)?;
    dict.set_item("updated_at_ms", item.updated_at_ms)?;
    Ok(dict)
}

#[pyclass(name = "InMemoryStore", module = "litgraph.store")]
pub(crate) struct PyInMemoryStore {
    inner: InMemoryStore,
}

#[pymethods]
impl PyInMemoryStore {
    #[new]
    fn new() -> Self {
        Self {
            inner: InMemoryStore::new(),
        }
    }

    /// Insert / replace a JSON document at `(namespace, key)`. `ttl_ms` is
    /// relative-from-now milliseconds; omit for non-expiring entries.
    #[pyo3(signature = (namespace, key, value, ttl_ms=None))]
    fn put<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
        value: Bound<'py, PyAny>,
        ttl_ms: Option<u64>,
    ) -> PyResult<()> {
        let ns = ns_from_py(&namespace)?;
        let v = py_to_json(&value)?;
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.put(&ns, &key, &v, ttl_ms).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Fetch the document at `(namespace, key)`. Returns `None` if absent or
    /// expired.
    fn get<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        let item = py
            .allow_threads(|| {
                block_on_compat(async move { store.get(&ns, &key).await })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        match item {
            Some(it) => Ok(Some(item_to_py(py, &it)?)),
            None => Ok(None),
        }
    }

    /// Delete `(namespace, key)`. Returns True if it existed.
    fn delete<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
    ) -> PyResult<bool> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ns, &key).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Pop a key — returns the value or raises KeyError.
    fn pop<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        let key_for_err = key.clone();
        let item = py
            .allow_threads(|| {
                block_on_compat(async move {
                    let got = store.get(&ns, &key).await?;
                    if got.is_some() {
                        store.delete(&ns, &key).await?;
                    }
                    Ok::<_, litgraph_core::Error>(got)
                })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        match item {
            Some(it) => item_to_py(py, &it),
            None => Err(PyKeyError::new_err(key_for_err)),
        }
    }

    /// List items in (or under) a namespace prefix. Filters: `query_text`
    /// (substring match, case-insensitive) and `matches` (list of
    /// `(json_pointer, expected_value)` tuples). Sort: most-recently-updated
    /// first.
    #[pyo3(signature = (namespace_prefix, query_text=None, matches=None, limit=None, offset=None))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        namespace_prefix: Bound<'py, PyAny>,
        query_text: Option<String>,
        matches: Option<Bound<'py, PyList>>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let ns = ns_from_py(&namespace_prefix)?;
        let mut filter = SearchFilter {
            limit,
            offset,
            query_text,
            ..Default::default()
        };
        if let Some(list) = matches {
            for item in list.iter() {
                let tup = item.downcast::<PyTuple>().map_err(|_| {
                    PyRuntimeError::new_err("matches entries must be (path, value) tuples")
                })?;
                if tup.len() != 2 {
                    return Err(PyRuntimeError::new_err(
                        "matches tuples must be length 2",
                    ));
                }
                let path: String = tup.get_item(0)?.extract()?;
                let value = py_to_json(&tup.get_item(1)?)?;
                filter.matches.push((path, value));
            }
        }
        let store = self.inner.clone();
        let hits = py
            .allow_threads(|| {
                block_on_compat(async move { store.search(&ns, &filter).await })
                    .map_err(|e| PyRuntimeError::new_err(e.to_string()))
            })?;
        hits.iter().map(|it| item_to_py(py, it)).collect()
    }

    /// List the distinct namespaces under a prefix. Useful for tenant
    /// discovery (e.g. all users with stored memory).
    #[pyo3(signature = (prefix=None, limit=None))]
    fn list_namespaces<'py>(
        &self,
        py: Python<'py>,
        prefix: Option<Bound<'py, PyAny>>,
        limit: Option<usize>,
    ) -> PyResult<Vec<Vec<String>>> {
        let p = match prefix {
            Some(b) => ns_from_py(&b)?,
            None => Vec::new(),
        };
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.list_namespaces(&p, limit).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __repr__(&self) -> String {
        format!("InMemoryStore(items={})", self.inner.len())
    }
}

/// Long-term memory store with semantic-search recall. Wraps any
/// `Store` (today: `InMemoryStore`) with an `Embeddings` provider so
/// `semantic_search` can rank items by meaning, not just exact match.
///
/// Internal storage shape: each item's value is wrapped as
/// `{"_emb": [...], "_text": "...", "value": <user value>}` — durable
/// across crashes / processes for backends that persist (Postgres,
/// Redis). `get` strips the wrapper before returning.
///
/// The `semantic_search` cosine pass runs in **Rayon-parallel** over
/// every item in the namespace. Brute force, fine up to ~10k items
/// per namespace; reach for `pgvector` / `hnsw` above that.
///
/// ```python
/// from litgraph.store import InMemoryStore, SemanticStore
/// from litgraph.embeddings import OpenAIEmbeddings
/// emb = OpenAIEmbeddings(api_key="sk-...")
/// store = InMemoryStore()
/// sem = SemanticStore(store, emb)
/// sem.put(("users", "alice"), "fact:1", "alice loves rust", {"id": 1})
/// hits = sem.search(("users", "alice"), "preferred languages?", k=3)
/// for h in hits:
///     print(h["score"], h["text"], h["value"])
/// ```
#[pyclass(name = "SemanticStore", module = "litgraph.store")]
pub(crate) struct PySemanticStore {
    inner: SemanticStore,
}

#[pymethods]
impl PySemanticStore {
    #[new]
    fn new(store: Bound<'_, PyAny>, embedder: Bound<'_, PyAny>) -> PyResult<Self> {
        let inner_store: Arc<dyn Store> = if let Ok(s) = store.extract::<PyRef<PyInMemoryStore>>() {
            Arc::new(s.inner.clone())
        } else {
            return Err(PyRuntimeError::new_err(
                "store must be a litgraph InMemoryStore (more backends to come)",
            ));
        };
        let emb = extract_embeddings(&embedder)?;
        Ok(Self {
            inner: SemanticStore::new(inner_store, emb),
        })
    }

    /// Embed `text`, store at `(namespace, key)` alongside `value`.
    /// Subsequent `search(namespace, query)` calls rank by cosine
    /// similarity between `embed(query)` and stored embeddings.
    #[pyo3(signature = (namespace, key, text, value=None, ttl_ms=None))]
    fn put<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
        text: String,
        value: Option<Bound<'py, PyAny>>,
        ttl_ms: Option<u64>,
    ) -> PyResult<()> {
        let ns = ns_from_py(&namespace)?;
        let v = match value {
            Some(v) => py_to_json(&v)?,
            None => serde_json::Value::Null,
        };
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.put(&ns, &key, &text, v, ttl_ms).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Fetch `(namespace, key)`. Returns `{"text", "value"}` or `None`.
    /// The internal `_emb` / `_text` wrapper is stripped before return.
    fn get<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        let got = py.allow_threads(|| {
            block_on_compat(async move { store.get(&ns, &key).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        match got {
            Some((text, value)) => {
                let d = PyDict::new_bound(py);
                d.set_item("text", text)?;
                d.set_item("value", json_to_py(py, &value)?)?;
                Ok(Some(d))
            }
            None => Ok(None),
        }
    }

    /// Delete `(namespace, key)`. True if it existed.
    fn delete<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        key: String,
    ) -> PyResult<bool> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        py.allow_threads(|| {
            block_on_compat(async move { store.delete(&ns, &key).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })
    }

    /// Cosine top-k under `namespace_prefix`. Returns a list of dicts
    /// `{"namespace", "key", "text", "value", "score", "created_at_ms",
    /// "updated_at_ms"}` sorted by score descending.
    #[pyo3(signature = (namespace, query, k=5))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        namespace: Bound<'py, PyAny>,
        query: String,
        k: usize,
    ) -> PyResult<Bound<'py, PyList>> {
        let ns = ns_from_py(&namespace)?;
        let store = self.inner.clone();
        let hits = py.allow_threads(|| {
            block_on_compat(async move { store.semantic_search(&ns, &query, k, None).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let out = PyList::empty_bound(py);
        for h in hits {
            let d = PyDict::new_bound(py);
            d.set_item("namespace", h.namespace)?;
            d.set_item("key", h.key)?;
            d.set_item("text", h.text)?;
            d.set_item("value", json_to_py(py, &h.value)?)?;
            d.set_item("score", h.score)?;
            d.set_item("created_at_ms", h.created_at_ms)?;
            d.set_item("updated_at_ms", h.updated_at_ms)?;
            out.append(d)?;
        }
        Ok(out)
    }

    fn __repr__(&self) -> String {
        "SemanticStore()".into()
    }
}

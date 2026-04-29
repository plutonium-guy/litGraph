//! Python bindings for `litgraph_core::evaluators` — string-distance and
//! semantic-similarity scoring functions for prompt evaluation.

use std::sync::Arc;

use litgraph_core::{
    contains_all as core_contains_all, contains_any as core_contains_any,
    embedding_cosine as core_embedding_cosine, exact_match as core_exact_match,
    exact_match_strict as core_exact_match_strict, jaccard_similarity as core_jaccard_similarity,
    json_validity as core_json_validity, levenshtein as core_levenshtein,
    levenshtein_ratio as core_levenshtein_ratio, luhn_valid as core_luhn_valid,
    regex_match as core_regex_match, run_eval as core_run_eval, ChatModel, ContainsAllScorer,
    EvalCase, EvalDataset, ExactMatchScorer, JaccardScorer, LevenshteinScorer, LlmJudge,
    LlmJudgeScorer, PiiScrubber as CorePiiScrubber, RegexScorer, Scorer,
};
use pyo3::types::PyList;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::runtime::block_on_compat;

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_match, m)?)?;
    m.add_function(wrap_pyfunction!(exact_match_strict, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein, m)?)?;
    m.add_function(wrap_pyfunction!(levenshtein_ratio, m)?)?;
    m.add_function(wrap_pyfunction!(jaccard_similarity, m)?)?;
    m.add_function(wrap_pyfunction!(regex_match, m)?)?;
    m.add_function(wrap_pyfunction!(json_validity, m)?)?;
    m.add_function(wrap_pyfunction!(embedding_cosine, m)?)?;
    m.add_function(wrap_pyfunction!(contains_all, m)?)?;
    m.add_function(wrap_pyfunction!(contains_any, m)?)?;
    m.add_class::<PyLlmJudge>()?;
    m.add_class::<PyPiiScrubber>()?;
    m.add_function(wrap_pyfunction!(luhn_valid, m)?)?;
    m.add_function(wrap_pyfunction!(run_eval, m)?)?;
    Ok(())
}

/// Standalone Luhn-check helper. `True` if the digit string (spaces /
/// dashes OK, they're stripped) passes the Luhn mod-10 checksum —
/// standard credit-card validity filter.
#[pyfunction]
fn luhn_valid(digits: &str) -> bool {
    core_luhn_valid(digits)
}

/// Exact match (trim + case-fold). Returns True if `actual` matches `expected`.
#[pyfunction]
fn exact_match(actual: &str, expected: &str) -> bool {
    core_exact_match(actual, expected)
}

/// Strict byte-equal match. No trimming, no case-folding.
#[pyfunction]
fn exact_match_strict(actual: &str, expected: &str) -> bool {
    core_exact_match_strict(actual, expected)
}

/// Raw Levenshtein edit distance (number of single-char edits).
#[pyfunction]
fn levenshtein(a: &str, b: &str) -> usize {
    core_levenshtein(a, b)
}

/// Levenshtein ratio in `[0.0, 1.0]`. 1.0 = identical.
#[pyfunction]
fn levenshtein_ratio(a: &str, b: &str) -> f32 {
    core_levenshtein_ratio(a, b)
}

/// Jaccard token-set similarity in `[0.0, 1.0]`. Word-order invariant.
#[pyfunction]
fn jaccard_similarity(a: &str, b: &str) -> f32 {
    core_jaccard_similarity(a, b)
}

/// Returns True if the regex pattern matches anywhere in `text`. Raises
/// `ValueError` if the pattern doesn't compile.
#[pyfunction]
fn regex_match(text: &str, pattern: &str) -> PyResult<bool> {
    core_regex_match(text, pattern).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// True if `text` parses as valid JSON.
#[pyfunction]
fn json_validity(text: &str) -> bool {
    core_json_validity(text)
}

/// Cosine similarity of two equal-length embedding vectors. Result in
/// `[-1.0, 1.0]`. Raises `ValueError` if the lengths don't match.
/// Zero-vector inputs return 0.0 (no NaN).
#[pyfunction]
fn embedding_cosine(a: Vec<f32>, b: Vec<f32>) -> PyResult<f32> {
    core_embedding_cosine(&a, &b).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// True if every substring in `needles` appears in `text`.
#[pyfunction]
fn contains_all(text: &str, needles: Vec<String>) -> bool {
    let refs: Vec<&str> = needles.iter().map(|s| s.as_str()).collect();
    core_contains_all(text, &refs)
}

/// True if at least one substring in `needles` appears in `text`.
#[pyfunction]
fn contains_any(text: &str, needles: Vec<String>) -> bool {
    let refs: Vec<&str> = needles.iter().map(|s| s.as_str()).collect();
    core_contains_any(text, &refs)
}

/// LLM-as-judge evaluator. Wraps a ChatModel to score prediction vs
/// reference on a 0.0–1.0 scale. Returns `{score, reasoning}` dict.
///
/// ```python
/// from litgraph.evaluators import LlmJudge
/// from litgraph.providers import OpenAIChat
/// judge_model = OpenAIChat(api_key=..., model="gpt-4o-mini")  # cheap
/// judge = LlmJudge(judge_model, criteria="Rate factual accuracy only.")
/// result = judge.judge(
///     prediction="The capital is Paris.",
///     reference="Paris is the capital of France.",
/// )
/// # {"score": 0.95, "reasoning": "Matches in both facts and intent."}
/// ```
///
/// `criteria=None` uses the default ("match in meaning and factual
/// content, paraphrases allowed"). Pair with a cheap model to keep
/// eval cost low (<$0.001/sample typical).
#[pyclass(name = "LlmJudge", module = "litgraph.evaluators")]
pub struct PyLlmJudge {
    inner: Arc<LlmJudge>,
}

#[pymethods]
impl PyLlmJudge {
    #[new]
    #[pyo3(signature = (model, criteria=None))]
    fn new(model: Py<PyAny>, criteria: Option<String>) -> PyResult<Self> {
        let chat_model: Arc<dyn ChatModel> =
            Python::with_gil(|py| crate::agents::extract_chat_model(model.bind(py)))?;
        let judge = LlmJudge::new(chat_model, criteria);
        Ok(Self { inner: Arc::new(judge) })
    }

    fn judge<'py>(
        &self,
        py: Python<'py>,
        prediction: String,
        reference: String,
    ) -> PyResult<Bound<'py, PyDict>> {
        let judge = self.inner.clone();
        let score = py.allow_threads(|| {
            block_on_compat(async move { judge.judge(&prediction, &reference).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let d = PyDict::new_bound(py);
        d.set_item("score", score.score)?;
        d.set_item("reasoning", score.reasoning)?;
        Ok(d)
    }

    /// Score a list of (prediction, reference) tuples. Serial — one call
    /// per pair. Use concurrent.futures.ThreadPoolExecutor or the agent
    /// stream API for parallelism at the Python level.
    fn judge_batch<'py>(
        &self,
        py: Python<'py>,
        pairs: Vec<(String, String)>,
    ) -> PyResult<Vec<Bound<'py, PyDict>>> {
        let judge = self.inner.clone();
        let scores = py.allow_threads(|| {
            block_on_compat(async move { judge.judge_batch(pairs).await })
                .map_err(|e| PyRuntimeError::new_err(e.to_string()))
        })?;
        let mut out = Vec::with_capacity(scores.len());
        for s in scores {
            let d = PyDict::new_bound(py);
            d.set_item("score", s.score)?;
            d.set_item("reasoning", s.reasoning)?;
            out.push(d);
        }
        Ok(out)
    }
}

/// PII scrubber. Detects emails, phones, SSNs, credit cards (Luhn-validated),
/// AWS access keys, JWTs, IPv4/IPv6 and replaces with `<KIND>` tokens.
/// Returns the scrubbed text plus a list of `{kind, original, redacted, start}`
/// replacements for audit logging.
///
/// ```python
/// from litgraph.evaluators import PiiScrubber
/// scrubber = PiiScrubber()
/// r = scrubber.scrub("Email alice@example.com or call 555-867-5309")
/// r["scrubbed"]       # "Email <EMAIL> or call <PHONE>"
/// r["replacements"]   # [{kind: "EMAIL", original: "alice@...", ...}, ...]
/// ```
///
/// Pass `validate_luhn=False` to mask any 13–19-digit sequence as a
/// credit card (useful for testing; default keeps Luhn on to avoid
/// flagging internal numeric IDs).
///
/// For custom patterns, pass `extra_patterns=[("KIND", "regex"), ...]`.
/// Pass `only_custom=True` to drop the built-in patterns entirely.
#[pyclass(name = "PiiScrubber", module = "litgraph.evaluators")]
pub struct PyPiiScrubber {
    inner: CorePiiScrubber,
}

#[pymethods]
impl PyPiiScrubber {
    #[new]
    #[pyo3(signature = (extra_patterns=None, only_custom=false, validate_luhn=true))]
    fn new(
        extra_patterns: Option<Vec<(String, String)>>,
        only_custom: bool,
        validate_luhn: bool,
    ) -> PyResult<Self> {
        let mut inner = if only_custom {
            CorePiiScrubber::new().only_custom()
        } else {
            CorePiiScrubber::new()
        };
        if !validate_luhn {
            inner = inner.without_luhn();
        }
        if let Some(pats) = extra_patterns {
            let mut compiled = Vec::with_capacity(pats.len());
            for (label, pattern) in pats {
                let re = regex::Regex::new(&pattern).map_err(|e| {
                    PyValueError::new_err(format!("invalid regex `{pattern}`: {e}"))
                })?;
                compiled.push((label, re));
            }
            inner = inner.with_patterns(compiled);
        }
        Ok(Self { inner })
    }

    fn scrub<'py>(&self, py: Python<'py>, text: &str) -> PyResult<Bound<'py, PyDict>> {
        let result = self.inner.scrub(text);
        let out = PyDict::new_bound(py);
        out.set_item("scrubbed", result.scrubbed)?;
        let reps = pyo3::types::PyList::empty_bound(py);
        for r in result.replacements {
            let d = PyDict::new_bound(py);
            d.set_item("kind", r.kind)?;
            d.set_item("original", r.original)?;
            d.set_item("redacted", r.redacted)?;
            d.set_item("start", r.start)?;
            reps.append(d)?;
        }
        out.set_item("replacements", reps)?;
        Ok(out)
    }
}

impl PyPiiScrubber {
    pub(crate) fn clone_inner(&self) -> CorePiiScrubber {
        self.inner.clone()
    }
}

/// Run a golden-dataset eval against a target callable. Returns a dict
/// `{per_case: [...], aggregate: {n_cases, n_errors, means: {...}}}`.
///
/// `cases` is a list of dicts, each `{"input": str, "expected": str?, "metadata": any?}`.
/// `target` is a Python callable: `(input_str) -> str`. Throws? → recorded
/// as per-case error, doesn't abort the run.
/// `scorers` is a list of dicts:
///   - `{"name": "exact_match"}`
///   - `{"name": "jaccard"}`
///   - `{"name": "levenshtein"}`
///   - `{"name": "contains_all", "required": ["foo", "bar"]}`
///   - `{"name": "regex", "pattern": r"\d+"}`
/// `max_parallel` (default 4) bounds in-flight target invocations.
///
/// ```python
/// from litgraph.evaluators import run_eval
/// def my_agent(question):
///     return chat.invoke([{"role": "user", "content": question}])["text"]
///
/// report = run_eval(
///     cases=[
///         {"input": "what is 2+2?", "expected": "4"},
///         {"input": "capital of france?", "expected": "Paris"},
///     ],
///     target=my_agent,
///     scorers=[{"name": "exact_match"}, {"name": "jaccard"}],
///     max_parallel=4,
/// )
/// print(report["aggregate"]["means"])
/// ```
#[pyfunction]
#[pyo3(signature = (cases, target, scorers, max_parallel=4))]
fn run_eval<'py>(
    py: Python<'py>,
    cases: Bound<'py, PyList>,
    target: Py<PyAny>,
    scorers: Bound<'py, PyList>,
    max_parallel: usize,
) -> PyResult<Bound<'py, PyDict>> {
    // Parse cases into EvalCase vec.
    let mut case_vec: Vec<EvalCase> = Vec::with_capacity(cases.len());
    for item in cases.iter() {
        let d: Bound<'_, PyDict> = item
            .downcast_into()
            .map_err(|_| PyValueError::new_err("each case must be a dict"))?;
        let input: String = d
            .get_item("input")?
            .ok_or_else(|| PyValueError::new_err("case missing `input`"))?
            .extract()?;
        let expected: Option<String> = match d.get_item("expected")? {
            Some(v) if !v.is_none() => Some(v.extract()?),
            _ => None,
        };
        let metadata = match d.get_item("metadata")? {
            Some(v) if !v.is_none() => crate::graph::py_to_json(py, &v)?,
            _ => serde_json::Value::Null,
        };
        let mut c = EvalCase::new(input);
        if let Some(e) = expected {
            c = c.with_expected(e);
        }
        c = c.with_metadata(metadata);
        case_vec.push(c);
    }
    let dataset = EvalDataset::new(case_vec);

    // Parse scorers into Arc<dyn Scorer> vec.
    let mut scorer_vec: Vec<Arc<dyn Scorer>> = Vec::new();
    for item in scorers.iter() {
        let d: Bound<'_, PyDict> = item
            .downcast_into()
            .map_err(|_| PyValueError::new_err("each scorer must be a dict"))?;
        let name: String = d
            .get_item("name")?
            .ok_or_else(|| PyValueError::new_err("scorer missing `name`"))?
            .extract()?;
        let scorer: Arc<dyn Scorer> = match name.as_str() {
            "exact_match" => Arc::new(ExactMatchScorer),
            "jaccard" => Arc::new(JaccardScorer),
            "levenshtein" => Arc::new(LevenshteinScorer),
            "contains_all" => {
                let required: Vec<String> = d
                    .get_item("required")?
                    .ok_or_else(|| PyValueError::new_err("contains_all scorer needs `required`"))?
                    .extract()?;
                Arc::new(ContainsAllScorer { required })
            }
            "regex" => {
                let pattern: String = d
                    .get_item("pattern")?
                    .ok_or_else(|| PyValueError::new_err("regex scorer needs `pattern`"))?
                    .extract()?;
                Arc::new(RegexScorer { pattern })
            }
            "llm_judge" => {
                let model_obj = d
                    .get_item("model")?
                    .ok_or_else(|| PyValueError::new_err("llm_judge scorer needs `model`"))?;
                let chat = crate::agents::extract_chat_model(&model_obj)?;
                let criteria: Option<String> = match d.get_item("criteria")? {
                    Some(v) if !v.is_none() => Some(v.extract()?),
                    _ => None,
                };
                let custom_name: Option<String> = match d.get_item("scorer_name")? {
                    Some(v) if !v.is_none() => Some(v.extract()?),
                    _ => None,
                };
                let mut scorer = LlmJudgeScorer::new(chat, criteria);
                if let Some(n) = custom_name {
                    scorer = scorer.with_name(n);
                }
                Arc::new(scorer)
            }
            other => {
                return Err(PyValueError::new_err(format!(
                    "unknown scorer `{other}` (valid: exact_match, jaccard, levenshtein, contains_all, regex, llm_judge)"
                )));
            }
        };
        scorer_vec.push(scorer);
    }

    // Wrap the Python callable as an async target (acquires GIL per call).
    let target_arc = Arc::new(target);
    let target_fn = move |input: String| {
        let cb = target_arc.clone();
        async move {
            Python::with_gil(|py| -> Result<String, litgraph_core::Error> {
                match cb.call1(py, (input,)) {
                    Ok(v) => v
                        .extract::<String>(py)
                        .map_err(|e| litgraph_core::Error::other(format!("target return type: {e}"))),
                    Err(e) => Err(litgraph_core::Error::other(format!("target threw: {e}"))),
                }
            })
        }
    };

    let report = py.allow_threads(|| {
        crate::runtime::block_on_compat(async move {
            core_run_eval(&dataset, &scorer_vec, target_fn, max_parallel).await
        })
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))
    })?;

    // Convert report → Python dict via serde_json round-trip.
    let json_str = serde_json::to_string(&report)
        .map_err(|e| PyRuntimeError::new_err(format!("report serialize: {e}")))?;
    let json_mod = py.import_bound("json")?;
    let parsed = json_mod.call_method1("loads", (json_str,))?;
    let dict: Bound<'_, PyDict> = parsed
        .downcast_into()
        .map_err(|_| PyRuntimeError::new_err("report wasn't a dict"))?;
    Ok(dict)
}

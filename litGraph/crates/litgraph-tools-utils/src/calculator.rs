//! Safe arithmetic + boolean expression evaluator built on `evalexpr`.
//!
//! Sandboxed: no I/O, no syscalls, no eval-of-arbitrary-code. The agent gives
//! us a math expression, we evaluate it numerically. Supports +, -, *, /, %,
//! ^ (power), parentheses, basic math functions exposed by evalexpr (math::sqrt,
//! math::sin, etc.), and boolean comparisons.

use async_trait::async_trait;
use evalexpr::eval;
use litgraph_core::tool::{Tool, ToolSchema};
use litgraph_core::{Error, Result};
use serde_json::{Value, json};

#[derive(Debug, Clone, Default)]
pub struct CalculatorTool;

impl CalculatorTool {
    pub fn new() -> Self { Self }
}

#[async_trait]
impl Tool for CalculatorTool {
    fn schema(&self) -> ToolSchema {
        ToolSchema {
            name: "calculator".into(),
            description: "Evaluate a math expression. Supports arithmetic (+ - * / % ^), \
                         parentheses, and math functions like math::sqrt, math::sin, math::cos, \
                         math::ln. Returns the numeric result.".into(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Math expression to evaluate. Example: '2 + 2 * 3' or 'math::sqrt(144)'."
                    }
                },
                "required": ["expression"]
            }),
        }
    }

    async fn run(&self, args: Value) -> Result<Value> {
        let expr = args
            .get("expression")
            .and_then(|v| v.as_str())
            .ok_or_else(|| Error::invalid("calculator: missing `expression`"))?;
        match eval(expr) {
            Ok(v) => {
                // evalexpr returns a Value enum; normalize to a JSON value.
                let out: Value = match v {
                    evalexpr::Value::Int(i) => json!(i),
                    evalexpr::Value::Float(f) => json!(f),
                    evalexpr::Value::Boolean(b) => json!(b),
                    evalexpr::Value::String(s) => json!(s),
                    evalexpr::Value::Empty => Value::Null,
                    evalexpr::Value::Tuple(t) => {
                        let xs: Vec<Value> = t.iter().map(eval_value_to_json).collect();
                        json!(xs)
                    }
                };
                Ok(json!({ "result": out }))
            }
            Err(e) => Err(Error::invalid(format!("calculator: {e}"))),
        }
    }
}

fn eval_value_to_json(v: &evalexpr::Value) -> Value {
    match v {
        evalexpr::Value::Int(i) => json!(*i),
        evalexpr::Value::Float(f) => json!(*f),
        evalexpr::Value::Boolean(b) => json!(*b),
        evalexpr::Value::String(s) => json!(s),
        evalexpr::Value::Empty => Value::Null,
        evalexpr::Value::Tuple(t) => json!(t.iter().map(eval_value_to_json).collect::<Vec<_>>()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn evaluates_arithmetic() {
        let t = CalculatorTool::new();
        let out = t.run(json!({"expression": "2 + 2 * 3"})).await.unwrap();
        assert_eq!(out["result"], json!(8));
    }

    #[tokio::test]
    async fn evaluates_with_parens_and_power() {
        let t = CalculatorTool::new();
        let out = t.run(json!({"expression": "(1 + 2) ^ 3"})).await.unwrap();
        // evalexpr's `^` is float power → Float(27.0); just assert numeric value.
        let n = out["result"].as_f64().unwrap_or_else(|| out["result"].as_i64().unwrap() as f64);
        assert!((n - 27.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn evaluates_math_functions() {
        let t = CalculatorTool::new();
        let out = t.run(json!({"expression": "math::sqrt(144.0)"})).await.unwrap();
        assert_eq!(out["result"].as_f64().unwrap(), 12.0);
    }

    #[tokio::test]
    async fn rejects_missing_expression() {
        let t = CalculatorTool::new();
        let err = t.run(json!({})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[tokio::test]
    async fn rejects_unknown_identifier() {
        // evalexpr returns an error for unbound vars — we surface as InvalidInput.
        let t = CalculatorTool::new();
        let err = t.run(json!({"expression": "open(/etc/passwd)"})).await.unwrap_err();
        assert!(matches!(err, Error::InvalidInput(_)));
    }

    #[test]
    fn schema_is_well_formed() {
        let t = CalculatorTool::new();
        let s = t.schema();
        assert_eq!(s.name, "calculator");
        assert_eq!(s.parameters["properties"]["expression"]["type"], json!("string"));
    }
}

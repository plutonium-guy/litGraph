use litgraph_core::Result as LgResult;
use litgraph_core::tool::Tool;
use litgraph_macros::tool;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

#[derive(Deserialize, JsonSchema)]
pub struct AddArgs { pub a: i64, pub b: i64 }

#[derive(Serialize, JsonSchema)]
pub struct AddOut { pub sum: i64 }

/// Add two integers.
#[tool]
async fn add(args: AddArgs) -> LgResult<AddOut> {
    Ok(AddOut { sum: args.a + args.b })
}

#[tokio::test]
async fn tool_macro_generates_working_tool() {
    let t = Add;
    let schema = t.schema();
    assert_eq!(schema.name, "add");
    assert_eq!(schema.description, "Add two integers.");
    // Schema should describe an object with `a` and `b` integer properties.
    let props = schema.parameters.get("properties").unwrap().as_object().unwrap();
    assert!(props.contains_key("a"));
    assert!(props.contains_key("b"));

    let out = t.run(json!({"a": 2, "b": 40})).await.unwrap();
    assert_eq!(out.get("sum").and_then(|v| v.as_i64()), Some(42));
}

#[derive(Deserialize, JsonSchema)]
pub struct GreetArgs { pub name: String }

#[derive(Serialize, JsonSchema)]
pub struct GreetOut { pub message: String }

#[tool(name = "say_hi", description = "Greet a person by name.")]
async fn greet(args: GreetArgs) -> LgResult<GreetOut> {
    Ok(GreetOut { message: format!("hi {}", args.name) })
}

#[tokio::test]
async fn tool_macro_respects_explicit_name_and_description() {
    let t = Greet;
    let s = t.schema();
    assert_eq!(s.name, "say_hi");
    assert_eq!(s.description, "Greet a person by name.");
    let out = t.run(json!({"name": "world"})).await.unwrap();
    assert_eq!(out.get("message").and_then(|v| v.as_str()), Some("hi world"));
}

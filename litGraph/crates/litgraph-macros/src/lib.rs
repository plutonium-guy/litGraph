//! Procedural macros for litGraph.
//!
//! # `#[tool]`
//!
//! Attribute macro that turns an async function into a `Tool`-implementing struct.
//! The argument must be a single type implementing `Deserialize + JsonSchema`; the
//! return must be `Result<T>` for some `T: Serialize + JsonSchema`.
//!
//! ```ignore
//! use litgraph_core::Result;
//! use litgraph_macros::tool;
//! use serde::{Deserialize, Serialize};
//! use schemars::JsonSchema;
//!
//! #[derive(Deserialize, JsonSchema)]
//! struct AddArgs { a: i64, b: i64 }
//!
//! #[derive(Serialize, JsonSchema)]
//! struct AddOut { sum: i64 }
//!
//! /// Add two integers.
//! #[tool]
//! async fn add(args: AddArgs) -> Result<AddOut> {
//!     Ok(AddOut { sum: args.a + args.b })
//! }
//!
//! // Generated: `struct Add;` and `impl Tool for Add`. Use as `Add`.
//! ```

use proc_macro::TokenStream;
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    Attribute, Error, FnArg, ItemFn, LitStr, Meta, PatType, ReturnType, Type,
    parse::Parse, parse_macro_input, spanned::Spanned,
};

struct ToolArgs {
    name: Option<String>,
    description: Option<String>,
}

impl Parse for ToolArgs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        // Accept: `#[tool]`, `#[tool(name = "...")]`, `#[tool(name = "...", description = "...")]`
        let mut name = None;
        let mut description = None;
        if input.is_empty() { return Ok(Self { name, description }); }
        let metas: syn::punctuated::Punctuated<Meta, syn::Token![,]> =
            syn::punctuated::Punctuated::parse_terminated(input)?;
        for m in metas {
            if let Meta::NameValue(nv) = m {
                let key = nv.path.get_ident()
                    .map(|i| i.to_string())
                    .unwrap_or_default();
                let val = match nv.value {
                    syn::Expr::Lit(expr_lit) => match expr_lit.lit {
                        syn::Lit::Str(s) => s.value(),
                        _ => return Err(Error::new(expr_lit.span(), "expected string literal")),
                    }
                    other => return Err(Error::new(other.span(), "expected string literal")),
                };
                match key.as_str() {
                    "name" => name = Some(val),
                    "description" => description = Some(val),
                    _ => return Err(Error::new(nv.path.span(), format!("unknown key `{key}`"))),
                }
            } else {
                return Err(Error::new(m.span(), "expected `name = \"...\"` or `description = \"...\"`"));
            }
        }
        Ok(Self { name, description })
    }
}

/// Extract the text of `///` doc-comments on the function as the tool description.
fn doc_comment(attrs: &[Attribute]) -> Option<String> {
    let mut lines: Vec<String> = Vec::new();
    for a in attrs {
        if !a.path().is_ident("doc") { continue; }
        if let Meta::NameValue(nv) = &a.meta {
            if let syn::Expr::Lit(lit) = &nv.value {
                if let syn::Lit::Str(s) = &lit.lit {
                    lines.push(s.value().trim().to_string());
                }
            }
        }
    }
    if lines.is_empty() { None } else { Some(lines.join("\n")) }
}

#[proc_macro_attribute]
pub fn tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as ToolArgs);
    let mut func = parse_macro_input!(item as ItemFn);

    // Sanity: single argument, async, returns Result<T>.
    if func.sig.asyncness.is_none() {
        return Error::new(func.sig.span(), "#[tool] functions must be `async`")
            .to_compile_error()
            .into();
    }
    if func.sig.inputs.len() != 1 {
        return Error::new(
            func.sig.span(),
            "#[tool] functions must take exactly one argument (the args struct)",
        )
        .to_compile_error()
        .into();
    }

    let arg_ty: &Type = match func.sig.inputs.first().unwrap() {
        FnArg::Typed(PatType { ty, .. }) => ty,
        FnArg::Receiver(r) => {
            return Error::new(r.span(), "#[tool] functions cannot have a receiver")
                .to_compile_error()
                .into();
        }
    };

    let _out_ty = match &func.sig.output {
        ReturnType::Type(_, ty) => ty.clone(),
        ReturnType::Default => {
            return Error::new(
                func.sig.span(),
                "#[tool] functions must return `Result<T>` with a serializable T",
            )
            .to_compile_error()
            .into();
        }
    };

    let fn_ident = func.sig.ident.clone();
    let struct_ident = format_ident!("{}", camel_case(&fn_ident.to_string()));
    let tool_name = args.name.unwrap_or_else(|| fn_ident.to_string());
    let tool_description = args
        .description
        .or_else(|| doc_comment(&func.attrs))
        .unwrap_or_else(|| format!("Tool `{tool_name}`"));

    // Rename user function to an implementation symbol to avoid collision with the
    // generated struct methods.
    let impl_fn_ident = format_ident!("__litgraph_tool_impl_{}", fn_ident);
    func.sig.ident = impl_fn_ident.clone();

    // Fix visibility of the impl fn — keep whatever the user had. We don't touch it.

    let tool_name_lit = LitStr::new(&tool_name, Span::call_site());
    let tool_desc_lit = LitStr::new(&tool_description, Span::call_site());

    let expanded = quote! {
        #func

        #[allow(non_camel_case_types)]
        #[doc = concat!("Generated tool struct for `", stringify!(#fn_ident), "`.")]
        pub struct #struct_ident;

        impl #struct_ident {
            pub fn schema() -> ::litgraph_core::tool::ToolSchema {
                let settings = ::schemars::gen::SchemaSettings::draft07();
                let mut gen = settings.into_generator();
                let schema = gen.root_schema_for::<#arg_ty>();
                let parameters = ::serde_json::to_value(&schema.schema)
                    .unwrap_or(::serde_json::Value::Null);
                ::litgraph_core::tool::ToolSchema {
                    name: #tool_name_lit.into(),
                    description: #tool_desc_lit.into(),
                    parameters,
                }
            }
        }

        #[::async_trait::async_trait]
        impl ::litgraph_core::tool::Tool for #struct_ident {
            fn schema(&self) -> ::litgraph_core::tool::ToolSchema {
                Self::schema()
            }

            async fn run(&self, args: ::serde_json::Value) -> ::litgraph_core::Result<::serde_json::Value> {
                let parsed: #arg_ty = ::serde_json::from_value(args)
                    .map_err(|e| ::litgraph_core::Error::invalid(format!(
                        "tool `{}` args: {e}", #tool_name_lit
                    )))?;
                let out = #impl_fn_ident(parsed).await?;
                ::serde_json::to_value(out).map_err(::litgraph_core::Error::from)
            }
        }
    };

    expanded.into()
}

fn camel_case(snake: &str) -> String {
    let mut out = String::with_capacity(snake.len());
    let mut upper = true;
    for c in snake.chars() {
        if c == '_' { upper = true; continue; }
        if upper { out.extend(c.to_uppercase()); upper = false; }
        else { out.push(c); }
    }
    out
}

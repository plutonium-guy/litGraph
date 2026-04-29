//! Graph visualisation — Mermaid + ASCII renderers.
//!
//! Why this lives next to the executor: the renderers walk the same `nodes` /
//! `edges` maps the scheduler does. Keeping them in-crate avoids duplicating
//! the structural model.
//!
//! Output is purely structural: node names + edge kinds. Conditional-edge
//! routers can branch dynamically at runtime, so we render them as a single
//! "?" outgoing edge. Subgraph nodes appear as plain nodes (the wrapper layer
//! decides whether to recurse).

use std::collections::BTreeSet;
use std::fmt::Write;

use serde::{Serialize, de::DeserializeOwned};

use crate::graph::{StateGraph, START, END};

/// Internal structural view used by both renderers. Sorted-stable so output
/// is deterministic — important for snapshot tests and version-control diffs.
#[derive(Debug, Clone)]
struct GraphSkeleton {
    nodes: BTreeSet<String>,
    static_edges: BTreeSet<(String, String)>,
    /// Sources whose outgoing edges include at least one `Conditional`.
    conditional_sources: BTreeSet<String>,
}

fn skeleton<S>(g: &StateGraph<S>) -> GraphSkeleton
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    let mut nodes: BTreeSet<String> = g.nodes.keys().cloned().collect();
    let mut static_edges: BTreeSet<(String, String)> = BTreeSet::new();
    let mut conditional_sources: BTreeSet<String> = BTreeSet::new();

    for (from, edges) in &g.edges {
        nodes.insert(from.clone());
        for e in edges {
            match e {
                crate::graph::EdgeKind::Static(to) => {
                    nodes.insert(to.clone());
                    static_edges.insert((from.clone(), to.clone()));
                }
                crate::graph::EdgeKind::Conditional(_) => {
                    conditional_sources.insert(from.clone());
                }
            }
        }
    }

    GraphSkeleton {
        nodes,
        static_edges,
        conditional_sources,
    }
}

fn mermaid_id(name: &str) -> String {
    // Replace characters Mermaid trips on; keep alnum + underscore.
    let mut out = String::with_capacity(name.len());
    for c in name.chars() {
        if c.is_ascii_alphanumeric() || c == '_' {
            out.push(c);
        } else {
            out.push('_');
        }
    }
    if out.chars().next().map(|c| c.is_ascii_digit()).unwrap_or(true) {
        out.insert(0, 'n');
    }
    out
}

fn render_label(name: &str) -> String {
    if name == START {
        "Start".to_string()
    } else if name == END {
        "End".to_string()
    } else {
        name.to_string()
    }
}

fn render_mermaid(sk: &GraphSkeleton) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "graph TD");

    // Node declarations — square-bracket label, with shape per role.
    for name in &sk.nodes {
        let id = mermaid_id(name);
        let label = render_label(name);
        let shape = if name == START {
            format!("(({label}))")
        } else if name == END {
            format!("(((End)))")
        } else {
            format!("[{label}]")
        };
        let _ = writeln!(out, "    {id}{shape}");
    }

    for (from, to) in &sk.static_edges {
        let _ = writeln!(out, "    {} --> {}", mermaid_id(from), mermaid_id(to));
    }

    // Conditional fan-outs render as a single dashed edge to a "?" diamond
    // because the actual targets depend on runtime state.
    for src in &sk.conditional_sources {
        let qid = format!("{}_q", mermaid_id(src));
        let _ = writeln!(out, "    {qid}{{?}}");
        let _ = writeln!(out, "    {} -.-> {qid}", mermaid_id(src));
    }

    out
}

fn render_ascii(sk: &GraphSkeleton) -> String {
    // Adjacency-list summary; keep output stable + scannable. Conditional
    // edges marked with "(?)" so debug logs don't lie about static structure.
    let mut out = String::new();
    let _ = writeln!(out, "litgraph StateGraph");
    let _ = writeln!(out, "  nodes ({}):", sk.nodes.len());
    for n in &sk.nodes {
        let _ = writeln!(out, "    - {}", render_label(n));
    }
    let _ = writeln!(out, "  edges:");
    for (from, to) in &sk.static_edges {
        let _ = writeln!(out, "    {} -> {}", render_label(from), render_label(to));
    }
    for src in &sk.conditional_sources {
        let _ = writeln!(out, "    {} -> ? (conditional)", render_label(src));
    }
    out
}

impl<S> StateGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    /// Render the graph as a Mermaid `graph TD` flowchart. Conditional edges
    /// appear as dashed arrows to a `{?}` diamond — their concrete targets
    /// can only be known at runtime.
    pub fn to_mermaid(&self) -> String {
        render_mermaid(&skeleton(self))
    }

    /// Render the graph as a plain-text adjacency listing. Useful for logs
    /// or quick `println!` debugging without a Mermaid renderer.
    pub fn to_ascii(&self) -> String {
        render_ascii(&skeleton(self))
    }
}

impl<S> crate::graph::CompiledGraph<S>
where
    S: Serialize + DeserializeOwned + Clone + Send + Sync + 'static,
{
    /// Mermaid `graph TD` rendering of the compiled-graph topology.
    pub fn to_mermaid(&self) -> String {
        self.inner.to_mermaid()
    }

    pub fn to_ascii(&self) -> String {
        self.inner.to_ascii()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{StateGraph, END};
    use crate::node::NodeOutput;
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Clone, Debug, Default, Serialize, Deserialize)]
    struct S {
        n: i64,
    }

    fn build_linear() -> StateGraph<S> {
        let mut g = StateGraph::<S>::new();
        g.add_node("a", |s: S| async move {
            NodeOutput::update(json!({"n": s.n + 1}))
        });
        g.add_node("b", |s: S| async move {
            NodeOutput::update(json!({"n": s.n + 1}))
        });
        g.set_entry("a").add_edge("a", "b").add_edge("b", END);
        g
    }

    #[test]
    fn mermaid_includes_header_and_all_nodes() {
        let g = build_linear();
        let m = g.to_mermaid();
        assert!(m.starts_with("graph TD"), "missing mermaid header: {m}");
        assert!(m.contains("[a]"));
        assert!(m.contains("[b]"));
        assert!(m.contains("Start"));
        assert!(m.contains("End"));
    }

    #[test]
    fn mermaid_renders_static_edges() {
        let g = build_linear();
        let m = g.to_mermaid();
        assert!(m.contains("__start__ --> a") || m.contains("n__start__ --> a"));
        assert!(m.contains("a --> b"));
        assert!(m.contains("b --> __end__") || m.contains("b --> n__end__"));
    }

    #[test]
    fn mermaid_marks_conditional_edges_with_diamond() {
        let mut g = StateGraph::<S>::new();
        g.add_node("router", |s: S| async move {
            NodeOutput::update(serde_json::to_value(s).unwrap())
        });
        g.add_node("leaf", |s: S| async move {
            NodeOutput::update(serde_json::to_value(s).unwrap())
        });
        g.set_entry("router");
        g.add_conditional_edges("router", |_s: &S| vec!["leaf".into()]);
        g.add_edge("leaf", END);
        let m = g.to_mermaid();
        assert!(m.contains("{?}"), "expected conditional diamond: {m}");
        assert!(m.contains("router_q"));
    }

    #[test]
    fn ascii_listing_lists_nodes_and_edges() {
        let g = build_linear();
        let ascii = g.to_ascii();
        assert!(ascii.contains("nodes ("), "no node count: {ascii}");
        assert!(ascii.contains("Start -> a"));
        assert!(ascii.contains("a -> b"));
        assert!(ascii.contains("b -> End"));
    }

    #[test]
    fn ascii_marks_conditional() {
        let mut g = StateGraph::<S>::new();
        g.add_node("r", |s: S| async move {
            NodeOutput::update(serde_json::to_value(s).unwrap())
        });
        g.set_entry("r");
        g.add_conditional_edges("r", |_| vec![END.into()]);
        let ascii = g.to_ascii();
        assert!(ascii.contains("conditional"), "{ascii}");
    }

    #[test]
    fn mermaid_id_safely_quotes_special_chars() {
        assert_eq!(mermaid_id("foo"), "foo");
        assert_eq!(mermaid_id("foo-bar"), "foo_bar");
        assert_eq!(mermaid_id("foo bar"), "foo_bar");
        assert_eq!(mermaid_id("__start__"), "__start__");
        assert_eq!(mermaid_id("1node"), "n1node");
    }

    #[test]
    fn output_is_deterministic_across_runs() {
        let g1 = build_linear();
        let g2 = build_linear();
        assert_eq!(g1.to_mermaid(), g2.to_mermaid());
        assert_eq!(g1.to_ascii(), g2.to_ascii());
    }

    #[test]
    fn compiled_graph_delegates_to_state_graph() {
        let g = build_linear();
        let m_state = g.to_mermaid();
        let compiled = g.compile().unwrap();
        assert_eq!(compiled.to_mermaid(), m_state);
    }
}

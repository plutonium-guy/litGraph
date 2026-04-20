//! Shared benchmark utilities.

pub fn lorem_words(n: usize) -> String {
    let stock = [
        "rust", "tokio", "async", "await", "graph", "state", "node", "edge",
        "superstep", "parallel", "reducer", "checkpoint", "interrupt", "agent",
        "tool", "stream", "token", "embedding", "vector", "retrieval", "chunk",
        "splitter", "loader", "document", "prompt", "model", "cache", "observability",
    ];
    (0..n).map(|i| stock[i % stock.len()]).collect::<Vec<_>>().join(" ")
}

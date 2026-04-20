//! How fast is the StateGraph Kahn scheduler? Fan-out N branches, join, assert.
//!
//! The key comparison (not in the bench itself): LangGraph's Python equivalent of
//! an N-way `Send` fan-out serializes callbacks through asyncio + GIL. The Rust
//! version runs N branches as tokio tasks on a worker pool with zero GIL contention.
//! This bench shows the scheduler overhead in isolation — it should be microseconds
//! per node for the trivial nodes used here.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litgraph_graph::{END, NodeOutput, START, StateGraph};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct State {
    #[serde(default)]
    items: Vec<u32>,
}

fn build_fanout(n: usize) -> litgraph_graph::CompiledGraph<State> {
    let mut g = StateGraph::<State>::new();
    for i in 0..n {
        let name = format!("n{i}");
        let i_u = i as u32;
        g.add_node(&name, move |_s: State| async move {
            NodeOutput::update(State { items: vec![i_u] })
        });
        g.add_edge(START, name.clone());
        g.add_edge(name, END);
    }
    g.with_max_parallel(n.max(1)).compile().unwrap()
}

fn fanout_bench(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("graph_fanout");
    group.measurement_time(Duration::from_secs(4));
    for n in [1usize, 4, 16, 64].iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let compiled = build_fanout(*n);
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let s = compiled.invoke(State::default(), None).await.unwrap();
                    criterion::black_box(s);
                });
            });
        });
    }
    group.finish();
}

criterion_group!(benches, fanout_bench);
criterion_main!(benches);

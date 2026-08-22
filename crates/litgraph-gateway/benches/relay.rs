use std::sync::Arc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use litgraph_gateway::testing::{bench_state, invoke_once, relay_n_chunks};
use tokio::runtime::Runtime;

fn non_streaming(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let state = Arc::new(bench_state());
    c.bench_function("non_streaming_round_trip", |bencher| {
        bencher
            .to_async(&runtime)
            .iter(|| invoke_once(state.clone()));
    });
}

fn streaming(c: &mut Criterion) {
    let runtime = Runtime::new().unwrap();
    let mut group = c.benchmark_group("sse_relay");
    for chunks in [100usize, 1_000] {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunks),
            &chunks,
            |bencher, &n| {
                bencher.to_async(&runtime).iter(|| relay_n_chunks(n));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, non_streaming, streaming);
criterion_main!(benches);

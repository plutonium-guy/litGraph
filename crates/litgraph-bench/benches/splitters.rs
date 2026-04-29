//! Recursive char splitter — single-doc and batch (rayon-parallel) throughput.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litgraph_core::Document;
use litgraph_splitters::{RecursiveCharacterSplitter, Splitter};

fn single_doc(c: &mut Criterion) {
    let text = litgraph_bench::lorem_words(20_000);
    let sp = RecursiveCharacterSplitter::new(1000, 200);
    let mut group = c.benchmark_group("split_single");
    group.throughput(Throughput::Bytes(text.len() as u64));
    group.bench_function("20k_words", |b| {
        b.iter(|| {
            let chunks = sp.split_text(&text);
            criterion::black_box(chunks);
        });
    });
    group.finish();
}

fn batch_parallel(c: &mut Criterion) {
    let sp = RecursiveCharacterSplitter::new(1000, 200);
    let mut group = c.benchmark_group("split_batch");
    for n in [100usize, 1_000].iter() {
        let docs: Vec<Document> = (0..*n)
            .map(|i| Document::new(litgraph_bench::lorem_words(500)).with_id(format!("d{i}")))
            .collect();
        group.throughput(Throughput::Elements(*n as u64));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let out = sp.split_documents(&docs);
                criterion::black_box(out);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, single_doc, batch_parallel);
criterion_main!(benches);

//! BM25 index throughput — indexing + search on a small corpus.
//!
//! Note: Rust BM25 scoring runs rayon-parallel across documents; Python BM25 libs
//! (rank_bm25, Whoosh) typically don't parallelize the score loop. Scaling to larger
//! corpora amplifies the gap.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litgraph_core::Document;
use litgraph_retrieval::Bm25Index;

fn make_corpus(n: usize) -> Vec<Document> {
    (0..n)
        .map(|i| {
            Document::new(litgraph_bench::lorem_words(40 + (i % 60)))
                .with_id(format!("d{i}"))
        })
        .collect()
}

fn bm25_index(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_index");
    for n in [1_000usize, 10_000].iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let corpus = make_corpus(*n);
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let idx = Bm25Index::new();
                idx.add(corpus.clone()).unwrap();
                criterion::black_box(idx.len());
            });
        });
    }
    group.finish();
}

fn bm25_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bm25_search");
    for n in [1_000usize, 10_000, 50_000].iter() {
        group.throughput(Throughput::Elements(*n as u64));
        let idx = Bm25Index::new();
        idx.add(make_corpus(*n)).unwrap();
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| {
                let r = idx.search("graph parallel tokio", 10).unwrap();
                criterion::black_box(r);
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bm25_index, bm25_search);
criterion_main!(benches);

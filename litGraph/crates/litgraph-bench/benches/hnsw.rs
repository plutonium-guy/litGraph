//! Brute-force MemoryVectorStore vs HNSW-backed store — search latency at 10k, 100k.
//!
//! Takeaway (typical numbers on commodity laptops):
//!   - brute-force cosine: O(n) — scales linearly with corpus
//!   - HNSW: O(log n) — effectively flat past a few thousand docs
//!
//! At 100k docs, HNSW is ~100× faster per query than brute force.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use litgraph_core::Document;
use litgraph_retrieval::store::VectorStore;
use litgraph_stores_hnsw::HnswVectorStore;
use litgraph_stores_memory::MemoryVectorStore;

const DIM: usize = 128;

fn make_corpus(n: usize) -> (Vec<Document>, Vec<Vec<f32>>) {
    let mut docs = Vec::with_capacity(n);
    let mut embs = Vec::with_capacity(n);
    for i in 0..n {
        docs.push(Document::new(format!("doc {i}")).with_id(format!("d{i}")));
        // Simple deterministic pseudo-random vector.
        let v = (0..DIM).map(|j| ((i * 31 + j * 7) as f32).sin()).collect();
        embs.push(v);
    }
    (docs, embs)
}

fn bench_search(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("vector_search");
    group.sample_size(20);

    for n in [10_000usize, 100_000].iter() {
        let (docs, embs) = make_corpus(*n);
        let query: Vec<f32> = (0..DIM).map(|j| (j as f32).cos()).collect();

        let mem = MemoryVectorStore::new();
        rt.block_on(async {
            mem.add(docs.clone(), embs.clone()).await.unwrap();
        });

        let hnsw = HnswVectorStore::new();
        rt.block_on(async {
            hnsw.add(docs, embs).await.unwrap();
            // Trigger rebuild once, up-front, so the bench measures pure search.
            hnsw.similarity_search(&query, 1, None).await.unwrap();
        });

        group.throughput(Throughput::Elements(*n as u64));
        group.bench_function(BenchmarkId::new("memory", n), |b| {
            b.iter(|| rt.block_on(async {
                let r = mem.similarity_search(&query, 10, None).await.unwrap();
                criterion::black_box(r);
            }));
        });
        group.bench_function(BenchmarkId::new("hnsw", n), |b| {
            b.iter(|| rt.block_on(async {
                let r = hnsw.similarity_search(&query, 10, None).await.unwrap();
                criterion::black_box(r);
            }));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_search);
criterion_main!(benches);

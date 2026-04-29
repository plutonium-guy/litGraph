//! Cache key hashing + memory-cache roundtrip throughput.

use criterion::{Criterion, criterion_group, criterion_main};
use litgraph_cache::{Cache, MemoryCache, cache_key};
use litgraph_core::model::{FinishReason, TokenUsage};
use litgraph_core::{ChatOptions, ChatResponse, Message};

fn key_hashing(c: &mut Criterion) {
    let msgs = vec![
        Message::system("be terse"),
        Message::user(litgraph_bench::lorem_words(200)),
    ];
    let opts = ChatOptions::default();
    c.bench_function("cache_key_200words", |b| {
        b.iter(|| {
            let k = cache_key("gpt-5", &msgs, &opts);
            criterion::black_box(k);
        });
    });
}

fn mem_cache_roundtrip(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    let cache = MemoryCache::new(10_000);
    let resp = ChatResponse {
        message: Message::assistant("hello"),
        finish_reason: FinishReason::Stop,
        usage: TokenUsage::default(),
        model: "gpt-5".into(),
    };
    let key = "k";

    c.bench_function("mem_cache_put_get", |b| {
        b.iter(|| {
            rt.block_on(async {
                cache.put(key, resp.clone()).await.unwrap();
                let got = cache.get(key).await.unwrap();
                criterion::black_box(got);
            });
        });
    });
}

criterion_group!(benches, key_hashing, mem_cache_roundtrip);
criterion_main!(benches);

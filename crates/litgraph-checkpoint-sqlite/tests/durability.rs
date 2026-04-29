use std::sync::Arc;

use litgraph_checkpoint_sqlite::SqliteCheckpointer;
use litgraph_graph::{END, NodeOutput, START, StateGraph};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Counter { n: u32 }

#[tokio::test]
async fn interrupt_and_resume_via_sqlite_survives_new_compile() {
    let tmp = tempdir_like();
    let db = tmp.join("cp.sqlite");

    // --- run A: start fresh, interrupt_before("step2"), expect an interrupt.
    let run_a = {
        let cp: Arc<dyn litgraph_graph::Checkpointer> =
            Arc::new(SqliteCheckpointer::open(&db).unwrap());

        let mut g = StateGraph::<Counter>::new();
        g.add_node("step1", |s: Counter| async move {
            NodeOutput::update(Counter { n: s.n + 1 })
        });
        g.add_node("step2", |s: Counter| async move {
            NodeOutput::update(Counter { n: s.n + 10 })
        });
        g.add_edge(START, "step1");
        g.add_edge("step1", "step2");
        g.add_edge("step2", END);
        g.interrupt_before("step2");

        let compiled = g.compile().unwrap().with_checkpointer(cp);
        let err = compiled.invoke(Counter::default(), Some("durable1".into())).await.unwrap_err();
        assert!(matches!(err, litgraph_graph::GraphError::Interrupted(ref n) if n == "step2"));
    };
    let _ = run_a;

    // --- run B: brand new CompiledGraph + brand new SQLite handle on same file. Resume.
    let cp: Arc<dyn litgraph_graph::Checkpointer> =
        Arc::new(SqliteCheckpointer::open(&db).unwrap());
    let mut g = StateGraph::<Counter>::new();
    g.add_node("step1", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_node("step2", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 10 })
    });
    g.add_edge(START, "step1");
    g.add_edge("step1", "step2");
    g.add_edge("step2", END);
    g.interrupt_before("step2");
    let compiled = g.compile().unwrap().with_checkpointer(cp);
    let final_state = compiled.resume("durable1".into(), serde_json::json!({})).await.unwrap();
    assert_eq!(final_state.n, 11);
}

fn tempdir_like() -> std::path::PathBuf {
    let base = std::env::temp_dir().join(format!("litgraph-cp-{}", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    base
}

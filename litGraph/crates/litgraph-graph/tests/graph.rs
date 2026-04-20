use litgraph_graph::{END, NodeOutput, START, StateGraph};
use serde::{Deserialize, Serialize};

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Counter {
    n: u32,
}

#[tokio::test]
async fn single_node_runs() {
    let mut g = StateGraph::<Counter>::new();
    g.add_node("inc", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_edge(START, "inc");
    g.add_edge("inc", END);
    let compiled = g.compile().unwrap();
    let s = compiled.invoke(Counter::default(), None).await.unwrap();
    assert_eq!(s.n, 1);
}

#[tokio::test]
async fn chained_nodes() {
    let mut g = StateGraph::<Counter>::new();
    g.add_node("a", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_node("b", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 10 })
    });
    g.add_edge(START, "a");
    g.add_edge("a", "b");
    g.add_edge("b", END);
    let s = g.compile().unwrap().invoke(Counter::default(), None).await.unwrap();
    assert_eq!(s.n, 11);
}

#[tokio::test]
async fn conditional_edges_route() {
    let mut g = StateGraph::<Counter>::new();
    g.add_node("start", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_node("big", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 100 })
    });
    g.add_node("small", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_edge(START, "start");
    g.add_conditional_edges("start", |s: &Counter| {
        vec![if s.n >= 5 { "big".into() } else { "small".into() }]
    });
    g.add_edge("big", END);
    g.add_edge("small", END);

    let s = g.compile().unwrap().invoke(Counter { n: 10 }, None).await.unwrap();
    assert_eq!(s.n, 11 + 100);
}

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct Bag {
    #[serde(default)]
    items: Vec<u32>,
}

#[tokio::test]
async fn parallel_fanout_reducer_appends() {
    // Two branches run concurrently in one superstep; `merge_append` reducer
    // concatenates the arrays they emit. Nodes return *deltas*, not full state.
    let mut g = StateGraph::<Bag>::new();
    g.add_node("a", |_s: Bag| async move {
        NodeOutput::update(Bag { items: vec![1] })
    });
    g.add_node("b", |_s: Bag| async move {
        NodeOutput::update(Bag { items: vec![2] })
    });
    g.add_node("join", |_s: Bag| async move {
        // Sink node — no delta, just marks completion.
        NodeOutput::empty()
    });
    g.add_edge(START, "a");
    g.add_edge(START, "b");
    g.add_edge("a", "join");
    g.add_edge("b", "join");
    g.add_edge("join", END);

    let s = g.compile().unwrap().invoke(Bag::default(), None).await.unwrap();
    let mut items = s.items;
    items.sort();
    assert_eq!(items, vec![1, 2]);
}

#[tokio::test]
async fn stream_emits_events() {
    use litgraph_graph::GraphEvent;

    let mut g = StateGraph::<Counter>::new();
    g.add_node("inc", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_edge(START, "inc");
    g.add_edge("inc", END);
    let compiled = g.compile().unwrap();
    let mut rx = compiled.stream(Counter::default(), Some("t1".into()));
    let mut seen_end = false;
    let mut saw_node_end = false;
    while let Some(ev) = rx.recv().await {
        match ev {
            GraphEvent::NodeEnd { node, .. } if node == "inc" => saw_node_end = true,
            GraphEvent::GraphEnd { .. } => { seen_end = true; break; }
            _ => {}
        }
    }
    assert!(saw_node_end);
    assert!(seen_end);
}

#[tokio::test]
async fn interrupt_before_checkpoints_and_resume_completes() {
    let mut g = StateGraph::<Counter>::new();
    g.add_node("inc", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    g.add_edge(START, "inc");
    g.add_edge("inc", END);
    g.interrupt_before("inc");

    let compiled = g.compile().unwrap();
    let err = compiled.invoke(Counter::default(), Some("t2".into())).await.unwrap_err();
    assert!(matches!(err, litgraph_graph::GraphError::Interrupted(ref n) if n == "inc"));

    // Resume with no extra state update.
    let final_state = compiled
        .resume("t2".into(), serde_json::json!({}))
        .await
        .unwrap();
    assert_eq!(final_state.n, 1);
}

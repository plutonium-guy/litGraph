use litgraph_graph::{Command, END, NodeOutput, START, StateGraph};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

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

// ---------- LangGraph-style Send fan-out / map-reduce (iter 77) ----------

#[derive(Clone, Default, Debug, Serialize, Deserialize)]
struct MapReduceState {
    /// Items to map over (set by the splitter).
    #[serde(default)]
    items: Vec<i64>,
    /// Per-item input the worker reads. Each Send carries one of these.
    #[serde(default)]
    item: Option<i64>,
    /// Reduced outputs from all workers, accumulated via merge_append.
    #[serde(default)]
    results: Vec<i64>,
}

#[tokio::test]
async fn send_fan_out_runs_worker_once_per_send_with_per_item_state() {
    // Splitter emits 4 Send commands, each carrying a distinct `item`.
    // Worker reads `state.item`, doubles it, returns as a 1-element results vec.
    // The reducer (merge_append on `results`) collects all 4 outputs.
    //
    // CRITICAL invariant: each worker invocation must see ONLY its own item
    // (not a merged-together state with all 4 items in the same field).
    let saw_items: Arc<std::sync::Mutex<Vec<i64>>> = Arc::new(std::sync::Mutex::new(Vec::new()));
    let invocations = Arc::new(AtomicUsize::new(0));

    let mut g = StateGraph::<MapReduceState>::new();

    g.add_node("split", |_s: MapReduceState| async move {
        let items = vec![1, 2, 3, 4];
        // Seed `items` then fan out one Send per item, each with its own `item`.
        let mut out = NodeOutput::update(json!({ "items": items.clone() }));
        for i in &items {
            out = out.send(Command::to("worker").with(json!({ "item": i })));
        }
        // No goto — sends carry routing.
        out
    });

    let saw = saw_items.clone();
    let invs = invocations.clone();
    g.add_node("worker", move |s: MapReduceState| {
        let saw = saw.clone();
        let invs = invs.clone();
        async move {
            invs.fetch_add(1, Ordering::SeqCst);
            let item = s.item.expect("worker must see its per-Send item");
            saw.lock().unwrap().push(item);
            // Critical assertion: each worker invocation sees its OWN item
            // only. Without per-fork state, all 4 sends would have
            // collapsed `state.item` to the last value (whichever the
            // reducer overwrote with last) — so all 4 workers would see
            // the same item.
            // Emit a single result.
            NodeOutput::update(json!({ "results": [item * 2] })).goto("join")
        }
    });

    g.add_node("join", |_s: MapReduceState| async move {
        NodeOutput::empty()
    });

    g.add_edge(START, "split");
    g.add_edge("join", END);

    // `with_reducer` consumes the graph, so do it last. Use merge_append on
    // `results` array (standard reducer for collected map outputs); `items`
    // and `item` use replace semantics.
    let g = g.with_reducer(|mut state: MapReduceState, update: serde_json::Value| {
        if let Some(obj) = update.as_object() {
            if let Some(item) = obj.get("item") {
                state.item = serde_json::from_value(item.clone()).ok();
            }
            if let Some(items) = obj.get("items") {
                state.items = serde_json::from_value(items.clone()).unwrap_or_default();
            }
            if let Some(arr) = obj.get("results").and_then(|v| v.as_array()) {
                for v in arr {
                    if let Some(n) = v.as_i64() {
                        state.results.push(n);
                    }
                }
            }
        }
        Ok(state)
    });

    let final_state = g
        .compile()
        .unwrap()
        .invoke(MapReduceState::default(), None)
        .await
        .unwrap();

    // Worker ran once per Send.
    assert_eq!(invocations.load(Ordering::SeqCst), 4);
    let mut seen = saw_items.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, vec![1, 2, 3, 4]);
    // Reduce phase: all 4 doubled values present in any order.
    let mut results = final_state.results;
    results.sort();
    assert_eq!(results, vec![2, 4, 6, 8]);
}

#[tokio::test]
async fn send_does_not_pollute_shared_state_during_fan_out() {
    // Two parallel Sends to the same worker with different items. After they
    // finish, `state.item` should reflect the last-reduced one (or be None
    // if we don't merge it back), but during execution NEITHER sibling should
    // see the OTHER sibling's item.
    let mut g = StateGraph::<MapReduceState>::new();
    g.add_node("split", |_s: MapReduceState| async move {
        NodeOutput::empty()
            .send(Command::to("w").with(json!({ "item": 100 })))
            .send(Command::to("w").with(json!({ "item": 200 })))
    });
    g.add_node("w", |s: MapReduceState| async move {
        let item = s.item.expect("Send override must reach worker as state.item");
        // Echo back as a result. Each invocation sees ONLY its own item.
        NodeOutput::update(json!({ "results": [item] }))
    });
    g.add_edge(START, "split");

    let g = g.with_reducer(|mut state: MapReduceState, update: serde_json::Value| {
        if let Some(obj) = update.as_object() {
            if let Some(arr) = obj.get("results").and_then(|v| v.as_array()) {
                for v in arr {
                    if let Some(n) = v.as_i64() {
                        state.results.push(n);
                    }
                }
            }
            if let Some(item) = obj.get("item") {
                state.item = serde_json::from_value(item.clone()).ok();
            }
        }
        Ok(state)
    });

    let final_state = g
        .compile()
        .unwrap()
        .invoke(MapReduceState::default(), None)
        .await
        .unwrap();
    let mut r = final_state.results;
    r.sort();
    assert_eq!(r, vec![100, 200]);
}

// ---------- Subgraph composition (iter 80) ----------

#[tokio::test]
async fn subgraph_embedded_as_node_runs_to_completion_and_state_merges_back() {
    // Child: inc twice (n=0→1→2). Parent: (inc) then (child subgraph) then
    // (inc). Final n should be child-result (2) + 1 from parent's second inc
    // = 3. This proves the child ran AND its output flowed through the
    // parent's reducer as a state update.
    let mut child = StateGraph::<Counter>::new();
    child.add_node("c_inc_1", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    child.add_node("c_inc_2", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 1 })
    });
    child.add_edge(START, "c_inc_1");
    child.add_edge("c_inc_1", "c_inc_2");
    child.add_edge("c_inc_2", END);
    let child_compiled = Arc::new(child.compile().unwrap());

    let mut parent = StateGraph::<Counter>::new();
    parent.add_node("p_start", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 10 })  // 0 → 10
    });
    parent.add_subgraph("team", child_compiled);
    parent.add_node("p_end", |s: Counter| async move {
        NodeOutput::update(Counter { n: s.n + 100 })  // (10+2) → 112
    });
    parent.add_edge(START, "p_start");
    parent.add_edge("p_start", "team");
    parent.add_edge("team", "p_end");
    parent.add_edge("p_end", END);

    let final_state = parent
        .compile()
        .unwrap()
        .invoke(Counter::default(), None)
        .await
        .unwrap();
    // 0 → 10 (p_start) → 10+2=12 (team: subgraph ran c_inc_1 + c_inc_2) → 112 (p_end).
    assert_eq!(final_state.n, 112);
}

#[tokio::test]
async fn subgraph_errors_bubble_up_to_parent() {
    // Child that fails mid-execution; parent must surface the error rather
    // than silently continuing with stale state.
    let mut child = StateGraph::<Counter>::new();
    child.add_fallible_node("boom", |_s: Counter| async move {
        Err(litgraph_graph::GraphError::Other("child exploded".into()))
    });
    child.add_edge(START, "boom");
    child.add_edge("boom", END);
    let child_compiled = Arc::new(child.compile().unwrap());

    let mut parent = StateGraph::<Counter>::new();
    parent.add_subgraph("team", child_compiled);
    parent.add_edge(START, "team");
    parent.add_edge("team", END);

    let err = parent
        .compile()
        .unwrap()
        .invoke(Counter::default(), None)
        .await
        .unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("child exploded"), "got: {msg}");
}

#[tokio::test]
async fn checkpoint_preserves_next_sends_for_resume_after_interrupt() {
    // Validate the next_sends round-trip: a node fan-outs to "worker" then
    // the graph hits interrupt_before on the worker. On resume, both Sends
    // must run with their original per-item payloads (NOT collapsed into one).
    let mut g = StateGraph::<MapReduceState>::new();
    g.add_node("split", |_s: MapReduceState| async move {
        NodeOutput::empty()
            .send(Command::to("worker").with(json!({ "item": 7 })))
            .send(Command::to("worker").with(json!({ "item": 9 })))
    });
    g.add_node("worker", |s: MapReduceState| async move {
        NodeOutput::update(json!({ "results": [s.item.unwrap()] }))
    });
    g.add_edge(START, "split");
    g.interrupt_before("worker");

    let g = g.with_reducer(|mut state: MapReduceState, update: serde_json::Value| {
        if let Some(obj) = update.as_object() {
            if let Some(arr) = obj.get("results").and_then(|v| v.as_array()) {
                for v in arr {
                    if let Some(n) = v.as_i64() {
                        state.results.push(n);
                    }
                }
            }
            if let Some(item) = obj.get("item") {
                state.item = serde_json::from_value(item.clone()).ok();
            }
        }
        Ok(state)
    });
    let compiled = g.compile().unwrap();
    let err = compiled
        .invoke(MapReduceState::default(), Some("send-tid".into()))
        .await
        .unwrap_err();
    assert!(matches!(err, litgraph_graph::GraphError::Interrupted(ref n) if n == "worker"));

    let final_state = compiled
        .resume("send-tid".into(), json!({}))
        .await
        .unwrap();
    let mut r = final_state.results;
    r.sort();
    assert_eq!(r, vec![7, 9], "both Send payloads must survive checkpoint round-trip");
}

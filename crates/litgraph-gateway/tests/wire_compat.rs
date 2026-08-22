//! End-to-end wire compatibility over a real TCP listener.

#[tokio::test]
async fn streaming_and_non_streaming_round_trip_over_real_http() {
    let (address, plaintext_key, shutdown) = litgraph_gateway::testing::spawn_test_gateway().await;
    let client = reqwest::Client::new();
    let base = format!("http://{address}");

    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&plaintext_key)
        .json(&serde_json::json!({
            "model": "ollama",
            "messages": [{"role": "user", "content": "hi"}],
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = response.json().await.unwrap();
    assert_eq!(body["object"], "chat.completion");
    assert_eq!(body["model"], "ollama");

    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth(&plaintext_key)
        .json(&serde_json::json!({
            "model": "ollama",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": true,
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::OK);
    let body = response.text().await.unwrap();
    assert!(body.contains("chat.completion.chunk"));
    assert!(body.contains("[DONE]"));

    let response = client
        .post(format!("{base}/v1/chat/completions"))
        .bearer_auth("lg-sk-deadbeef.notarealsecret")
        .json(&serde_json::json!({"model": "ollama", "messages": []}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), reqwest::StatusCode::UNAUTHORIZED);
    let _ = shutdown.send(());
}

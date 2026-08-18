use super::*;

#[tokio::test]
async fn request_history_summarizes_image_payloads() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-request-history-images");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "vision-test-model");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "vision-test-model"}),
    )
    .await;

    tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "vision-test-model",
                "messages": [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}
                    ]
                }],
                "stream": false
            }))
            .unwrap();
    })
    .await
    .unwrap();

    let history = wait_for_history(port, "admin", "vision-test-model").await;
    assert_eq!(
        history["data"][0]["request"]["messages"][0]["content"][1]["image_url"]["url"]["omitted"],
        "data_url"
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn request_history_captures_streaming_chat_response() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-request-history-stream");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _state_guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    remote_admin_post(
        port,
        "/omni/model/select",
        "admin",
        json!({"model": "qwen3.5-4b-q4_k_m"}),
    )
    .await;

    let stream_text = tokio::task::spawn_blocking(move || {
        let response = ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "qwen3.5-4b-q4_k_m",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": true,
                "stream_options": {"include_usage": true}
            }))
            .unwrap();
        assert_eq!(response.status().as_u16(), 200);
        let mut text = String::new();
        response
            .into_body()
            .into_reader()
            .read_to_string(&mut text)
            .unwrap();
        text
    })
    .await
    .unwrap();
    assert!(stream_text.contains("fake"));
    assert!(stream_text.contains("[DONE]"));

    let history = wait_for_history(port, "admin", "qwen3.5-4b-q4_k_m").await;
    assert_eq!(history["data"][0]["status"], 200);
    assert_eq!(
        history["data"][0]["response"]["choices"][0]["message"]["content"],
        "fake backend"
    );
    assert_eq!(history["data"][0]["usage"]["prompt_tokens"], 3);
    assert_eq!(history["data"][0]["usage"]["completion_tokens"], 2);
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["mode"],
        "stream_passthrough"
    );
    assert_eq!(
        history["data"][0]["response"]["omniinfer_metrics"]["response_truncated"],
        false
    );

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[test]
fn normalizes_openai_usage_total_tokens() {
    let mut payload = json!({
        "choices": [{"message": {"content": "ok"}}],
        "usage": {"prompt_tokens": 11, "completion_tokens": 7}
    });
    normalize_openai_usage(&mut payload);
    assert_eq!(payload["usage"]["total_tokens"], 18);
}

#[test]
fn keeps_existing_openai_usage_total_tokens() {
    let mut payload = json!({
        "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 99}
    });
    normalize_openai_usage(&mut payload);
    assert_eq!(payload["usage"]["total_tokens"], 99);
}

#[tokio::test]
async fn remote_admins_can_load_multiple_models_with_owner_unload_policy() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-multi-model-owner");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-35b-a3b-q4_k_m");
    write_public_model_manifest(&root, "gemma-4-e4b-it-q4_k_m");
    install_fake_llama_server(&temp, external_test_backend_id());
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_keys: vec![
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminA".to_string(),
                    key: "admin-a".to_string(),
                },
                omniinfer_core::gateway_auth::GatewayAdminApiKey {
                    id: "adminB".to_string(),
                    key: "admin-b".to_string(),
                },
            ],
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let qwen_load = remote_admin_post(
        port,
        "/omni/model/load",
        "admin-a",
        json!({
            "model": "qwen3.5-35b-a3b-q4_k_m",
            "request_defaults": {"max_tokens": 11}
        }),
    )
    .await;
    assert_eq!(qwen_load["model"], "qwen3.5-35b-a3b-q4_k_m");
    assert_eq!(qwen_load["owner_admin_id"], "adminA");

    let gemma_load = remote_admin_post(
        port,
        "/omni/model/load",
        "admin-b",
        json!({
            "model": "gemma-4-e4b-it-q4_k_m",
            "request_defaults": {"max_tokens": 22}
        }),
    )
    .await;
    assert_eq!(gemma_load["model"], "gemma-4-e4b-it-q4_k_m");
    assert_eq!(gemma_load["owner_admin_id"], "adminB");
    assert_ne!(qwen_load["backend_port"], gemma_load["backend_port"]);

    let denied = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/omni/model/unload"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-b")
            .send_json(json!({"model": "qwen3.5-35b-a3b-q4_k_m"}))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(denied.to_string().contains("403"));

    let qwen_chat = remote_chat(port, "inference", "qwen3.5-35b-a3b-q4_k_m").await;
    assert_eq!(qwen_chat["model_echo"], "qwen3.5-35b-a3b-q4_k_m");
    assert_eq!(qwen_chat["max_tokens_echo"], 11);
    let gemma_chat = remote_chat(port, "inference", "gemma-4-e4b-it-q4_k_m").await;
    assert_eq!(gemma_chat["model_echo"], "gemma-4-e4b-it-q4_k_m");
    assert_eq!(gemma_chat["max_tokens_echo"], 22);

    let missing_chat = tokio::task::spawn_blocking(move || {
        ureq::post(format!("http://127.0.0.1:{port}/v1/chat/completions"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer inference")
            .send_json(json!({
                "model": "not-loaded-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": false
            }))
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(missing_chat.to_string().contains("404"));

    let default_chat = remote_chat_without_model(port, "inference").await;
    assert_eq!(default_chat["model_echo"], Value::Null);

    let loaded = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/loaded-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer admin-a")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    let loaded_body: Value = loaded.into_body().read_json().unwrap();
    assert_eq!(loaded_body["data"].as_array().unwrap().len(), 2);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}

#[tokio::test]
async fn gateway_hot_reloads_admin_keys_file() {
    let _env_lock = TEST_ENV_LOCK.lock().await;
    let temp = temp_root("rust-gateway-admin-keys-file");
    let root = temp.join("public_models");
    write_public_model_manifest(&root, "qwen3.5-4b-q4_k_m");
    let _guard = EnvGuard::set("OMNIINFER_RUST_STATE_ROOT", temp.display().to_string());

    let upstream = spawn_test_upstream().await;
    let gateway = spawn_test_gateway_with_public_root(
        GatewayAccessPolicy {
            api_key: "inference".to_string(),
            admin_api_key: "old-admin".to_string(),
            allow_remote_management: true,
            trust_proxy_headers: true,
            ..GatewayAccessPolicy::default()
        },
        Some(root),
    )
    .await;
    let port = gateway.port;

    let before = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer alice-key")
            .call()
            .unwrap_err()
    })
    .await
    .unwrap();
    assert!(before.to_string().contains("401"));

    let config_dir = temp.join(".local").join("config");
    std::fs::create_dir_all(&config_dir).unwrap();
    std::fs::write(
        config_dir.join("admin_keys.json"),
        r#"{"keys":{"alice":"alice-key","bob":"bob-key"}}"#,
    )
    .unwrap();

    let after = tokio::task::spawn_blocking(move || {
        ureq::get(format!("http://127.0.0.1:{port}/omni/public-models"))
            .header("CF-Connecting-IP", "203.0.113.10")
            .header("Authorization", "Bearer alice-key")
            .call()
            .unwrap()
    })
    .await
    .unwrap();
    assert_eq!(after.status().as_u16(), 200);

    gateway.stop().await;
    upstream.stop().await;
    std::fs::remove_dir_all(temp).ok();
}
